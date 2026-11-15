"""Experimental bucketed TensorRT AR neural graph, with native sampling.

Install, enable, disable and close only while the runner is idle. The plan
captures only the neural decoder and its KV writeback; sampling remains native.
Positions above the declared bucket dispatch to the previously installed graph.
Plans must carry the matching bucket name, except legacy unmarked 512 plans.
The fixed input mask is the native boolean mask; additive masks are unsupported.
No model, CUDA allocation or engine loading occurs merely by importing this file.
"""
from __future__ import annotations

import time
import weakref


class TensorRTARNeural:
    def __init__(self, runner, plan, *, bucket_size=512, native_attention=False, compact_host_state=False,
                 weight_l2_shim=None, weight_l2_role='ffn1'):
        if type(bucket_size) is not int or bucket_size not in (256, 512, 1024):
            raise ValueError("TensorRT AR bucket_size must be 256, 512 or 1024")
        if weight_l2_role not in ('ffn1', 'ffn2') or (weight_l2_role != 'ffn1' and weight_l2_shim is None):
            raise ValueError('AR weight L2 role must be ffn1 or ffn2 and ffn2 requires a shim')
        import torch
        import tensorrt as trt
        import triton
        import triton.language as tl
        from aniflive_tts.backend.trt_runtime import TensorRTRunner
        if native_attention:
            from . import trt_ar_native_attention as trt_ar_native_attention  # Register before engine deserialization.

        @triton.jit
        def writeback(addresses, keys, values, position, BLOCK: tl.constexpr):
            layer = tl.program_id(0)
            i = tl.arange(0, BLOCK)
            key_ptr = tl.load(addresses + 2 * layer).to(tl.pointer_type(tl.float16))
            value_ptr = tl.load(addresses + 2 * layer + 1).to(tl.pointer_type(tl.float16))
            pos = tl.load(position).to(tl.int64) - 1
            offset = (i // 32) * (2048 * 32) + pos * 32 + i % 32
            packed = layer * 512 + i
            tl.store(key_ptr + offset, tl.load(keys + packed))
            tl.store(value_ptr + offset, tl.load(values + packed))

        self.runner = runner
        if type(compact_host_state) is not bool:
            raise TypeError('compact_host_state must be a boolean')
        self.compact_host_state = compact_host_state
        self.weight_l2 = None
        self.weight_l2_inspection = None
        self._weight_l2_saved_limit = None
        self.weight_l2_last_saved = None
        self.weight_l2_last_restored = None
        self.weight_l2_enables = 0
        self.weight_l2_restorations = 0
        self._validated_session = None
        self._validated_request = None
        self._position_base = None
        self.static_session_validations = 0
        self.original_acquire_graph = runner._acquire_graph
        self.bucket_size = bucket_size
        self.enabled = False
        self.closed = False
        self._trt_counter = f"trt_{bucket_size}"
        self._original_counter = f"original_above_{bucket_size}"
        self.counts = {self._trt_counter: 0, self._original_counter: 0}
        self._require_idle()
        original_owner = getattr(self.original_acquire_graph, "__self__", None)
        if isinstance(original_owner, TensorRTARNeural):
            if not original_owner.enabled or original_owner.bucket_size <= bucket_size:
                raise ValueError("A chained TensorRT AR graph must have a larger enabled bucket")
        decoder = runner.decoder
        if (runner.max_batch_size != 1 or runner.device.type != "cuda"
                or not decoder.__class__.__module__.endswith(".torch_static_cuda_graph")
                or (decoder.n_layer, decoder.n_head, decoder.head_dim, decoder.max_seq_length)
                != (24, 16, 32, 2048)
                or not runner._should_use_graph() or 1 not in runner.graph_cache):
            raise ValueError("TensorRT AR requires the warmed single-slot static CUDA decoder")
        self._require_boolean_graph()
        self.mask = runner.extra_buffers.get("attn_mask")
        expected = ((runner.xy_pos_buf, (1, 1, 512), torch.float16),
                    (runner.xy_dec_buf, (1, 1, 512), torch.float16),
                    (runner.input_pos_buf, (1,), torch.int32),
                    (self.mask, (1, 16, 1, 2048), torch.bool))
        for tensor, shape, dtype in expected:
            if (tensor is None or tuple(tensor.shape) != shape or tensor.dtype != dtype
                    or tensor.device != runner.device or not tensor.is_contiguous()):
                raise ValueError("Unexpected AR input/output buffer layout")
        self.kv = tuple((key, value) for key, value in runner.kv_cache_buf)
        if len(self.kv) != 24:
            raise ValueError("Expected exactly 24 native KV layers")
        for pair in self.kv:
            for tensor in pair:
                if (tuple(tensor.shape) != (1, 16, 2048, 32) or tensor.dtype != torch.float16
                        or tensor.device != runner.device or not tensor.is_contiguous()):
                    raise ValueError("Unexpected native contiguous FP16 HND KV buffer")

        self.trt = TensorRTRunner(plan, device=str(runner.device))
        # A plan compiled with explicit GEMV pointer divisibility must prove
        # that condition on every created graph before it can be selected.
        import json
        import os
        from pathlib import Path
        plan_report = Path(plan).resolve(strict=True).parent / 'report.json'
        plan_metadata = json.loads(plan_report.read_text(encoding='utf-8')) if plan_report.is_file() else {}
        self.requires_prefix_mask = plan_metadata.get('requires_fused_native_prefix_mask', False)
        if type(self.requires_prefix_mask) is not bool:
            raise ValueError('Invalid prefix-mask contract in AR plan metadata')
        self.prefix_guard_fallbacks = 0
        grouped_alignment = plan_metadata.get('requires_grouped_pointer_alignment_bytes')
        if grouped_alignment not in (None, 16):
            raise ValueError('Unsupported grouped-attention pointer-alignment requirement')
        self.grouped_alignment_guard = os.environ.get('PERF_AR_GROUPED_ATTENTION_ALIGNMENT_GUARD') if grouped_alignment else None
        self.grouped_alignment_validated = False
        if grouped_alignment and (not self.grouped_alignment_guard or weight_l2_shim is None):
            raise ValueError('Grouped-attention plan requires its graph alignment guard and L2 shim')
        if grouped_alignment:
            import hashlib
            guard_path = Path(self.grouped_alignment_guard).resolve(strict=True)
            guard_digest = hashlib.sha256(guard_path.read_bytes()).hexdigest()
            if guard_digest != plan_metadata.get('grouped_alignment_guard_sha256'):
                raise ValueError('Grouped-attention guard does not match plan metadata')
        native_k8_alignment = plan_metadata.get('requires_native_k8_pointer_alignment_bytes')
        if native_k8_alignment not in (None, 16):
            raise ValueError('Unsupported native-K8 attention pointer-alignment requirement')
        self.native_k8_alignment_guard = os.environ.get('PERF_AR_NATIVE_K8_ALIGNMENT_GUARD') if native_k8_alignment else None
        self.native_k8_alignment_validated = False
        if native_k8_alignment and (not self.native_k8_alignment_guard or weight_l2_shim is None):
            raise ValueError('Native-K8 attention plan requires its graph alignment guard and L2 shim')
        if native_k8_alignment:
            if plan_metadata.get('native_k8_alignment_guard_version') not in (1, 2):
                raise ValueError('Unsupported native-K8 alignment guard version')
            import hashlib
            guard_path = Path(self.native_k8_alignment_guard).resolve(strict=True)
            if hashlib.sha256(guard_path.read_bytes()).hexdigest() != plan_metadata.get('native_k8_alignment_guard_sha256'):
                raise ValueError('Native-K8 attention guard does not match plan metadata')
        required_alignment = plan_metadata.get('requires_gemv_pointer_alignment_bytes')
        evenodd_alignment = plan_metadata.get('requires_evenodd_ffn2_pointer_alignment_bytes')
        if evenodd_alignment not in (None, 16):
            raise ValueError('Unsupported even/odd FFN2 pointer alignment requirement')
        self.evenodd_alignment_guard = os.environ.get('PERF_AR_EVENODD_ALIGNMENT_GUARD') if evenodd_alignment else None
        self.evenodd_alignment_validated = False
        if evenodd_alignment and (not self.evenodd_alignment_guard or weight_l2_shim is None):
            raise ValueError('Even/odd FFN2 plan requires its graph alignment guard and L2 shim')
        if required_alignment not in (None, 16):
            raise ValueError('Unsupported GEMV pointer-alignment requirement')
        self.gemv_alignment_guard = os.environ.get('PERF_AR_GEMV_ALIGNMENT_GUARD') if required_alignment else None
        self.gemv_alignment_validated = False
        if required_alignment and (not self.gemv_alignment_guard or weight_l2_shim is None):
            raise ValueError('Aligned GEMV plan requires its graph alignment guard and L2 shim')
        if weight_l2_shim is not None:
            import hashlib
            import json
            from pathlib import Path
            from .ar_weight_l2 import WeightL2
            plan_path = Path(plan).resolve(strict=True)
            report_path = plan_path.parent / 'report.json'
            expected = json.loads(report_path.read_text(encoding='utf-8'))['build']['sha256']
            digest = hashlib.sha256()
            with plan_path.open('rb') as f:
                for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError('L2 plan does not match the qualified build report')
            self.weight_l2 = WeightL2(weight_l2_shim, report_path, role=weight_l2_role)
        engine_name = self.trt.engine.name or ""
        expected_name = f"v5-ar-neural-bucket{bucket_size}"
        marker_present = "v5-ar-neural-bucket" in engine_name
        # Legacy 512 plans have no bucket marker; TensorRT may still give such
        # a network a nonempty default name.
        if (marker_present and engine_name != expected_name) or (not marker_present and bucket_size != 512):
            raise ValueError(f"TensorRT AR engine name {engine_name!r} does not match "
                             f"declared bucket {bucket_size}; expected {expected_name!r}")
        names = ["x", "pos", "mask"] + [f"{kind}_{i:02d}" for i in range(24) for kind in ("k", "v")]
        base_outputs = {"hidden", "key_delta", "value_delta"}
        projection_shapes = {f'projection_{i:02d}': (1, 1, (1536, 512, 2048, 512)[i % 4])
                             for i in range(96)}
        actual_outputs = set(self.trt.output_names)
        if (set(self.trt.input_names) != set(names)
                or actual_outputs not in (base_outputs, base_outputs | set(projection_shapes))
                or self.trt.engine.num_aux_streams != 0):
            raise ValueError("Unexpected TensorRT AR plan bindings or auxiliary streams")
        if actual_outputs != base_outputs:
            for name, shape in projection_shapes.items():
                if (tuple(self.trt.engine.get_tensor_shape(name)) != shape
                        or self.trt.engine.get_tensor_dtype(name) != trt.float16):
                    raise ValueError(f"Unexpected materialized projection output {name!r}")
        self.inputs = dict(zip(names, (runner.xy_pos_buf, runner.input_pos_buf, self.mask,
                                       *(tensor for pair in self.kv for tensor in pair))))
        self.outputs = {"hidden": runner.xy_dec_buf,
                        "key_delta": torch.empty((24, 1, 16, 1, 32), device=runner.device, dtype=torch.float16),
                        "value_delta": torch.empty((24, 1, 16, 1, 32), device=runner.device, dtype=torch.float16)}
        if actual_outputs != base_outputs:
            self.outputs.update({name: torch.empty(shape, device=runner.device, dtype=torch.float16)
                                 for name, shape in projection_shapes.items()})
        self.addresses = torch.tensor([tensor.data_ptr() for pair in self.kv for tensor in pair],
                                      dtype=torch.int64, device=runner.device)
        self.writeback = writeback
        self.capture_stream = torch.cuda.Stream(device=runner.device)
        self.capture_stream.wait_stream(torch.cuda.current_stream(runner.device))
        changed = (runner.xy_pos_buf, runner.xy_dec_buf, runner.input_pos_buf, self.mask,
                   *(tensor for pair in self.kv for tensor in pair))
        started = time.perf_counter()
        with torch.cuda.device(runner.device), torch.cuda.stream(self.capture_stream), torch.inference_mode():
            snapshots = [tensor.clone() for tensor in changed]
            try:
                runner.input_pos_buf.fill_(1)
                self.mask.zero_()
                self.mask[..., 0] = True
                # Bind all fixed addresses and compile the writeback before capture.
                if self.native_k8_alignment_guard or self.grouped_alignment_guard or self.gemv_alignment_guard or self.evenodd_alignment_guard:
                    # Bind without executing: alignment is checked on the captured graph first.
                    # Native affine plugins need a cuBLAS handle before stream capture.
                    warm_input = torch.zeros((1, 32), device=runner.device, dtype=torch.float16)
                    warm_weight = torch.zeros((32, 32), device=runner.device, dtype=torch.float16)
                    torch.addmm(warm_input, warm_input, warm_weight)
                    for name, tensor in self.inputs.items():
                        self.trt._validate_tensor(name, tensor)
                        if (not self.trt.context.set_input_shape(name, tuple(tensor.shape))
                                or not self.trt.context.set_tensor_address(name, tensor.data_ptr())):
                            raise RuntimeError('Aligned AR input binding failed: ' + name)
                    if self.trt.context.infer_shapes():
                        raise RuntimeError('Aligned AR shape inference failed')
                    for name, tensor in self.outputs.items():
                        shape = tuple(self.trt.context.get_tensor_shape(name))
                        self.trt._validate_tensor(name, tensor, expected_shape=shape)
                        if not self.trt.context.set_tensor_address(name, tensor.data_ptr()):
                            raise RuntimeError('Aligned AR output binding failed: ' + name)
                else:
                    self.trt.infer(self.inputs, outputs=self.outputs, stream=self.capture_stream,
                                   synchronize=False, profile=False)
                writeback[(24,)](self.addresses, self.outputs["key_delta"], self.outputs["value_delta"],
                                 runner.input_pos_buf, BLOCK=512)
                self.capture_stream.synchronize()
                self.graph = (torch.cuda.CUDAGraph(keep_graph=True) if self.weight_l2 is not None
                              else torch.cuda.CUDAGraph())
                with torch.cuda.graph(self.graph, stream=self.capture_stream):
                    if not self.trt.context.execute_async_v3(stream_handle=self.capture_stream.cuda_stream):
                        raise RuntimeError("TensorRT AR neural graph enqueue failed")
                    writeback[(24,)](self.addresses, self.outputs["key_delta"], self.outputs["value_delta"],
                                     runner.input_pos_buf, BLOCK=512)
                if self.gemv_alignment_guard:
                    import ctypes
                    self.capture_stream.synchronize()
                    guard = ctypes.CDLL(str(Path(self.gemv_alignment_guard).resolve(strict=True)))
                    guard.l2_init.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
                    guard.qa_validate_gemv_alignment.argtypes = [ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t]
                    guard.l2_init.restype = guard.qa_validate_gemv_alignment.restype = ctypes.c_int
                    error = ctypes.create_string_buffer(1024)
                    if guard.l2_init(error, len(error)) or guard.qa_validate_gemv_alignment(
                            int(self.graph.raw_cuda_graph()), error, len(error)):
                        raise RuntimeError('Aligned GEMV graph validation failed: ' + error.value.decode(errors='replace'))
                    self.gemv_alignment_validated = True
                if self.evenodd_alignment_guard:
                    import ctypes
                    self.capture_stream.synchronize()
                    guard = ctypes.CDLL(str(Path(self.evenodd_alignment_guard).resolve(strict=True)))
                    guard.l2_init.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
                    guard.qa_validate_evenodd_alignment.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t]
                    guard.l2_init.restype = guard.qa_validate_evenodd_alignment.restype = ctypes.c_int
                    error = ctypes.create_string_buffer(1024)
                    if guard.l2_init(error, len(error)) or guard.qa_validate_evenodd_alignment(int(self.graph.raw_cuda_graph()), 24, error, len(error)):
                        raise RuntimeError('Even/odd FFN2 graph validation failed: ' + error.value.decode(errors='replace'))
                    self.evenodd_alignment_validated = True
                if self.grouped_alignment_guard:
                    import ctypes
                    self.capture_stream.synchronize()
                    guard = ctypes.CDLL(str(Path(self.grouped_alignment_guard).resolve(strict=True)))
                    guard.l2_init.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
                    guard.qa_validate_grouped_attention.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t]
                    guard.l2_init.restype = guard.qa_validate_grouped_attention.restype = ctypes.c_int
                    error = ctypes.create_string_buffer(1024)
                    if guard.l2_init(error, len(error)) or guard.qa_validate_grouped_attention(
                            int(self.graph.raw_cuda_graph()), 24, error, len(error)):
                        raise RuntimeError('Grouped-attention graph validation failed: ' + error.value.decode(errors='replace'))
                    self.grouped_alignment_validated = True
                if self.native_k8_alignment_guard:
                    import ctypes
                    self.capture_stream.synchronize()
                    guard = ctypes.CDLL(str(Path(self.native_k8_alignment_guard).resolve(strict=True)))
                    guard.l2_init.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
                    guard.qa_validate_native_k8_attention.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t]
                    guard.l2_init.restype = guard.qa_validate_native_k8_attention.restype = ctypes.c_int
                    error = ctypes.create_string_buffer(1024)
                    if guard.l2_init(error, len(error)) or guard.qa_validate_native_k8_attention(
                            int(self.graph.raw_cuda_graph()), 24, error, len(error)):
                        raise RuntimeError('Native-K8 attention graph validation failed: ' + error.value.decode(errors='replace'))
                    self.native_k8_alignment_validated = True
                if self.weight_l2 is not None:
                    self.capture_stream.synchronize()
                    self.weight_l2_inspection = self.weight_l2.inspect_and_apply(self.graph, hit_ratio=1.0)
                    self.graph.instantiate()
            finally:
                self.capture_stream.synchronize()
                for tensor, saved in zip(changed, snapshots):
                    tensor.copy_(saved)
                self.capture_stream.synchronize()
        self.capture_seconds = time.perf_counter() - started

    def _require_idle(self):
        runner = self.runner
        if runner.bound_sessions or runner.waiting_bind or any(runner.slot_sessions):
            raise RuntimeError("Switch or capture TensorRT AR only when the runner is idle")

    def _require_boolean_graph(self):
        owner = getattr(self.original_acquire_graph, "__self__", None)
        if isinstance(owner, TensorRTARNeural):
            owner._require_boolean_graph()
        if getattr(owner, "additive_mask", None) is not None:
            raise ValueError("TensorRT AR does not support additive attention masks")

    def acquire_graph(self, batch_size):
        runner = self.runner
        if self.requires_prefix_mask:
            # The component consumes a mask binding but computes with device
            # seqlen; select it only when the existing qualified producer owns
            # that mask and the original native prefix hook is unchanged.
            from .ar_sampling_graph import SamplingGraph
            producer = getattr(runner.decode_step, '__self__', None)
            if (type(producer) is not SamplingGraph or producer.runner is not runner or not producer.enabled
                    or runner.decode_step != producer.decode_step or not producer.fuse_attention_mask
                    or not producer.fused_post or producer.additive_mask is not None
                    or producer.attention_mask is not self.mask
                    or runner.decoder.pre_forward_slots_hook != producer.native_pre_forward_hook):
                from .ar_attention_buckets import AttentionBuckets
                owner = getattr(self.original_acquire_graph, '__self__', None)
                self.prefix_guard_fallbacks += 1
                native = owner.original_acquire_graph if isinstance(owner, AttentionBuckets) else self.original_acquire_graph
                return native(batch_size)
        if (batch_size != 1 or len(runner.bound_sessions) != 1 or runner.waiting_bind
                or runner.extra_buffers.get("attn_mask") is not self.mask):
            raise ValueError("TensorRT AR requires one active slot and the original boolean mask")
        session = runner.slot_sessions[0]
        if (session is None or session not in runner.bound_sessions or session.bsz != 1
                or session.slot_indices != [0]):
            raise ValueError("TensorRT AR session must occupy slot zero")
        request = session.request
        cached = (self.compact_host_state and self._validated_session is not None
                  and self._validated_session() is session and self._validated_request is request)
        if not cached:
            self._require_boolean_graph()
            if (not request.use_cuda_graph or len(request.x) != 1 or request.x[0].ndim != 1
                    or request.x_lens.device.type != "cpu" or tuple(request.x_lens.shape) != (1,)):
                raise ValueError("Unsupported TensorRT AR request layout")
            text_length = int(request.x_lens[0].item())
            if text_length != request.x[0].numel() or session.prompt_len != request.prompts.shape[-1]:
                raise RuntimeError('Native single-slot immutable input layout is inconsistent')
            self._position_base = text_length + int(session.prompt_len)
            self.static_session_validations += 1
            if self.compact_host_state:
                # Submitted T2S request layout and prompt length are immutable;
                # step/slot ownership and output counters remain checked below.
                self._validated_session = weakref.ref(session)
                self._validated_request = request
        if (session.step_count <= 0 or session.output_lengths != [session.step_count]
                or session.total_tokens != session.step_count):
            raise RuntimeError("Native single-slot input-position invariant is not satisfied")
        position = self._position_base + int(session.step_count)
        if not 1 <= position <= 2048:
            raise ValueError("Native AR input position is outside the 2048-token cache")
        if position <= self.bucket_size:
            self.counts[self._trt_counter] += 1
            return self.graph
        self.counts[self._original_counter] += 1
        return self.original_acquire_graph(batch_size)

    def _restore_weight_l2(self):
        if self.weight_l2 is None or self._weight_l2_saved_limit is None:
            return
        import torch
        saved = self._weight_l2_saved_limit
        errors = []
        with torch.cuda.device(self.runner.device):
            try:
                torch.cuda.synchronize(self.runner.device)
            except Exception as error:
                errors.append(f'synchronize: {error}')
            try:
                self.weight_l2.reset()
            except Exception as error:
                errors.append(f'reset: {error}')
            try:
                self.weight_l2.set_limit(saved)
                actual = self.weight_l2.get_limit()
                self.weight_l2_last_restored = actual
                if actual != saved:
                    errors.append(f'limit restored as {actual}, expected {saved}')
            except Exception as error:
                errors.append(f'restore: {error}')
        self._weight_l2_saved_limit = None
        if errors:
            raise RuntimeError('AR weight L2 cleanup failed: ' + '; '.join(errors))
        self.weight_l2_restorations += 1

    def enable(self):
        self._require_idle()
        if self.closed:
            raise RuntimeError("TensorRT AR handle is closed")
        if self.enabled:
            if self.runner._acquire_graph != self.acquire_graph:
                raise RuntimeError("TensorRT AR graph binding changed externally")
            return
        if self.runner._acquire_graph != self.original_acquire_graph:
            raise RuntimeError("Original AR graph binding changed externally")
        self._require_boolean_graph()
        try:
            if self.weight_l2 is not None:
                import torch
                with torch.cuda.device(self.runner.device):
                    self._weight_l2_saved_limit = self.weight_l2.get_limit()
                    self.weight_l2_last_saved = self._weight_l2_saved_limit
                    self.weight_l2.set_limit(30 * 1024**2)
                    if self.weight_l2.get_limit() != 30 * 1024**2:
                        raise RuntimeError('CUDA context did not accept the 30 MiB AR weight L2 limit')
            self.runner._acquire_graph = self.acquire_graph
            self.enabled = True
            if self.weight_l2 is not None:
                self.weight_l2_enables += 1
        except BaseException:
            self._restore_weight_l2()
            raise

    def disable(self):
        self._require_idle()
        if self.enabled:
            try:
                if self.runner._acquire_graph != self.acquire_graph:
                    raise RuntimeError("TensorRT AR graph binding changed externally")
                self.runner._acquire_graph = self.original_acquire_graph
            finally:
                self.enabled = False
                self._validated_session = None
                self._validated_request = None
                self._position_base = None
                self._restore_weight_l2()
        else:
            self._restore_weight_l2()

    def close(self):
        self._require_idle()
        try:
            self.disable()
        finally:
            if not self.closed:
                import torch
                torch.cuda.synchronize(self.runner.device)
                self.graph.reset()
                self.closed = True


def install(runner, plan, *, bucket_size=512, native_attention=False, compact_host_state=False,
            weight_l2_shim=None, weight_l2_role='ffn1'):
    """Capture and enable a matching-plan neural graph on an idle runner."""
    handle = TensorRTARNeural(runner, plan, bucket_size=bucket_size,native_attention=native_attention,
                             compact_host_state=compact_host_state, weight_l2_shim=weight_l2_shim,
                             weight_l2_role=weight_l2_role)
    handle.enable()
    return handle
