"""V5 Turbo CUDA service using the qualified native streaming path."""
from __future__ import annotations
from contextlib import ExitStack, nullcontext
import io
import json
import os
from pathlib import Path
import threading
import time
import wave
from aniflive_tts import service as core
from aniflive_tts.expression import ExpressionSegment, PreparedReference
from aniflive_tts.voices import VoiceError
tracer = None


class _CompleteCancelled(BaseException):
    """Escape the upstream model-reload handler on an intentional cancellation."""


def _complete_cancellation(engine, accelerator, wake, stop):
    """Install request-scoped checks around existing AR, CFM and vocoder calls."""
    if wake is None or wake.engine is not accelerator.engine:
        raise RuntimeError("Complete synthesis requires the owner-thread AR wakeup")
    stack = ExitStack()

    def check():
        if stop.is_set():
            raise _CompleteCancelled()

    def replace(obj, name, value):
        had_value = name in vars(obj)
        old_value = vars(obj).get(name)
        def restore():
            if had_value:
                setattr(obj, name, old_value)
            else:
                delattr(obj, name)
        stack.callback(restore)
        setattr(obj, name, value)

    original_ar = accelerator.infer_batch
    def infer_batch(*args, **kwargs):
        check()
        try:
            result = original_ar(*args, **kwargs)
        except RuntimeError as error:
            if stop.is_set() and str(error) == 'T2S request cancelled':
                raise _CompleteCancelled() from error
            raise
        check()
        return result

    def checked_call(original):
        def call(*args, **kwargs):
            check()
            result = original(*args, **kwargs)
            check()
            return result
        return call

    try:
        wake.set_complete_stop(stop)
        stack.callback(wake.set_complete_stop, None)
        replace(accelerator, 'infer_batch', infer_batch)
        cfm = engine.vits_model.cfm
        replace(cfm, 'inference', checked_call(cfm.inference))
        replace(engine, 'vocoder', checked_call(engine.vocoder))
    except BaseException:
        stack.close()
        raise
    return stack


class V5Service:
    sample_rate = 48000
    ready = False
    switching = False
    expression_available = False

    def __init__(self):
        from types import SimpleNamespace
        self.settings = SimpleNamespace(dynamic_voices=True, reference_wav=Path("/nonexistent"))
        self._engine = None
        self._lease = None
        self._lease_handle = None
        self._lock = threading.Lock()
        self._active_lock = threading.Lock()
        self._owner = None
        self._stop = None
        self._cfm_shape_plans = {}
        self._prefill_shape_plans = {}
        self._engine_manifest = {}
        self._previous_cwd = None
        from aniflive_tts.inference_lease import InferenceGPUResourceLease
        self._gpu_resource_lease = InferenceGPUResourceLease.from_env()

    def load(self):
        try:
            self._load()
        except BaseException:
            if self._previous_cwd is not None:
                os.chdir(self._previous_cwd)
                self._previous_cwd = None
            raise

    def _load(self):
        self._configure()
        from .engine import load_engine, shared_gpu_lease
        raw_plans = json.loads(os.environ.get('PERF_CFM_SHAPE_PLANS') or '{}')
        if not isinstance(raw_plans, dict):
            raise ValueError('PERF_CFM_SHAPE_PLANS must be a frame-to-plan object')
        self._cfm_shape_plans = {int(frames): path for frames, path in raw_plans.items()}
        raw_prefill = json.loads(os.environ.get('PERF_PREFILL_SHAPE_PLANS') or '{}')
        if not isinstance(raw_prefill, dict):
            raise ValueError('PERF_PREFILL_SHAPE_PLANS must be a length-to-plan object')
        self._prefill_shape_plans = {int(length): path for length, path in raw_prefill.items()}
        if self._prefill_shape_plans and (os.environ.get('PERF_AR_PREFILL_GRAPH') != '1'
                or os.environ.get('PERF_AR_PREFILL_WORKSPACE') != '1'):
            raise ValueError('TensorRT prefill requires the persistent native prefill workspace')
        if self._cfm_shape_plans and os.environ.get('PERF_CFM_GRAPH') != '1':
            raise ValueError('CFM shape plans require the CFM graph adapter')
        for name in ('PERF_CFM_GRAPH', 'PERF_ENCODER_GRAPH', 'PERF_AR_FUSED_MASK', 'PERF_AR_FUSED_POST', 'PERF_AR_TILED_AOT', 'PERF_AR_GEMV_AOT', 'PERF_AR_KV_OVERLAY_AOT', 'PERF_AR_DELTA_QUALIFIED_AOT', 'PERF_AR_DELTA_MASKED_TILES_AOT',
                     'PERF_AR_COMPILED_NATIVE', 'PERF_AR_COMPACT_HOST', 'PERF_AR_FAST_TOPK',
                     'PERF_AR_PREFILL_CACHE', 'PERF_AR_PREFILL_GRAPH', 'PERF_AR_PREFILL_WORKSPACE', 'PERF_AR_PREFILL_BATCH_BIND_COPY', 'PERF_AR_EOS_PINNED_POLL', 'PERF_AR_DELTA_SUFFIX_TILES_AOT', 'PERF_AR_NEURAL_COMPACT_HOST', 'PERF_CFM_POSITIVE_ONLY', 'PERF_AR_DEFER_INACTIVE_PARTIALS', 'PERF_AR_PREFILL_BIND_INDEX', 'PERF_AR_WORKER_EVENT_WAKE', 'PERF_AR_JOINT_GRAPH', 'PERF_AR_RETAIN_SELECTION'):
            if os.environ.get(name, '0') not in ('0', '1'):
                raise ValueError(f'{name} must be 0 or 1')
        if os.environ.get('PERF_OUTPUT_HEAD', 'raw') not in ('raw', 'v2'):
            raise ValueError('PERF_OUTPUT_HEAD must be raw or v2')
        if os.environ.get('PERF_AR_BUCKETS', '0') not in ('0', '1'):
            raise ValueError('PERF_AR_BUCKETS must be 0 or 1')
        if os.environ.get('PERF_AR_BUCKETS') == '1' and os.environ.get('PERF_AR_SAMPLING_GRAPH') != '1':
            raise ValueError('Attention buckets require the validated AR sampling graph configuration')
        if any(os.environ.get(name) == '1' for name in ('PERF_AR_FUSED_MASK', 'PERF_AR_COMPACT_HOST', 'PERF_AR_FAST_TOPK')) and os.environ.get('PERF_AR_SAMPLING_GRAPH') != '1':
            raise ValueError('Configured AR sampler options require PERF_AR_SAMPLING_GRAPH=1')
        if os.environ.get('PERF_AR_FUSED_POST') == '1' and os.environ.get('PERF_AR_FUSED_MASK') != '1':
            raise ValueError('Fused AR post-sampling requires PERF_AR_FUSED_MASK=1')
        if os.environ.get('PERF_AR_EOS_PINNED_POLL') == '1' and os.environ.get('PERF_AR_FUSED_MASK') != '1':
            raise ValueError('Pinned EOS polling requires the fused single-slot sampling graph')
        if os.environ.get('PERF_AR_EOS_NATIVE_POLL_LIBRARY') and os.environ.get('PERF_AR_EOS_PINNED_POLL') != '1':
            raise ValueError('Native EOS polling requires pinned EOS polling')
        if os.environ.get('PERF_AR_COMPILED_NATIVE') == '1':
            if os.environ.get('PERF_AR_BUCKETS') != '1':
                raise ValueError('Compiled native AR requires PERF_AR_BUCKETS=1')
            if os.environ.get('PERF_AR_NEURAL_ENGINE') or os.environ.get('PERF_AR_NEURAL_SMALL_ENGINE'):
                raise ValueError('Compiled native AR and TensorRT AR neural plans are mutually exclusive')
        bucket_size = os.environ.get('PERF_AR_NEURAL_BUCKET_SIZE', '512')
        if bucket_size not in ('256', '512', '1024'):
            raise ValueError('PERF_AR_NEURAL_BUCKET_SIZE must be 256, 512 or 1024')
        if os.environ.get('PERF_AR_NEURAL_SMALL_ENGINE') and (
                not os.environ.get('PERF_AR_NEURAL_ENGINE') or int(bucket_size) <= 256):
            raise ValueError('The 256 neural plan requires a larger main neural plan')
        if os.environ.get('PERF_AR_NEURAL_ENGINE') and os.environ.get('PERF_AR_BUCKETS') != '1':
            raise ValueError('TensorRT AR neural plans require attention buckets')
        if os.environ.get('PERF_AR_WEIGHT_L2_SHIM') and (not os.environ.get('PERF_AR_NEURAL_ENGINE') or os.environ.get('PERF_AR_NEURAL_SMALL_ENGINE')):
            raise ValueError('Experimental AR weight L2 requires one qualified neural plan')
        if os.environ.get('PERF_AR_WEIGHT_L2_ROLE', 'ffn1') not in ('ffn1', 'ffn2'):
            raise ValueError('AR weight L2 role must be ffn1 or ffn2')
        if os.environ.get('PERF_CUDA_ONLY_SEED', '0') not in ('0', '1'):
            raise ValueError('PERF_CUDA_ONLY_SEED must be0 or1')
        if os.environ.get('PERF_AR_TOPK_THRESHOLD', '0') not in ('0', '1'):
            raise ValueError('PERF_AR_TOPK_THRESHOLD must be0 or1')
        if os.environ.get('PERF_AR_TOPK_THRESHOLD') == '1' and os.environ.get('PERF_AR_FAST_TOPK') != '1':
            raise ValueError('Exact top15 threshold requires the qualified fast top-k path')
        if os.environ.get('PERF_AR_JOINT_GRAPH') == '1' and (
                not os.environ.get('PERF_AR_WEIGHT_L2_SHIM') or os.environ.get('PERF_AR_FUSED_POST') != '1'
                or os.environ.get('PERF_AR_FUSED_MASK') != '1' or os.environ.get('PERF_AR_NEURAL_SMALL_ENGINE')
                or os.environ.get('PERF_AR_EOS_PINNED_POLL') == '1'):
            raise ValueError('Joint graph requires one retained L2 neural graph and the fused native sampler')
        if os.environ.get('PERF_AR_TILED_AOT') == '1' and not os.environ.get('PERF_AR_NEURAL_ENGINE'):
            raise ValueError('Tiled attention AOT requires PERF_AR_NEURAL_ENGINE')
        if os.environ.get('PERF_AR_GEMV_AOT') == '1' and not os.environ.get('PERF_AR_NEURAL_ENGINE'):
            raise ValueError('GEMV AOT requires PERF_AR_NEURAL_ENGINE')
        if os.environ.get('PERF_AR_KV_OVERLAY_AOT') == '1' and not os.environ.get('PERF_AR_NEURAL_ENGINE'):
            raise ValueError('KV overlay AOT requires PERF_AR_NEURAL_ENGINE')
        if os.environ.get('PERF_AR_DELTA_QUALIFIED_AOT') == '1' and not os.environ.get('PERF_AR_NEURAL_ENGINE'):
            raise ValueError('Qualified direct-KV attention requires PERF_AR_NEURAL_ENGINE')
        if os.environ.get('PERF_AR_DELTA_MASKED_TILES_AOT') == '1':
            if not os.environ.get('PERF_AR_NEURAL_ENGINE') or os.environ.get('PERF_AR_DELTA_QUALIFIED_AOT') == '1':
                raise ValueError('Masked-tile direct-KV attention requires its own neural plan and creator')
        if os.environ.get('PERF_AR_DELTA_SUFFIX_TILES_AOT') == '1':
            if not os.environ.get('PERF_AR_NEURAL_ENGINE') or any(os.environ.get(n) == '1' for n in ('PERF_AR_DELTA_QUALIFIED_AOT', 'PERF_AR_DELTA_MASKED_TILES_AOT')):
                raise ValueError('Suffix-tile attention requires its own neural plan and creator')
        if os.environ.get('PERF_CFM_GRAPH') == '1' and not os.environ.get('PERF_DIT_ENGINE'):
            raise ValueError('CFM graph requires PERF_DIT_ENGINE')
        if os.environ.get('PERF_ENCODER_GRAPH') == '1' and not os.environ.get('PERF_ENCODER_ENGINE'):
            raise ValueError('Encoder graph requires PERF_ENCODER_ENGINE')
        if os.environ.get('PERF_VOCODER_GRAPH', '0') not in ('0', '1'):
            raise ValueError('PERF_VOCODER_GRAPH must be 0 or 1')
        if os.environ.get('PERF_VOCODER_GRAPH') == '1' and not os.environ.get('PERF_VOCODER_ENGINE'):
            raise ValueError('Vocoder graph requires PERF_VOCODER_ENGINE')
        if os.environ.get('PERF_AR_PREFILL_GRAPH') == '1' and os.environ.get('PERF_AR_PREFILL_CACHE') == '1':
            raise ValueError('Fresh-computation prefill graph and target-content prefill cache are mutually exclusive')
        if os.environ.get('PERF_AR_PREFILL_WORKSPACE') == '1' and os.environ.get('PERF_AR_PREFILL_GRAPH') != '1':
            raise ValueError('Persistent prefill workspace requires the fresh-computation graph')
        if os.environ.get('PERF_AR_PREFILL_BATCH_BIND_COPY') == '1' and os.environ.get('PERF_AR_PREFILL_WORKSPACE') != '1':
            raise ValueError('Batched KV binding requires the persistent prefill workspace')
        if os.environ.get('PERF_AR_PREFILL_BIND_INDEX') == '1' and (
                os.environ.get('PERF_AR_PREFILL_WORKSPACE') != '1'
                or os.environ.get('PERF_AR_PREFILL_BATCH_BIND_COPY') == '1'):
            raise ValueError('Persistent bind index requires workspace and excludes batched KV binding')
        self._lease = shared_gpu_lease(self._gpu_resource_lease)
        handle, engines = self._lease.__enter__()
        self._lease_handle = handle
        try:
            handle.ensure_valid()
            self._engine = load_engine(engines)
            self._engine.performance_cuda_only_seed = os.environ.get('PERF_CUDA_ONLY_SEED') == '1'
            if self._engine.performance_cuda_only_seed:
                from .cuda_seed import initialize_cuda_only_seed
                initialize_cuda_only_seed()
            if os.environ.get('PERF_QUIET_STATUS') == '1':
                # The pinned frontend prints only request/progress status.
                # Route those messages to DEBUG; errors still propagate and
                # retain the service/uvicorn error loggers.
                import logging
                import sys
                frontend_module = sys.modules[type(self._engine.text_preprocessor).__module__]
                frontend_module.print = logging.getLogger('aniflive.v5.frontend').debug
            self._engine.performance_emit_budget = os.environ.get('PERF_AR_DEFER_INACTIVE_PARTIALS') == '1'
            if tracer is not None:
                tracer.install_engine_hooks(self._engine)
            if os.environ.get('PERF_VOCODER_GRAPH') == '1':
                from vocoder_cuda_graph import VocoderGraph
                graph = VocoderGraph(self._engine.vocoder,
                                     idle_guard=lambda: self._ar_idle(self._engine))
                self._engine.performance_vocoder_graph = graph
                self._engine.vocoder = graph
            self.ready = True
        except BaseException as error:
            # Failed constructors can retain large model tensors in traceback
            # frames while the manager tries to restore the previous package.
            import traceback
            traceback.clear_frames(error.__traceback__)
            self._lease.__exit__(*__import__("sys").exc_info())
            self._lease = None
            self._lease_handle = None
            raise

    def unload(self):
        self.cancel_active_stream()
        self.ready = False
        # Let any active iterator close before bucket/AR graph cleanup.
        with self._lock:
            vocoder_graph = getattr(self._engine, 'performance_vocoder_graph', None)
            if vocoder_graph is not None:
                vocoder_graph.close()
                self._engine.vocoder = vocoder_graph.base
                self._engine.performance_vocoder_graph = None
            encoder_graph = getattr(self._engine, 'performance_encoder_graph', None)
            if encoder_graph is not None:
                encoder_graph.close()
                self._engine.performance_encoder_graph = None
                encoder_graph = None
            if self._lease is not None:
                self._lease.__exit__(None, None, None)
                self._lease = None
                self._lease_handle = None
            self._engine = None
            if self._previous_cwd is not None:
                os.chdir(self._previous_cwd)
                self._previous_cwd = None

    def prepare_dynamic_reference(self, *, reference_id, reference_wav, reference_text, reference_language):
        self._lease_handle.ensure_valid()
        import torch
        from TTS_infer_pack.streaming import v5_reference
        from TTS_infer_pack.text_segmentation_method import splits
        language = core._normalise_language(reference_language, "reference_language")
        text = reference_text.strip("\n")
        if not text:
            raise ValueError("Reference transcript is empty")
        if text[-1] not in splits:
            text += "。"
        with self._lock, torch.inference_mode():
            engine = self._engine
            engine.set_ref_audio(str(reference_wav))
            phones, bert, normalized = engine.text_preprocessor.segment_and_extract_feature_for_text(
                text, language, engine.configs.version)
            engine.prompt_cache.update(prompt_text=text, prompt_lang=language, phones=phones,
                                       bert_features=bert, norm_text=normalized)
            ref = v5_reference(engine)
            if getattr(engine, 'performance_cfm_compiled', False) and not getattr(engine, 'performance_cfm_warmed', False):
                # Compile during explicit Reference preparation, never hide
                # the cold cost inside a purported warm user request.
                started = time.perf_counter()
                for target_frames in (32, 212):
                    mu = torch.cat((ref[4], ref[4][..., :target_frames]), -1).transpose(1, 2)
                    lengths = torch.full((1,), mu.shape[1], device=mu.device, dtype=torch.long)
                    generator = torch.Generator(device=mu.device).manual_seed(3407)
                    engine.vits_model.cfm.inference(mu, lengths, ref[6], 4,
                                                   inference_cfg_rate=0, generator=generator)
                torch.cuda.synchronize()
                engine.performance_cfm_prepare_seconds = time.perf_counter() - started
                engine.performance_cfm_warmed = True
                import gc
                import ctypes
                gc.collect()
                ctypes.CDLL('libc.so.6').malloc_trim(0)
            prompt = dict(engine.prompt_cache)
            # set_ref_audio mutates this list in place on the next reference.
            prompt["refer_spec"] = list(prompt["refer_spec"])
            prompt["aux_ref_audio_paths"] = list(prompt.get("aux_ref_audio_paths") or [])
            if not getattr(engine, 'performance_ar_warmed', False):
                from .engine import SAMPLER
                from .native_stream import stream
                calibration_language = language.removeprefix('all_')
                warm_cache = {'key': (str(reference_wav), text, language), 'reference': ref}
                # Match the validated probe: initialize and warm the native
                # AR sampler before capturing its sampling/post-step graph.
                for warm_index, optimized in enumerate((False, False, os.environ.get('PERF_AR_SAMPLING_GRAPH') == '1')):
                    values = dict(SAMPLER, text=core.STREAM_CALIBRATION_TEXT[calibration_language],
                                  text_lang=language, seed=3407, sampling_graph=optimized,
                                  ref_audio_path=str(reference_wav), prompt_text=text, prompt_lang=language)
                    for _rate, _pcm in stream(engine, values, cache=warm_cache):
                        pass
                    if warm_index == 0 and os.environ.get('PERF_AR_WORKER_EVENT_WAKE') == '1':
                        self._wait_ar_idle(engine)
                        if tracer is not None:
                            tracer.install_ar_hooks(engine)
                        from .ar_worker_wake import WorkerWake
                        engine.performance_worker_wake = WorkerWake(engine.streaming_accel.engine)
                        engine.performance_worker_wake.select(True)
                    if warm_index == 0 and os.environ.get('PERF_AR_PREFILL_GRAPH') == '1':
                        self._wait_ar_idle(engine)
                        if tracer is not None and os.environ.get('PERF_AR_WORKER_EVENT_WAKE') != '1':
                            tracer.install_ar_hooks(engine)
                        from .ar_prefill_graph import PrefillGraph
                        engine.performance_ar_prefill_graph = PrefillGraph(
                            engine.streaming_accel.engine.model_runner,
                            idle_guard=lambda: self._ar_idle(engine), collect_misses=True,
                            persistent_workspace=os.environ.get('PERF_AR_PREFILL_WORKSPACE') == '1',
                            batch_bind_copy=os.environ.get('PERF_AR_PREFILL_BATCH_BIND_COPY') == '1',
                            persistent_bind_index=os.environ.get('PERF_AR_PREFILL_BIND_INDEX') == '1',
                            trt_shape_plans=self._prefill_shape_plans or None)
                        engine.performance_ar_prefill_graph.enable()
                acceleration = engine.streaming_accel
                if acceleration is None or acceleration.engine is None:
                    raise RuntimeError('Reference calibration did not initialize the AR worker')
                runner = acceleration.engine.model_runner
                sampling = getattr(engine, 'performance_sampling_graph', None)
                if sampling is not None and any(os.environ.get(name) == '1' for name in
                                                ('PERF_AR_FUSED_MASK', 'PERF_AR_FUSED_POST',
                                                 'PERF_AR_COMPACT_HOST', 'PERF_AR_FAST_TOPK')):
                    from .ar_sampling_graph import SamplingGraph
                    sampling.disable()
                    sampling = SamplingGraph(runner,
                        fuse_attention_mask=os.environ.get('PERF_AR_FUSED_MASK') == '1',
                        compact_host_state=os.environ.get('PERF_AR_COMPACT_HOST') == '1',
                        fast_topk=os.environ.get('PERF_AR_FAST_TOPK') == '1',
                        fused_post=os.environ.get('PERF_AR_FUSED_POST') == '1',
                        eos_pinned_poll=os.environ.get('PERF_AR_EOS_PINNED_POLL') == '1',
                        eos_native_poll_library=os.environ.get('PERF_AR_EOS_NATIVE_POLL_LIBRARY') or None,
                        capture_for_joint=os.environ.get('PERF_AR_JOINT_GRAPH') == '1',
                        exact_topk_threshold=os.environ.get('PERF_AR_TOPK_THRESHOLD') == '1')
                    engine.performance_sampling_graph = sampling
                    sampling.enable()
                if os.environ.get('PERF_AR_BUCKETS') == '1':
                    from .ar_attention_buckets import install_attention_buckets
                    if (acceleration is None or acceleration.engine is None or sampling is None
                            or not sampling.enabled or sampling.runner is not acceleration.engine.model_runner):
                        raise RuntimeError('Attention buckets require the warmed native AR and sampling graphs')
                    compiled = os.environ.get('PERF_AR_COMPILED_NATIVE') == '1'
                    engine.performance_attention_buckets = install_attention_buckets(
                        runner, compact_host_state=os.environ.get('PERF_AR_COMPACT_HOST') == '1',
                        compile_neural=compiled, eager_rounding=compiled,
                        native_norm=compiled, native_projections=compiled)
                    if compiled:
                        engine.performance_attention_buckets.select_compiled(True)
                if os.environ.get('PERF_AR_NEURAL_ENGINE'):
                    if os.environ.get('PERF_AR_GROUPED_ATTENTION_AOT') == '1':
                        import attention_grouped_aot as attention_grouped_aot
                    if os.environ.get('PERF_AR_NATIVE_K8_ATTENTION_AOT') == '1':
                        import attention_native_k8_aot as attention_native_k8_aot
                    if os.environ.get('PERF_AR_NATIVE_K8_SHARED_ATTENTION_AOT') == '1':
                        from . import attention_native_k8_shared_aot as attention_native_k8_shared_aot
                    if os.environ.get('PERF_AR_NATIVEORDER_RESIDUAL_GEMV_AOT') == '1':
                        from . import trt_gemv_nativeorder_warp as trt_gemv_nativeorder_warp
                        from . import trt_gemv_nativeorder_residual as trt_gemv_nativeorder_residual
                    if os.environ.get('PERF_AR_FFN2_STRIDE32_AOT') == '1':
                        import trt_gemv_stride32 as trt_gemv_stride32
                    if os.environ.get('PERF_AR_NATIVEORDER_GEMV_AOT') == '1':
                        import trt_gemv_stride32 as trt_gemv_stride32
                        import trt_gemv_k512_serial16 as trt_gemv_k512_serial16
                    # Explicit private precision-repair libraries, registered
                    # before deserializing their matching neural plan.
                    import importlib
                    for variable, module in (
                        ('PERF_AR_CPP_ATTENTION_PROJECTION_LIBRARY', 'trt_cpp_native_attention_projection'),
                        ('PERF_AR_CPP_FFN_LIBRARY', 'trt_cpp_native_ffn'),
                        ('PERF_AR_CPP_LAYERNORM_LIBRARY', 'trt_cpp_native_layernorm'),
                        ('PERF_AR_CPP_LAYERNORM_DIRECT_LIBRARY', 'trt_cpp_native_layernorm_direct'),
                        ('PERF_AR_CPP_GUARDED_ATTENTION_LIBRARY', 'trt_cpp_native_guarded_attention'),
                    ):
                        library = os.environ.get(variable)
                        if library:
                            importlib.import_module('.' + module, __package__).load(library)
                    if os.environ.get('PERF_AR_DELTA_SUFFIX_TILES_AOT') == '1':
                        import tensorrt as trt
                        import trt_ar_delta_suffix_tiles as trt_ar_delta_suffix_tiles
                        creator = trt.get_plugin_registry().get_creator('delta_attention_suffix_tiles_aot', '1', 'aniflive_research')
                        if creator is None:
                            raise RuntimeError('TensorRT suffix-tile direct-KV creator was not registered')
                        engine.performance_ar_delta_suffix_tiles_aot_creator = True
                    if os.environ.get('PERF_AR_DELTA_MASKED_TILES_AOT') == '1':
                        import tensorrt as trt
                        import trt_ar_delta_masked_tiles as trt_ar_delta_masked_tiles
                        creator = trt.get_plugin_registry().get_creator('delta_attention_masked_tiles_aot', '1', 'aniflive_research')
                        if creator is None:
                            raise RuntimeError('TensorRT masked-tile direct-KV creator was not registered')
                        engine.performance_ar_delta_masked_tiles_aot_creator = True
                    if os.environ.get('PERF_AR_DELTA_QUALIFIED_AOT') == '1':
                        import tensorrt as trt
                        import trt_ar_delta_qualified as trt_ar_delta_qualified
                        creator = trt.get_plugin_registry().get_creator('delta_attention_qualified_aot', '1', 'aniflive_research')
                        if creator is None:
                            raise RuntimeError('TensorRT qualified direct-KV attention creator was not registered')
                        engine.performance_ar_delta_qualified_aot_creator = True
                    if os.environ.get('PERF_AR_KV_OVERLAY_AOT') == '1':
                        import tensorrt as trt
                        import trt_ar_kv_overlay as trt_ar_kv_overlay
                        creator = trt.get_plugin_registry().get_creator('ar_kv_overlay_aot', '1', 'aniflive_research')
                        if creator is None:
                            raise RuntimeError('TensorRT KV overlay AOT creator was not registered')
                        engine.performance_ar_kv_overlay_aot_creator = True
                    if os.environ.get('PERF_AR_GEMV_AOT') == '1':
                        import tensorrt as trt
                        import trt_ar_gemv as trt_ar_gemv
                        creator = trt.get_plugin_registry().get_creator('ar_gemv_aot', '1', 'aniflive_research')
                        if creator is None:
                            raise RuntimeError('TensorRT GEMV AOT creator was not registered')
                        engine.performance_ar_gemv_aot_creator = True
                    if os.environ.get('PERF_AR_TILED_AOT') == '1':
                        import tensorrt as trt
                        import trt_ar_tiled_attention as trt_ar_tiled_attention  # Registers creators before plan deserialization.
                        registry = trt.get_plugin_registry()
                        creators = {name: registry.get_creator(name, '1', 'aniflive_research') is not None
                                    for name in ('tiled_attention_aot', 'tiled_attention_vector_aot')}
                        if not all(creators.values()):
                            raise RuntimeError('TensorRT tiled attention AOT creators were not registered')
                        engine.performance_ar_tiled_aot_creators = creators
                    from .trt_ar_neural import install
                    engine.performance_ar_neural = install(runner, os.environ['PERF_AR_NEURAL_ENGINE'],
                        bucket_size=int(os.environ.get('PERF_AR_NEURAL_BUCKET_SIZE', '512')),
                        compact_host_state=os.environ.get('PERF_AR_NEURAL_COMPACT_HOST') == '1',
                        weight_l2_shim=os.environ.get('PERF_AR_WEIGHT_L2_SHIM') or None,
                        weight_l2_role=os.environ.get('PERF_AR_WEIGHT_L2_ROLE', 'ffn1'))
                    if os.environ.get('PERF_AR_NEURAL_SMALL_ENGINE'):
                        engine.performance_ar_neural_small = install(runner, os.environ['PERF_AR_NEURAL_SMALL_ENGINE'], bucket_size=256,
                            compact_host_state=os.environ.get('PERF_AR_NEURAL_COMPACT_HOST') == '1')
                if os.environ.get('PERF_AR_JOINT_GRAPH') == '1':
                    from .ar_joint_graph import JointGraph
                    joint = JointGraph(engine.performance_sampling_graph, engine.performance_ar_neural,
                                       audit_library=os.environ.get('PERF_AR_JOINT_AUDIT_LIBRARY') or None)
                    joint.select(True)
                if os.environ.get('PERF_AR_PREFILL_CACHE') == '1':
                    from ar_prefill_cache import install_prefill_cache
                    engine.performance_ar_prefill = install_prefill_cache(
                        runner, model_epoch=id(engine), idle_guard=lambda: self._ar_idle(engine))
                if os.environ.get('PERF_AR_PREFILL_GRAPH') == '1':
                    prefill_graph = engine.performance_ar_prefill_graph
                    started = time.perf_counter()
                    prefill_graph.prepare_pending()
                    engine.performance_prefill_graph_prepare_seconds = time.perf_counter() - started
                engine.performance_ar_warmed = True
                import gc
                import ctypes
                gc.collect()
                ctypes.CDLL('libc.so.6').malloc_trim(0)
            if any(os.environ.get(name) == '1' for name in ('PERF_CFM_GRAPH', 'PERF_ENCODER_GRAPH')):
                from .engine import SAMPLER
                from .native_stream import stream
                calibration_started = time.perf_counter()
                use_cfm = os.environ.get('PERF_CFM_GRAPH') == '1'
                use_encoder = os.environ.get('PERF_ENCODER_GRAPH') == '1'
                prefill_graph = getattr(engine, 'performance_ar_prefill_graph', None)
                if os.environ.get('PERF_AR_PREFILL_GRAPH') == '1':
                    if prefill_graph is None:
                        raise RuntimeError('Configured prefill graph was not installed')
                    prefill_graph.enable()
                cfm = engine.vits_model.cfm
                original = cfm.inference
                captured = {}
                encoder_inputs = {}
                def observe(mu, lengths, prompt_mel, *args, **kwargs):
                    key = (int(mu.shape[1]), int(prompt_mel.shape[-1]))
                    if key not in captured:
                        if len(captured) == 2:
                            captured.pop(next(iter(captured)))
                        captured[key] = (mu.detach().clone(), lengths.detach().clone(), prompt_mel.detach().clone())
                    return original(mu, lengths, prompt_mel, *args, **kwargs)
                encoder = getattr(engine, 'performance_encoder', None)
                encoder_graph = getattr(engine, 'performance_encoder_graph', None)
                if use_encoder:
                    if encoder is None:
                        raise RuntimeError('Configured TensorRT target encoder was not installed')
                    if 'infer' in encoder.runner.__dict__:
                        raise RuntimeError('Target encoder runner already has an inference hook')
                    if encoder_graph is not None and encoder_graph.enabled:
                        encoder_graph.disable()
                    encoder_infer = encoder.runner.infer
                    def observe_encoder(inputs, **kwargs):
                        key = (int(inputs['codes'].shape[-1]), int(inputs['phones'].shape[-1]))
                        if key not in encoder_inputs and len(encoder_inputs) < 2:
                            encoder_inputs[key] = tuple(inputs[k].detach().clone() for k in ('codes', 'phones', 'ge'))
                        return encoder_infer(inputs, **kwargs)
                    encoder.runner.infer = observe_encoder
                if use_cfm:
                    cfm.inference = observe
                try:
                    calibration_language = language.removeprefix('all_')
                    values = dict(SAMPLER, text=core.STREAM_CALIBRATION_TEXT[calibration_language],
                        text_lang=language, seed=3407, sampling_graph=os.environ.get('PERF_AR_SAMPLING_GRAPH') == '1',
                        ref_audio_path=str(reference_wav), prompt_text=text, prompt_lang=language)
                    cache = {'key': (str(reference_wav), text, language), 'reference': ref}
                    calibration = stream(engine, values, cache=cache)
                    try:
                        for _rate, _pcm in calibration:
                            pass
                    finally:
                        calibration.close()
                        self._wait_ar_idle(engine)
                finally:
                    if use_cfm:
                        cfm.inference = original
                    if use_encoder:
                        del encoder.runner.__dict__['infer']
                        if encoder_graph is not None:
                            encoder_graph.enable()
                calibration_seconds = time.perf_counter() - calibration_started
                if prefill_graph is not None:
                    started = time.perf_counter()
                    prefill_graph.prepare_pending()
                    engine.performance_prefill_graph_prepare_seconds = time.perf_counter() - started
                    engine.performance_prefill_graph_calibration_seconds = calibration_seconds
                if use_cfm:
                    engine.performance_cfm_graph_calibration_seconds = calibration_seconds
                    if not captured:
                        raise RuntimeError('CFM reference calibration produced no acoustic context')
                    adapter = getattr(engine, 'performance_cfm_graph', None)
                    if adapter is None:
                        from .cfm_cuda_graph import install
                        adapter = install(cfm, os.environ['PERF_DIT_ENGINE'],
                                          ar_is_idle=lambda: self._ar_idle(engine), collect_misses=True,
                                          shape_plans=self._cfm_shape_plans or None,
                                          positive_only=os.environ.get('PERF_CFM_POSITIVE_ONLY') == '1')
                        engine.performance_cfm_graph = adapter
                    target = ref[4].new_zeros((*ref[4].shape[:-1], 212))
                    mu = torch.cat((ref[4], target), -1).transpose(1, 2)
                    lengths = torch.full((1,), mu.shape[1], device=mu.device, dtype=torch.long)
                    started = time.perf_counter()
                    for tensors in captured.values():
                        adapter.prepare(*tensors)
                    adapter.prepare(mu, lengths, ref[6])
                    engine.performance_cfm_graph_prepare_seconds = time.perf_counter() - started
                    engine.performance_cfm_graph_total_seconds = time.perf_counter() - calibration_started
                if use_encoder:
                    if not encoder_inputs:
                        raise RuntimeError('Reference calibration produced no target encoder context')
                    if encoder_graph is None:
                        from .encoder_cuda_graph import install
                        encoder_graph = install(encoder, ar_is_idle=lambda: self._ar_idle(engine),
                                                collect_misses=True)
                        engine.performance_encoder_graph = encoder_graph
                    started = time.perf_counter()
                    for tensors in encoder_inputs.values():
                        encoder_graph.prepare(*tensors)
                    engine.performance_encoder_graph_prepare_seconds = time.perf_counter() - started
                    engine.performance_encoder_graph_calibration_seconds = calibration_seconds
            vocoder_graph = getattr(engine, 'performance_vocoder_graph', None)
            if vocoder_graph is not None and vocoder_graph.pending:
                self._wait_ar_idle(engine)
                vocoder_graph.prepare_pending()
            if tracer is not None:
                tracer.install_ar_hooks(engine)
            return PreparedReference(prompt_semantic=ref[2], spectrogram=ref[3],
                speaker_embedding=ref[5], phones=phones,
                bert={"prompt": prompt, "reference": ref, "path": str(reference_wav),
                      "text": text, "language": language}, id=reference_id)

    @staticmethod
    def _ar_idle(engine):
        accel = getattr(engine, 'streaming_accel', None)
        if accel is None or accel.engine is None:
            return False
        ar = accel.engine
        runner = ar.model_runner
        return not (runner.bound_sessions or runner.waiting_bind or any(runner.slot_sessions)
                    or ar.handles or not ar.pending_requests.empty())

    @staticmethod
    def _wait_ar_idle(engine):
        # Closing the vendor iterator signals cancellation asynchronously.
        # Hooks and graph bindings must remain alive until its worker exits.
        deadline = time.monotonic() + 5.0
        while not V5Service._ar_idle(engine):
            if time.monotonic() >= deadline:
                accel = engine.streaming_accel
                ar = accel.engine
                ar._stop_event.set()
                if ar._loop_thread is not None:
                    ar._loop_thread.join(timeout=5.0)
                accel._text_history_mask_failed = True
                raise RuntimeError('AR worker did not quiesce; v5 runtime requires restart')
            time.sleep(0.001)

    def _expression_plan(self, options):
        return options.expression_segments or (ExpressionSegment(
            text=options.text, enabled=options.expression_enabled,
            profile=options.expression_profile, intensity=options.expression_intensity,
            policy=options.expression_policy,
        ),)

    def validate_expression(self, options, binding=None):
        binding = binding if binding is not None else options.voice_binding
        plan = self._expression_plan(options)
        if options.expression_segments:
            core._validate_expression_segment_boundaries(plan)
        selected = []
        for segment in plan:
            if binding is None:
                if segment.policy is not core.ConditioningPolicy.FULL_SWITCH:
                    raise core.RequestError("Reference conditioning requires a prepared voice")
                selected.append((segment, None))
                continue
            if segment.enabled and segment.profile in binding.bank.references:
                bundle = binding.bank.conditioning(reference_id=segment.profile,
                    policy=segment.policy)
            else:
                bundle = binding.conditioning(enabled=segment.enabled, profile=segment.profile,
                    intensity=segment.intensity, language=options.text_language,
                    policy=segment.policy)
            from .conditioning import resolve_conditioning
            selected.append((segment, resolve_conditioning(bundle, binding.bank)))
        return selected

    def expression_metadata(self):
        return {"enabled": True, "default": None, "preferred_policy": "full-switch",
                "profiles": [], "policies": [policy.value for policy in core.ConditioningPolicy],
                "reference_selection": "voice-scoped-reference-id",
                "conditioning_components": ["AR semantic prompt", "GE/spectrum style", "CFM voice prompt"],
                "legacy_sv_equivalence": "native_v5_mapping_accepted_by_owner"}

    def preferred_expression_policy(self, **_):
        return core.ConditioningPolicy.FULL_SWITCH

    def recommended_stream_prebuffer_ms(self, segments):
        return core._recommended_stream_prebuffer_ms(segments)

    def _segments(self, options):
        return [segment.text for segment in self._expression_plan(options)]

    def prepare_stream(self, options, requested_model, *, stream_owner=None,
                       continuity_input=None, continuity_capture=None, continuity_policy="A", _complete=False):
        self._lease_handle.ensure_valid()
        if requested_model not in (None, core.MODEL_ID):
            raise core.RequestError("Requested model is not the v5 Turbo model")
        if _complete and (continuity_input is not None or continuity_capture is not None):
            raise core.RequestError("Complete synthesis does not support neural continuation")
        policy = core.parse_continuity_policy(continuity_policy)
        if policy is core.ContinuityPolicy.NONE:
            if continuity_input is not None or continuity_capture is not None:
                raise core.RequestError("Policy A does not accept neural continuation")
        elif continuity_capture is None:
            raise core.RequestError("Neural continuation requires a session capture")
        # Match the identity used by core's session planning/commit logic;
        # a neutral Reference ID is not an enabled expression profile.
        expression = options.expression_profile if options.expression_enabled else None
        if expression is None:
            expression = next((segment.profile for segment in reversed(options.expression_segments)
                               if segment.enabled), None)
        binding = options.voice_binding
        owned = None
        if binding is None:
            if not options.voice_profile:
                raise VoiceError("voice_profile_required", "Select a prepared voice_profile")
            owned = core.VOICE_LIBRARY.bind(options.voice_profile)
            binding = owned
        try:
            selected = self.validate_expression(options, binding)
            if not self._lock.acquire(blocking=False):
                raise core.ServiceBusy("v5 Turbo engine is busy")
        except BaseException:
            if owned is not None:
                owned.close()
            raise
        stop = threading.Event()
        completed = False
        with self._active_lock:
            self._owner, self._stop = stream_owner, stop

        def release():
            try:
                prefill_graph = getattr(self._engine, 'performance_ar_prefill_graph', None)
                encoder_graph = getattr(self._engine, 'performance_encoder_graph', None)
                cfm_graph = getattr(self._engine, 'performance_cfm_graph', None)
                vocoder_graph = getattr(self._engine, 'performance_vocoder_graph', None)
                adapters = (prefill_graph, encoder_graph, cfm_graph, vocoder_graph)
                if stop.is_set() or not completed:
                    if vocoder_graph is not None:
                        vocoder_graph.discard_pending()
                elif any(adapter is not None and adapter.pending for adapter in adapters):
                    self._wait_ar_idle(self._engine)
                    for adapter in adapters:
                        if stop.is_set():
                            break
                        if adapter is not None and adapter.pending:
                            adapter.prepare_pending()
            finally:
                with self._active_lock:
                    if self._stop is stop:
                        self._owner = self._stop = None
                self._lock.release()
                if owned is not None:
                    owned.close()

        def generate():
            nonlocal completed
            import torch
            from .engine import SAMPLER
            from .native_stream import stream
            engine = self._engine
            reference_caches = {}
            captured = None
            for index, (segment, prepared) in enumerate(selected):
                if stop.is_set():
                    return
                data = prepared.bert
                engine.prompt_cache = dict(data["prompt"])
                engine.prompt_cache["refer_spec"] = list(data["prompt"]["refer_spec"])
                values = dict(SAMPLER, text=segment.text, text_lang=options.text_language,
                    seed=options.seed, top_k=options.top_k, top_p=options.top_p,
                    temperature=options.temperature, speed_factor=options.speed,
                    fragment_interval=options.pause_length, ref_audio_path=data["path"],
                    prompt_text=data["text"], prompt_lang=data["language"])
                # Capability dispatch preserves the native path for other
                # supported SDK settings; failures in the graph never fall back.
                values['sampling_graph'] = (os.environ.get('PERF_AR_SAMPLING_GRAPH') == '1'
                                            and options.top_k == 15 and options.top_p == 1
                                            and options.temperature == 1)
                segment_history = continuity_input if index == 0 else None
                text_only_history = segment_history is not None and segment_history.semantic is None
                wake = getattr(engine, 'performance_worker_wake', None)
                if wake is not None:
                    wake.select(True)
                if text_only_history:
                    values['sampling_graph'] = False
                prefill_graph = getattr(engine, 'performance_ar_prefill_graph', None)
                if prefill_graph is not None:
                    (prefill_graph.disable if text_only_history else prefill_graph.enable)()
                buckets = getattr(engine, 'performance_attention_buckets', None)
                if os.environ.get('PERF_AR_BUCKETS') == '1' and buckets is None:
                    raise RuntimeError('Configured attention buckets were not prepared')
                neural = getattr(engine, 'performance_ar_neural', None)
                neural_small = getattr(engine, 'performance_ar_neural_small', None)
                encoder_graph = getattr(engine, 'performance_encoder_graph', None)
                if tracer is not None:
                    tracer.mark('ar.backend_selection.before')
                handles = [h for h in (buckets, neural, neural_small) if h is not None]
                want_compiled = values['sampling_graph'] and os.environ.get('PERF_AR_COMPILED_NATIVE') == '1'
                retain = (os.environ.get('PERF_AR_RETAIN_SELECTION') == '1' and values['sampling_graph']
                          and handles and all(h.enabled for h in handles)
                          and (buckets is None or buckets.use_compiled == want_compiled))
                if retain:
                    if engine.streaming_accel.engine.model_runner._acquire_graph != handles[-1].acquire_graph:
                        raise RuntimeError('Retained AR graph binding changed externally')
                else:
                    # Restore bindings from the innermost handle out before
                    # selecting the declared optimized or native path.
                    for handle in reversed(handles):
                        if handle.enabled:
                            handle.disable()
                    if buckets is not None:
                        buckets.select_compiled(want_compiled)
                    if values['sampling_graph']:
                        for handle in handles:
                            handle.enable()
                sampling = getattr(engine, 'performance_sampling_graph', None)
                joint = sampling.joint_handle if sampling is not None else None
                if joint is not None:
                    joint.select(values['sampling_graph'])
                if tracer is not None:
                    tracer.mark('ar.backend_selection.after')
                prefill = getattr(engine, 'performance_ar_prefill', None)
                if prefill is not None and segment_history is None:
                    import json
                    ref_info = core.VOICE_LIBRARY.store.get_reference(binding.voice_id, prepared.id)
                    identity = json.dumps([binding.voice_id, prepared.id, ref_info['conditioning_revision'],
                                               ref_info['audio_sha256'], data.get('_semantic_identity_id', id(prepared)), id(engine)], separators=(',', ':'))
                    prefill.enable(no_history=True, reference_identity=identity)
                cache = reference_caches.setdefault(
                    data.get('_conditioning_key', prepared.id),
                    {"key": (data["path"], data["text"], data["language"]),
                     "reference": data["reference"]},
                )
                if tracer is not None:
                    tracer.mark('reference.cache_bound', reference_id=prepared.id)
                def remember(*, phones, bert, semantic):
                    nonlocal captured
                    captured = (phones, bert, semantic)

                borrowed = None
                previous_t2s_accel = None
                cancellation = None
                try:
                    if _complete:
                        # TTS.run's full V5 acoustic path owns its original
                        # RNG/window/postprocessing policy. Borrow the already
                        # prepared worker only while this iterator is active.
                        candidate = engine.streaming_accel
                        if (candidate is None or candidate.engine is None or candidate.max_batch_size != 1
                                or candidate.backend != 'torch_static_cuda_graph' or not candidate.graph_enabled
                                or not values['use_cuda_graph'] or values['use_flash_attention']):
                            raise RuntimeError('Complete synthesis requires the prepared batch-one native AR worker')
                        self._wait_ar_idle(engine)
                        worker = candidate.engine
                        if (not candidate.prepare(True, False) or candidate.engine is not worker
                                or candidate.backend != 'torch_static_cuda_graph' or not candidate.graph_enabled):
                            raise RuntimeError('Complete synthesis could not retain the prepared native AR worker')
                        previous_t2s_accel = engine.t2s_accel
                        borrowed = candidate
                        engine.t2s_accel = borrowed
                        cancellation = _complete_cancellation(engine, borrowed, wake, stop)
                        if data.get('_mixed_conditioning'):
                            from .full_conditioning import synthesize_complete
                            had_acoustic = 'using_vocoder_synthesis' in vars(engine)
                            old_acoustic = vars(engine).get('using_vocoder_synthesis')
                            def restore_acoustic():
                                if had_acoustic:
                                    engine.using_vocoder_synthesis = old_acoustic
                                else:
                                    del engine.using_vocoder_synthesis
                            cancellation.callback(restore_acoustic)
                            def check_conditioning_stop():
                                if stop.is_set():
                                    raise _CompleteCancelled()
                            def conditioned_acoustic(tokens, phones, speed=1.0, sample_steps=None, cfg_rate=None):
                                return synthesize_complete(engine, tokens, phones, cache['reference'],
                                    speed=speed, sample_steps=4 if sample_steps is None else sample_steps,
                                    cfg_rate=0.0 if cfg_rate is None else cfg_rate,
                                    check_cancel=check_conditioning_stop)
                            engine.using_vocoder_synthesis = conditioned_acoustic
                        chunks = engine.run(dict(values, streaming_mode=False, return_fragment=False))
                    elif continuity_capture is None and continuity_input is None:
                        chunks = stream(engine, values, cache=cache, stop=stop)
                    else:
                        chunks = stream(engine, values, cache=cache, stop=stop,
                            continuity_input=segment_history,
                            continuity_capture=remember if continuity_capture is not None else None)
                    head = None
                    if not _complete and os.environ.get('PERF_OUTPUT_HEAD', 'raw') == 'v2':
                        from .output_head import V2Head
                        from aniflive_tts.streaming import INITIAL_LEADING_SILENCE_RETAIN_SECONDS, FOLLOWING_LEADING_SILENCE_RETAIN_SECONDS
                        head = V2Head(retained_seconds=(INITIAL_LEADING_SILENCE_RETAIN_SECONDS if index == 0
                                                       else FOLLOWING_LEADING_SILENCE_RETAIN_SECONDS))
                    try:
                        while True:
                            if _complete and stop.is_set():
                                return
                            # Starlette may resume this iterator on another
                            # worker. End the thread-local inference guard
                            # before yielding, while matching benchmark mode.
                            try:
                                with torch.inference_mode(), (tracer.span('native_pcm_next') if tracer else nullcontext()):
                                    rate, pcm = next(chunks)
                            except StopIteration:
                                break
                            if _complete and stop.is_set():
                                return
                            if _complete and (engine.t2s_accel is not borrowed or borrowed.engine is not worker
                                              or borrowed.backend != 'torch_static_cuda_graph' or not borrowed.graph_enabled):
                                raise RuntimeError('Complete synthesis did not retain the prepared AR worker')
                            if rate != 48000:
                                raise RuntimeError("Native stream returned an unexpected sample rate")
                            import numpy as np
                            samples = np.asarray(pcm)
                            if samples.dtype != np.int16 or samples.ndim != 1:
                                raise RuntimeError("Native stream returned invalid PCM16")
                            if tracer is not None:
                                tracer.mark('pcm.cpu_ready', samples=int(samples.size), sample_rate=rate)
                            if head is not None:
                                with tracer.span('pcm.output_head') if tracer else nullcontext():
                                    samples = head.feed(samples)
                            if samples.size:
                                with tracer.span('pcm.pack') if tracer else nullcontext():
                                    payload = samples.astype("<i2", copy=False).tobytes()
                                if tracer is not None:
                                    tracer.mark('pcm.yield', bytes=len(payload))
                                yield payload
                    finally:
                        with torch.inference_mode():
                            chunks.close()
                finally:
                    try:
                        if borrowed is not None or any(handle is not None for handle in
                               (prefill, prefill_graph, neural_small, neural, buckets, encoder_graph,
                                getattr(engine, 'performance_vocoder_graph', None))):
                            self._wait_ar_idle(engine)
                    finally:
                        changed = False
                        try:
                            if cancellation is not None:
                                cancellation.close()
                        finally:
                            if borrowed is not None:
                                changed = engine.t2s_accel is not borrowed
                                engine.t2s_accel = previous_t2s_accel
                            if prefill is not None and prefill.enabled:
                                prefill.disable()
                        if changed:
                            raise RuntimeError('Complete synthesis changed the borrowed AR binding')
                if not _complete and index < len(selected) - 1 and options.pause_length and not stop.is_set():
                    yield b"\x00\x00" * round(48000 * options.pause_length)
            if continuity_capture is not None:
                with self._active_lock:
                    if stop.is_set():
                        return
                    if captured is None:
                        raise RuntimeError("Completed stream produced no neural continuation")
                    phones, bert, semantic = captured
                    continuity_capture.publish(language=options.text_language, expression=expression,
                                               phones=phones, bert=bert, semantic=semantic)
                    # Publication clones retained tensors on this iterator
                    # thread's stream; the next segment may use another one.
                    import torch
                    torch.cuda.current_stream(engine.configs.device).synchronize()
            if not stop.is_set():
                completed = True

        segments = [segment.text for segment, _ in selected]
        return segments, core._OwnedPCMIterator(generate(), release), 48000, core.MODEL_ID

    def synthesize_request(self, options, requested_model):
        start = time.perf_counter()
        segments, pcm, rate, model = self.prepare_stream(options, requested_model, _complete=True)
        with self._active_lock:
            cancelled = self._stop
        try:
            try:
                data = b"".join(pcm)
            except _CompleteCancelled as error:
                raise core.SynthesisCancelled("Speech synthesis was cancelled") from error
        finally:
            pcm.close()
        if cancelled is not None and cancelled.is_set():
            raise core.SynthesisCancelled("Speech synthesis was cancelled")
        if not data:
            raise RuntimeError("v5 Turbo synthesis returned no audio")
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(data)
        return core.SynthesisResult(output.getvalue(), rate, len(data)//2,
            len(segments), time.perf_counter()-start, {}), model

    def cancel_stream(self, owner):
        with self._active_lock:
            if self._owner != owner or self._stop is None:
                return False
            self._stop.set()
            return True

    @property
    def active_stream_owner(self):
        with self._active_lock:
            return self._owner

    def cancel_active_stream(self):
        with self._active_lock:
            if self._stop is None:
                return False
            self._stop.set()
            return True

    def _configure(self):
        import sys
        from aniflive_tts.model_package import resolve_contained_path
        source = Path(os.environ["ANIFLIVE_TTS_V5_SOURCE_DIR"]).resolve()
        engine_dir = Path(os.environ["ANIFLIVE_TTS_ENGINE_DIR"]).resolve()
        self._engine_manifest = json.loads((engine_dir / "engine-manifest.json").read_text(encoding="utf-8"))
        files = self._engine_manifest["files"]
        paths = {key: str(resolve_contained_path(engine_dir, value, "v5 engine resource"))
                 for key, value in files.items()}
        # Use only the qualified numerical and graph configuration.  Clear inherited
        # tuning variables, which may belong to another service in this process.
        for key in list(os.environ):
            if key.startswith("PERF_"):
                del os.environ[key]
        options = {
            "PERF_AR_BUCKETS": "1", "PERF_AR_COMPACT_HOST": "1",
            "PERF_AR_FAST_TOPK": "1", "PERF_AR_FUSED_MASK": "1",
            "PERF_AR_FUSED_POST": "1", "PERF_AR_JOINT_GRAPH": "1",
            "PERF_AR_NEURAL_BUCKET_SIZE": "512", "PERF_AR_PREFILL_GRAPH": "1",
            "PERF_AR_PREFILL_WORKSPACE": "1", "PERF_AR_RETAIN_SELECTION": "1",
            "PERF_AR_SAMPLING_GRAPH": "1", "PERF_AR_TOPK_THRESHOLD": "1",
            "PERF_AR_WEIGHT_L2_ROLE": "ffn2", "PERF_AR_WORKER_EVENT_WAKE": "1",
            "PERF_CFM_COMPILE": "dynamic", "PERF_CFM_GRAPH": "1",
            "PERF_CFM_POSITIVE_ONLY": "1", "PERF_CUDA_ONLY_SEED": "1",
            "PERF_ENCODER_GRAPH": "1", "PERF_OUTPUT_HEAD": "v2",
            "PERF_QUIET_STATUS": "1", "PERF_VRAM_GIB": "6",
            "PERF_AR_NATIVE_K8_SHARED_ATTENTION_AOT": "1",
            "PERF_AR_NATIVEORDER_RESIDUAL_GEMV_AOT": "1",
            "PERF_AR_NEURAL_ENGINE": paths["ar_neural"], "PERF_ENCODER_ENGINE": paths["encoder"],
            "PERF_DIT_ENGINE": paths["dit"], "PERF_VOCODER_ENGINE": paths["vocoder"],
            "PERF_CFM_SHAPE_PLANS": json.dumps({"579": paths["dit_579"]}),
            "PERF_AR_NATIVE_K8_ALIGNMENT_GUARD": paths["attention_guard"],
            "PERF_AR_GEMV_ALIGNMENT_GUARD": paths["gemv_guard"],
            "PERF_AR_CPP_LAYERNORM_DIRECT_LIBRARY": paths["layernorm_plugin"],
            "PERF_AR_WEIGHT_L2_SHIM": paths["weight_l2"],
            "PERF_AR_JOINT_AUDIT_LIBRARY": paths["joint_graph"],
        }
        os.environ.update(options)
        cache = Path(os.environ.get("ANIFLIVE_TTS_CACHE_DIR", "/data/cache")).resolve()
        os.environ["ANIFLIVE_TTS_CACHE_DIR"] = str(cache)
        (cache / "v5").mkdir(parents=True, exist_ok=True)
        sys.path[:0] = [str(source), str(source / "GPT_SoVITS")]
        self._previous_cwd = Path.cwd()
        os.chdir(source)

    def _is_busy(self):
        with self._active_lock:
            return self._stop is not None or self._lock.locked()

    def health(self):
        return {"service": core.SERVICE_NAME, "version": core.SERVICE_VERSION,
                "status": "ok" if self.ready else "not_ready", "ready": self.ready,
                "model": core.MODEL_ID, "backend": "hybrid-cuda", "sample_rate": 48000}
