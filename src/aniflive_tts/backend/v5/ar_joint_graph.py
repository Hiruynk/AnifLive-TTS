"""A retained non-RNG neural child followed by the unchanged Torch sampler.

Insert the child through CUDA capture dependencies, not cuGraphLaunch (which
rejects stream capture), and let the outer Torch graph own its RNG prologue.
"""
import ctypes
import time


class JointGraph:
    def __init__(self, sampler, neural, *, audit_library=None):
        import torch
        runner = sampler.runner
        if (neural.runner is not runner or neural.weight_l2 is None
                or not hasattr(sampler, '_joint_capture_body') or sampler.joint_handle is not None
                or not sampler.fused_post or not sampler.fuse_attention_mask):
            raise ValueError('Joint graph requires one retained L2 neural graph and the qualified fused sampler')
        if runner.bound_sessions or runner.waiting_bind or any(runner.slot_sessions):
            raise RuntimeError('Joint graph capture requires an idle runner')
        self.sampler, self.neural = sampler, neural
        self.enabled, self.closed, self.hits = False, False, 0
        self.graph = torch.cuda.CUDAGraph(keep_graph=True)
        library = ctypes.CDLL('libcuda.so.1')
        signatures = {
            'cuStreamGetCaptureInfo_v2': [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)), ctypes.POINTER(ctypes.c_size_t)],
            'cuGraphAddChildGraphNode': [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_void_p],
            'cuStreamUpdateCaptureDependencies': [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint],
            'cuGraphGetNodes': [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)],
            'cuGraphNodeGetType': [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)],
        }
        for name, types in signatures.items():
            function = getattr(library, name)
            function.argtypes, function.restype = types, ctypes.c_int
        def call(name, *values):
            code = getattr(library, name)(*values)
            if code:
                raise RuntimeError(f'{name} failed with CUDA status {code}')
        buffers = list(sampler._joint_changed) + [runner.xy_dec_buf]
        buffers += [t for pair in neural.kv for t in pair] + list(neural.outputs.values())
        buffers = list({id(t): t for t in buffers}.values())
        started = time.perf_counter()
        with torch.cuda.device(runner.device), torch.inference_mode():
            torch.cuda.synchronize(runner.device)
            saved = [t.clone() for t in buffers]
            rng = torch.cuda.get_rng_state(runner.device)
            side = torch.cuda.Stream(device=runner.device)
            side.wait_stream(torch.cuda.current_stream(runner.device))
            try:
                with torch.cuda.graph(self.graph, stream=side):
                    status, capture_id = ctypes.c_int(), ctypes.c_uint64()
                    capture_graph, child = ctypes.c_void_p(), ctypes.c_void_p()
                    deps, count = ctypes.POINTER(ctypes.c_void_p)(), ctypes.c_size_t()
                    call('cuStreamGetCaptureInfo_v2', side.cuda_stream, ctypes.byref(status), ctypes.byref(capture_id),
                         ctypes.byref(capture_graph), ctypes.byref(deps), ctypes.byref(count))
                    if status.value != 1 or not capture_graph.value:
                        raise RuntimeError('Joint neural insertion requires active stream capture')
                    call('cuGraphAddChildGraphNode', ctypes.byref(child), capture_graph, deps, count,
                         neural.graph.raw_cuda_graph())
                    call('cuStreamUpdateCaptureDependencies', side.cuda_stream, ctypes.byref(child), 1, 0)
                    sampler._joint_capture_body()
                self.graph.instantiate()
                count = ctypes.c_size_t()
                call('cuGraphGetNodes', self.graph.raw_cuda_graph(), None, ctypes.byref(count))
                nodes = (ctypes.c_void_p * count.value)()
                call('cuGraphGetNodes', self.graph.raw_cuda_graph(), nodes, ctypes.byref(count))
                self.node_types = []
                for node in nodes:
                    kind = ctypes.c_int()
                    call('cuGraphNodeGetType', node, ctypes.byref(kind))
                    self.node_types.append(kind.value)
                if self.node_types.count(4) != 1:
                    raise RuntimeError('Joint capture must contain exactly one neural child graph')
                self.l2_audit = None
                if audit_library is not None:
                    from .ar_joint_l2_audit import audit
                    self.l2_audit = audit(neural.graph, self.graph, audit_library)
                    self.l2_audit['context_limit_at_capture'] = neural.weight_l2.get_limit()
                torch.cuda.synchronize(runner.device)
                # Real loaded weights/runner buffers: compare every modified
                # buffer and complete RNG, including changed input and poisoning.
                self.parity = []
                for changed, seed in ((False, 1234), (True, 3407)):
                    def reset():
                        for target, value in zip(buffers, saved):
                            target.copy_(value)
                        runner.completed_buf.zero_()
                        runner.step_buf.fill_(1)
                        position = min(int(runner.prefill_len_buf[0].item()) + 1, neural.bucket_size - 1)
                        runner.input_pos_buf.fill_(max(position, 1))
                        neural.mask.copy_((sampler.mask_positions < max(position, 1)).view(1, 1, 1, -1).expand_as(neural.mask))
                        if changed:
                            runner.xy_pos_buf.mul_(.99).add_(.001)
                        torch.cuda.manual_seed(seed)
                    reset()
                    neural.graph.replay()
                    sampler.graph.replay()
                    torch.cuda.synchronize(runner.device)
                    expected = [t.clone() for t in buffers]
                    expected_rng = torch.cuda.get_rng_state(runner.device)
                    reset()
                    for target in neural.outputs.values():
                        target.fill_(float('nan'))
                    self.graph.replay()
                    torch.cuda.synchronize(runner.device)
                    row = {'changed_input': changed, 'seed': seed,
                           'all_modified_buffers_exact': all(torch.equal(a, b) for a, b in zip(buffers, expected)),
                           'cuda_rng_exact': bool(torch.equal(expected_rng, torch.cuda.get_rng_state(runner.device)))}
                    self.parity.append(row)
                    if not row['all_modified_buffers_exact'] or not row['cuda_rng_exact']:
                        raise RuntimeError('Joint neural/sampler graph failed buffer or RNG parity')
            finally:
                torch.cuda.synchronize(runner.device)
                for target, value in zip(buffers, saved):
                    target.copy_(value)
                torch.cuda.set_rng_state(rng, runner.device)
                torch.cuda.synchronize(runner.device)
        self.capture_seconds = time.perf_counter() - started
        sampler.joint_handle = self

    def select(self, enabled):
        runner = self.neural.runner
        if self.closed or runner.bound_sessions or runner.waiting_bind or any(runner.slot_sessions):
            raise RuntimeError('Joint graph selection requires an idle live runner')
        self.enabled = bool(enabled)

    def replay(self):
        if not self.enabled or self.closed:
            raise RuntimeError('Joint graph is not enabled')
        self.graph.replay()
        self.hits += 1

    def close(self):
        import torch
        if self.closed:
            return
        self.select(False)
        if self.sampler.joint_handle is not self:
            raise RuntimeError('Joint graph binding changed externally')
        torch.cuda.synchronize(self.neural.runner.device)
        self.graph.reset()
        self.sampler.joint_handle = None
        self.closed = True
