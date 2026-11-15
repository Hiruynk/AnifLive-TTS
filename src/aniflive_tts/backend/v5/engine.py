"""V5 Turbo model construction and coordinated GPU lifecycle."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
@contextmanager
def _compatible_interop_setting(torch):
    """Preserve a process-wide inter-op setting when upstream is imported again."""
    original = torch.set_num_interop_threads
    def set_once(count):
        try:
            original(count)
        except RuntimeError as error:
            if ("cannot set number of interop threads" not in str(error)
                    or torch.get_num_interop_threads() != count):
                raise
    torch.set_num_interop_threads = set_once
    try:
        yield
    finally:
        torch.set_num_interop_threads = original

SAMPLER = {
    'top_k': 15, 'top_p': 1.0, 'temperature': 1.0, 'repetition_penalty': 1.35,
    'text_split_method': 'cut0', 'batch_size': 1, 'split_bucket': False,
    'parallel_infer': False, 'use_cuda_graph': True, 'use_flash_attention': False,
    'sample_steps': 4, 'cfg_rate': 0.0, 'speed_factor': 1.0,
    'fragment_interval': 0.3,
}


def effective_precision_settings():
    """Read the active per-operator policy without mixing legacy TF32 getters."""
    import torch
    return {'cudnn_deterministic': bool(torch.backends.cudnn.deterministic),
                'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
                'cudnn_enabled': bool(torch.backends.cudnn.enabled),
                'deterministic_algorithms': bool(torch.are_deterministic_algorithms_enabled()),
                'global_fp32_precision': str(torch.backends.fp32_precision),
                'matmul_fp32_precision': str(torch.backends.cuda.matmul.fp32_precision),
                'cudnn_conv_fp32_precision': str(torch.backends.cudnn.conv.fp32_precision),
                'cudnn_rnn_fp32_precision': str(torch.backends.cudnn.rnn.fp32_precision)}


def configure_reference_precision():
    """Set and record the explicit reference-computation policy."""
    import torch
    before = effective_precision_settings()
    torch.backends.fp32_precision = 'ieee'
    torch.backends.cuda.matmul.fp32_precision = 'ieee'
    torch.backends.cudnn.conv.fp32_precision = 'ieee'
    torch.backends.cudnn.rnn.fp32_precision = 'ieee'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    after = effective_precision_settings()
    expected = {'global_fp32_precision': 'ieee', 'matmul_fp32_precision': 'ieee',
                'cudnn_conv_fp32_precision': 'ieee', 'cudnn_rnn_fp32_precision': 'ieee',
                'cudnn_benchmark': False, 'cudnn_deterministic': True}
    if any(after[key] != value for key, value in expected.items()):
        raise RuntimeError(f'Unable to apply reference precision policy: {after}')
    return {'before': before, 'after': after}

@contextmanager
def shared_gpu_lease(manager=None):
    from aniflive_tts.inference_lease import InferenceGPUResourceLease
    manager = manager or InferenceGPUResourceLease.from_env()
    context = manager.acquire('inference:v5turbo')
    with context as handle:
        engines = []
        try:
            yield handle, engines
        finally:
            # Close asynchronous AR workers and clear CUDA state BEFORE lease release.
            for engine in engines:
                encoder_graph = getattr(engine, 'performance_encoder_graph', None)
                if encoder_graph is not None:
                    encoder_graph.close()
                    engine.performance_encoder_graph = None
                    encoder_graph = None
                encoder = getattr(engine, 'performance_encoder', None)
                if encoder is not None:
                    encoder.disable()
                    engine.performance_encoder = None
                    encoder = None
                sampling = getattr(engine, 'performance_sampling_graph', None)
                if sampling is not None and sampling.joint_handle is not None:
                    sampling.joint_handle.close()
                for name in ('performance_ar_neural_small', 'performance_ar_neural'):
                    neural = getattr(engine, name, None)
                    if neural is not None:
                        neural.close()
                        setattr(engine, name, None)
                    neural = None
                buckets = getattr(engine, 'performance_attention_buckets', None)
                if buckets is not None:
                    buckets.disable()
                    engine.performance_attention_buckets = None
                    buckets = None
                prefill = getattr(engine, 'performance_ar_prefill', None)
                if prefill is not None:
                    prefill.close()
                    engine.performance_ar_prefill = None
                    prefill = None
                prefill_graph = getattr(engine, 'performance_ar_prefill_graph', None)
                if prefill_graph is not None:
                    prefill_graph.close()
                    engine.performance_ar_prefill_graph = None
                    prefill_graph = None
                sampling = getattr(engine, 'performance_sampling_graph', None)
                if sampling is not None and sampling.enabled:
                    sampling.disable()
                wake = getattr(engine, 'performance_worker_wake', None)
                if wake is not None:
                    wake.close()
                    engine.performance_worker_wake = None
                    wake = None
                cfm_graph = getattr(engine, 'performance_cfm_graph', None)
                if cfm_graph is not None:
                    cfm_graph.close()
                    engine.performance_cfm_graph = None
                    cfm_graph = None
                for name in ('t2s_accel', 'streaming_accel'):
                    accelerator = getattr(engine, name, None)
                    if accelerator is not None:
                        accelerator.close()
                        setattr(engine, name, None)
                    accelerator = None
                for name in ('t2s_model', 'vits_model', 'vocoder', 'bert_model', 'cnhuhbert_model',
                             'text_preprocessor', 'using_vocoder_synthesis', 'set_ref_audio'):
                    if hasattr(engine, name):
                        setattr(engine, name, None)
                engine.prompt_cache.clear()
                cache = getattr(engine, 'performance_reference_cache', None)
                if cache is not None:
                    cache.clear()
                engine.performance_sampling_graph = None
                sampling = None
            module = __import__('sys').modules.get('TTS_infer_pack.TTS')
            if module is not None:
                module.resample_transform_dict.clear()
                module.resampy_filter_cache.clear()
            import gc
            import sys
            gc.collect()
            import ctypes
            ctypes.CDLL('libc.so.6').malloc_trim(0)
            torch = sys.modules.get('torch')
            if torch is not None and torch.cuda.is_initialized():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()


def _refresh_reference_activation(source_root):
    """Reset the restored source's TorchScript profiling state for a new model."""
    import importlib
    import sys
    from pathlib import Path
    module = sys.modules.get("module.commons")
    expected = (Path(source_root) / "GPT_SoVITS/module/commons.py").resolve()
    if module is None or Path(getattr(module, "__file__", "")).resolve() != expected:
        raise RuntimeError("Unexpected V5 reference activation source")
    previous = module.fused_add_tanh_sigmoid_multiply
    module = importlib.reload(module)
    if (Path(getattr(module, "__file__", "")).resolve() != expected
            or module.fused_add_tanh_sigmoid_multiply is previous):
        raise RuntimeError("V5 reference activation could not be reset")

def load_engine(engines):
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(Path(os.environ['ANIFLIVE_TTS_CACHE_DIR']) / 'v5' / 'torchinductor')
    os.environ['TRITON_CACHE_DIR'] = str(Path(os.environ['ANIFLIVE_TTS_CACHE_DIR']) / 'v5' / 'triton')
    os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
    os.environ['MAX_JOBS'] = '1'
    import torch
    import sys
    restored_reference_source = "module.commons" in sys.modules
    # Import the package itself explicitly before any model/reference work:
    # its initializer sets global TF32 and cuDNN benchmark and owns inter-op.
    with _compatible_interop_setting(torch):
        import GPT_SoVITS.Accel.PyTorch  # noqa: F401
    from GPT_SoVITS.Accel.adapter import AccelInference as AccelInference
    from TTS_infer_pack.TTS import TTS, TTS_Config
    import sys
    source_root = Path(os.environ["ANIFLIVE_TTS_V5_SOURCE_DIR"]).resolve()
    for name in ("TTS_infer_pack.TTS", "GPT_SoVITS.Accel.PyTorch"):
        module_file = Path(vars(sys.modules[name]).get("__file__", "")).resolve()
        if not module_file.is_relative_to(source_root):
            raise RuntimeError(f"Unexpected V5 source module origin: {name}")
    if restored_reference_source:
        # Restored source modules retain TorchScript's warmed profiling state.
        # Re-create the scripted activation before model construction so every
        # activation has the same native FP16 first-use behavior as startup.
        _refresh_reference_activation(source_root)
    precision_policy = configure_reference_precision()
    print(json.dumps({'reference_precision_policy': precision_policy}), flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required; no CPU inference fallback')
    torch.set_num_threads(2)
    budget = float(os.environ.get('PERF_VRAM_GIB', '6')) * 1024**3
    torch.cuda.set_per_process_memory_fraction(budget / torch.cuda.get_device_properties(0).total_memory)
    root = Path(os.environ['ANIFLIVE_TTS_V5_MODEL_DIR'])
    config = TTS_Config({'custom': {
        'version': 'v5turbo', 'device': 'cuda:0', 'is_half': True,
        't2s_weights_path': str(root / 's1v3.ckpt'),
        'vits_weights_path': str(root / 'gsv-v5-pretrained/s2Gv5turbo.pth'),
        'bert_base_path': str(root / 'chinese-roberta-wwm-ext-large'),
        'cnhuhbert_base_path': str(root / 'chinese-hubert-base'),
    }})
    config.configs_path = str(Path(os.environ['ANIFLIVE_TTS_CACHE_DIR']) / 'v5' / 'tts_infer.generated.yaml')
    engine = TTS(config)
    engines.append(engine)
    actual_after_load = effective_precision_settings()
    if actual_after_load != precision_policy['after']:
        raise RuntimeError(f'Model construction changed the reference precision policy: {actual_after_load}')
    engine.performance_precision_policy = {**precision_policy, 'after_model_load': actual_after_load}
    for model in (engine.t2s_model, engine.vits_model, engine.vocoder,
                  engine.bert_model, engine.cnhuhbert_model):
        model.eval().requires_grad_(False)
    compile_mode = os.environ.get('PERF_CFM_COMPILE', '')
    dit_engine = os.environ.get('PERF_DIT_ENGINE', '')
    if dit_engine:
        from .trt_dit import TensorRTDiT
        engine.vits_model.cfm.estimator = TensorRTDiT(engine.vits_model.cfm.estimator, dit_engine).eval()
        engine.performance_cfm_backend = 'TensorRT-11'
    elif compile_mode:
        if compile_mode != 'dynamic':
            raise ValueError('Only the validated dynamic CFM compile mode is supported')
        torch._dynamo.config.suppress_errors = False
        engine.vits_model.cfm.estimator = torch.compile(
            engine.vits_model.cfm.estimator, mode='default', dynamic=True, fullgraph=True)
        engine.performance_cfm_compiled = True
        engine.performance_cfm_backend = 'compiled-pytorch'
    vocoder_engine = os.environ.get('PERF_VOCODER_ENGINE', '')
    if vocoder_engine:
        from .trt_vocoder import V5TRTVocoder
        engine.vocoder = V5TRTVocoder(vocoder_engine)
        engine.performance_vocoder_backend = 'TensorRT-11'
    encoder_engine = os.environ.get('PERF_ENCODER_ENGINE', '')
    if encoder_engine:
        from .trt_encoder import install_target_encoder
        engine.performance_encoder = install_target_encoder(engine.vits_model, encoder_engine)
        engine.performance_encoder_backend = 'TensorRT-11'
    # CPU checkpoint staging is no longer needed after GPU placement. Return
    # freed glibc arenas before another multilingual frontend loads its weights.
    import gc
    import ctypes
    gc.collect()
    ctypes.CDLL('libc.so.6').malloc_trim(0)
    return engine
