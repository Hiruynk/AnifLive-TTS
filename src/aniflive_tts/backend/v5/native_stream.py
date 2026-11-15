"""Wire the pinned vendor v5 stream and optional research window schedule."""
from __future__ import annotations

from contextlib import nullcontext
import math
import time


def stream(engine, values, *, cache=None, events=None, stop=None,
           continuity_input=None, continuity_capture=None):
    from TTS_infer_pack.TTS import set_seed, denorm_spec
    if getattr(engine, 'performance_cuda_only_seed', False):
        from .cuda_seed import set_seed
    from TTS_infer_pack.streaming import (
        PipelineStop, validate_streaming_options, v5_reference,
        semantic_updates, stream_v5_windows,
    )
    from TTS_infer_pack.text_segmentation_method import splits

    values = dict(values)
    options = getattr(engine, 'performance_streaming_options', None)
    if options is not None:
        for name in ('first_window_seconds', 'later_window_seconds', 'semantic_interval'):
            if name in options:
                values.setdefault('_' + name, options[name])
    first_seconds = values.pop('_first_window_seconds', None)
    later_seconds = values.pop('_later_window_seconds', None)
    interval = values.pop('_semantic_interval', 16)
    if type(interval) is not int or not 1 <= interval <= 16:
        raise ValueError('Semantic interval must be an integer between 1 and 16')
    if first_seconds is None and later_seconds is not None:
        raise ValueError('A later window requires an explicit first window')
    if first_seconds is not None:
        if later_seconds is None:
            later_seconds = 2.5
        for name, value in (('first_window_seconds', first_seconds), ('later_window_seconds', later_seconds)):
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        first_seconds, later_seconds = float(first_seconds), float(later_seconds)
    acoustic_stage_event = None
    if '_acoustic_stage_event' in values:
        from threading import Event
        acoustic_stage_event = values['_acoustic_stage_event']
        if not isinstance(acoustic_stage_event, Event):
            raise TypeError('_acoustic_stage_event must be a threading.Event')
    req = dict(values, gpt_streaming=True, streaming_mode=True, media_type='raw',
               streaming_window_mode=values.get('streaming_window_mode', 'fixed'),
               streaming_chunk_seconds=values.get('streaming_chunk_seconds', 2.0),
               streaming_lookahead_seconds=values.get('streaming_lookahead_seconds', 0.12),
               streaming_overlap_seconds=values.get('streaming_overlap_seconds', 0.04),
               gpt_streaming_lookahead_seconds=values.get('gpt_streaming_lookahead_seconds', 0.5), min_chunk_length=interval,
               overlap_length=2)
    req.pop('_acoustic_stage_event', None)
    tracer = None
    validate_streaming_options(req, 'v5turbo')
    if first_seconds is not None and req['streaming_window_mode'] != 'fixed':
        raise ValueError('First-window specialization requires fixed streaming mode')
    if req['sample_steps'] != 4 or req['cfg_rate'] != 0:
        raise ValueError('Native benchmark requires four-step Turbo, CFG 0')
    if tracer is not None:
        core_frames = max(1, round((first_seconds if first_seconds is not None else req['streaming_chunk_seconds']) * 100))
        later_core_frames = max(1, round((later_seconds if first_seconds is not None else req['streaming_chunk_seconds']) * 100))
        future_frames = round(req['streaming_lookahead_seconds'] * 100)
        holdback_frames = max(16, round(req['gpt_streaming_lookahead_seconds'] * 100))
        tracer.mark('streaming.budget', core_frames=core_frames, later_core_frames=later_core_frames, future_frames=future_frames,
            holdback_frames=holdback_frames, mode=req['streaming_window_mode'],
            overlap_seconds=req['streaming_overlap_seconds'], speed=req['speed_factor'],
            first_threshold_tokens=math.ceil((core_frames + future_frames + holdback_frames) * req['speed_factor'] / 4),
            semantic_interval=req['min_chunk_length'])
    engine.stop_flag = False
    stop = stop if stop is not None else PipelineStop(engine)
    events = events if events is not None else []
    start = time.perf_counter()
    if tracer is not None:
        tracer.mark('rng.seed.before')
    set_seed(req['seed'])
    if tracer is not None:
        tracer.mark('rng.seed.after')
    def rng_evidence(stage):
        if values.get('diagnostic_rng'):
            import torch
            import hashlib
            events.append(dict(stage=stage, seconds=time.perf_counter()-start,
                               rng_sha256=hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest()))
    rng_evidence('rng_before_reference')
    prompt = engine.prompt_cache
    text = req['prompt_text'].strip('\n')
    if text[-1] not in splits:
        text += '。'
    key = (req['ref_audio_path'], text, req['prompt_lang'])
    if cache is not None and cache.get('key') == key:
        reference = cache['reference']
        if tracer is not None:
            tracer.mark('reference.cache_hit', hit=True)
    else:
        if tracer is not None:
            tracer.mark('reference.cache_hit', hit=False)
        if prompt['ref_audio_path'] != req['ref_audio_path']:
            engine.set_ref_audio(req['ref_audio_path'])
        if prompt['prompt_text'] != text or prompt['prompt_lang'] != req['prompt_lang']:
            phones, bert, normalized = engine.text_preprocessor.segment_and_extract_feature_for_text(
                text, req['prompt_lang'], engine.configs.version)
            prompt.update(prompt_text=text, prompt_lang=req['prompt_lang'], phones=phones,
                          bert_features=bert, norm_text=normalized)
        reference = v5_reference(engine)
        if cache is not None:
            cache.update(key=key, reference=reference)
    rng_evidence('rng_after_reference')
    if not hasattr(engine, 'streaming_accel'):
        engine.streaming_accel = None
    if getattr(engine.streaming_accel, '_text_history_mask_failed', False):
        raise RuntimeError('AR worker requires restart after a text-history mask failure')
    sampling_graph = getattr(engine, 'performance_sampling_graph', None)
    if values.get('sampling_graph'):
        flash = bool(req.get('use_flash_attention', False))
        from tools.acceleration import create_acceleration
        from .ar_sampling_graph import SamplingGraph
        if engine.streaming_accel is None:
            engine.streaming_accel = create_acceleration(
                engine.configs.t2s_weights_path, device=engine.configs.device,
                dtype=engine.precision, max_batch_size=1, max_seq_length=2048,
                use_cuda_graph=True, use_flash_attention=flash)
        if engine.streaming_accel is None or not engine.streaming_accel.prepare(True, flash):
            raise RuntimeError('Requested sampling graph has no native neural graph')
        runner = engine.streaming_accel.engine.model_runner
        if sampling_graph is None:
            sampling_graph = SamplingGraph(runner)
            engine.performance_sampling_graph = sampling_graph
        elif sampling_graph.runner is not runner:
            raise RuntimeError('AR runner changed after sampling graph installation')
        if not sampling_graph.enabled:
            sampling_graph.enable()
    elif sampling_graph is not None and sampling_graph.enabled:
        sampling_graph.disable()
    data = engine.text_preprocessor.preprocess(
        req['text'], req['text_lang'], req['text_split_method'], engine.configs.version)
    if not data:
        raise ValueError('Text frontend produced no segments')
    # Target BERT tensors may have been produced on a different iterator
    # thread's CUDA stream. Only continuation requests need this handoff.
    text_ready = None
    if continuity_input is not None or continuity_capture is not None:
        import torch
        text_ready = torch.cuda.Event()
        text_ready.record(torch.cuda.current_stream(engine.configs.device))
    captured = None
    rng_evidence('rng_after_text')
    for index, item in enumerate(data):
        if stop.is_set():
            return
        if text_ready is not None:
            torch.cuda.current_stream(engine.configs.device).wait_event(text_ready)
        prompt_data = prompt
        semantic_prompt = prompt['prompt_semantic'].reshape(1, -1)
        if index == 0 and continuity_input is not None:
            history_phones = list(continuity_input.phones)
            history_bert = continuity_input.bert
            history_semantic = continuity_input.semantic
            if not history_phones or history_bert.ndim != 2 or history_bert.shape[-1] != len(history_phones):
                raise ValueError('Invalid committed phone/BERT continuation')
            if history_semantic is not None and (history_semantic.ndim != 2 or history_semantic.shape[0] != 1):
                raise ValueError('Invalid committed semantic continuation')
            prompt_data = dict(prompt)
            prompt_data['phones'] = list(prompt['phones']) + history_phones
            prompt_data['bert_features'] = torch.cat((prompt['bert_features'],
                history_bert.to(device=prompt['bert_features'].device, dtype=prompt['bert_features'].dtype)), dim=1)
            if history_semantic is not None:
                semantic_prompt = torch.cat((semantic_prompt,
                    history_semantic.to(device=semantic_prompt.device, dtype=semantic_prompt.dtype)), dim=1)
            max_sequence = getattr(engine.streaming_accel, 'max_seq_length', None) or 2048
            if len(prompt_data['phones']) + len(item['phones']) + semantic_prompt.shape[-1] >= max_sequence - 1:
                raise ValueError('Committed continuation exceeds native AR context capacity')
        batches, _ = engine.to_batch([item], prompt_data=prompt_data, batch_size=1,
                                     split_bucket=False, device=engine.configs.device,
                                     precision=engine.precision)
        batch = batches[0]
        # AR takes [batch, semantic], while reference[2] has the acoustic [B,1,S] shape.
        mask_scope = nullcontext()
        masked_ar_engine = None
        if index == 0 and continuity_input is not None and continuity_input.semantic is None:
            if not req.get('use_cuda_graph', True) or req.get('use_flash_attention', True):
                raise ValueError('Text-only continuity requires the native Torch-static CUDA graph backend')
            if engine.streaming_accel is None:
                from tools.acceleration import create_acceleration
                engine.streaming_accel = create_acceleration(
                    engine.configs.t2s_weights_path, device=engine.configs.device,
                    dtype=engine.precision, max_batch_size=1, max_seq_length=2048,
                    use_cuda_graph=True, use_flash_attention=False)
            if engine.streaming_accel is None or not engine.streaming_accel.prepare(True, False):
                raise RuntimeError('Text-only continuity has no native Torch-static AR graph')
            masked_ar_engine = engine.streaming_accel.engine
            from .text_history_mask import text_history_mask
            mask_scope = text_history_mask(engine.streaming_accel, len(prompt['phones']),
                                           len(continuity_input.phones), int(semantic_prompt.shape[-1]))
        updates = semantic_updates(engine, batch['all_phones'][0], semantic_prompt,
                                   batch['all_bert_features'][0], req, stop)
        semantic_state = {'final': False, 'tokens': 0}

        def observed():
            try:
                for tokens, finished in updates:
                    if masked_ar_engine is not None and engine.streaming_accel.engine is not masked_ar_engine:
                        raise RuntimeError('Text-only continuity AR runner changed while the mask was active')
                    semantic_state.update(final=finished, tokens=int(tokens.numel()))
                    if tracer is not None:
                        tracer.mark('semantic.update', segment=index, tokens=int(tokens.numel()), final=bool(finished))
                    event = dict(stage='semantic_update', segment=index,
                                 seconds=time.perf_counter()-start,
                                 tokens=int(tokens.numel()), final=finished)
                    if finished and (values.get('diagnostic_rng') or values.get('capture_semantic')):
                        import hashlib
                        event['tokens_sha256'] = hashlib.sha256(tokens.detach().cpu().numpy().tobytes()).hexdigest()
                    if finished and continuity_capture is not None:
                        semantic_state['capture'] = {
                            'phones': tuple(item['phones']),
                            'bert': item['bert_features'].detach().clone(),
                            'semantic': tokens.detach().reshape(1, -1).clone(),
                        }
                        semantic_state['capture_ready'] = torch.cuda.Event()
                        semantic_state['capture_ready'].record(torch.cuda.current_stream(engine.configs.device))
                    events.append(event)
                    if acoustic_stage_event is None:
                        yield tokens, finished
                    else:
                        acoustic_stage_event.set()
                        try:
                            yield tokens, finished
                        finally:
                            acoustic_stage_event.clear()
            finally:
                updates.close()

        emit_scope = nullcontext()
        emit_budget = getattr(engine, 'performance_emit_budget', False)
        if type(emit_budget) is not bool:
            raise TypeError('performance_emit_budget must be a boolean')
        if emit_budget:
            from .ar_emit_budget import defer_inactive_partials
            emit_scope = defer_inactive_partials(engine.streaming_accel, req,
                                                 first_seconds=first_seconds, events=events)
        with mask_scope, emit_scope:
            observed_updates = observed()
            if first_seconds is None:
                chunks = stream_v5_windows(engine.vits_model, engine.vocoder, denorm_spec,
                                          engine.configs.device, observed_updates, item['phones'],
                                          reference, req, stop)
            else:
                from .first_window_stream import stream_first_windows
                chunks = stream_first_windows(engine.vits_model, engine.vocoder, denorm_spec,
                    engine.configs.device, observed_updates, item['phones'], reference, req, stop,
                    first_seconds=first_seconds, later_seconds=later_seconds, events=events, tracer=tracer)
            try:
                for pcm in chunks:
                    events.append(dict(stage='generated_pcm', segment=index,
                                       seconds=time.perf_counter()-start, samples=int(pcm.size),
                                       consumed_semantic_final=semantic_state['final'],
                                       consumed_semantic_tokens=semantic_state['tokens']))
                    yield 48000, pcm
            finally:
                try:
                    chunks.close()
                finally:
                    try:
                        observed_updates.close()
                    finally:
                        updates.close()
        if continuity_capture is not None:
            if stop.is_set():
                return
            if not semantic_state['final'] or 'capture' not in semantic_state:
                raise RuntimeError('Native stream ended without final continuation tokens')
            semantic_state['capture_ready'].synchronize()
            captured = semantic_state['capture']
        if index < len(data)-1 and req['fragment_interval']:
            import numpy as np
            yield 48000, np.zeros(round(48000 * req['fragment_interval']), dtype=np.int16)
    if continuity_capture is not None and not stop.is_set():
        if captured is None:
            raise RuntimeError('Native stream produced no continuation state')
        continuity_capture(**captured)
