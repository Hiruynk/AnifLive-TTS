"""Fixed first/later window schedule, derived from pinned GPT-SoVITS streaming.

Source: vendor/GPT_SoVITS/TTS_infer_pack/streaming.py, commit
04a44eb977586645f33594b72af13a035a2056c6. MIT License, Copyright (c) 2024
RVC-Boss; see the accompanying vendor/LICENSE for the complete license.
"""
import math

import torch


@torch.inference_mode()
def stream_first_windows(vq_model, vocoder, denorm_spec, device, token_updates, phones,
                         reference, req, stop, *, first_seconds, later_seconds, events, tracer=None):
    for name, value in (('first_window_seconds', first_seconds), ('later_window_seconds', later_seconds)):
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if req.get('streaming_window_mode', 'fixed') != 'fixed':
        raise ValueError('First-window specialization requires fixed streaming mode')
    first_core = max(1, round(float(first_seconds) * 100))
    later_core = max(1, round(float(later_seconds) * 100))
    future = round(req['streaming_lookahead_seconds'] * 100)
    holdback = max(16, round(req['gpt_streaming_lookahead_seconds'] * 100))
    fade_frames = min(round(req['streaming_overlap_seconds'] * 100), first_core - 1, later_core - 1)
    fade = fade_frames * 480
    left_frames = max(24, fade_frames)
    ref_features, ref_mel = reference[4], reference[6]
    count = min(500, ref_features.shape[-1], ref_mel.shape[-1])
    ref_features, ref_mel = ref_features[..., -count:], ref_mel[..., -count:]
    if count + max(first_core, later_core) + future > 1000:
        raise ValueError('Reference plus streaming window exceeds the V5 1000-frame context')
    rng = torch.Generator(device=device).manual_seed(torch.initial_seed())
    phone_tensor = torch.tensor(phones, device=device, dtype=torch.long).unsqueeze(0)
    alpha = torch.linspace(0, 1, fade, device=ref_mel.device, dtype=torch.float32) if fade else None
    history = ref_mel[..., :0]
    feature_history = ref_features[..., :0]
    pending = None
    cursor = previous_tokens = 0
    for tokens, finished in token_updates:
        if stop.is_set():
            return
        if tokens.numel() < previous_tokens:
            raise RuntimeError('Semantic stream moved backwards')
        previous_tokens = tokens.numel()
        core = first_core if cursor == 0 else later_core
        needed = math.ceil((cursor + core + future + holdback) * req['speed_factor'] / 4)
        if not finished and tokens.numel() < needed:
            continue
        target, _ = vq_model.decode_encp(
            tokens.reshape(1, 1, -1).long(), phone_tensor, reference[3], reference[5], req['speed_factor'])
        total = target.shape[-1]
        safe_end = total if finished else max(0, total - future - holdback)
        while cursor < total:
            core = first_core if cursor == 0 else later_core
            if not finished and cursor + core > safe_end:
                break
            if stop.is_set():
                return
            end = min(cursor + core, total)
            if cursor == 0:
                prompt, features = ref_mel, ref_features
            else:
                tail = min(32, count, history.shape[-1])
                prefix = count - tail
                prompt = torch.cat((ref_mel[..., :prefix], history[..., -tail:]), -1)
                features = torch.cat((ref_features[..., :prefix], target[..., cursor-tail:cursor]), -1)
            mu = torch.cat((features, target[..., cursor:min(end + future, total)]), -1).transpose(2, 1)
            lengths = torch.full((mu.shape[0],), mu.shape[1], device=mu.device, dtype=torch.long)
            window = dict(stage='acoustic_window', cursor=cursor, end=end, core=core,
                          ref_frames=count, cfm_frames=mu.shape[1], semantic_tokens=tokens.numel(),
                          final=bool(finished))
            events.append(window)
            if tracer is not None:
                tracer.mark('acoustic.window', **{key: value for key, value in window.items() if key != 'stage'})
            generated = vq_model.cfm.inference(
                mu, lengths, prompt, req['sample_steps'], inference_cfg_rate=req['cfg_rate'], generator=rng
            )[..., prompt.shape[-1]:].to(ref_mel.dtype)
            accepted = generated[..., :end-cursor]
            left = history[..., -left_frames:]
            audio = vocoder(denorm_spec(torch.cat((left, generated), -1)))[0, 0].float()
            offset = left.shape[-1] * 480
            current = audio[offset:offset+(end-cursor)*480]
            if current.numel() != (end-cursor)*480:
                raise RuntimeError('Vocoder returned an incomplete acoustic window')
            prefix_audio = current[:0] if pending is None else (
                pending * (1-alpha) + audio[offset-fade:offset] * alpha)
            if fade and (not finished or end != total):
                emitted = torch.cat((prefix_audio, current[:-fade]))
                pending = current[-fade:]
            else:
                emitted = torch.cat((prefix_audio, current))
                pending = None
            history = torch.cat((history, accepted), -1)[..., -max(left_frames, 32):]
            feature_history = torch.cat((feature_history, target[..., cursor:end]), -1)[..., -max(left_frames, 32):]
            cursor = end
            if stop.is_set():
                return
            if not torch.isfinite(emitted).all():
                raise RuntimeError('Non-finite V5 audio')
            yield (emitted.clamp(-1, 1) * 32767).to(torch.int16).cpu().numpy()
        if finished:
            if cursor != total:
                raise RuntimeError('Semantic streaming did not flush all acoustic frames')
            if pending is not None:
                yield (pending.clamp(-1, 1) * 32767).to(torch.int16).cpu().numpy()
            return
    if not stop.is_set():
        raise RuntimeError('Semantic stream ended without a final result')
