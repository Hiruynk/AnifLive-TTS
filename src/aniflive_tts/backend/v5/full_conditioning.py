"""V5 complete acoustic path with explicitly selected reference components."""


def synthesize_complete(engine, semantic_tokens, phones, reference, speed=1.0,
                        sample_steps=4, cfg_rate=0.0, check_cancel=None):
    import torch
    from TTS_infer_pack.TTS import denorm_spec, synthesize_v5_mel

    if sample_steps != 4 or cfg_rate != 0.0:
        raise ValueError('V5 Turbo conditioning requires four-step CFM and CFG 0')
    if check_cancel is not None:
        check_cancel()
    target, _ = engine.vits_model.decode_encp(
        semantic_tokens, phones, reference[3], reference[5], speed)
    if check_cancel is not None:
        check_cancel()
    mel = synthesize_v5_mel(engine.vits_model, reference[4], target,
                            reference[6].to(engine.precision), sample_steps, cfg_rate)
    if check_cancel is not None:
        check_cancel()
    with torch.inference_mode():
        audio = engine.vocoder(denorm_spec(mel).to(engine.precision))[0, 0]
    if check_cancel is not None:
        check_cancel()
    return audio
