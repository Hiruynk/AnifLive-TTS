"""Opt-in deterministic unique-valid-token repetition penalty for research.

Unlike the native padded scatter, invalid positions cannot overwrite a valid
token-zero update. The operation keeps native FP16 penalty arithmetic and
mutates the original logits. Installing the native reference hook is explicit
and must be done only when no AR request is active.
"""
from __future__ import annotations

import torch


def apply_repetition_penalty(logits, previous_tokens, valid_lens, repetition_penalty):
    valid = torch.arange(previous_tokens.shape[1], device=logits.device)[None, :] < valid_lens.long()[:, None]
    tokens = previous_tokens.long()
    safe = torch.where(valid, tokens, torch.zeros_like(tokens))
    seen = torch.zeros_like(logits, dtype=torch.int32)
    seen.scatter_reduce_(1, safe, valid.to(torch.int32), reduce='amax', include_self=True)
    rp = repetition_penalty.to(logits.dtype)[:, None]
    penalized = torch.where(logits < 0, logits * rp, logits / rp)
    logits.copy_(torch.where(seen.bool(), penalized, logits))
    return logits


def install_native_reference():
    """Patch Python sample_naive's module global; return an idle-only restore callable."""
    from GPT_SoVITS.Accel.PyTorch.AR import sample_funcs
    if sample_funcs.sample_naive.__call__.__globals__ is not vars(sample_funcs):
        raise RuntimeError('Native Python sampler global binding changed')
    original = sample_funcs.apply_repetition_penalty
    if original is apply_repetition_penalty:
        raise RuntimeError('Deterministic repetition penalty is already installed')
    sample_funcs.apply_repetition_penalty = apply_repetition_penalty

    def restore():
        if sample_funcs.apply_repetition_penalty is not apply_repetition_penalty:
            raise RuntimeError('Native repetition penalty binding changed externally')
        sample_funcs.apply_repetition_penalty = original
    return restore


def _cpu_oracle_check():
    # Includes valid zero mixed with padding, padding alone, repeated nonzero,
    # token 1024 (EOS in this vocabulary), and the FP16 negative-zero bit.
    history = torch.tensor([[0, 29, 87, 0, 0], [0, 0, 0, 0, 0],
                            [29, 29, 0, 0, 0], [1024, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]], dtype=torch.long)
    lengths = torch.tensor([4, 0, 2, 1, 1])
    logits = torch.ones((5, 1025), dtype=torch.float16)
    logits[:, 0] = torch.tensor(-0.0, dtype=torch.float16)
    logits[:2, 0] = 2
    logits[:, 29] = -2
    logits[:, 87] = 2
    logits[:, 1024] = -3
    ratio = torch.full((5,), 1.35, dtype=torch.float32)
    expected = logits.clone()
    for row in range(len(lengths)):
        for token in set(history[row, :int(lengths[row])].tolist()):
            old = expected[row, token]
            rp = ratio[row].to(expected.dtype)
            expected[row, token] = old * rp if old < 0 else old / rp
    actual = logits.clone()
    returned = apply_repetition_penalty(actual, history, lengths, ratio)
    if returned is not actual or not torch.equal(actual.view(torch.int16), expected.view(torch.int16)):
        raise AssertionError('CPU unique-valid-token oracle mismatch')
    print('CPU repetition-penalty oracle examples passed')


if __name__ == '__main__':
    _cpu_oracle_check()
