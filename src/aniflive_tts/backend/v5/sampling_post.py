"""Opt-in single-slot Triton post-sampling state update for research.

The native TorchScript sampling ratio is computed before this state update.
Only the native full boolean attention mask is supported.
"""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _post(ratios, logits, embedding, positions, alpha, y, xy, input_pos,
          step, prompt_len, max_step, complete, attention_mask,
          X_SCALE: tl.constexpr, HALF_POSITION_PRODUCT: tl.constexpr):
    i = tl.arange(0, 2048)
    valid_vocab = i < 1025
    ratio = tl.load(ratios + i, mask=valid_vocab, other=float('-inf')).to(tl.float32)
    sample_nan = tl.min(tl.where(valid_vocab & (ratio != ratio), i, 2048), 0)
    best_ratio = tl.max(tl.where(valid_vocab & (ratio == ratio), ratio, float('-inf')), 0)
    sample_first = tl.min(tl.where(valid_vocab & (ratio == best_ratio), i, 2048), 0)
    sample = tl.where(sample_nan < 2048, sample_nan, sample_first)
    raw = tl.load(logits + i, mask=valid_vocab, other=float('-inf')).to(tl.float32)
    raw_nan = tl.min(tl.where(valid_vocab & (raw != raw), i, 2048), 0)
    best_raw = tl.max(tl.where(valid_vocab & (raw == raw), raw, float('-inf')), 0)
    raw_first = tl.min(tl.where(valid_vocab & (raw == best_raw), i, 2048), 0)
    argmax_logit = tl.where(raw_nan < 2048, raw_nan, raw_first)

    old_step = tl.load(step)
    prompt = tl.load(prompt_len)
    old_pos = tl.load(input_pos)
    was_complete = tl.load(complete)
    eos = (sample == 1024) | (argmax_logit == 1024)
    write = (~was_complete) & (~eos)
    new_step = old_step + write.to(tl.int32)
    done = was_complete | eos | (new_step >= tl.load(max_step))
    new_pos = tl.where(done, 1, old_pos + write.to(tl.int32))
    # Native clamps the sampled EOS embedding index before its conditional
    # token scatter; the write condition excludes EOS and completed sequences.
    tl.store(y + prompt + old_step, sample, mask=write)
    token = tl.minimum(sample, 1023)
    emb = tl.load(embedding + token * 512 + i, mask=i < 512, other=0).to(tl.float32)
    pe = tl.load(positions + (prompt + old_step) * 512 + i, mask=i < 512, other=0).to(tl.float32)
    a = tl.load(alpha).to(tl.float32)
    scaled_emb = (emb * X_SCALE).to(tl.float16).to(tl.float32)
    product = a * pe
    if HALF_POSITION_PRODUCT:
        product = product.to(tl.float16).to(tl.float32)
    next_xy = (scaled_emb + product).to(tl.float16)
    tl.store(xy + i, next_xy, mask=(i < 512) & (~done))
    tl.store(step, new_step)
    tl.store(input_pos, new_pos)
    tl.store(complete, done)
    for head in tl.static_range(16):
        tl.store(attention_mask + head * 2048 + i, i < new_pos)


class SamplingPost:
    def __init__(self, runner, attention_mask):
        import torch
        decoder = runner.decoder
        position = decoder.ar_audio_position
        weight = decoder.ar_audio_embedding.weight
        if (runner.max_batch_size != 1 or runner.device.type != 'cuda'
                or (decoder.vocab_size, decoder.EOS, decoder.embedding_dim, decoder.n_head, decoder.max_seq_length)
                != (1025, 1024, 512, 16, 2048)):
            raise ValueError('Fused post-sampling requires the native batch-one v5 AR layout')
        specs = ((weight, (1025, 512), (torch.float16,)),
                 (position.pe, (1, 2048, 512), (torch.float16, torch.float32)),
                 (position.alpha, (1,), (torch.float16, torch.float32)),
                 (runner.y_buf, (1, 2048), (torch.int32,)),
                 (runner.xy_pos_buf, (1, 1, 512), (torch.float16,)),
                 (runner.input_pos_buf, (1,), (torch.int32,)),
                 (runner.step_buf, (1,), (torch.int32,)),
                 (runner.prompt_len_buf, (1,), (torch.int32,)),
                 (runner.max_step_buf, (1,), (torch.int32,)),
                 (runner.completed_buf, (1,), (torch.bool,)),
                 (attention_mask, (1, 16, 1, 2048), (torch.bool,)))
        for tensor, shape, dtypes in specs:
            if (tensor is None or tuple(tensor.shape) != shape or tensor.dtype not in dtypes
                    or tensor.device != runner.device or not tensor.is_contiguous()):
                raise ValueError('Unexpected fused post-sampling buffer layout, device or dtype')
        if position.x_scale != 1.0:
            raise ValueError('Fused post-sampling requires the native unit embedding scale')
        # Retain every tensor whose address is embedded in the captured graph.
        self.runner, self.weight, self.positions, self.alpha = runner, weight, position.pe, position.alpha
        self.mask = attention_mask
        self.x_scale = float(position.x_scale)
        self.half_product = position.pe.dtype == position.alpha.dtype == torch.float16

    def __call__(self, ratios, logits):
        import torch
        if any(t.dtype != torch.float16 or tuple(t.shape) != (1, 1025)
               or t.device != self.runner.device or not t.is_contiguous() for t in (ratios, logits)):
            raise ValueError('Fused post-sampling requires contiguous [1,1025] FP16 logits/ratios')
        r = self.runner
        _post[(1,)](ratios, logits, self.weight, self.positions, self.alpha,
                    r.y_buf, r.xy_pos_buf, r.input_pos_buf, r.step_buf, r.prompt_len_buf,
                    r.max_step_buf, r.completed_buf, self.mask,
                    X_SCALE=self.x_scale, HALF_POSITION_PRODUCT=self.half_product,
                    num_warps=8, enable_fp_fusion=False)
