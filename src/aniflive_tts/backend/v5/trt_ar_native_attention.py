"""TensorRT Python-JIT diagnostic plugin for native Torch SDPA.

Import before parsing or deserializing any engine containing this plugin.
Those engines require this module and Python at execution time. The plugin is
experimental; its numerical parity and graph capture must be measured on GPU.
"""
from typing import List, Tuple

import tensorrt.plugin as trtp
import torch
import torch.nn.functional as F


@trtp.register('aniflive_research::native_attention')
def native_attention_desc(query: trtp.TensorDesc, key: trtp.TensorDesc,
                          value: trtp.TensorDesc, mask: trtp.TensorDesc) -> trtp.TensorDesc:
    return query.like()


@trtp.autotune('aniflive_research::native_attention')
def native_attention_formats(query: trtp.TensorDesc, key: trtp.TensorDesc,
                             value: trtp.TensorDesc, mask: trtp.TensorDesc,
                             outputs: Tuple[trtp.TensorDesc]) -> List[trtp.AutoTuneCombination]:
    return [trtp.AutoTuneCombination('FP16,FP16,FP16,BOOL,FP16', 'LINEAR')]


@trtp.impl('aniflive_research::native_attention')
def native_attention_impl(query: trtp.Tensor, key: trtp.Tensor, value: trtp.Tensor,
                          mask: trtp.Tensor, outputs: Tuple[trtp.Tensor], stream: int) -> None:
    external = torch.cuda.ExternalStream(stream, device='cuda:0')
    with torch.cuda.stream(external), torch.inference_mode():
        q, k, v, attn_mask = (torch.as_tensor(tensor, device='cuda:0')
                              for tensor in (query, key, value, mask))
        output = torch.as_tensor(outputs[0], device='cuda:0')
        if (q.dtype != torch.float16 or k.dtype != torch.float16 or v.dtype != torch.float16
                or attn_mask.dtype != torch.bool or output.dtype != torch.float16):
            raise ValueError('Native attention plugin requires FP16 Q/K/V/output and a boolean mask')
        output.copy_(F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                                    dropout_p=0.0, is_causal=False))
