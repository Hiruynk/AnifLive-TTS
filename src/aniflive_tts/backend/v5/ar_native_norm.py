"""Keep the original CUDA LayerNorm kernel opaque during AR compilation."""
from contextlib import contextmanager
import torch


@torch.library.custom_op('aniflive_research::ar_native_norm',mutates_args=())
def ar_native_norm(x:torch.Tensor,weight:torch.Tensor,bias:torch.Tensor,epsilon:float)->torch.Tensor:
    return torch.nn.functional.layer_norm(x,(512,),weight,bias,epsilon)


@ar_native_norm.register_fake
def _fake(x,weight,bias,epsilon):
    return torch.empty_like(x,memory_format=torch.contiguous_format)


@torch.library.custom_op('aniflive_research::ar_native_linear',mutates_args=())
def ar_native_linear(x:torch.Tensor,weight:torch.Tensor,bias:torch.Tensor)->torch.Tensor:
    return torch.nn.functional.linear(x,weight,bias)


@ar_native_linear.register_fake
def _linear_fake(x,weight,bias):
    return x.new_empty((*x.shape[:-1],weight.shape[0]))


@contextmanager
def native_projections(decoder):
    """Preserve native fused biased GEMM rounding instead of mm-half then bias."""
    saved=[]
    try:
        for block in decoder.layers:
            for linear in (block.attention.in_proj,block.attention.out_proj,
                           block.feed_forward.linear1,block.feed_forward.linear2):
                if linear.bias is None:
                    raise ValueError('Expected biased native AR projection')
                saved.append((linear,linear.__dict__.get('forward')))
                linear.forward=lambda x,linear=linear: ar_native_linear(x,linear.weight,linear.bias)
        if len(saved)!=96:
            raise ValueError('Expected exactly 96 native AR projections')
        yield
    finally:
        for linear,old in reversed(saved):
            if old is None:
                del linear.__dict__['forward']
            else:
                linear.forward=old


@contextmanager
def native_norms(decoder):
    """Scope forward hooks to idle compilation/capture; weight bytes stay unchanged."""
    saved=[]
    try:
        for block in decoder.layers:
            for norm in (block.attention_norm,block.ffn_norm):
                if tuple(norm.normalized_shape)!=(512,) or norm.weight is None or norm.bias is None:
                    raise ValueError('Expected the original affine 512-feature AR LayerNorm')
                # Remember whether this was an instance override, so restoration
                # also restores native method lookup for later eager requests.
                saved.append((norm,norm.__dict__.get('forward')))
                norm.forward=lambda x,norm=norm: ar_native_norm(x,norm.weight,norm.bias,norm.eps)
        if len(saved)!=48:
            raise ValueError('Expected exactly 48 AR LayerNorm operations')
        yield
    finally:
        for norm,old in reversed(saved):
            if old is None:
                del norm.__dict__['forward']
            else:
                norm.forward=old
