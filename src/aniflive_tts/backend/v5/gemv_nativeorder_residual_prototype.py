"""Private Native-order GEMV + explicit Half boundary + residual addition."""
import triton
import triton.language as tl
from .gemv_nativeorder_warp_prototype import _accumulate

@triton.jit
def gemv_nativeorder_residual(X,W,B,R,Y,IN:tl.constexpr):
    row,lane,acc=_accumulate(X,W,IN,512)
    valid=(row<512)&(lane==0)
    bias=tl.load(B+row,valid,0).to(tl.float32)
    rounded=((acc+0.)+bias).to(tl.float16)
    residual=tl.load(R+row,valid,0)
    # Keep the required Half round and explicitly perform the residual add in
    # FP32; otherwise Triton may legally lower this expression to add.rn.f16.
    summed=tl.inline_asm_elementwise(asm='{ .reg .f32 a,b,c; cvt.f32.f16 a,$1; cvt.f32.f16 b,$2; add.rn.f32 c,a,b; cvt.rn.f16.f32 $0,c; }',constraints='=h,h,h',args=[rounded,residual],dtype=tl.float16,is_pure=False,pack=1)
    tl.store(Y+row,summed,valid)
