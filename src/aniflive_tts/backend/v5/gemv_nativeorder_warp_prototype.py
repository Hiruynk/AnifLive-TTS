"""Private one-warp-per-output Native-order 1D layout diagnostic."""
import triton
import triton.language as tl

@triton.jit
def _accumulate(X,W,IN:tl.constexpr,OUT:tl.constexpr):
    i=tl.arange(0,128)
    row=tl.program_id(0)*4+i//32
    lane=i%32
    acc=tl.full((128,),0.,tl.float32)
    if IN==512:
        for j in tl.static_range(0,16):
            col=lane*16+j
            x=tl.load(X+col).to(tl.float32)
            w=tl.load(W+row*IN+col,row<OUT,0).to(tl.float32)
            acc=acc+w*x
    else:
        for j in range(64):
            col=lane+32*j
            x=tl.load(X+col).to(tl.float32)
            w=tl.load(W+row*IN+col,row<OUT,0).to(tl.float32)
            acc=acc+w*x
    for offset in tl.static_range(4,-1,-1):
        shift=1<<offset
        bits=acc.to(tl.uint32,bitcast=True)
        shuffled=tl.inline_asm_elementwise(asm='shfl.sync.down.b32 $0, $1, $2, 31, -1;',constraints='=r,r,r',args=[bits,tl.full((128,),shift,tl.uint32)],dtype=tl.uint32,is_pure=False,pack=1)
        other=shuffled.to(tl.float32,bitcast=True)
        acc=tl.where(lane+shift<32,acc+other,acc)
    return row,lane,acc

@triton.jit
def gemv_nativeorder_warp(X,W,B,Y,IN:tl.constexpr,OUT:tl.constexpr,RELU:tl.constexpr):
    row,lane,acc=_accumulate(X,W,IN,OUT)
    # Prior lane-zero extraction sums with positive zeros before bias.
    result=(acc+0.)+tl.load(B+row,(row<OUT)&(lane==0),0).to(tl.float32)
    rounded=result.to(tl.float16)
    if RELU:
        rounded=tl.where(rounded<0.,tl.full((128,),0.,tl.float16),rounded)
    tl.store(Y+row,rounded,(row<OUT)&(lane==0))

@triton.jit
def gemv_nativeorder_warp_debug(X,W,B,Y,LANES,TIDS,IN:tl.constexpr,OUT:tl.constexpr,RELU:tl.constexpr):
    i=tl.arange(0,128)
    actual_lane,actual_tid=tl.inline_asm_elementwise(asm='mov.u32 $0, %laneid; mov.u32 $1, %tid.x;',constraints='=r,=r,r',args=[i],dtype=(tl.uint32,tl.uint32),is_pure=False,pack=1)
    tl.store(LANES+tl.program_id(0)*128+i,actual_lane)
    tl.store(TIDS+tl.program_id(0)*128+i,actual_tid)
    row,lane,acc=_accumulate(X,W,IN,OUT)
    result=(acc+0.)+tl.load(B+row,(row<OUT)&(lane==0),0).to(tl.float32)
    rounded=result.to(tl.float16)
    if RELU:
        rounded=tl.where(rounded<0.,tl.full((128,),0.,tl.float16),rounded)
    tl.store(Y+row,rounded,(row<OUT)&(lane==0))
