"""Diagnostic 64-key online attention with actual FP16 m16n8k8 MMA."""
import triton
import triton.language as tl


@triton.jit
def _pack(lo, hi):
    return lo.to(tl.uint16,bitcast=True).to(tl.uint32) | (hi.to(tl.uint16,bitcast=True).to(tl.uint32) << 16)


@triton.jit
def _mma(a0,a1,b,c0,c1,c2,c3):
    return tl.inline_asm_elementwise(
        asm="mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {$0,$1,$2,$3}, {$4,$5}, {$6}, {$7,$8,$9,$10};",
        constraints="=f,=f,=f,=f,r,r,r,f,f,f,f",args=[a0,a1,b,c0,c1,c2,c3],
        dtype=(tl.float32,tl.float32,tl.float32,tl.float32),is_pure=False,pack=1)


@triton.jit
def native_k8_attention(Q,K_NEW,V_NEW,K_OLD,V_OLD,POS,MASK,OUT,SCORES,PV,LANES,TIDS,BUCKET:tl.constexpr):
    head=tl.program_id(0)
    i=tl.arange(0,128)
    warp=i//32
    lane=i%32
    g=lane//4
    t=lane%4
    actual_lane,actual_tid=tl.inline_asm_elementwise(asm="mov.u32 $0, %laneid; mov.u32 $1, %tid.x;",constraints="=r,=r,r",args=[i],dtype=(tl.uint32,tl.uint32),is_pure=False,pack=1)
    tl.store(LANES+head*128+i,actual_lane)
    tl.store(TIDS+head*128+i,actual_tid)
    current=tl.load(POS).to(tl.int32)-1
    mask_indices=tl.arange(0,BUCKET)
    valid=tl.load(MASK+head*2048+mask_indices)!=0
    end=tl.max(tl.where(valid,mask_indices+1,0),axis=0)
    maximum=tl.full((),float('-inf'),tl.float32)
    normalizer=tl.full((),0.,tl.float32)
    p0=tl.zeros((128,),tl.float32)
    p1=tl.zeros((128,),tl.float32)
    p2=tl.zeros((128,),tl.float32)
    p3=tl.zeros((128,),tl.float32)
    for start in range(0,end,64):
        scores=tl.zeros((128,),tl.float32)
        for segment in tl.static_range(2):
            s0=tl.zeros((128,),tl.float32)
            s1=tl.zeros((128,),tl.float32)
            s2=tl.zeros((128,),tl.float32)
            s3=tl.zeros((128,),tl.float32)
            token=start+segment*32+warp*8+g
            for part in tl.static_range(4):
                d=part*8+2*t
                q0=tl.load(Q+head*32+d)
                q1=tl.load(Q+head*32+d+1)
                ok0=tl.load(K_OLD+head*2048*32+token*32+d)
                ok1=tl.load(K_OLD+head*2048*32+token*32+d+1)
                nk0=tl.load(K_NEW+head*32+d)
                nk1=tl.load(K_NEW+head*32+d+1)
                kp=_pack(tl.where(token==current,nk0,ok0),tl.where(token==current,nk1,ok1))
                qp=_pack(q0,q1)
                s0,s1,s2,s3=_mma(qp,qp,kp,s0,s1,s2,s3)
            col=i%32
            source=(col//8)*32+(col%8)//2
            lo=tl.gather(s0,source,axis=0)
            hi=tl.gather(s1,source,axis=0)
            scores=tl.where((i%64)//32==segment,tl.where(col%2==0,lo,hi),scores)
        col=i%64
        tl.store(SCORES+head*BUCKET+start+col,scores,mask=i<64)
        allowed=tl.load(MASK+head*2048+start+col)!=0
        scaled=scores*0.1767766952966369
        scores2=(scaled+tl.where(allowed,0.,float('-inf')))*1.4426950408889634
        new_max=tl.maximum(maximum,tl.max(scores2,axis=0))
        all_masked=new_max==float('-inf')
        safe_max=tl.where(all_masked,0.,new_max)
        rescale=tl.where(all_masked,1.,tl.exp2(maximum-new_max))
        exps=tl.exp2(scores2-safe_max)
        # Source-derived serial fragment addition, lane XOR reductions, then N-warp order.
        offset=((i%8)//4)*32+(i%4)*2
        part_sum=tl.zeros((128,),tl.float32)
        for j in tl.static_range(4):
            part_sum=part_sum+tl.gather(exps,offset+j*8,axis=0)
            part_sum=part_sum+tl.gather(exps,offset+j*8+1,axis=0)
        part_sum=part_sum+tl.gather(part_sum,i^1,axis=0)
        part_sum=part_sum+tl.gather(part_sum,i^2,axis=0)
        partial0=tl.sum(tl.where(i==0,part_sum,0.),axis=0)
        partial1=tl.sum(tl.where(i==4,part_sum,0.),axis=0)
        normalizer=normalizer*rescale
        normalizer=normalizer+partial0
        normalizer=normalizer+partial1
        p0=p0*rescale
        p1=p1*rescale
        p2=p2*rescale
        p3=p3*rescale
        for part in tl.static_range(8):
            k=part*8+2*t
            a0=tl.gather(exps,k,axis=0).to(tl.float16)
            a1=tl.gather(exps,k+1,axis=0).to(tl.float16)
            ap=_pack(a0,a1)
            dim=warp*8+g
            token0=start+k
            token1=token0+1
            ov0=tl.load(V_OLD+head*2048*32+token0*32+dim)
            ov1=tl.load(V_OLD+head*2048*32+token1*32+dim)
            nv=tl.load(V_NEW+head*32+dim)
            vp=_pack(tl.where(token0==current,nv,ov0),tl.where(token1==current,nv,ov1))
            p0,p1,p2,p3=_mma(ap,ap,vp,p0,p1,p2,p3)
        outdim=warp*8+2*t
        tl.store(PV+head*(BUCKET//64)*32+(start//64)*32+outdim,p0,mask=g==0)
        tl.store(PV+head*(BUCKET//64)*32+(start//64)*32+outdim+1,p1,mask=g==0)
        maximum=new_max
    denominator=tl.where(normalizer==0,1.,normalizer)
    inverse=tl.div_rn(1.,denominator)
    outdim=warp*8+2*t
    tl.store(OUT+head*32+outdim,(p0*inverse).to(tl.float16),mask=g==0)
    tl.store(OUT+head*32+outdim+1,(p1*inverse).to(tl.float16),mask=g==0)
