"""Private native-K8 AOT with a statically allocated CTA score stage."""
from typing import List,Tuple,Union
import tensorrt.plugin as trtp
import triton
import triton.language as tl
from .attention_native_k8_kernel import _pack,_mma


@triton.jit
def score_segment(Q,K_NEW,K_OLD,head,start,current,warp,g,t,SEG:tl.constexpr):
    c0=tl.zeros((128,),tl.float32)
    c1=tl.zeros((128,),tl.float32)
    c2=tl.zeros((128,),tl.float32)
    c3=tl.zeros((128,),tl.float32)
    token=start+32*SEG+8*warp+g
    for part in tl.static_range(4):
        d=8*part+2*t
        a=_pack(tl.load(Q+head*32+d),tl.load(Q+head*32+d+1))
        k0=tl.load(K_OLD+head*2048*32+token*32+d)
        k1=tl.load(K_OLD+head*2048*32+token*32+d+1)
        n0=tl.load(K_NEW+head*32+d)
        n1=tl.load(K_NEW+head*32+d+1)
        b=_pack(tl.where(token==current,n0,k0),tl.where(token==current,n1,k1))
        c0,c1,c2,c3=_mma(a,a,b,c0,c1,c2,c3)
    return c0,c1


@triton.jit
def score_stage(a,b,c,d,i):
    return tl.inline_asm_elementwise(
        asm="""{
        .shared .align 4 .b8 native_score_stage[256];
        .reg .u32 lane, warp, grp, col, offset, base, addr;
        .reg .pred writer;
        and.b32 lane, $6, 31; shr.u32 warp, $6, 5; shr.u32 grp, lane, 2;
        and.b32 col, lane, 3; shl.b32 col, col, 1; shl.b32 offset, warp, 3;
        add.u32 offset, offset, col; shl.b32 offset, offset, 2;
        mov.u32 base, native_score_stage; add.u32 addr, base, offset;
        setp.eq.u32 writer, grp, 0;
        @writer st.shared.f32 [addr], $2; @writer st.shared.f32 [addr+4], $3;
        @writer st.shared.f32 [addr+128], $4; @writer st.shared.f32 [addr+132], $5;
        bar.sync 0;
        shl.b32 offset, lane, 3; add.u32 addr, base, offset;
        ld.shared.f32 $0, [addr]; ld.shared.f32 $1, [addr+4];
        bar.sync 0;
        }""",constraints="=f,=f,f,f,f,f,r,~{memory}",args=[a,b,c,d,i],
        dtype=(tl.float32,tl.float32),is_pure=False,pack=1)


@triton.jit
def shuffle_idx(value,index):
    bits=value.to(tl.uint32,bitcast=True)
    result=tl.inline_asm_elementwise(asm="shfl.sync.idx.b32 $0, $1, $2, 31, -1;",constraints="=r,r,r",args=[bits,index],dtype=tl.uint32,is_pure=False,pack=1)
    return result.to(tl.float32,bitcast=True)


@triton.jit
def shuffle_xor(value,delta:tl.constexpr):
    bits=value.to(tl.uint32,bitcast=True)
    change=tl.full((128,),delta,tl.uint32)
    result=tl.inline_asm_elementwise(asm="shfl.sync.bfly.b32 $0, $1, $2, 31, -1;",constraints="=r,r,r",args=[bits,change],dtype=tl.uint32,is_pure=False,pack=1)
    return result.to(tl.float32,bitcast=True)


@triton.jit
def native_k8_attention_aot(Q,K_NEW,V_NEW,K_OLD,V_OLD,POS,MASK,OUT,BUCKET:tl.constexpr):
    head=tl.program_id(0)
    i=tl.arange(0,128)
    warp=i//32
    lane=i%32
    g=lane//4
    t=lane%4
    current=tl.load(POS).to(tl.int32)-1
    idx=tl.arange(0,BUCKET)
    valid=tl.load(MASK+head*2048+idx)!=0
    end=tl.max(tl.where(valid,idx+1,0),axis=0)
    maximum=tl.full((128,),float('-inf'),tl.float32)
    normalizer=tl.zeros((128,),tl.float32)
    p0=tl.zeros((128,),tl.float32)
    p1=tl.zeros((128,),tl.float32)
    p2=tl.zeros((128,),tl.float32)
    p3=tl.zeros((128,),tl.float32)
    for start in range(0,end,64):
        a0,a1=score_segment(Q,K_NEW,K_OLD,head,start,current,warp,g,t,0)
        b0,b1=score_segment(Q,K_NEW,K_OLD,head,start,current,warp,g,t,1)
        a,b=score_stage(a0,a1,b0,b1,i)
        tok=start+2*lane
        ok0=tl.load(MASK+head*2048+tok)!=0
        ok1=tl.load(MASK+head*2048+tok+1)!=0
        s0=(a*0.1767766952966369+tl.where(ok0,0.,float('-inf')))*1.4426950408889634
        s1=(b*0.1767766952966369+tl.where(ok1,0.,float('-inf')))*1.4426950408889634
        m=tl.maximum(maximum,tl.maximum(s0,s1))
        m=tl.maximum(m,shuffle_xor(m,16))
        m=tl.maximum(m,shuffle_xor(m,8))
        m=tl.maximum(m,shuffle_xor(m,4))
        m=tl.maximum(m,shuffle_xor(m,2))
        m=tl.maximum(m,shuffle_xor(m,1))
        all_masked=m==float('-inf')
        safe=tl.where(all_masked,0.,m)
        rescale=tl.where(all_masked,1.,tl.exp2(maximum-m))
        e0=tl.exp2(s0-safe)
        e1=tl.exp2(s1-safe)
        sum0=tl.zeros((128,),tl.float32)
        sum1=tl.zeros((128,),tl.float32)
        for part in tl.static_range(4):
            owner=t+4*part
            sum0=sum0+shuffle_idx(e0,owner)
            sum0=sum0+shuffle_idx(e1,owner)
            sum1=sum1+shuffle_idx(e0,owner+16)
            sum1=sum1+shuffle_idx(e1,owner+16)
        sum0=sum0+shuffle_xor(sum0,1)
        sum0=sum0+shuffle_xor(sum0,2)
        sum1=sum1+shuffle_xor(sum1,1)
        sum1=sum1+shuffle_xor(sum1,2)
        normalizer=normalizer*rescale
        normalizer=normalizer+sum0
        normalizer=normalizer+sum1
        p0=p0*rescale
        p1=p1*rescale
        p2=p2*rescale
        p3=p3*rescale
        for part in tl.static_range(8):
            owner=t+4*part
            ap=_pack(shuffle_idx(e0,owner).to(tl.float16),shuffle_idx(e1,owner).to(tl.float16))
            tok0=start+8*part+2*t
            tok1=tok0+1
            dim=warp*8+g
            v0=tl.load(V_OLD+head*2048*32+tok0*32+dim)
            v1=tl.load(V_OLD+head*2048*32+tok1*32+dim)
            nv=tl.load(V_NEW+head*32+dim)
            vp=_pack(tl.where(tok0==current,nv,v0),tl.where(tok1==current,nv,v1))
            p0,p1,p2,p3=_mma(ap,ap,vp,p0,p1,p2,p3)
        maximum=m
    inverse=tl.div_rn(1.,tl.where(normalizer==0,1.,normalizer))
    dim=warp*8+2*t
    tl.store(OUT+head*32+dim,(p0*inverse).to(tl.float16),mask=g==0)
    tl.store(OUT+head*32+dim+1,(p1*inverse).to(tl.float16),mask=g==0)


@trtp.register('aniflive_research::delta_attention_native_k8_shared_aot')
def desc(query:trtp.TensorDesc,new_key:trtp.TensorDesc,new_value:trtp.TensorDesc,old_key:trtp.TensorDesc,old_value:trtp.TensorDesc,position:trtp.TensorDesc,mask:trtp.TensorDesc,bucket:int)->trtp.TensorDesc:
    if bucket!=512 or any(t.ndim!=4 for t in (query,new_key,new_value,old_key,old_value,mask)) or position.ndim!=1:
        raise ValueError('Expected bucket 512 attention ranks')
    return query.like()


@trtp.autotune('aniflive_research::delta_attention_native_k8_shared_aot')
def formats(query:trtp.TensorDesc,new_key:trtp.TensorDesc,new_value:trtp.TensorDesc,old_key:trtp.TensorDesc,old_value:trtp.TensorDesc,position:trtp.TensorDesc,mask:trtp.TensorDesc,bucket:int,outputs:Tuple[trtp.TensorDesc])->List[trtp.AutoTuneCombination]:
    expected=((1,16,1,32),(1,16,1,32),(1,16,1,32),(1,16,2048,32),(1,16,2048,32),(1,),(1,16,1,2048))
    if bucket!=512 or any(not t.has_shape or tuple(t.shape)!=shape for t,shape in zip((query,new_key,new_value,old_key,old_value,position,mask),expected)):
        raise ValueError('Expected fixed bucket 512 cache shapes')
    return [trtp.AutoTuneCombination('FP16,FP16,FP16,FP16,FP16,INT32,BOOL,FP16','LINEAR')]


def compile_native_k8():
    source=triton.compiler.ASTSource(fn=native_k8_attention_aot,signature={'Q':'*fp16','K_NEW':'*fp16','V_NEW':'*fp16','K_OLD':'*fp16','V_OLD':'*fp16','POS':'*i32','MASK':'*u1','OUT':'*fp16'},constexprs={'BUCKET':512},attrs={(i,):[['tt.divisibility',16]] for i in range(8)})
    compiled=triton.compile(source,options={'num_warps':4,'enable_fp_fusion':False})
    if compiled.metadata.shared!=16 or 'native_score_stage[256]' not in compiled.asm['ptx']:
        raise RuntimeError('Shared memory compilation contract changed')
    return compiled


@trtp.aot_impl('aniflive_research::delta_attention_native_k8_shared_aot')
def aot(query:trtp.TensorDesc,new_key:trtp.TensorDesc,new_value:trtp.TensorDesc,old_key:trtp.TensorDesc,old_value:trtp.TensorDesc,position:trtp.TensorDesc,mask:trtp.TensorDesc,bucket:int,outputs:Tuple[trtp.TensorDesc],tactic:int)->Tuple[Union[str,bytes],Union[str,bytes],trtp.KernelLaunchParams,trtp.SymIntExprs]:
    if bucket!=512 or tactic!=0:
        raise ValueError('Unsupported native K8 shared attention configuration')
    compiled=compile_native_k8()
    launch=trtp.KernelLaunchParams(grid_x=16,block_x=compiled.metadata.num_warps*32,shared_mem=compiled.metadata.shared)
    return compiled.metadata.name,compiled.asm['ptx'],launch,trtp.SymIntExprs.from_tuple(())
