"""Private one-warp-per-row Native-order AOT creator; not production-registered."""
from typing import List,Tuple,Union
import tensorrt.plugin as trtp
import triton
from .gemv_nativeorder_warp_prototype import gemv_nativeorder_warp

def valid(in_features:int,out_features:int,relu:int)->bool:
    return ((in_features==512 and out_features in (512,1536,2048) and relu==int(out_features==2048)) or (in_features==2048 and out_features==512 and relu==0))

@trtp.register('aniflive_research::gemv_nativeorder_warp_aot')
def desc(x:trtp.TensorDesc,w:trtp.TensorDesc,b:trtp.TensorDesc,in_features:int,out_features:int,relu:int)->trtp.TensorDesc:
    if not valid(in_features,out_features,relu) or x.ndim!=2 or w.ndim!=2 or b.ndim!=1:
        raise ValueError('Unsupported Native-order warp projection')
    y=x.like()
    y.shape_expr[1]=w.shape_expr[0]
    return y

@trtp.autotune('aniflive_research::gemv_nativeorder_warp_aot')
def formats(x:trtp.TensorDesc,w:trtp.TensorDesc,b:trtp.TensorDesc,in_features:int,out_features:int,relu:int,outputs:Tuple[trtp.TensorDesc])->List[trtp.AutoTuneCombination]:
    if not valid(in_features,out_features,relu) or any(not t.has_shape or tuple(t.shape)!=s for t,s in zip((x,w,b),((1,in_features),(out_features,in_features),(out_features,)))):
        raise ValueError('Unexpected Native-order warp shapes')
    return [trtp.AutoTuneCombination('FP16,FP16,FP16,FP16','LINEAR')]

@trtp.aot_impl('aniflive_research::gemv_nativeorder_warp_aot')
def aot(x:trtp.TensorDesc,w:trtp.TensorDesc,b:trtp.TensorDesc,in_features:int,out_features:int,relu:int,outputs:Tuple[trtp.TensorDesc],tactic:int)->Tuple[Union[str,bytes],Union[str,bytes],trtp.KernelLaunchParams,trtp.SymIntExprs]:
    if tactic!=0 or not valid(in_features,out_features,relu):
        raise ValueError('Unsupported Native-order warp role/tactic')
    source=triton.compiler.ASTSource(fn=gemv_nativeorder_warp,signature={'X':'*fp16','W':'*fp16','B':'*fp16','Y':'*fp16'},constexprs={'IN':in_features,'OUT':out_features,'RELU':bool(relu)},attrs={(i,):[['tt.divisibility',16]] for i in range(4)})
    compiled=triton.compile(source,options={'num_warps':4,'enable_fp_fusion':False})
    launch=trtp.KernelLaunchParams(grid_x=triton.cdiv(out_features,4),block_x=compiled.metadata.num_warps*32,shared_mem=compiled.metadata.shared)
    return compiled.metadata.name,compiled.asm['ptx'],launch,trtp.SymIntExprs.from_tuple(())
