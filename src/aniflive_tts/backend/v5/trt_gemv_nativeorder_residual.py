"""Private AOT for Native-order GEMV with explicit Half-before-residual boundary."""
from typing import List,Tuple,Union
import tensorrt.plugin as trtp
import triton
from .gemv_nativeorder_residual_prototype import gemv_nativeorder_residual

@trtp.register('aniflive_research::gemv_nativeorder_residual_aot')
def desc(x:trtp.TensorDesc,w:trtp.TensorDesc,b:trtp.TensorDesc,residual:trtp.TensorDesc,in_features:int)->trtp.TensorDesc:
    if in_features not in (512,2048) or x.ndim!=2 or w.ndim!=2 or b.ndim!=1 or residual.ndim!=3:
        raise ValueError('Unsupported residual projection shapes')
    return residual.like()

@trtp.autotune('aniflive_research::gemv_nativeorder_residual_aot')
def formats(x:trtp.TensorDesc,w:trtp.TensorDesc,b:trtp.TensorDesc,residual:trtp.TensorDesc,in_features:int,outputs:Tuple[trtp.TensorDesc])->List[trtp.AutoTuneCombination]:
    expected=((1,in_features),(512,in_features),(512,),(1,1,512))
    if in_features not in (512,2048) or any(not t.has_shape or tuple(t.shape)!=s for t,s in zip((x,w,b,residual),expected)):
        raise ValueError('Unexpected residual projection layout')
    return [trtp.AutoTuneCombination('FP16,FP16,FP16,FP16,FP16','LINEAR')]

@trtp.aot_impl('aniflive_research::gemv_nativeorder_residual_aot')
def aot(x:trtp.TensorDesc,w:trtp.TensorDesc,b:trtp.TensorDesc,residual:trtp.TensorDesc,in_features:int,outputs:Tuple[trtp.TensorDesc],tactic:int)->Tuple[Union[str,bytes],Union[str,bytes],trtp.KernelLaunchParams,trtp.SymIntExprs]:
    if tactic!=0 or in_features not in (512,2048):
        raise ValueError('Unsupported residual projection role/tactic')
    source=triton.compiler.ASTSource(fn=gemv_nativeorder_residual,signature={'X':'*fp16','W':'*fp16','B':'*fp16','R':'*fp16','Y':'*fp16'},constexprs={'IN':in_features},attrs={(i,):[['tt.divisibility',16]] for i in range(5)})
    compiled=triton.compile(source,options={'num_warps':4,'enable_fp_fusion':False})
    launch=trtp.KernelLaunchParams(grid_x=128,block_x=compiled.metadata.num_warps*32,shared_mem=compiled.metadata.shared)
    return compiled.metadata.name,compiled.asm['ptx'],launch,trtp.SymIntExprs.from_tuple(())
