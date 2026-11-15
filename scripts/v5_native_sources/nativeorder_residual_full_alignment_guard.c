/* Copyright 2026 Hiruynk. See the repository LICENSE. */
/* Host-only strict 48 four-pointer + 48 five-pointer AOT graph guard. */
#include "ar_weight_l2.c"
int qa_validate_gemv_alignment(uintptr_t graph,char *error,size_t error_size){
    size_t count=0,plain=0,residual=0;CUgraphNode *nodes=NULL;if(!driver){snprintf(error,error_size,"invalid guard state");return -1;}
    CHECK(p_cuGraphGetNodes((CUgraph)graph,NULL,&count));nodes=calloc(count?count:1,sizeof(*nodes));if(!nodes){snprintf(error,error_size,"allocation failed");return -1;}
    CUresult s=p_cuGraphGetNodes((CUgraph)graph,nodes,&count);if(s!=CUDA_SUCCESS){free(nodes);snprintf(error,error_size,"node list failed %d",(int)s);return -1;}
    for(size_t i=0;i<count;i++){
        CUgraphNodeType type;s=p_cuGraphNodeGetType(nodes[i],&type);if(s!=CUDA_SUCCESS){free(nodes);snprintf(error,error_size,"node type failed %d",(int)s);return -1;}
        if(type!=CU_GRAPH_NODE_TYPE_KERNEL)continue;CUDA_KERNEL_NODE_PARAMS p={0};s=p_cuGraphKernelNodeGetParams(nodes[i],&p);if(s!=CUDA_SUCCESS){free(nodes);snprintf(error,error_size,"kernel params failed %d",(int)s);return -1;}
        const char *name=NULL;if(p.func)s=p_cuFuncGetName(&name,p.func);else if(p.kern)s=p_cuKernelGetName(&name,p.kern);else s=CUDA_ERROR_INVALID_VALUE;
        if(s!=CUDA_SUCCESS||!name){free(nodes);snprintf(error,error_size,"kernel name unavailable");return -1;}
        size_t pointers=0;if(!strcmp(name,"gemv_nativeorder_warp")){plain++;pointers=4;}else if(!strcmp(name,"gemv_nativeorder_residual")){residual++;pointers=5;}else continue;
        if(!p.kernelParams){free(nodes);snprintf(error,error_size,"kernel arguments unavailable");return -1;}
        for(size_t j=0;j<pointers;j++){size_t offset=0,size=0;s=p.func?p_cuFuncGetParamInfo(p.func,j,&offset,&size):p_cuKernelGetParamInfo(p.kern,j,&offset,&size);if(s!=CUDA_SUCCESS||size!=sizeof(CUdeviceptr)||!p.kernelParams[j]){free(nodes);snprintf(error,error_size,"argument metadata invalid");return -1;}CUdeviceptr ptr=0;memcpy(&ptr,p.kernelParams[j],sizeof(ptr));if(!ptr||ptr%16){free(nodes);snprintf(error,error_size,"unaligned projection pointer");return -1;}}
    }
    free(nodes);if(plain!=48||residual!=48){snprintf(error,error_size,"expected 48 plain / 48 residual nodes, found %zu/%zu",plain,residual);return -1;}return 0;
}
