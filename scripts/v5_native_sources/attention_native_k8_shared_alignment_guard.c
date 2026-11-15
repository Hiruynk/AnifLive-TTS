/* Copyright 2026 Hiruynk. See the repository LICENSE. */
/* Per-created-graph check for the eight 16-byte-aligned grouped attention arguments. */
#include "ar_weight_l2.c"

int qa_validate_native_k8_attention(uintptr_t graph, size_t expected, char *error, size_t error_size) {
    size_t count = 0, found = 0;
    CUgraphNode *nodes = NULL;
    if (!driver) { snprintf(error,error_size,"CUDA driver was not initialized"); return -1; }
    __typeof__(&cuFuncGetAttribute) get_func_attr=(__typeof__(&cuFuncGetAttribute))dlsym(driver,"cuFuncGetAttribute");
    __typeof__(&cuKernelGetAttribute) get_kernel_attr=(__typeof__(&cuKernelGetAttribute))dlsym(driver,"cuKernelGetAttribute");
    __typeof__(&cuCtxGetDevice) get_device=(__typeof__(&cuCtxGetDevice))dlsym(driver,"cuCtxGetDevice");
    if (!get_func_attr||!get_kernel_attr||!get_device) { snprintf(error,error_size,"CUDA shared-memory attribute functions unavailable"); return -1; }
    CHECK(p_cuGraphGetNodes((CUgraph)graph,NULL,&count));
    nodes=(CUgraphNode*)calloc(count?count:1,sizeof(*nodes));
    if (!nodes) { snprintf(error,error_size,"node allocation failed"); return -1; }
    CUresult s=p_cuGraphGetNodes((CUgraph)graph,nodes,&count);
    if (s!=CUDA_SUCCESS) { free(nodes); snprintf(error,error_size,"graph node query: %d",(int)s); return -1; }
    for (size_t i=0;i<count;++i) {
        CUgraphNodeType type; s=p_cuGraphNodeGetType(nodes[i],&type);
        if (s!=CUDA_SUCCESS) { free(nodes); snprintf(error,error_size,"graph node type: %d",(int)s); return -1; }
        if (type!=CU_GRAPH_NODE_TYPE_KERNEL) continue;
        CUDA_KERNEL_NODE_PARAMS p={0}; s=p_cuGraphKernelNodeGetParams(nodes[i],&p);
        if (s!=CUDA_SUCCESS) { free(nodes); snprintf(error,error_size,"kernel params: %d",(int)s); return -1; }
        const char *name=NULL;
        if (p.func) s=p_cuFuncGetName(&name,p.func);
        else if (p.kern) s=p_cuKernelGetName(&name,p.kern);
        else { free(nodes); snprintf(error,error_size,"kernel has no function"); return -1; }
        if (s!=CUDA_SUCCESS||!name) { free(nodes); snprintf(error,error_size,"kernel name: %d",(int)s); return -1; }
        if (strcmp(name,"native_k8_attention_aot")) continue;
        ++found;
        int static_bytes=0; CUdevice device=0;
        if (p.func) s=get_func_attr(&static_bytes,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,p.func);
        else { s=get_device(&device); if(s==CUDA_SUCCESS) s=get_kernel_attr(&static_bytes,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,p.kern,device); }
        if (s!=CUDA_SUCCESS||static_bytes!=256||p.sharedMemBytes!=16) { free(nodes); snprintf(error,error_size,"attention shared-memory contract failed at node %zu: status=%d static=%d dynamic=%u",i,(int)s,static_bytes,p.sharedMemBytes); return -1; }
        if (!p.kernelParams) { free(nodes); snprintf(error,error_size,"attention arguments unavailable"); return -1; }
        for (size_t j=0;j<8;++j) {
            size_t offset=0,size=0;
            s=p.func?p_cuFuncGetParamInfo(p.func,j,&offset,&size):p_cuKernelGetParamInfo(p.kern,j,&offset,&size);
            if (s!=CUDA_SUCCESS||size!=sizeof(CUdeviceptr)||!p.kernelParams[j]) { free(nodes); snprintf(error,error_size,"invalid attention argument %zu",j); return -1; }
            CUdeviceptr pointer=0; memcpy(&pointer,p.kernelParams[j],sizeof(pointer));
            if (!pointer||pointer%16) { free(nodes); snprintf(error,error_size,"node %zu argument %zu is not 16-byte aligned",i,j); return -1; }
        }
    }
    free(nodes);
    if (found!=expected) { snprintf(error,error_size,"expected %zu grouped attention nodes, found %zu",expected,found); return -1; }
    return 0;
}
