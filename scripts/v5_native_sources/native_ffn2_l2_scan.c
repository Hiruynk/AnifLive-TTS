/* Copyright 2026 Hiruynk. See the repository LICENSE. */
/* Host-only, actual CUDA argument metadata + immutable FFN2 weight hashes.
 * No cuBLAS argument index or packed-parameter layout is assumed. */
#define l2_inspect_ffn2 legacy_l2_inspect_ffn2
#include "ar_weight_l2.c"
#undef l2_inspect_ffn2
#include "ffn2_candidate_hashes.h"

int qa_scan_native_ffn2(uintptr_t graph, uintptr_t *nodes_out,
        uintptr_t *weights_out, size_t capacity, size_t *found,
        char *error, size_t error_size) {
    CUgraphNode *nodes=NULL;
    unsigned char *buffer=NULL;
    void *crypto=NULL;
    size_t count=0, matches=0;
    int seen[24]={0}, rc=-1;
    CUresult status;
    if (!driver) { snprintf(error,error_size,"CUDA driver not initialized"); return -1; }
    status=p_cuGraphGetNodes((CUgraph)graph,NULL,&count);
    if(status!=CUDA_SUCCESS) {snprintf(error,error_size,"graph nodes: %d",status);return -1;}
    nodes=calloc(count?count:1,sizeof(*nodes));
    buffer=malloc(2*1024*1024);
    crypto=dlopen("libcrypto.so.3",RTLD_NOW|RTLD_LOCAL);
    if(!nodes||!buffer||!crypto) {snprintf(error,error_size,"native FFN2 scan allocation/library failure");goto done;}
    unsigned char *(*sha256)(const unsigned char*,size_t,unsigned char*)=NULL;
    *(void **)(&sha256)=dlsym(crypto,"SHA256");
    if(!sha256) {snprintf(error,error_size,"SHA256 unavailable");goto done;}
    status=p_cuGraphGetNodes((CUgraph)graph,nodes,&count);
    if(status!=CUDA_SUCCESS) {snprintf(error,error_size,"graph nodes read: %d",status);goto done;}
    for(size_t i=0;i<count;++i) {
        CUgraphNodeType type;
        status=p_cuGraphNodeGetType(nodes[i],&type);
        if(status!=CUDA_SUCCESS) {snprintf(error,error_size,"node type: %d",status);goto done;}
        if(type==CU_GRAPH_NODE_TYPE_GRAPH||type==CU_GRAPH_NODE_TYPE_CONDITIONAL) {
            snprintf(error,error_size,"native scan requires a flat neural graph");goto done;
        }
        if(type!=CU_GRAPH_NODE_TYPE_KERNEL) continue;
        CUDA_KERNEL_NODE_PARAMS p={0};
        status=p_cuGraphKernelNodeGetParams(nodes[i],&p);
        if(status!=CUDA_SUCCESS||!p.kernelParams||(!p.func&&!p.kern)) {
            snprintf(error,error_size,"kernel argument metadata unavailable: %d",status);goto done;
        }
        int node_index=-1;
        CUdeviceptr node_weight=0;
        for(size_t ordinal=0;;++ordinal) {
            size_t offset=0,size=0;
            status=p.func?p_cuFuncGetParamInfo(p.func,ordinal,&offset,&size)
                         :p_cuKernelGetParamInfo(p.kern,ordinal,&offset,&size);
            if(status==CUDA_ERROR_INVALID_VALUE) break; /* End of argument list. */
            if(status!=CUDA_SUCCESS||!p.kernelParams[ordinal]) {
                snprintf(error,error_size,"kernel parameter %zu: %d",ordinal,status);goto done;
            }
            for(size_t byte=0;byte+sizeof(CUdeviceptr)<=size;byte+=sizeof(CUdeviceptr)) {
                CUdeviceptr candidate=0,base=0;
                size_t allocation=0;
                memcpy(&candidate,(const unsigned char*)p.kernelParams[ordinal]+byte,sizeof(candidate));
                if(!candidate) continue;
                status=p_cuMemGetAddressRange(&base,&allocation,candidate);
                if(status!=CUDA_SUCCESS||candidate<base||candidate-base>allocation
                    ||allocation-(size_t)(candidate-base)<2*1024*1024) continue;
                status=p_cuMemcpyDtoH(buffer,candidate,2*1024*1024);
                if(status!=CUDA_SUCCESS) {snprintf(error,error_size,"bounded argument memory read: %d",status);goto done;}
                unsigned char digest[32];char hex[65];
                if(!sha256(buffer,2*1024*1024,digest)) {snprintf(error,error_size,"weight SHA failure");goto done;}
                for(size_t j=0;j<32;++j) snprintf(hex+2*j,3,"%02x",digest[j]);
                int index=-1;
                for(int j=0;j<24;++j) if(strcmp(hex,expected_ffn2_hashes[j])==0) {index=j;break;}
                if(index<0) continue;
                if(node_index>=0&&(node_index!=index||node_weight!=candidate)) {
                    snprintf(error,error_size,"kernel refers to multiple expected FFN2 weights");goto done;
                }
                node_index=index;node_weight=candidate;
            }
        }
        if(node_index<0) continue;
        if(seen[node_index]||matches>=capacity) {
            snprintf(error,error_size,"duplicate FFN2 node or insufficient scan capacity");goto done;
        }
        seen[node_index]=1;
        nodes_out[matches]=(uintptr_t)nodes[i];weights_out[matches]=(uintptr_t)node_weight;
        ++matches;
    }
    *found=matches;rc=0;
done:
    if(crypto) dlclose(crypto);
    free(buffer);free(nodes);
    return rc;
}

int l2_inspect_ffn2(uintptr_t graph, uintptr_t *nodes_out,
        uintptr_t *weights_out, size_t capacity, size_t *found,
        char *error, size_t error_size) {
    if(qa_scan_native_ffn2(graph,nodes_out,weights_out,capacity,found,error,error_size)) return -1;
    if(*found!=24) {snprintf(error,error_size,"Native FFN2 hash inventory: %zu nodes, expected24",*found);return -1;}
    return 0;
}
