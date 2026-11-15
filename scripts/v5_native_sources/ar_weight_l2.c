/* Copyright 2026 Hiruynk. See the repository LICENSE. */
/* Header-typed CUDA driver graph inspection and per-node L2 policy shim. */
#include <cuda.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STR2(x) #x
#define STR(x) STR2(x)
#define LOAD(name) do { *(void **)(&p_##name) = dlsym(driver, STR(name)); \
    if (!p_##name) { snprintf(error, error_size, "missing CUDA symbol %s", STR(name)); return -1; } } while (0)
#define CHECK(call) do { CUresult status_ = (call); if (status_ != CUDA_SUCCESS) { \
    snprintf(error, error_size, "%s failed with CUDA status %d", #call, (int)status_); return -1; } } while (0)

static void *driver;
static __typeof__(&cuGraphGetNodes) p_cuGraphGetNodes;
static __typeof__(&cuGraphNodeGetType) p_cuGraphNodeGetType;
static __typeof__(&cuGraphNodeGetDependentNodes) p_cuGraphNodeGetDependentNodes;
static __typeof__(&cuGraphKernelNodeGetParams) p_cuGraphKernelNodeGetParams;
static __typeof__(&cuFuncGetName) p_cuFuncGetName;
static __typeof__(&cuFuncGetParamInfo) p_cuFuncGetParamInfo;
static __typeof__(&cuKernelGetName) p_cuKernelGetName;
static __typeof__(&cuKernelGetParamInfo) p_cuKernelGetParamInfo;
static __typeof__(&cuMemGetAddressRange) p_cuMemGetAddressRange;
static __typeof__(&cuMemcpyDtoH) p_cuMemcpyDtoH;
static __typeof__(&cuGraphKernelNodeSetAttribute) p_cuGraphKernelNodeSetAttribute;
static __typeof__(&cuCtxGetLimit) p_cuCtxGetLimit;
static __typeof__(&cuCtxSetLimit) p_cuCtxSetLimit;
static __typeof__(&cuCtxResetPersistingL2Cache) p_cuCtxResetPersistingL2Cache;

int l2_init(char *error, size_t error_size) {
    driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!driver) { snprintf(error, error_size, "cannot load libcuda.so.1: %s", dlerror()); return -1; }
    LOAD(cuGraphGetNodes); LOAD(cuGraphNodeGetType); LOAD(cuGraphKernelNodeGetParams);
    LOAD(cuFuncGetName); LOAD(cuFuncGetParamInfo); LOAD(cuKernelGetName); LOAD(cuKernelGetParamInfo);
    LOAD(cuMemGetAddressRange); LOAD(cuMemcpyDtoH); LOAD(cuGraphKernelNodeSetAttribute);
    LOAD(cuCtxGetLimit); LOAD(cuCtxSetLimit); LOAD(cuCtxResetPersistingL2Cache);
    return 0;
}

int l2_inspect(uintptr_t graph, uintptr_t *nodes_out, uintptr_t *weights_out,
               size_t capacity, size_t *found, char *error, size_t error_size) {
    size_t count = 0, matches = 0;
    CUgraphNode *nodes = NULL;
    CUresult status = p_cuGraphGetNodes((CUgraph)graph, NULL, &count);
    if (status != CUDA_SUCCESS) { snprintf(error, error_size, "cuGraphGetNodes count: %d", (int)status); return -1; }
    nodes = (CUgraphNode *)calloc(count ? count : 1, sizeof(*nodes));
    if (!nodes) { snprintf(error, error_size, "cannot allocate graph node list"); return -1; }
    status = p_cuGraphGetNodes((CUgraph)graph, nodes, &count);
    if (status != CUDA_SUCCESS) { snprintf(error, error_size, "cuGraphGetNodes: %d", (int)status); free(nodes); return -1; }
    for (size_t i = 0; i < count; ++i) {
        CUgraphNodeType type;
        status = p_cuGraphNodeGetType(nodes[i], &type);
        if (status != CUDA_SUCCESS) { snprintf(error, error_size, "cuGraphNodeGetType: %d", (int)status); free(nodes); return -1; }
        if (type == CU_GRAPH_NODE_TYPE_GRAPH || type == CU_GRAPH_NODE_TYPE_CONDITIONAL) {
            snprintf(error, error_size, "nested or conditional graph inspection is unsupported"); free(nodes); return -1;
        }
        if (type != CU_GRAPH_NODE_TYPE_KERNEL) continue;
        CUDA_KERNEL_NODE_PARAMS params = {0};
        status = p_cuGraphKernelNodeGetParams(nodes[i], &params);
        if (status != CUDA_SUCCESS) { snprintf(error, error_size, "cuGraphKernelNodeGetParams: %d", (int)status); free(nodes); return -1; }
        const char *name = NULL;
        if (params.func) status = p_cuFuncGetName(&name, params.func);
        else if (params.kern) status = p_cuKernelGetName(&name, params.kern);
        else { snprintf(error, error_size, "kernel node has neither function nor kernel handle"); free(nodes); return -1; }
        if (status != CUDA_SUCCESS || !name) { snprintf(error, error_size, "CUDA kernel name: %d", (int)status); free(nodes); return -1; }
        if (strcmp(name, "_gemv_relu") != 0) continue;
        if (matches >= capacity || !params.kernelParams || !params.kernelParams[1]) {
            snprintf(error, error_size, "unsupported GEMV parameter array or too many nodes"); free(nodes); return -1;
        }
        size_t offset = 0, size = 0;
        status = params.func ? p_cuFuncGetParamInfo(params.func, 1, &offset, &size)
                             : p_cuKernelGetParamInfo(params.kern, 1, &offset, &size);
        if (status != CUDA_SUCCESS || size != sizeof(CUdeviceptr)) {
            snprintf(error, error_size, "GEMV weight argument metadata unsupported: status %d size %zu", (int)status, size);
            free(nodes); return -1;
        }
        CUdeviceptr weight = 0, base = 0;
        size_t allocation = 0;
        memcpy(&weight, params.kernelParams[1], sizeof(weight));
        status = p_cuMemGetAddressRange(&base, &allocation, weight);
        if (status != CUDA_SUCCESS || weight < base || weight - base > allocation ||
                allocation - (size_t)(weight - base) < 2 * 1024 * 1024) {
            snprintf(error, error_size, "GEMV weight allocation/range invalid: status %d", (int)status); free(nodes); return -1;
        }
        nodes_out[matches] = (uintptr_t)nodes[i];
        weights_out[matches] = (uintptr_t)weight;
        ++matches;
    }
    free(nodes);
    *found = matches;
    return 0;
}

/* FFN2 is the first _gemv reachable along each _gemv_relu's unique successor
   chain. No graph enumeration ordering is used to associate the two kernels. */
int l2_inspect_ffn2(uintptr_t graph, uintptr_t *nodes_out, uintptr_t *weights_out,
                    size_t capacity, size_t *found, char *error, size_t error_size) {
    uintptr_t relus[24], unused_weights[24];
    size_t relu_count = 0, total_nodes = 0;
    if (!driver) { snprintf(error, error_size, "CUDA driver was not initialized"); return -1; }
    *(void **)(&p_cuGraphNodeGetDependentNodes) = dlsym(driver, STR(cuGraphNodeGetDependentNodes));
    if (!p_cuGraphNodeGetDependentNodes) {
        snprintf(error, error_size, "missing CUDA symbol cuGraphNodeGetDependentNodes"); return -1;
    }
    if (l2_inspect(graph, relus, unused_weights, 24, &relu_count, error, error_size)) return -1;
    if (relu_count != 24 || capacity < 24) {
        snprintf(error, error_size, "expected 24 FFN1 nodes and FFN2 output capacity"); return -1;
    }
    CHECK(p_cuGraphGetNodes((CUgraph)graph, NULL, &total_nodes));
    CUgraphNode *visited = (CUgraphNode *)calloc(total_nodes ? total_nodes : 1, sizeof(*visited));
    if (!visited) { snprintf(error, error_size, "cannot allocate dependency path"); return -1; }
    for (size_t i = 0; i < 24; ++i) {
        CUgraphNode current = (CUgraphNode)relus[i];
        size_t used = 0;
        int matched = 0;
        while (used < total_nodes) {
            for (size_t j = 0; j < used; ++j) {
                if (visited[j] == current) {
                    snprintf(error, error_size, "cycle in FFN2 dependency chain at FFN1 %zu", i);
                    free(visited); return -1;
                }
            }
            visited[used++] = current;
            size_t successors = 0;
            CUresult status = p_cuGraphNodeGetDependentNodes(current, NULL, &successors);
            if (status != CUDA_SUCCESS || successors != 1) {
                snprintf(error, error_size, "FFN2 dependency chain has %zu successors (status %d) at FFN1 %zu",
                         successors, (int)status, i); free(visited); return -1;
            }
            CUgraphNode next = NULL;
            status = p_cuGraphNodeGetDependentNodes(current, &next, &successors);
            if (status != CUDA_SUCCESS || successors != 1 || !next) {
                snprintf(error, error_size, "cannot read unique FFN2 successor: status %d", (int)status);
                free(visited); return -1;
            }
            current = next;
            CUgraphNodeType type;
            status = p_cuGraphNodeGetType(current, &type);
            if (status != CUDA_SUCCESS || type == CU_GRAPH_NODE_TYPE_GRAPH || type == CU_GRAPH_NODE_TYPE_CONDITIONAL) {
                snprintf(error, error_size, "unsupported node in FFN2 dependency chain: status %d", (int)status);
                free(visited); return -1;
            }
            if (type != CU_GRAPH_NODE_TYPE_KERNEL) continue;
            CUDA_KERNEL_NODE_PARAMS params = {0};
            status = p_cuGraphKernelNodeGetParams(current, &params);
            if (status != CUDA_SUCCESS) { snprintf(error, error_size, "FFN2 kernel params: %d", (int)status); free(visited); return -1; }
            const char *name = NULL;
            if (params.func) status = p_cuFuncGetName(&name, params.func);
            else if (params.kern) status = p_cuKernelGetName(&name, params.kern);
            else { snprintf(error, error_size, "FFN2 chain kernel has no handle"); free(visited); return -1; }
            if (status != CUDA_SUCCESS || !name) { snprintf(error, error_size, "FFN2 kernel name: %d", (int)status); free(visited); return -1; }
            if (strcmp(name, "_gemv_relu") == 0) {
                snprintf(error, error_size, "another FFN1 occurred before FFN2 for layer %zu", i); free(visited); return -1;
            }
            if (strcmp(name, "_gemv") != 0) continue;
            if (params.gridDimX != 64 || params.gridDimY != 1 || params.gridDimZ != 1 ||
                    !params.kernelParams || !params.kernelParams[1]) {
                snprintf(error, error_size, "FFN2 GEMV grid or argument array is unexpected"); free(visited); return -1;
            }
            for (size_t j = 0; j < i; ++j) {
                if (nodes_out[j] == (uintptr_t)current) {
                    snprintf(error, error_size, "multiple FFN1 nodes map to the same FFN2"); free(visited); return -1;
                }
            }
            size_t offset = 0, size = 0;
            status = params.func ? p_cuFuncGetParamInfo(params.func, 1, &offset, &size)
                                 : p_cuKernelGetParamInfo(params.kern, 1, &offset, &size);
            if (status != CUDA_SUCCESS || size != sizeof(CUdeviceptr)) {
                snprintf(error, error_size, "FFN2 weight argument metadata unsupported: status %d size %zu", (int)status, size);
                free(visited); return -1;
            }
            CUdeviceptr weight = 0, base = 0;
            size_t allocation = 0;
            memcpy(&weight, params.kernelParams[1], sizeof(weight));
            status = p_cuMemGetAddressRange(&base, &allocation, weight);
            if (status != CUDA_SUCCESS || weight < base || weight - base > allocation ||
                    allocation - (size_t)(weight - base) < 2 * 1024 * 1024) {
                snprintf(error, error_size, "FFN2 weight allocation/range invalid: status %d", (int)status);
                free(visited); return -1;
            }
            nodes_out[i] = (uintptr_t)current;
            weights_out[i] = (uintptr_t)weight;
            matched = 1;
            break;
        }
        if (!matched) {
            snprintf(error, error_size, "FFN2 dependency chain exhausted at FFN1 %zu", i); free(visited); return -1;
        }
    }
    free(visited);
    *found = 24;
    return 0;
}

int l2_copy_weight(uintptr_t address, void *out, size_t size, char *error, size_t error_size) {
    CHECK(p_cuMemcpyDtoH(out, (CUdeviceptr)address, size));
    return 0;
}

int l2_apply(uintptr_t node, uintptr_t weight, size_t size, float ratio, char *error, size_t error_size) {
    CUkernelNodeAttrValue value = {0};
    value.accessPolicyWindow.base_ptr = (void *)weight;
    value.accessPolicyWindow.num_bytes = size;
    value.accessPolicyWindow.hitRatio = ratio;
    value.accessPolicyWindow.hitProp = CU_ACCESS_PROPERTY_PERSISTING;
    value.accessPolicyWindow.missProp = CU_ACCESS_PROPERTY_NORMAL;
    CHECK(p_cuGraphKernelNodeSetAttribute((CUgraphNode)node, CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW, &value));
    return 0;
}

int l2_get_limit(size_t *out, char *error, size_t error_size) {
    CHECK(p_cuCtxGetLimit(out, CU_LIMIT_PERSISTING_L2_CACHE_SIZE));
    return 0;
}
int l2_set_limit(size_t value, char *error, size_t error_size) {
    CHECK(p_cuCtxSetLimit(CU_LIMIT_PERSISTING_L2_CACHE_SIZE, value));
    return 0;
}
int l2_reset(char *error, size_t error_size) {
    CHECK(p_cuCtxResetPersistingL2Cache());
    return 0;
}
