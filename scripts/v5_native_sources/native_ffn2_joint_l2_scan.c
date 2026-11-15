/* Copyright 2026 Hiruynk. See the repository LICENSE. */
/* Read-only comparison of neural FFN2 L2 windows before and inside a joint graph. */
#include "native_ffn2_l2_scan.c"
#include <stdarg.h>

static __typeof__(&cuGraphChildGraphNodeGetGraph) p_cuGraphChildGraphNodeGetGraph;
static __typeof__(&cuGraphKernelNodeGetAttribute) p_cuGraphKernelNodeGetAttribute;

typedef struct {
    uintptr_t node;
    uintptr_t weightptr;
    uintptr_t base;
    size_t bytes;
    float hit_ratio;
    int hit_prop;
    int miss_prop;
} audit_row;

static int appendf(char *out, size_t capacity, size_t *used,
                   char *error, size_t error_size, const char *format, ...) {
    va_list args;
    int written;
    if (*used >= capacity) {
        snprintf(error, error_size, "JSON output buffer is full");
        return -1;
    }
    va_start(args, format);
    written = vsnprintf(out + *used, capacity - *used, format, args);
    va_end(args);
    if (written < 0 || (size_t)written >= capacity - *used) {
        snprintf(error, error_size, "JSON output buffer is too small");
        return -1;
    }
    *used += (size_t)written;
    return 0;
}

static int find_child_graph(CUgraph outer, CUgraph *child_out,
                            char *error, size_t error_size) {
    size_t count = 0, graph_nodes = 0;
    CUgraphNode *nodes = NULL;
    CUgraph child = NULL;
    CUresult status = p_cuGraphGetNodes(outer, NULL, &count);
    if (status != CUDA_SUCCESS) {
        snprintf(error, error_size, "cuGraphGetNodes count: %d", (int)status);
        return -1;
    }
    nodes = (CUgraphNode *)calloc(count ? count : 1, sizeof(*nodes));
    if (!nodes) {
        snprintf(error, error_size, "cannot allocate outer graph node list");
        return -1;
    }
    status = p_cuGraphGetNodes(outer, nodes, &count);
    if (status != CUDA_SUCCESS) {
        snprintf(error, error_size, "cuGraphGetNodes: %d", (int)status);
        free(nodes);
        return -1;
    }
    for (size_t i = 0; i < count; ++i) {
        CUgraphNodeType type;
        status = p_cuGraphNodeGetType(nodes[i], &type);
        if (status != CUDA_SUCCESS) {
            snprintf(error, error_size, "cuGraphNodeGetType: %d", (int)status);
            free(nodes);
            return -1;
        }
        if (type != CU_GRAPH_NODE_TYPE_GRAPH) continue;
        ++graph_nodes;
        status = p_cuGraphChildGraphNodeGetGraph(nodes[i], &child);
        if (status != CUDA_SUCCESS) {
            snprintf(error, error_size, "cuGraphChildGraphNodeGetGraph: %d", (int)status);
            free(nodes);
            return -1;
        }
    }
    free(nodes);
    if (graph_nodes != 1 || !child) {
        snprintf(error, error_size, "expected exactly one child graph node; found %zu", graph_nodes);
        return -1;
    }
    *child_out = child;
    return 0;
}

static int collect_rows(CUgraph graph, audit_row rows[24],
                        char *error, size_t error_size) {
    uintptr_t nodes[24], weights[24];
    size_t found = 0;
    if (l2_inspect_ffn2((uintptr_t)graph, nodes, weights, 24, &found, error, error_size)) return -1;
    if (found != 24) {
        snprintf(error, error_size, "expected 24 FFN2 nodes; found %zu", found);
        return -1;
    }
    for (size_t i = 0; i < 24; ++i) {
        CUkernelNodeAttrValue value = {0};
        CUresult status = p_cuGraphKernelNodeGetAttribute((CUgraphNode)nodes[i],
                CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW, &value);
        if (status != CUDA_SUCCESS) {
            snprintf(error, error_size, "cuGraphKernelNodeGetAttribute row %zu: %d", i, (int)status);
            return -1;
        }
        rows[i].node = nodes[i];
        rows[i].weightptr = weights[i];
        rows[i].base = (uintptr_t)value.accessPolicyWindow.base_ptr;
        rows[i].bytes = value.accessPolicyWindow.num_bytes;
        rows[i].hit_ratio = value.accessPolicyWindow.hitRatio;
        rows[i].hit_prop = (int)value.accessPolicyWindow.hitProp;
        rows[i].miss_prop = (int)value.accessPolicyWindow.missProp;
    }
    return 0;
}

static int append_rows(char *out, size_t capacity, size_t *used,
                       const char *name, const audit_row rows[24],
                       char *error, size_t error_size) {
    if (appendf(out, capacity, used, error, error_size, "\"%s\":[", name)) return -1;
    for (size_t i = 0; i < 24; ++i) {
        if (appendf(out, capacity, used, error, error_size,
                "%s{\"weightptr\":\"0x%llx\",\"base\":\"0x%llx\",\"bytes\":%zu,"
                "\"hitRatio\":%.9g,\"hitProp\":%d,\"missProp\":%d}",
                i ? "," : "",
                (unsigned long long)rows[i].weightptr,
                (unsigned long long)rows[i].base, rows[i].bytes,
                (double)rows[i].hit_ratio, rows[i].hit_prop, rows[i].miss_prop)) return -1;
    }
    return appendf(out, capacity, used, error, error_size, "]");
}

int ar_joint_l2_audit(uintptr_t original_graph, uintptr_t outer_graph,
                      char *json_out, size_t json_capacity,
                      char *error, size_t error_size) {
    audit_row original[24], embedded[24];
    CUgraph child = NULL;
    size_t used = 0;
    if (!original_graph || !outer_graph || !json_out || !json_capacity || !error || !error_size) {
        if (error && error_size) snprintf(error, error_size, "invalid audit argument");
        return -1;
    }
    if (!driver && l2_init(error, error_size)) return -1;
    *(void **)(&p_cuGraphChildGraphNodeGetGraph) = dlsym(driver, "cuGraphChildGraphNodeGetGraph");
    if (!p_cuGraphChildGraphNodeGetGraph) {
        snprintf(error, error_size, "missing CUDA symbol cuGraphChildGraphNodeGetGraph");
        return -1;
    }
    *(void **)(&p_cuGraphKernelNodeGetAttribute) = dlsym(driver, "cuGraphKernelNodeGetAttribute");
    if (!p_cuGraphKernelNodeGetAttribute) {
        snprintf(error, error_size, "missing CUDA symbol cuGraphKernelNodeGetAttribute");
        return -1;
    }
    if (find_child_graph((CUgraph)outer_graph, &child, error, error_size)) return -1;
    if (collect_rows((CUgraph)original_graph, original, error, error_size)) return -1;
    if (collect_rows(child, embedded, error, error_size)) return -1;
    json_out[0] = '\0';
    if (appendf(json_out, json_capacity, &used, error, error_size, "{")) return -1;
    if (append_rows(json_out, json_capacity, &used, "original", original, error, error_size)) return -1;
    if (appendf(json_out, json_capacity, &used, error, error_size, ",")) return -1;
    if (append_rows(json_out, json_capacity, &used, "embedded", embedded, error, error_size)) return -1;
    return appendf(json_out, json_capacity, &used, error, error_size, "}");
}
