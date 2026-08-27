/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Hiruynk
 * SPDX-License-Identifier: Apache-2.0
 *
 * The update equation is adapted from the Apache-2.0 Mamba-2 generation
 * kernel in NVIDIA TensorRT-LLM v0.18.2 and the original Mamba project.
 */

#include "mamba2_update_plugin.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace aniflive::mamba
{
namespace
{

__device__ __forceinline__ float softplus(float value)
{
    return value <= 20.0F ? __logf(1.0F + __expf(value)) : value;
}

__global__ __launch_bounds__(128, 2) void mamba2UpdateKernel(__half const* x, __half const* state,
    __half const* delta, float const* deltaBias, float const* a, __half const* b, __half const* c,
    float const* d, __half const* z, __half* y, __half* stateOut, int32_t dim, int32_t nHeads,
    int32_t nGroups, int32_t dState, bool deltaSoftplus)
{
    int32_t const channel = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (channel >= dim)
    {
        return;
    }

    int32_t const batchIndex = static_cast<int32_t>(blockIdx.y);
    int32_t const headDim = dim / nHeads;
    int32_t const head = channel / headDim;
    int32_t const headChannel = channel % headDim;
    int32_t const group = head / (nHeads / nGroups);

    float const xValue = __half2float(x[batchIndex * dim + channel]);
    float const zValue = __half2float(z[batchIndex * dim + channel]);
    float const deltaValue = __half2float(delta[batchIndex * nHeads + head]) + deltaBias[head];
    float const dt = deltaSoftplus ? softplus(deltaValue) : deltaValue;
    float const transition = __expf(a[head] * dt);

    float output = d[head] * xValue;
    int64_t const stateBase
        = (static_cast<int64_t>(batchIndex) * nHeads + head) * dState * headDim + headChannel;
    int64_t const bcBase = (static_cast<int64_t>(batchIndex) * nGroups + group) * dState;

#pragma unroll 8
    for (int32_t index = 0; index < 128; ++index)
    {
        if (index >= dState)
        {
            break;
        }
        int64_t const stateIndex = stateBase + static_cast<int64_t>(index) * headDim;
        int64_t const bcIndex = bcBase + index;
        float const oldState = __half2float(state[stateIndex]);
        float const bValue = __half2float(b[bcIndex]);
        float const cValue = __half2float(c[bcIndex]);
        float const newState = oldState * transition + bValue * dt * xValue;
        stateOut[stateIndex] = __float2half_rn(newState);
        output += newState * cValue;
    }

    float const sigmoid = 1.0F / (1.0F + __expf(-zValue));
    output *= zValue * sigmoid;
    y[batchIndex * dim + channel] = __float2half_rn(output);
}

} // namespace

int32_t launchMamba2Update(void const* x, void const* state, void const* delta, void const* deltaBias, void const* a,
    void const* b, void const* c, void const* d, void const* z, void* y, void* stateOut, int32_t batch,
    int32_t dim, int32_t nHeads, int32_t nGroups, int32_t dState, bool deltaSoftplus,
    cudaStream_t stream) noexcept
{
    if (x == nullptr || state == nullptr || delta == nullptr || deltaBias == nullptr || a == nullptr || b == nullptr
        || c == nullptr || d == nullptr || z == nullptr || y == nullptr || stateOut == nullptr || batch <= 0
        || dim <= 0 || nHeads <= 0 || nGroups <= 0 || dState <= 0 || dState > 128 || dim % nHeads != 0
        || nHeads % nGroups != 0)
    {
        return 1;
    }

    dim3 const block(128, 1, 1);
    dim3 const grid(static_cast<uint32_t>((dim + 127) / 128), static_cast<uint32_t>(batch), 1);
    mamba2UpdateKernel<<<grid, block, 0, stream>>>(static_cast<__half const*>(x),
        static_cast<__half const*>(state), static_cast<__half const*>(delta), static_cast<float const*>(deltaBias),
        static_cast<float const*>(a), static_cast<__half const*>(b), static_cast<__half const*>(c),
        static_cast<float const*>(d), static_cast<__half const*>(z), static_cast<__half*>(y),
        static_cast<__half*>(stateOut), dim, nHeads, nGroups, dState, deltaSoftplus);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
}

} // namespace aniflive::mamba
