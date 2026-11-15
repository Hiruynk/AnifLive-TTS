//! Experimental functional QKV projection plus K/V prefix copies.
//!
//! Adapted from MLX v0.32.2 `mlx/backend/metal/kernels/gemv.h`.
//! MIT License
//! Copyright © 2023-2024 Apple Inc.
//! Permission is hereby granted, free of charge, to any person obtaining a copy
//! of this software and associated documentation files (the "Software"), to deal
//! in the Software without restriction, including without limitation the rights
//! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//! copies of the Software, and to permit persons to whom the Software is
//! furnished to do so, subject to the following conditions:
//! The above copyright notice and this permission notice shall be included in all
//! copies or substantial portions of the Software.
//! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//! SOFTWARE.
use mlx_rs::{
    error::{Exception, Result},
    fast::{MetalKernel, MetalKernelApplyOptions, MetalKernelOptions, MetalKernelOutput},
    Array, Dtype,
};

// The original TM1 GEMV reduction and bias expression are retained. Each
// thread also copies a disjoint part of the two prior K/V arrays. Appended
// elements are written by the same lane that produces each K or V QKV row.
const SOURCE: &str = r#"
    const int in_vec_size = weights_shape[1];
    const int out_vec_size = weights_shape[0];
    const int matrix_ld = in_vec_size;
    thread float result[1] = {0};
    thread float inter[4];
    thread float v_coeff[4];
    const int thrN = int(thread_index_in_simdgroup);
    const int bm = int(simdgroup_index_in_threadgroup);
    int bn = thrN * 4;
    const int original_row = int(threadgroup_position_in_grid.x) * 4 + bm;
    if (original_row >= out_vec_size) return;
    int out_row = original_row;
    const device half* mat = weights + size_t(out_row) * matrix_ld;
    constexpr const uniform<int> loop_stride = make_uniform(128);
    const uniform<int> in_size = make_uniform(in_vec_size);
    const uniform<int> n_iter = in_size / loop_stride;
    for (int i = 0; i < n_iter; ++i) {
        #pragma clang loop unroll(full)
        for (int tn = 0; tn < 4; tn++) v_coeff[tn] = vec[bn + tn];
        int mat_offset = 0;
        #pragma clang loop unroll(full)
        for (int tm = 0; tm < 1; tm++) {
            #pragma clang loop unroll(full)
            for (int tn = 0; tn < 4; tn++) inter[tn] = float(mat[mat_offset + bn + tn]);
            #pragma clang loop unroll(full)
            for (int tn = 0; tn < 4; tn++) result[tm] += inter[tn] * v_coeff[tn];
            mat_offset += matrix_ld;
        }
        bn += 128;
    }
    #pragma clang loop unroll(full)
    for (int tm = 0; tm < 1; tm++) {
        #pragma clang loop unroll(full)
        for (ushort sn = 16; sn >= 1; sn >>= 1) {
            result[tm] += simd_shuffle_down(result[tm], sn);
        }
    }
    if (thrN == 0) {
        float value = result[0] + bias[out_row];
        qkv[out_row] = value;
        const size_t tail = size_t(old_k_shape[1]) * 512;
        if (out_row >= 512 && out_row < 1024) new_k[tail + out_row - 512] = value;
        if (out_row >= 1024) new_v[tail + out_row - 1024] = value;
    }
    const size_t total = size_t(old_k_shape[1]) * 512;
    for (size_t i = size_t(original_row) * 32 + thrN; i < total; i += 1536 * 32) {
        const size_t row = i / 512;
        const size_t col = i % 512;
        new_k[i] = old_k[row * old_k_strides[1] + col * old_k_strides[2]];
        new_v[i] = old_v[row * old_v_strides[1] + col * old_v_strides[2]];
    }
"#;

pub struct QkvCacheFused {
    kernel: MetalKernel,
}
impl QkvCacheFused {
    pub fn new() -> Result<Self> {
        Ok(Self{kernel:MetalKernel::new("ar_qkv_cache_fused_tm1",
        &["weights","vec","bias","old_k","old_v"],&["qkv","new_k","new_v"],SOURCE,
        MetalKernelOptions{header:"#include <metal_stdlib>\n#include <metal_simdgroup>\nusing namespace metal;\n".into(),ensure_row_contiguous:false,..Default::default()})?})
    }
    pub fn forward(
        &self,
        weights: &Array,
        vec: &Array,
        bias: &Array,
        old_k: &Array,
        old_v: &Array,
    ) -> Result<Vec<Array>> {
        if weights.dtype() != Dtype::Float16
            || weights.shape() != [1536, 512]
            || weights.strides() != [512, 1]
            || vec.dtype() != Dtype::Float32
            || vec.shape() != [1, 1, 512]
            || bias.dtype() != Dtype::Float32
            || bias.shape() != [1536]
            || bias.strides() != [1]
            || old_k.dtype() != Dtype::Float32
            || old_k.ndim() != 3
            || old_k.shape()[0] != 1
            || old_k.shape()[1] <= 0
            || old_k.shape()[2] != 512
            || old_v.dtype() != Dtype::Float32
            || old_v.shape() != old_k.shape()
        {
            return Err(Exception::custom("invalid fused QKV/KV inputs"));
        }
        let next = old_k.shape()[1]
            .checked_add(1)
            .ok_or_else(|| Exception::custom("KV length overflow"))?;
        let vec = vec.contiguous()?;
        // SAFETY: fixed 1536 SIMD rows each have one output writer; 49,152
        // unique lanes partition all logical old-cache elements. Tail writes
        // are disjoint from the copied prefix. Array strides are read at run time.
        unsafe {
            self.kernel.apply(
                &[weights, &vec, bias, old_k, old_v],
                &[
                    MetalKernelOutput {
                        shape: vec![1, 1, 1536],
                        dtype: Dtype::Float32,
                    },
                    MetalKernelOutput {
                        shape: vec![1, next, 512],
                        dtype: Dtype::Float32,
                    },
                    MetalKernelOutput {
                        shape: vec![1, next, 512],
                        dtype: Dtype::Float32,
                    },
                ],
                (384 * 32, 1, 4),
                (32, 1, 4),
                MetalKernelApplyOptions::default(),
            )
        }
    }
}
