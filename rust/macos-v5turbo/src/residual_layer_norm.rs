//! Experimental single-row FP32 residual add + LayerNorm.
//!
//! Adapted from MLX v0.32.2 `mlx/backend/metal/kernels/layer_norm.metal`.
//! MIT License
//! Copyright © 2024 Apple Inc.
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

// 512 values: 64 threads, two SIMD groups, eight adjacent values per thread.
// Store the residual sum through volatile FP32 before the copied norm math.
const SOURCE: &str = r#"
    uint lid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    thread float thread_x[8] = {0};
    threadgroup float local_buffer[32] = {0};
    if (simd_group == 0) local_buffer[lane] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base = lid * 8;
    for (int i = 0; i < 8; i++) {
        thread volatile float rounded = lhs[base + i] + rhs[base + i];
        thread_x[i] = rounded;
    }
    float mean = 0;
    for (int i = 0; i < 8; i++) mean += thread_x[i];
    mean = simd_sum(mean);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) local_buffer[simd_group] = mean;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mean = simd_sum(local_buffer[lane]);
    mean /= 512;
    float normalizer = 0;
    for (int i = 0; i < 8; i++) {
        thread_x[i] -= mean;
        normalizer += thread_x[i] * thread_x[i];
    }
    normalizer = simd_sum(normalizer);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) local_buffer[simd_group] = normalizer;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    normalizer = simd_sum(local_buffer[lane]);
    normalizer = metal::precise::rsqrt(normalizer / 512 + 1.0e-5f);
    for (int i = 0; i < 8; i++) {
        thread_x[i] *= normalizer;
        out[base + i] = weight[base + i] * thread_x[i] + bias[base + i];
    }
"#;

pub struct ResidualLayerNorm {
    kernel: MetalKernel,
    // Half I/O: Half + Half residual add rounds in Half (official order), the
    // normalization runs in float, and the output rounds to Half once.
    kernel_h: MetalKernel,
}
impl ResidualLayerNorm {
    pub fn new() -> Result<Self> {
        let make = |name: &str| MetalKernel::new(name,&["lhs","rhs","weight","bias"],&["out"],SOURCE,
            MetalKernelOptions {header:"#include <metal_stdlib>\n#include <metal_simdgroup>\nusing namespace metal;\n".into(),..Default::default()});
        Ok(Self {kernel: make("ar_residual_layer_norm512")?, kernel_h: make("ar_residual_layer_norm512_f16io")?})
    }
    pub fn forward(&self, lhs: &Array, rhs: &Array, weight: &Array, bias: &Array) -> Result<Array> {
        let io = lhs.dtype();
        if !matches!(io, Dtype::Float32 | Dtype::Float16)
            || lhs.shape() != [1, 1, 512]
            || rhs.dtype() != io
            || rhs.shape() != lhs.shape()
            || weight.dtype() != io
            || weight.shape() != [512]
            || bias.dtype() != io
            || bias.shape() != [512]
        {
            return Err(Exception::custom(
                "residual LayerNorm requires FP32 [1,1,512] rows and [512] affine",
            ));
        }
        let lhs = lhs.contiguous()?;
        let rhs = rhs.contiguous()?;
        let weight = weight.contiguous()?;
        let bias = bias.contiguous()?;
        // SAFETY: 64 threads each access eight distinct elements of four
        // contiguous length-512 inputs and write eight distinct outputs.
        let outputs = unsafe {
            (if io == Dtype::Float16 { &self.kernel_h } else { &self.kernel }).apply(
                &[&lhs, &rhs, &weight, &bias],
                &[MetalKernelOutput {
                    shape: vec![1, 1, 512],
                    dtype: io,
                }],
                (64, 1, 1),
                (64, 1, 1),
                MetalKernelApplyOptions::default(),
            )?
        };
        Ok(outputs.into_iter().next().expect("one declared output"))
    }
}
