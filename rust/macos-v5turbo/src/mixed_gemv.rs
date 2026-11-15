//! Experimental F16-storage, F32-arithmetic single-vector GEMV.
//!
//! This kernel adapts GEMVKernel::run from MLX v0.32.2,
//! mlx/backend/metal/kernels/gemv.h (SHA-256
//! 296ef622ea3c67d7485aa7cdf102bbfcb582e07bd3207793179a9d1e65ee5a65).
//!
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

// Same 32 lanes, four K elements per lane, and reduction order as MLX.
// The rows per simdgroup may be 1 or 4; matrix storage alone is half.
// Shift the final partial row tile and suppress duplicate writes (N=1025).
const SOURCE: &str = r#"
    const int in_vec_size = weights_shape[1];
    const int out_vec_size = weights_shape[0];
    const int matrix_ld = in_vec_size;
    thread float result[ROWS_PER_SIMD] = {0};
    thread float inter[4];
    thread float v_coeff[4];
    const int thrN = int(thread_index_in_simdgroup);
    const int bm = int(simdgroup_index_in_threadgroup) * ROWS_PER_SIMD;
    int bn = thrN * 4;
    const int original_row = int(threadgroup_position_in_grid.x) * ROWS_PER_GROUP + bm;
    if (original_row >= out_vec_size) return;
    int out_row = original_row + ROWS_PER_SIMD <= out_vec_size ? original_row : out_vec_size - ROWS_PER_SIMD;
    const device half* mat = weights + size_t(out_row) * matrix_ld;
    constexpr const uniform<int> loop_stride = make_uniform(128);
    const uniform<int> in_size = make_uniform(in_vec_size);
    const uniform<int> n_iter = in_size / loop_stride;
    for (int i = 0; i < n_iter; ++i) {
        #pragma clang loop unroll(full)
        for (int tn = 0; tn < 4; tn++) v_coeff[tn] = vec[bn + tn];
        int mat_offset = 0;
        #pragma clang loop unroll(full)
        for (int tm = 0; tm < ROWS_PER_SIMD; tm++) {
            #pragma clang loop unroll(full)
            for (int tn = 0; tn < 4; tn++) inter[tn] = float(mat[mat_offset + bn + tn]);
            #pragma clang loop unroll(full)
            for (int tn = 0; tn < 4; tn++) result[tm] += inter[tn] * v_coeff[tn];
            mat_offset += matrix_ld;
        }
        bn += 128;
    }
    #pragma clang loop unroll(full)
    for (int tm = 0; tm < ROWS_PER_SIMD; tm++) {
        #pragma clang loop unroll(full)
        for (ushort sn = 16; sn >= 1; sn >>= 1) {
            result[tm] += simd_shuffle_down(result[tm], sn);
        }
    }
    if (thrN == 0) {
        #pragma clang loop unroll(full)
        for (int tm = 0; tm < ROWS_PER_SIMD; tm++) {
            if (out_row + tm >= original_row) out[out_row + tm] = result[tm];
        }
    }
"#;
/// A GPU kernel restricted to the five verified single-token AR matrix shapes.
pub struct MixedGemv {
    kernel: MetalKernel,
    biased: MetalKernel,
    biased_relu: MetalKernel,
    // Half-I/O instances of the same source: Half vector/bias are widened to
    // float inside the unchanged F32 reduction and the result is rounded to
    // Half once at the store (FP16 model path).
    kernel_h: MetalKernel,
    biased_h: MetalKernel,
    biased_relu_h: MetalKernel,
    rows_per_simdgroup: i32,
}
impl MixedGemv {
    pub fn rows_per_simdgroup(&self) -> i32 {
        self.rows_per_simdgroup
    }

    pub fn new() -> Result<Self> {
        Self::with_rows_per_simdgroup(4)
    }

    /// Construct the original four-row kernel or the one-row diagnostic variant.
    pub fn with_rows_per_simdgroup(rows: i32) -> Result<Self> {
        if rows != 1 && rows != 4 {
            return Err(Exception::custom(
                "mixed GEMV rows per simdgroup must be 1 or 4",
            ));
        }
        let source = SOURCE
            .replace("ROWS_PER_SIMD", &rows.to_string())
            .replace("ROWS_PER_GROUP", &(4 * rows).to_string());
        // Kernel names include specialization to keep the runtime JIT cache distinct.
        let suffix = if rows == 4 { "" } else { "_tm1" };
        let plain = |name: &str| {
            MetalKernel::new(&format!("{name}{suffix}"), &["weights", "vec"], &["out"], &source,
                MetalKernelOptions {
                    header: "#include <metal_stdlib>\n#include <metal_simdgroup>\nusing namespace metal;\n".into(),
                    ..Default::default()
                })
        };
        let epilogue = |name: &str, expression: &str| {
            let epilogue_source = source.replace("out[out_row + tm] = result[tm];", expression);
            MetalKernel::new(&format!("{name}{suffix}"), &["weights", "vec", "bias"], &["out"], &epilogue_source,
                MetalKernelOptions {
                    header: "#include <metal_stdlib>\n#include <metal_simdgroup>\nusing namespace metal;\n".into(),
                    ..Default::default()
                })
        };
        Ok(Self { kernel: MetalKernel::new(
            &format!("ar_mixed_gemv{suffix}"), &["weights", "vec"], &["out"], &source,
            MetalKernelOptions {
                header: "#include <metal_stdlib>\n#include <metal_simdgroup>\nusing namespace metal;\n".into(),
                ..Default::default()
            },
        )?,
        biased: epilogue("ar_mixed_gemv_bias", "out[out_row + tm] = result[tm] + bias[out_row + tm];")?,
        biased_relu: epilogue("ar_mixed_gemv_bias_relu", "out[out_row + tm] = max(result[tm] + bias[out_row + tm], 0.0f);")?,
        kernel_h: plain("ar_mixed_gemv_f16io")?,
        biased_h: epilogue("ar_mixed_gemv_bias_f16io", "out[out_row + tm] = result[tm] + float(bias[out_row + tm]);")?,
        biased_relu_h: epilogue("ar_mixed_gemv_bias_relu_f16io", "out[out_row + tm] = max(result[tm] + float(bias[out_row + tm]), 0.0f);")?,
        rows_per_simdgroup: rows,
        })
    }

    /// Apply an original row-major F16 matrix [N,K] to one F32 vector [...,K].
    /// Output is F32 [...,N]; no batching or activation downcasting is performed.
    /// This custom kernel is not compatible with shapeless MLX compilation.
    pub fn apply(&self, matrix: &Array, vector: &Array) -> Result<Array> {
        self.dispatch(matrix, vector, None, false)
    }

    /// Research epilogue: the original F32 add and optional ReLU after the
    /// unchanged reduction. Requires full-runtime bit verification before use.
    pub fn apply_with_bias(
        &self,
        matrix: &Array,
        vector: &Array,
        bias: &Array,
        relu: bool,
    ) -> Result<Array> {
        self.dispatch(matrix, vector, Some(bias), relu)
    }

    fn dispatch(
        &self,
        matrix: &Array,
        vector: &Array,
        bias: Option<&Array>,
        relu: bool,
    ) -> Result<Array> {
        let shape = matrix.shape();
        if shape.len() != 2 || matrix.dtype() != Dtype::Float16 {
            return Err(Exception::custom("mixed GEMV requires F16 matrix [N,K]"));
        }
        let (n, k) = (shape[0], shape[1]);
        if !matches!(
            (k, n),
            (512, 1536) | (512, 512) | (512, 2048) | (2048, 512) | (512, 1025)
        ) || matrix.strides() != [k as usize, 1]
        {
            return Err(Exception::custom(
                "unsupported mixed GEMV shape or matrix layout",
            ));
        }
        let vector_shape = vector.shape();
        let io = vector.dtype();
        if !matches!(io, Dtype::Float32 | Dtype::Float16)
            || vector_shape.last() != Some(&k)
            || vector.size() != k as usize
        {
            return Err(Exception::custom(
                "mixed GEMV requires one F32 vector [...,K]",
            ));
        }
        let mut output_shape = vector_shape.to_vec();
        *output_shape
            .last_mut()
            .expect("vector has a last dimension") = n;
        let vector = vector.contiguous()?;
        let (kernel, inputs) = if let Some(bias) = bias {
            if bias.shape() != [n] || bias.dtype() != io || bias.strides() != [1] {
                return Err(Exception::custom(
                    "mixed GEMV bias must be contiguous F32 [N]",
                ));
            }
            (
                match (relu, io == Dtype::Float16) {
                    (true, false) => &self.biased_relu,
                    (false, false) => &self.biased,
                    (true, true) => &self.biased_relu_h,
                    (false, true) => &self.biased_h,
                },
                vec![matrix, &vector, bias],
            )
        } else {
            (if io == Dtype::Float16 { &self.kernel_h } else { &self.kernel }, vec![matrix, &vector])
        };
        // SAFETY: only the listed contiguous [N,K] F16 layouts are accepted;
        // K is a multiple of 128. The contiguous vector has exactly K F32 values.
        // Full 128-thread groups bound reads, and the tail writes each output once.
        let output = unsafe {
            kernel.apply(
                &inputs,
                &[MetalKernelOutput {
                    shape: output_shape,
                    dtype: io,
                }],
                (
                    ((n + 4 * self.rows_per_simdgroup - 1) / (4 * self.rows_per_simdgroup)) * 32,
                    1,
                    4,
                ),
                (32, 1, 4),
                MetalKernelApplyOptions::default(),
            )?
        };
        Ok(output.into_iter().next().expect("one declared output"))
    }
}
