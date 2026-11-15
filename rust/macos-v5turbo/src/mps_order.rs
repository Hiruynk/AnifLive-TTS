//! Half matrix products with the reduction order of official MPS FP16.
//!
//! Cancellation probes and random-shape checks on torch 2.9.1 MPS show:
//! * linear/matmul with more than 16 rows: K is reduced in 16-wide blocks; each
//!   block adds four groups (c,c+1)+(c+8,c+9) of exact Float32 products to one
//!   Float32 accumulator in order; bias is added last; one rounding to Half
//!   (100% exact on random shapes and on captured AR prefill linears).
//! * linear with at most 16 rows: 64-wide K segments are dealt round-robin to
//!   eight lanes; each lane accumulates its segments in the block/group order
//!   above; lanes are then added in order 0..7; bias last (100% exact).
//! The K tail is treated as zero products. Kernels compile in MLX Safe math
//! mode, so the written association is kept.
use mlx_rs::{
    error::{Exception, Result},
    ops::indexing::IndexOp,
    fast::{MetalKernel, MetalKernelApplyOptions, MetalKernelOptions, MetalKernelOutput},
    Array, Dtype,
};

const HEADER: &str = "#include <metal_stdlib>\n#include <metal_simdgroup>\nusing namespace metal;\n";

// One simdgroup = 4 rows x 8 lanes; a threadgroup (32,1,4) covers 16 rows.
// Grid z indexes input vectors (M <= 16).
const GEMV: &str = r#"
    const int K = dims[0], N = dims[1], has_bias = dims[2], relu = dims[3];
    const int tid = int(thread_index_in_simdgroup);
    const int lane = tid & 7;
    const int sub = tid >> 3;
    const int row = int(threadgroup_position_in_grid.x) * 16 + int(simdgroup_index_in_threadgroup) * 4 + sub;
    const int m = int(threadgroup_position_in_grid.z);
    const bool valid = row < N;
    const device half* w = weights + size_t(valid ? row : 0) * K;
    const device half* v = vec + size_t(m) * K;
    float acc = 0.0f;
    const int nseg = (K + 63) / 64;
    if ((K & 63) == 0) {
        for (int s = lane; s < nseg; s += 8) {
            for (int blk = 0; blk < 4; blk++) {
                const int base = s * 64 + blk * 16;
                const device half4* w4 = (const device half4*)(w + base);
                const device half4* v4 = (const device half4*)(v + base);
                float4 a0 = float4(w4[0]) * float4(v4[0]);
                float4 a1 = float4(w4[1]) * float4(v4[1]);
                float4 a2 = float4(w4[2]) * float4(v4[2]);
                float4 a3 = float4(w4[3]) * float4(v4[3]);
                // groups (c,c+1)+(c+8,c+9) for c = 0,2,4,6
                float g0 = (a0.x + a0.y) + (a2.x + a2.y);
                acc = acc + g0;
                float g1 = (a0.z + a0.w) + (a2.z + a2.w);
                acc = acc + g1;
                float g2 = (a1.x + a1.y) + (a3.x + a3.y);
                acc = acc + g2;
                float g3 = (a1.z + a1.w) + (a3.z + a3.w);
                acc = acc + g3;
            }
        }
    } else
    for (int s = lane; s < nseg; s += 8) {
        for (int blk = 0; blk < 4; blk++) {
            const int base = s * 64 + blk * 16;
            for (int g = 0; g < 4; g++) {
                const int c = base + 2 * g;
                float t0 = c < K ? float(w[c]) * float(v[c]) : 0.0f;
                float t1 = c + 1 < K ? float(w[c + 1]) * float(v[c + 1]) : 0.0f;
                float t8 = c + 8 < K ? float(w[c + 8]) * float(v[c + 8]) : 0.0f;
                float t9 = c + 9 < K ? float(w[c + 9]) * float(v[c + 9]) : 0.0f;
                float p1 = t0 + t1;
                float p2 = t8 + t9;
                float grp = p1 + p2;
                acc = acc + grp;
            }
        }
    }
    float total = simd_shuffle(acc, ushort(sub * 8));
    for (int l = 1; l < 8; l++) total = total + simd_shuffle(acc, ushort(sub * 8 + l));
    if (lane == 0 && valid) {
        if (has_bias) total = total + float(bias[row]);
        if (relu) total = max(total, 0.0f);
        out[size_t(m) * N + row] = half(total);
    }
"#;

// out[b,m,n] = half(acc(a[b,m,:], bt[b,n,:]) * alpha (+ bias[n])), 16x16 tiles.
const GEMM: &str = r#"
    const int M = dims[0], N = dims[1], K = dims[2], has_bias = dims[3], b_shared = dims[4];
    const float alpha = scale[0];
    const int tx = int(thread_position_in_threadgroup.x);
    const int ty = int(thread_position_in_threadgroup.y);
    const int n = int(threadgroup_position_in_grid.x) * 16 + tx;
    const int m = int(threadgroup_position_in_grid.y) * 16 + ty;
    const int b = int(threadgroup_position_in_grid.z);
    threadgroup float As[16][17];
    threadgroup float Bs[16][17];
    const device half* ap = a + size_t(b) * M * K;
    const device half* bp = bt + (b_shared ? 0 : size_t(b) * N * K);
    const int arow = int(threadgroup_position_in_grid.y) * 16 + ty;
    const int brow = int(threadgroup_position_in_grid.x) * 16 + ty;
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += 16) {
        const int kk = k0 + tx;
        As[ty][tx] = (arow < M && kk < K) ? float(ap[size_t(arow) * K + kk]) : 0.0f;
        Bs[ty][tx] = (brow < N && kk < K) ? float(bp[size_t(brow) * K + kk]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int g = 0; g < 4; g++) {
            const int c = 2 * g;
            float t0 = As[ty][c] * Bs[tx][c];
            float t1 = As[ty][c + 1] * Bs[tx][c + 1];
            float t8 = As[ty][c + 8] * Bs[tx][c + 8];
            float t9 = As[ty][c + 9] * Bs[tx][c + 9];
            float p1 = t0 + t1;
            float p2 = t8 + t9;
            float grp = p1 + p2;
            acc = acc + grp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (m < M && n < N) {
        float r = acc * alpha;
        if (has_bias) r = r + float(bias[n]);
        out[(size_t(b) * M + m) * N + n] = half(r);
    }
"#;

pub struct MpsOrder {
    gemv: MetalKernel,
    gemm: MetalKernel,
}

thread_local! {
    static KERNELS: std::cell::OnceCell<MpsOrder> = const { std::cell::OnceCell::new() };
}

fn with<T>(f: impl FnOnce(&MpsOrder) -> Result<T>) -> Result<T> {
    KERNELS.with(|cell| {
        if cell.get().is_none() {
            let _ = cell.set(MpsOrder::new()?);
        }
        f(cell.get().expect("initialized"))
    })
}

impl MpsOrder {
    fn new() -> Result<Self> {
        let opts = || MetalKernelOptions { header: HEADER.into(), ..Default::default() };
        Ok(Self {
            gemv: MetalKernel::new("mps_order_gemv_f16", &["weights", "vec", "bias", "dims"], &["out"], GEMV, opts())?,
            gemm: MetalKernel::new("mps_order_gemm_f16", &["a", "bt", "bias", "dims", "scale"], &["out"], GEMM, opts())?,
        })
    }
}

fn half_check(x: &Array, what: &str) -> Result<()> {
    if x.dtype() != Dtype::Float16 {
        return Err(Exception::custom(format!("mps_order {what} must be Half")));
    }
    Ok(())
}

/// Official MPS F.linear on Half: x [..., K], weight [N, K], optional bias [N].
/// Rows <= 16 use the lane GEMV order, more rows the blocked GEMM order.
pub fn linear(x: &Array, weight: &Array, bias: Option<&Array>, relu: bool) -> Result<Array> {
    half_check(x, "input")?;
    half_check(weight, "weight")?;
    let (n, k) = (weight.shape()[0], weight.shape()[1]);
    if weight.ndim() != 2 || x.shape().last() != Some(&k) {
        return Err(Exception::custom("mps_order linear shape mismatch"));
    }
    let rows = (x.size() / k as usize) as i32;
    let mut out_shape = x.shape().to_vec();
    *out_shape.last_mut().expect("rank >= 1") = n;
    let x2 = x.reshape(&[rows, k])?.contiguous()?;
    let w = weight.contiguous()?;
    let zero_bias;
    let b = match bias {
        Some(b) => {
            half_check(b, "bias")?;
            b.contiguous()?
        }
        None => {
            zero_bias = Array::from_slice(&[0f32], &[1]).as_dtype(Dtype::Float16)?;
            zero_bias.clone()
        }
    };
    let y = if rows <= 16 {
        let dims = Array::from_slice(&[k, n, bias.is_some() as i32, relu as i32], &[4]);
        with(|ks| {
            // SAFETY: reads are bounds-checked against K/N and the M vectors;
            // each (row, m) output is written once by lane 0 of its group.
            let o = unsafe {
                ks.gemv.apply(
                    &[&w, &x2, &b, &dims],
                    &[MetalKernelOutput { shape: vec![rows, n], dtype: Dtype::Float16 }],
                    (((n + 15) / 16) * 32, 1, 4 * rows),
                    (32, 1, 4),
                    MetalKernelApplyOptions::default(),
                )?
            };
            Ok(o.into_iter().next().expect("one output"))
        })?
    } else {
        let y = gemm(&x2.reshape(&[1, rows, k])?, &w.reshape(&[1, n, k])?, 1.0, bias.map(|_| &b))?;
        let y = y.reshape(&[rows, n])?;
        if relu { mlx_rs::ops::maximum(&y, Array::from_f32(0.0).as_dtype(Dtype::Float16)?)? } else { y }
    };
    Ok(y.reshape(&out_shape)?)
}

/// Batched a [B, M, K] times bt [B or 1, N, K]^T with the blocked GEMM order;
/// the Float32 accumulator is scaled by `alpha` before the optional bias and
/// the single rounding to Half.
pub fn gemm(a: &Array, bt: &Array, alpha: f32, bias: Option<&Array>) -> Result<Array> {
    half_check(a, "a")?;
    half_check(bt, "bt")?;
    if a.ndim() != 3 || bt.ndim() != 3 || a.shape()[2] != bt.shape()[2] || !(bt.shape()[0] == 1 || bt.shape()[0] == a.shape()[0]) {
        return Err(Exception::custom("mps_order gemm expects a [B,M,K], bt [B|1,N,K]"));
    }
    let (batch, m, k) = (a.shape()[0], a.shape()[1], a.shape()[2]);
    let n = bt.shape()[1];
    let shared = (bt.shape()[0] == 1 && batch > 1) as i32;
    let a = a.contiguous()?;
    let bt = bt.contiguous()?;
    let b = match bias {
        Some(b) => b.contiguous()?,
        None => Array::from_slice(&[0f32], &[1]).as_dtype(Dtype::Float16)?,
    };
    let dims = Array::from_slice(&[m, n, k, bias.is_some() as i32, shared], &[5]);
    let scale = Array::from_slice(&[alpha], &[1]);
    with(|ks| {
        // SAFETY: tile loads are bounds-checked against M/N/K; each output
        // element is written by exactly one thread.
        let o = unsafe {
            ks.gemm.apply(
                &[&a, &bt, &b, &dims, &scale],
                &[MetalKernelOutput { shape: vec![batch, m, n], dtype: Dtype::Float16 }],
                (((n + 15) / 16) * 16, ((m + 15) / 16) * 16, batch),
                (16, 16, 1),
                MetalKernelApplyOptions::default(),
            )?
        };
        Ok(o.into_iter().next().expect("one output"))
    })
}

/// Reduction layout official MPS Half conv1d uses for a measured shape, as the
/// order of the (tap, channel) products along K: tap-outer 16-channel blocks,
/// or 32/16-channel blocks with the taps inside. Thresholds are the
/// output lengths at which MPS switched kernels on the qualification host
/// (fp16_conv_mps_sweep.py); unmeasured shapes return None.
fn conv_layout(cin: i32, cout: i32, k: i32, t: i32) -> Option<i32> {
    let (tap_max, blk32_max) = match (cin, cout, k) {
        (_, _, 1) => return Some(0),
        (192, 768, 3) => (148, 384),
        (768, 192, 3) => (148, 1664),
        (512, 1024, 5) => (119, 256),
        _ => return None,
    };
    Some(if t <= tap_max { 0 } else if t <= blk32_max { 32 } else { 16 })
}

/// Stride-1, same-padding Half conv1d: x [1,T,Cin], w [Cout,k,Cin], b [Cout]
/// -> [1,T,Cout], computed as an MPS-order GEMM over the K layout official
/// MPS uses for this shape. Returns None when the shape was not measured.
pub fn conv1d_same(x: &Array, w: &Array, b: &Array) -> Result<Option<Array>> {
    let (cout, k, cin) = (w.shape()[0], w.shape()[1], w.shape()[2]);
    let t = x.shape()[1];
    if x.dtype() != Dtype::Float16 || x.ndim() != 3 || x.shape()[0] != 1 || x.shape()[2] != cin || k % 2 != 1 {
        return Ok(None);
    }
    let Some(layout) = conv_layout(cin, cout, k, t) else { return Ok(None) };
    if layout != 0 && cin % layout != 0 {
        return Ok(None);
    }
    let pad = (k - 1) / 2;
    let x2 = x.reshape(&[t, cin])?;
    let (a, wk) = if k == 1 {
        (x2, w.reshape(&[cout, cin])?)
    } else {
        let zeros = Array::zeros::<f32>(&[pad, cin])?.as_dtype(Dtype::Float16)?;
        let xp = mlx_rs::ops::concatenate(&[&zeros, &x2, &zeros], 0)?;
        let taps: Vec<Array> = (0..k)
            .map(|j| xp.index((j..j + t, ..)))
            .collect();
        let cols = mlx_rs::ops::stack(&taps, 1)?; // [T, k, Cin]
        let order = |z: Array, rows: i32| -> Result<Array> {
            Ok(if layout == 0 {
                z.reshape(&[rows, k * cin])?
            } else {
                z.reshape(&[rows, k, cin / layout, layout])?
                    .transpose_axes(&[0, 2, 1, 3])?
                    .reshape(&[rows, k * cin])?
            })
        };
        (order(cols, t)?, order(w.clone(), cout)?)
    };
    let y = gemm(&a.reshape(&[1, t, k * cin])?, &wk.reshape(&[1, cout, k * cin])?, 1.0, Some(b))?;
    Ok(Some(y.reshape(&[1, t, cout])?))
}
