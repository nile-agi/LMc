/*
 * src/math_ops.c — LMc Core Math Primitives
 *
 * These are the hot paths for inference. Every function here is
 * called millions of times per token. Keep them tight.
 *
 * Optimization layers (in order of portability):
 *   1. Scalar C99       — always compiled (this file)
 *   2. OpenMP parallel  — activated with -fopenmp
 *   3. SIMD intrinsics  — future: AVX2, NEON, RVV
 *   4. Hardware backend — future: CUDA, Metal, Vulkan, NNAPI
 *
 * SPDX-License-Identifier: MIT
 */

#include "lmc_internal.h"

/* ============================================================
 * FP16 → FP32 CONVERSION
 * IEEE 754 half-precision to single-precision, no FPU required.
 * ============================================================ */
float lmc_f16_to_f32(uint16_t h) {
    uint32_t sign     = (uint32_t)(h >> 15) << 31;
    uint32_t exponent = (uint32_t)((h >> 10) & 0x1F);
    uint32_t mantissa = (uint32_t)(h & 0x3FF);
    uint32_t f;

    if (exponent == 0) {
        if (mantissa == 0) {
            f = sign; /* ±0 */
        } else {
            /* Denormal → normalize */
            exponent = 1;
            while (!(mantissa & 0x400)) { mantissa <<= 1; exponent--; }
            mantissa &= 0x3FF;
            f = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        f = sign | 0x7F800000 | (mantissa << 13); /* Inf or NaN */
    } else {
        f = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }

    float result;
    memcpy(&result, &f, 4);
    return result;
}

/* ============================================================
 * GELU ACTIVATION
 * Gaussian Error Linear Unit — GPT-2 uses the tanh approximation.
 * gelu(x) ≈ 0.5x * (1 + tanh(√(2/π) * (x + 0.044715x³)))
 * ============================================================ */
float lmc_gelu(float x) {
    const float c = 0.7978845608028654f; /* sqrt(2/π) */
    const float k = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(c * (x + k * x3)));
}

/* ============================================================
 * SOFTMAX (in-place, numerically stable)
 * Subtracts max before exp to prevent overflow.
 * Falls back to uniform if sum underflows.
 * ============================================================ */
void lmc_softmax(float *x, int n) {
    /* Find max for numerical stability */
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    /* Exponentiate and sum */
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    /* Guard against underflow */
    if (sum < 1e-30f) {
        float inv = 1.0f / (float)n;
        for (int i = 0; i < n; i++) x[i] = inv;
        return;
    }

    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv_sum;
}

/* ============================================================
 * LAYER NORMALIZATION
 * GPT-2 uses pre-normalization (norm before each sub-layer).
 * out[i] = (x[i] - mean) / sqrt(var + eps) * weight[i] + bias[i]
 * ============================================================ */
void lmc_layer_norm(float *out, const float *x,
                    const float *weight, const float *bias, int dim) {
    const float eps = 1e-5f;

    /* Compute mean */
    float mean = 0.0f;
    for (int i = 0; i < dim; i++) mean += x[i];
    mean /= (float)dim;

    /* Compute variance */
    float var = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= (float)dim;

    float inv_std = 1.0f / sqrtf(var + eps);

    /* Normalize and scale */
    for (int i = 0; i < dim; i++) {
        out[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

/* ============================================================
 * MATRIX-VECTOR MULTIPLY
 *   out[M] = weight[M,K] * in[K] + bias[M]
 *
 * This is the single most-called function in inference (~90% of FLOPS).
 * Uses a tiled loop over K for better cache utilization.
 *
 * Future optimization slots:
 *   - AVX2/AVX-512 for x86-64
 *   - NEON/SVE for ARM64
 *   - RVV for RISC-V
 *   - CUDA/OpenCL for GPU
 * ============================================================ */

/* Tile size for K-dimension. 64 floats = 256 bytes = 4 cache lines.
 * Tune this for your target CPU's L1 cache size. */
#define LMC_MATMUL_BLOCK_K 64

void lmc_matmul_vec(float *out, const float *weight, const float *bias,
                    const float *in, int M, int K) {
    /* Initialize output: copy bias or zero */
    if (bias) {
        memcpy(out, bias, (size_t)M * sizeof(float));
    } else {
        memset(out, 0, (size_t)M * sizeof(float));
    }

    /* Tiled matrix-vector product */
    for (int kb = 0; kb < K; kb += LMC_MATMUL_BLOCK_K) {
        int k_end = kb + LMC_MATMUL_BLOCK_K;
        if (k_end > K) k_end = K;
        int k_len = k_end - kb;

#ifdef _OPENMP
        /* Parallelize output rows for large weight matrices */
        #pragma omp parallel for schedule(static) if(M >= 512)
#endif
        for (int m = 0; m < M; m++) {
            const float *w_row = weight + (size_t)m * K + kb;
            const float *x_blk = in + kb;
            float acc = 0.0f;

            /* 8-way unrolled inner loop for scalar throughput */
            int k = 0;
            for (; k <= k_len - 8; k += 8) {
                acc += w_row[k+0]*x_blk[k+0] + w_row[k+1]*x_blk[k+1]
                     + w_row[k+2]*x_blk[k+2] + w_row[k+3]*x_blk[k+3]
                     + w_row[k+4]*x_blk[k+4] + w_row[k+5]*x_blk[k+5]
                     + w_row[k+6]*x_blk[k+6] + w_row[k+7]*x_blk[k+7];
            }
            /* Handle tail */
            for (; k < k_len; k++) {
                acc += w_row[k] * x_blk[k];
            }

            out[m] += acc;
        }
    }
}
