/*
 * src/quantization.c — LMc Quantization Dequantization Routines
 *
 * Converts quantized weights stored on disk into float32 for inference.
 * All formats are compatible with llama.cpp / GGUF specification.
 *
 * Supported formats:
 *   F16   — IEEE 754 half-precision float
 *   Q8_0  — 8-bit symmetric quantization, block size 32
 *   Q5_K  — 5-bit K-quant (S and M variants), super-block size 256
 *   Q6_K  — 6-bit K-quant, super-block size 256
 *
 * Adding a new quantization format:
 *   1. Define BLOCK_SIZE and BYTES_PER_BLOCK constants
 *   2. Implement lmc_dequant_<name>(src, dst, n_elements)
 *   3. Add a case in backends/gguf_loader.c load loop
 *   4. Add the LmcQuantType enum value in lmc.h
 *
 * SPDX-License-Identifier: MIT
 */

#include "lmc_internal.h"

/* ============================================================
 * BLOCK SIZE CONSTANTS (must match GGUF spec)
 * ============================================================ */
#define Q8_0_BLOCK_SIZE       32
#define Q8_0_BYTES_PER_BLOCK  34   /* 2 (f16 scale) + 32 (int8 values)  */

#define Q5_K_BLOCK_SIZE      256
#define Q5_K_BYTES_PER_BLOCK 176   /* 2+2+12+32+128                      */

#define Q6_K_BLOCK_SIZE      256
#define Q6_K_QL_BYTES        128   /* lower 4 bits, 2/byte               */
#define Q6_K_QH_BYTES         64   /* upper 2 bits, 4/byte               */
#define Q6_K_SC_BYTES         16   /* int8 sub-block scales              */
#define Q6_K_BYTES_PER_BLOCK 210   /* 128+64+16+2                        */

/* ============================================================
 * F16 DEQUANTIZATION
 * Simply converts each FP16 element to FP32.
 * ============================================================ */
void lmc_dequant_f16(const uint8_t *src, float *dst, size_t n) {
    const uint16_t *s = (const uint16_t*)src;
    for (size_t i = 0; i < n; i++) {
        dst[i] = lmc_f16_to_f32(s[i]);
    }
}

/* ============================================================
 * Q8_0 DEQUANTIZATION
 *
 * Block layout (34 bytes for 32 elements):
 *   [uint16 scale (f16)][int8 q[32]]
 *
 * Dequantize: x[i] = q[i] * f16_to_f32(scale)
 * ============================================================ */
void lmc_dequant_q8_0(const uint8_t *src, float *dst, size_t n_elements) {
    size_t n_blocks = n_elements / Q8_0_BLOCK_SIZE;

    for (size_t b = 0; b < n_blocks; b++) {
        /* Read f16 scale */
        uint16_t d_raw;
        memcpy(&d_raw, src, sizeof(uint16_t));
        float scale = lmc_f16_to_f32(d_raw);
        src += 2;

        /* Dequantize 32 int8 values */
        const int8_t *qs  = (const int8_t*)src;
        float        *out = dst + b * Q8_0_BLOCK_SIZE;
        for (int i = 0; i < Q8_0_BLOCK_SIZE; i++) {
            out[i] = (float)qs[i] * scale;
        }
        src += Q8_0_BLOCK_SIZE;
    }
}

/* ============================================================
 * Q5_K DEQUANTIZATION (S and M variants)
 *
 * Q5_K_S and Q5_K_M share the SAME binary format (GGUF type 13).
 * The S/M distinction is a quantization-time choice only.
 *
 * Block layout (176 bytes for 256 elements):
 *   offset  0: d      [2]  f16  quant scale
 *   offset  2: dmin   [2]  f16  min scale
 *   offset  4: scales [12] 6-bit packed: 8 scale values + 8 min values
 *   offset 16: qh     [32] high bit (bit 4) of each 5-bit quant, 8/byte
 *   offset 48: qs    [128] low nibble (bits 0-3), packed 2/byte
 *
 * The qs/qh packing is interleaved — see comments in loop below.
 *
 * Verified against llama.cpp ggml-quants.c dequantize_row_q5_K().
 * ============================================================ */

/*
 * Helper: decode one (scale, min) pair from the 12-byte packed field.
 *
 * The field stores 8 scale values and 8 min values in 6 bits each (96 bits total).
 * Bit layout differs for indices 0-3 vs 4-7.
 */
static void q5k_get_scale_min(int j, const uint8_t *scales,
                               uint8_t *out_sc, uint8_t *out_m) {
    if (j < 4) {
        *out_sc = scales[j]   & 0x3F;
        *out_m  = scales[j+4] & 0x3F;
    } else {
        *out_sc = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4);
        *out_m  = (scales[j+4] >>  4)  | ((scales[j  ] >> 6) << 4);
    }
}

void lmc_dequant_q5k(const uint8_t *src, float *dst, size_t n_elements) {
    size_t n_blocks = n_elements / Q5_K_BLOCK_SIZE;

    for (size_t b = 0; b < n_blocks; b++) {
        /* Decode block header */
        uint16_t d_raw, dmin_raw;
        memcpy(&d_raw,    src + 0, sizeof(uint16_t));
        memcpy(&dmin_raw, src + 2, sizeof(uint16_t));
        float d    = lmc_f16_to_f32(d_raw);
        float dmin = lmc_f16_to_f32(dmin_raw);

        const uint8_t *scales = src + 4;   /* [12] packed 6-bit scales+mins */
        const uint8_t *qh     = src + 16;  /* [32] high bit of each element  */
        const uint8_t *qs     = src + 48;  /* [128] low nibbles              */
        float         *y      = dst + b * Q5_K_BLOCK_SIZE;

        /*
         * Four outer iterations (j = 0, 64, 128, 192):
         *   ql_base = j % 128   → 0, 64, 0, 64
         *   nibble  = j >= 128  → 0, 0, 4, 4  (lo or hi nibble of qs byte)
         *   shift   = j >> 5    → 0, 2, 4, 6  (which bit of qh byte)
         * Each outer iteration covers two groups of 32 elements.
         */
        int is_  = 0;   /* scale group index  */
        int y_off = 0;  /* output element idx */

        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            q5k_get_scale_min(is_++, scales, &sc, &m);
            float d1 = d * (float)sc,  m1 = dmin * (float)m;
            q5k_get_scale_min(is_++, scales, &sc, &m);
            float d2 = d * (float)sc,  m2 = dmin * (float)m;

            int ql_base = j & 127;
            int nibble  = (j >= 128) ? 4 : 0;
            int shift   = j >> 5;

            /* First group of 32 */
            for (int l = 0; l < 32; l++) {
                int hi = (qh[l] >> shift) & 1;
                int lo = (qs[ql_base + l] >> nibble) & 0x0F;
                y[y_off + l] = d1 * (float)(lo | (hi << 4)) - m1;
            }
            y_off += 32;

            /* Second group of 32 */
            for (int l = 0; l < 32; l++) {
                int hi = (qh[l] >> (shift + 1)) & 1;
                int lo = (qs[ql_base + l + 32] >> nibble) & 0x0F;
                y[y_off + l] = d2 * (float)(lo | (hi << 4)) - m2;
            }
            y_off += 32;
        }

        src += Q5_K_BYTES_PER_BLOCK;
    }
}

/* ============================================================
 * Q6_K DEQUANTIZATION
 *
 * Block layout (210 bytes for 256 elements):
 *   ql    [128] lower 4 bits of each quant, packed 2/byte
 *   qh    [64]  upper 2 bits of each quant, packed 4/byte
 *   scales[16]  int8 sub-block scales, one per 16 elements
 *   d     [2]   f16 super-block scale
 *
 * The bit layout uses an interleaved scheme (not sequential):
 * each inner iteration l=0..31 writes four outputs at stride 32.
 *
 * Verified against llama.cpp ggml-quants.c dequantize_row_q6_K().
 * ============================================================ */
void lmc_dequant_q6k(const uint8_t *src, float *dst, size_t n_elements) {
    size_t n_blocks = n_elements / Q6_K_BLOCK_SIZE;

    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *ql_base = src;
        const uint8_t *qh_base = src + Q6_K_QL_BYTES;
        const int8_t  *sc_base = (const int8_t*)(src + Q6_K_QL_BYTES + Q6_K_QH_BYTES);
        uint16_t d_raw;
        memcpy(&d_raw,
               src + Q6_K_QL_BYTES + Q6_K_QH_BYTES + Q6_K_SC_BYTES,
               sizeof(uint16_t));
        float d = lmc_f16_to_f32(d_raw);

        float *y = dst + b * Q6_K_BLOCK_SIZE;

        /*
         * Two passes: pass 0 → elements [0..127]
         *             pass 1 → elements [128..255]
         * Each pass: ql advances 64 bytes, qh advances 32 bytes, sc advances 8.
         */
        for (int pass = 0; pass < 2; pass++) {
            const uint8_t *ql = ql_base + pass * 64;
            const uint8_t *qh = qh_base + pass * 32;
            const int8_t  *sc = sc_base + pass * 8;
            float         *yp = y + pass * 128;

            /*
             * Each l=0..31 writes four positions at stride 32:
             *   q1 → yp[l +  0]   q2 → yp[l + 32]
             *   q3 → yp[l + 64]   q4 → yp[l + 96]
             *
             * Scale index is = l >> 4 (0 for l=0..15, 1 for l=16..31)
             */
            for (int l = 0; l < 32; l++) {
                int is = l >> 4;

                int q1 = (int)((ql[l]      & 0x0F) | (((qh[l] >> 0) & 0x03) << 4)) - 32;
                int q2 = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 0x03) << 4)) - 32;
                int q3 = (int)((ql[l]      >>    4) | (((qh[l] >> 4) & 0x03) << 4)) - 32;
                int q4 = (int)((ql[l + 32] >>    4) | (((qh[l] >> 6) & 0x03) << 4)) - 32;

                yp[l +  0] = d * (float)sc[is + 0] * (float)q1;
                yp[l + 32] = d * (float)sc[is + 2] * (float)q2;
                yp[l + 64] = d * (float)sc[is + 4] * (float)q3;
                yp[l + 96] = d * (float)sc[is + 6] * (float)q4;
            }
        }

        src += Q6_K_BYTES_PER_BLOCK;
    }
}
