/* quant.h — GGUF quantisation type IDs, block geometry constants,
 *            float16 conversion, and dequantisation API.
 *
 * Every supported quant type has three entries in this file:
 *   1. GGUF_TYPE_*          — integer type ID as encoded in the GGUF tensor header.
 *   2. *_BLOCK_SIZE         — number of weights per quantisation block.
 *   3. *_BYTES_PER_BLOCK    — serialised byte width of one block.
 *
 * The dequantisation functions share a uniform signature:
 *   void dequant_*(const uint8_t *src, float *dst, size_t n_elements);
 *
 * To add a new quantisation type:
 *   1. Add GGUF_TYPE_*, BLOCK_SIZE, and BYTES_PER_BLOCK constants below.
 *   2. Implement dequant_<name>(src, dst, n_elements) in quant.c.
 *   3. Add a LOAD_QUANT() case in gguf.c's tensor loading switch.
 *   See docs/adding_new_quant.md for a walkthrough.
 */
#ifndef LMC_QUANT_H
#define LMC_QUANT_H

#include <stddef.h>
#include <stdint.h>

/* ── GGUF tensor type IDs ─────────────────────────────────────────────────── *
 * Values match the gguf_type enum in the GGUF spec v3.                       *
 * Non-contiguous numbering reflects the historical growth of the format.     */
#define GGUF_TYPE_F32        0
#define GGUF_TYPE_F16        1
#define GGUF_TYPE_Q4_0       2
#define GGUF_TYPE_Q4_1       3
/* 4 and 5 are legacy / unused */
#define GGUF_TYPE_Q5_0       6
#define GGUF_TYPE_Q5_1       7
#define GGUF_TYPE_Q8_0       8
/* 9 = Q8_1 (not implemented) */
#define GGUF_TYPE_Q2_K       10
#define GGUF_TYPE_Q3_K       11
#define GGUF_TYPE_Q4_K       12
#define GGUF_TYPE_Q5_K       13
#define GGUF_TYPE_Q6_K       14
/* 15–17 unused */
#define GGUF_TYPE_IQ3_XXS    18
/* 19–20 unused */
#define GGUF_TYPE_IQ3_S      21
/* 22 = IQ3_M (not implemented) */
#define GGUF_TYPE_IQ4_XS     23

/* ── Block geometry ───────────────────────────────────────────────────────── *
 * BLOCK_SIZE      = number of float32 weights encoded per block.             *
 * BYTES_PER_BLOCK = serialised size of one block in bytes.                   *
 *                                                                             *
 * n_elements passed to dequant_*() must be a multiple of BLOCK_SIZE.        */

/* --- Legacy fixed-point (per-block scalar) --- */
#define Q4_0_BLOCK_SIZE          32
#define Q4_0_BYTES_PER_BLOCK     18    /* fp16 scale (2) + 4-bit nibbles (16) */

#define Q4_1_BLOCK_SIZE          32
#define Q4_1_BYTES_PER_BLOCK     20    /* fp16 scale (2) + fp16 min (2) + nibbles (16) */

#define Q5_0_BLOCK_SIZE          32
#define Q5_0_BYTES_PER_BLOCK     22    /* fp16 scale (2) + 5th-bit mask (4) + nibbles (16) */

#define Q5_1_BLOCK_SIZE          32
#define Q5_1_BYTES_PER_BLOCK     24    /* fp16 scale (2) + fp16 min (2) + mask (4) + nibbles (16) */

#define Q8_0_BLOCK_SIZE          32
#define Q8_0_BYTES_PER_BLOCK     34    /* fp16 scale (2) + int8 values (32) */

/* --- K-quants (super-block with nested sub-block scales) --- */
#define Q2_K_BLOCK_SIZE         256
#define Q2_K_BYTES_PER_BLOCK     84    /* scales (16) + mins (16) + 2-bit qs (64) + fp16 d/dmin (4) */

#define Q3_K_BLOCK_SIZE         256
#define Q3_K_BYTES_PER_BLOCK    110    /* hmask (32) + 2-bit qs (64) + scales (12) + fp16 d (2) */

#define Q4_K_BLOCK_SIZE         256
#define Q4_K_BYTES_PER_BLOCK    144    /* fp16 d (2) + fp16 dmin (2) + scales (12) + 4-bit qs (128) */

#define Q5_K_BLOCK_SIZE         256
#define Q5_K_BYTES_PER_BLOCK    176    /* fp16 d (2) + fp16 dmin (2) + scales (12) + hmask (32) + nibbles (128) */

#define Q6_K_BLOCK_SIZE         256
#define Q6_K_BYTES_PER_BLOCK    210    /* 2-bit high (64) + 4-bit low (128) + int8 scales (16) + fp16 d (2) */
/* Sub-field sizes within a Q6_K block (used in quant.c): */
#define Q6_K_QL_BYTES           128    /* lower 4 bits of each weight          */
#define Q6_K_QH_BYTES            64    /* upper 2 bits of each weight          */
#define Q6_K_SC_BYTES            16    /* int8 sub-block scales                */
#define Q6_K_D_BYTES              2    /* fp16 super-block scale               */

/* --- i-quants (importance-matrix quantisation) --- */
#define IQ3_XXS_BLOCK_SIZE      256
#define IQ3_XXS_BYTES_PER_BLOCK  98    /* fp16 d (2) + 3-bit packed qs (64) + 32-bit aux (32) */

#define IQ3_S_BLOCK_SIZE        256
#define IQ3_S_BYTES_PER_BLOCK   110    /* fp16 d (2) + 3-bit qs (96) + signs (32) + ... */

#define IQ4_XS_BLOCK_SIZE       256
#define IQ4_XS_BYTES_PER_BLOCK  136    /* fp16 d (2) + sub-scale (4) + 4-bit qs (128) + pad */

/* ── Float16 → Float32 ────────────────────────────────────────────────────── *
 * IEEE 754 half-precision conversion via bit manipulation.  Used by all      *
 * dequant kernels that store block scales in fp16.                           */
float f16_to_f32(uint16_t h);

/* ── Dequantisation functions ─────────────────────────────────────────────── *
 * All functions:                                                              *
 *   src         — pointer to raw quantised bytes for this tensor             *
 *   dst         — output float32 array (caller-allocated, n_elements floats) *
 *   n_elements  — total number of weights to decode; must be a multiple      *
 *                 of the corresponding *_BLOCK_SIZE                          */

/* Legacy fixed-point */
void dequant_q4_0   (const uint8_t *src, float *dst, size_t n_elements);
void dequant_q4_1   (const uint8_t *src, float *dst, size_t n_elements);
void dequant_q5_0   (const uint8_t *src, float *dst, size_t n_elements);
void dequant_q5_1   (const uint8_t *src, float *dst, size_t n_elements);
void dequant_q8_0   (const uint8_t *src, float *dst, size_t n_elements);

/* K-quants */
void dequant_q2k    (const uint8_t *src, float *dst, size_t n_elements);
void dequant_q3k    (const uint8_t *src, float *dst, size_t n_elements);
void dequant_q4k    (const uint8_t *src, float *dst, size_t n_elements);
void dequant_q5k    (const uint8_t *src, float *dst, size_t n_elements);
void dequant_q6k    (const uint8_t *src, float *dst, size_t n_elements);

/* I-quants */
void dequant_iq3xxs (const uint8_t *src, float *dst, size_t n_elements);
void dequant_iq3s   (const uint8_t *src, float *dst, size_t n_elements);
void dequant_iq4_xs (const uint8_t *src, float *dst, size_t n_elements);

#endif /* LMC_QUANT_H */
