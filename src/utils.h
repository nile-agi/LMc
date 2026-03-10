/* utils.h — Arena allocator, logging macros, and shared C99 utilities.
 *
 * Design goals:
 *   • Zero external dependencies beyond the C standard library.
 *   • Single contiguous malloc slab for all model weights (the arena),
 *     avoiding thousands of small allocations and their associated overhead.
 *   • Logging macros that include level tags and flush to the right stream,
 *     with FATAL immediately calling exit(1).
 *
 * All declarations here are pure C99; no GCC/Clang extensions are required
 * (though ALIGN16 uses __attribute__((aligned)) when available).
 */
#ifndef LMC_UTILS_H
#define LMC_UTILS_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Portability helpers ──────────────────────────────────────────────────── */

/* Silence "unused parameter" warnings without removing the parameter name. */
#define UNUSED(x) ((void)(x))

/* Branch prediction hints (no-ops on compilers that don't support them).   */
#ifdef __GNUC__
#  define LMC_LIKELY(x)    __builtin_expect(!!(x), 1)
#  define LMC_UNLIKELY(x)  __builtin_expect(!!(x), 0)
#else
#  define LMC_LIKELY(x)    (x)
#  define LMC_UNLIKELY(x)  (x)
#endif

/* 16-byte alignment annotation for data that benefits from SIMD loads.
 * Ignored silently on compilers that don't support __attribute__.          */
#ifdef __GNUC__
#  define ALIGN16 __attribute__((aligned(16)))
#else
#  define ALIGN16
#endif

/* ── Arithmetic utilities ─────────────────────────────────────────────────── */
#define LMC_MIN(a, b)  ((a) < (b) ? (a) : (b))
#define LMC_MAX(a, b)  ((a) > (b) ? (a) : (b))
#define LMC_CLAMP(x, lo, hi) LMC_MAX((lo), LMC_MIN((x), (hi)))

/* Round x up to the nearest multiple of align (must be a power of two).   */
#define LMC_ALIGN_UP(x, align) (((size_t)(x) + (align) - 1) & ~((align) - 1))

/* ── Logging ──────────────────────────────────────────────────────────────── *
 * INFO  → stdout (shown during normal operation)                             *
 * WARN  → stderr (non-fatal anomaly; execution continues)                    *
 * ERROR → stderr (recoverable error; caller decides whether to abort)        *
 * FATAL → stderr + exit(1) (unrecoverable; used for OOM, corrupt files)      *
 *                                                                             *
 * All macros append a newline; callers must not add one in the format string.*/
#define LMC_INFO(fmt,  ...) \
    fprintf(stdout, "[INFO]  " fmt "\n", ##__VA_ARGS__)

#define LMC_WARN(fmt,  ...) \
    fprintf(stderr, "[WARN]  " fmt "\n", ##__VA_ARGS__)

#define LMC_ERROR(fmt, ...) \
    fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)

#define LMC_FATAL(fmt, ...) \
    do { fprintf(stderr, "[FATAL] " fmt "\n", ##__VA_ARGS__); exit(1); } while (0)

/* ── Memory arena ─────────────────────────────────────────────────────────── *
 * A single large malloc() slab holds all model weight floats.               *
 *                                                                             *
 * Usage pattern:                                                              *
 *   arena_init(total_floats);          // one malloc                         *
 *   float *w = arena_alloc(n_floats);  // bump pointer; never fails          *
 *   …                                  // load weights into w                *
 *   arena_free();                      // one free at exit                   *
 *                                                                             *
 * arena_alloc() calls LMC_FATAL if the request would exceed capacity.        *
 * Individual slices cannot be freed — the whole slab is freed together.      *
 *                                                                             *
 * The arena stores float32 values.  Quantised weights are dequantised into   *
 * the arena at load time; quantised bytes are never retained past loading.   */
typedef struct {
    float  *data;       /* base pointer returned by malloc()                  */
    size_t  capacity;   /* total float32 slots allocated                      */
    size_t  used;       /* float32 slots handed out so far                    */
} Arena;

extern Arena g_arena;

/* Allocate the slab.  Calls LMC_FATAL on OOM.
 * n_floats should come from gpt2_total_params() / llama_total_params().    */
void   arena_init (size_t n_floats);

/* Bump-allocate n_floats from the slab and return a pointer to them.
 * The returned memory is zero-initialised (calloc semantics).
 * Calls LMC_FATAL if n_floats would exceed remaining capacity.             */
float *arena_alloc(size_t n_floats);

/* Release the entire arena slab.  Safe to call multiple times.             */
void   arena_free (void);

#endif /* LMC_UTILS_H */
