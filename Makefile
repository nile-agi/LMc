# =============================================================================
# LMc — Local Model Compute
# Edge-first AI inference engine
#
# Compatible with: GNU make, BSD make (macOS), and all standard make versions.
# Uses explicit per-file compile rules instead of wildcard pattern rules to
# ensure maximum portability across make implementations.
#
# Targets:
#   make              Build optimized binary (auto-detects CPU)
#   make debug        Debug build with sanitizers
#   make openmp       Multi-threaded build (requires OpenMP)
#   make clean        Remove build artifacts
#   make install      Install to PREFIX (default: /usr/local)
#   make test         Run basic smoke tests
#   make help         Show all targets
#
# Tunables:
#   make CC=clang
#   make PREFIX=/opt/lmc
#   make EXTRA_CFLAGS="-march=armv8-a+simd"
# =============================================================================

CC      ?= cc
PREFIX  ?= /usr/local
BINDIR  := $(PREFIX)/bin
BINARY  := lmc
B       := build

# --- Detect CPU architecture ---
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),x86_64)
    MARCH := -march=native
else ifeq ($(UNAME_M),aarch64)
    MARCH := -march=native
else ifeq ($(UNAME_M),arm64)
    MARCH := -march=native
else ifeq ($(UNAME_M),armv7l)
    MARCH := -march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard
else
    MARCH :=
endif

# --- Compiler flags ---
CFLAGS_BASE := -std=c99 -Iinclude -Wall -Wextra -Wno-unused-parameter \
               -D_POSIX_C_SOURCE=200809L

CFLAGS_OPT  := $(CFLAGS_BASE) -O3 $(MARCH) -ffast-math -funroll-loops \
               -fomit-frame-pointer $(EXTRA_CFLAGS)

CFLAGS_DBG  := $(CFLAGS_BASE) -O0 -g3 -DLMC_DEBUG \
               -fsanitize=address,undefined \
               -fno-omit-frame-pointer $(EXTRA_CFLAGS)

CFLAGS_OMP  := $(CFLAGS_OPT) -fopenmp

LDFLAGS     := -lm
LDFLAGS_OMP := -lm -fopenmp

# =============================================================================
# SOURCE → OBJECT LISTS  (explicit, portable across all make versions)
# =============================================================================

OBJS_OPT := \
	$(B)/main.o \
	$(B)/lmc.o \
	$(B)/arena.o \
	$(B)/math_ops.o \
	$(B)/quantization.o \
	$(B)/sampling.o \
	$(B)/tokenizer.o \
	$(B)/models/gpt2_weights.o \
	$(B)/models/gpt2_inference.o \
	$(B)/backends/bin_loader.o \
	$(B)/backends/gguf_loader.o

OBJS_DBG := \
	$(B)/debug/main.o \
	$(B)/debug/lmc.o \
	$(B)/debug/arena.o \
	$(B)/debug/math_ops.o \
	$(B)/debug/quantization.o \
	$(B)/debug/sampling.o \
	$(B)/debug/tokenizer.o \
	$(B)/debug/models/gpt2_weights.o \
	$(B)/debug/models/gpt2_inference.o \
	$(B)/debug/backends/bin_loader.o \
	$(B)/debug/backends/gguf_loader.o

OBJS_OMP := \
	$(B)/omp/main.o \
	$(B)/omp/lmc.o \
	$(B)/omp/arena.o \
	$(B)/omp/math_ops.o \
	$(B)/omp/quantization.o \
	$(B)/omp/sampling.o \
	$(B)/omp/tokenizer.o \
	$(B)/omp/models/gpt2_weights.o \
	$(B)/omp/models/gpt2_inference.o \
	$(B)/omp/backends/bin_loader.o \
	$(B)/omp/backends/gguf_loader.o

# =============================================================================
# DEFAULT TARGET
# =============================================================================
.PHONY: all
all: $(B)/$(BINARY)
	@echo ""
	@echo "  Built:  $(B)/$(BINARY)"
	@echo "  Usage:  ./$(B)/$(BINARY) \"Your prompt here\""
	@echo ""

$(B)/$(BINARY): $(OBJS_OPT)
	$(CC) $(CFLAGS_OPT) -o $@ $^ $(LDFLAGS)

# --- Optimized compile rules ---
$(B)/main.o: src/main.c include/lmc.h include/lmc_internal.h
	@mkdir -p $(B)
	$(CC) $(CFLAGS_OPT) -c src/main.c -o $@

$(B)/lmc.o: src/lmc.c include/lmc.h include/lmc_internal.h
	@mkdir -p $(B)
	$(CC) $(CFLAGS_OPT) -c src/lmc.c -o $@

$(B)/arena.o: src/arena.c include/lmc_internal.h
	@mkdir -p $(B)
	$(CC) $(CFLAGS_OPT) -c src/arena.c -o $@

$(B)/math_ops.o: src/math_ops.c include/lmc_internal.h
	@mkdir -p $(B)
	$(CC) $(CFLAGS_OPT) -c src/math_ops.c -o $@

$(B)/quantization.o: src/quantization.c include/lmc_internal.h
	@mkdir -p $(B)
	$(CC) $(CFLAGS_OPT) -c src/quantization.c -o $@

$(B)/sampling.o: src/sampling.c include/lmc_internal.h
	@mkdir -p $(B)
	$(CC) $(CFLAGS_OPT) -c src/sampling.c -o $@

$(B)/tokenizer.o: src/tokenizer.c include/lmc_internal.h
	@mkdir -p $(B)
	$(CC) $(CFLAGS_OPT) -c src/tokenizer.c -o $@

$(B)/models/gpt2_weights.o: src/models/gpt2_weights.c include/lmc_internal.h
	@mkdir -p $(B)/models
	$(CC) $(CFLAGS_OPT) -c src/models/gpt2_weights.c -o $@

$(B)/models/gpt2_inference.o: src/models/gpt2_inference.c include/lmc_internal.h
	@mkdir -p $(B)/models
	$(CC) $(CFLAGS_OPT) -c src/models/gpt2_inference.c -o $@

$(B)/backends/bin_loader.o: src/backends/bin_loader.c include/lmc_internal.h
	@mkdir -p $(B)/backends
	$(CC) $(CFLAGS_OPT) -c src/backends/bin_loader.c -o $@

$(B)/backends/gguf_loader.o: src/backends/gguf_loader.c include/lmc_internal.h
	@mkdir -p $(B)/backends
	$(CC) $(CFLAGS_OPT) -c src/backends/gguf_loader.c -o $@

# =============================================================================
# DEBUG BUILD
# =============================================================================
.PHONY: debug
debug: $(B)/$(BINARY)_debug
	@echo "  Debug build: $(B)/$(BINARY)_debug"

$(B)/$(BINARY)_debug: $(OBJS_DBG)
	$(CC) $(CFLAGS_DBG) -o $@ $^ $(LDFLAGS)

$(B)/debug/main.o: src/main.c include/lmc.h include/lmc_internal.h
	@mkdir -p $(B)/debug
	$(CC) $(CFLAGS_DBG) -c src/main.c -o $@

$(B)/debug/lmc.o: src/lmc.c include/lmc.h include/lmc_internal.h
	@mkdir -p $(B)/debug
	$(CC) $(CFLAGS_DBG) -c src/lmc.c -o $@

$(B)/debug/arena.o: src/arena.c include/lmc_internal.h
	@mkdir -p $(B)/debug
	$(CC) $(CFLAGS_DBG) -c src/arena.c -o $@

$(B)/debug/math_ops.o: src/math_ops.c include/lmc_internal.h
	@mkdir -p $(B)/debug
	$(CC) $(CFLAGS_DBG) -c src/math_ops.c -o $@

$(B)/debug/quantization.o: src/quantization.c include/lmc_internal.h
	@mkdir -p $(B)/debug
	$(CC) $(CFLAGS_DBG) -c src/quantization.c -o $@

$(B)/debug/sampling.o: src/sampling.c include/lmc_internal.h
	@mkdir -p $(B)/debug
	$(CC) $(CFLAGS_DBG) -c src/sampling.c -o $@

$(B)/debug/tokenizer.o: src/tokenizer.c include/lmc_internal.h
	@mkdir -p $(B)/debug
	$(CC) $(CFLAGS_DBG) -c src/tokenizer.c -o $@

$(B)/debug/models/gpt2_weights.o: src/models/gpt2_weights.c include/lmc_internal.h
	@mkdir -p $(B)/debug/models
	$(CC) $(CFLAGS_DBG) -c src/models/gpt2_weights.c -o $@

$(B)/debug/models/gpt2_inference.o: src/models/gpt2_inference.c include/lmc_internal.h
	@mkdir -p $(B)/debug/models
	$(CC) $(CFLAGS_DBG) -c src/models/gpt2_inference.c -o $@

$(B)/debug/backends/bin_loader.o: src/backends/bin_loader.c include/lmc_internal.h
	@mkdir -p $(B)/debug/backends
	$(CC) $(CFLAGS_DBG) -c src/backends/bin_loader.c -o $@

$(B)/debug/backends/gguf_loader.o: src/backends/gguf_loader.c include/lmc_internal.h
	@mkdir -p $(B)/debug/backends
	$(CC) $(CFLAGS_DBG) -c src/backends/gguf_loader.c -o $@

# =============================================================================
# OPENMP MULTI-THREADED BUILD
# =============================================================================
.PHONY: openmp
openmp: $(B)/$(BINARY)_omp
	@echo "  OpenMP build: $(B)/$(BINARY)_omp"

$(B)/$(BINARY)_omp: $(OBJS_OMP)
	$(CC) $(CFLAGS_OMP) -o $@ $^ $(LDFLAGS_OMP)

$(B)/omp/main.o: src/main.c include/lmc.h include/lmc_internal.h
	@mkdir -p $(B)/omp
	$(CC) $(CFLAGS_OMP) -c src/main.c -o $@

$(B)/omp/lmc.o: src/lmc.c include/lmc.h include/lmc_internal.h
	@mkdir -p $(B)/omp
	$(CC) $(CFLAGS_OMP) -c src/lmc.c -o $@

$(B)/omp/arena.o: src/arena.c include/lmc_internal.h
	@mkdir -p $(B)/omp
	$(CC) $(CFLAGS_OMP) -c src/arena.c -o $@

$(B)/omp/math_ops.o: src/math_ops.c include/lmc_internal.h
	@mkdir -p $(B)/omp
	$(CC) $(CFLAGS_OMP) -c src/math_ops.c -o $@

$(B)/omp/quantization.o: src/quantization.c include/lmc_internal.h
	@mkdir -p $(B)/omp
	$(CC) $(CFLAGS_OMP) -c src/quantization.c -o $@

$(B)/omp/sampling.o: src/sampling.c include/lmc_internal.h
	@mkdir -p $(B)/omp
	$(CC) $(CFLAGS_OMP) -c src/sampling.c -o $@

$(B)/omp/tokenizer.o: src/tokenizer.c include/lmc_internal.h
	@mkdir -p $(B)/omp
	$(CC) $(CFLAGS_OMP) -c src/tokenizer.c -o $@

$(B)/omp/models/gpt2_weights.o: src/models/gpt2_weights.c include/lmc_internal.h
	@mkdir -p $(B)/omp/models
	$(CC) $(CFLAGS_OMP) -c src/models/gpt2_weights.c -o $@

$(B)/omp/models/gpt2_inference.o: src/models/gpt2_inference.c include/lmc_internal.h
	@mkdir -p $(B)/omp/models
	$(CC) $(CFLAGS_OMP) -c src/models/gpt2_inference.c -o $@

$(B)/omp/backends/bin_loader.o: src/backends/bin_loader.c include/lmc_internal.h
	@mkdir -p $(B)/omp/backends
	$(CC) $(CFLAGS_OMP) -c src/backends/bin_loader.c -o $@

$(B)/omp/backends/gguf_loader.o: src/backends/gguf_loader.c include/lmc_internal.h
	@mkdir -p $(B)/omp/backends
	$(CC) $(CFLAGS_OMP) -c src/backends/gguf_loader.c -o $@

# =============================================================================
# INSTALL / UNINSTALL
# =============================================================================
.PHONY: install
install: all
	install -d $(BINDIR)
	install -m 755 $(B)/$(BINARY) $(BINDIR)/$(BINARY)
	@echo "  Installed to $(BINDIR)/$(BINARY)"

.PHONY: uninstall
uninstall:
	rm -f $(BINDIR)/$(BINARY)
	@echo "  Removed $(BINDIR)/$(BINARY)"

# =============================================================================
# TESTS
# =============================================================================
.PHONY: test
test: all
	@echo "Running smoke tests..."
	@if [ -f gpt2_124m.bin ] || [ -f gpt2.f16.gguf ] || [ -f gpt2.Q8_0.gguf ]; then \
	    ./$(B)/$(BINARY) "Hello" --tokens 5 --seed 42 --quiet && \
	    echo "  Smoke test passed" ; \
	else \
	    echo "  No model file found — skipping generation test" ; \
	    echo "  Run: python3 scripts/converter.py  to get gpt2_124m.bin" ; \
	fi

# =============================================================================
# CLEAN
# =============================================================================
.PHONY: clean
clean:
	rm -rf $(B)
	@echo "  Cleaned"

# =============================================================================
# HELP
# =============================================================================
.PHONY: help
help:
	@echo ""
	@echo "LMc — Local Model Compute"
	@echo ""
	@echo "Build targets:"
	@echo "  make              Optimized build (auto-detect CPU arch)"
	@echo "  make debug        Debug build with ASan/UBSan"
	@echo "  make openmp       Multi-threaded build (OpenMP)"
	@echo "  make clean        Remove build/ directory"
	@echo "  make install      Install to PREFIX=$(PREFIX)"
	@echo "  make uninstall    Remove installed binary"
	@echo "  make test         Run smoke tests"
	@echo ""
	@echo "Variables:"
	@echo "  CC=gcc|clang      Compiler  (default: cc)"
	@echo "  PREFIX=/path      Install prefix (default: /usr/local)"
	@echo "  EXTRA_CFLAGS=...  Additional compiler flags"
	@echo ""
	@echo "Examples:"
	@echo "  make CC=clang"
	@echo "  make openmp"
	@echo "  make EXTRA_CFLAGS=\"-march=armv8-a+simd\""
	@echo "  make install PREFIX=\$$HOME/.local"
	@echo ""