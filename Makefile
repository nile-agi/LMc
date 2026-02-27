# =============================================================================
# LMc — Local Model Compute
# Edge-first AI inference engine
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
# Tunables (override on command line):
#   make CC=clang
#   make PREFIX=/opt/lmc
#   make EXTRA_CFLAGS="-march=armv8-a+simd"
# =============================================================================

# --- Compiler ---
CC      ?= gcc
PREFIX  ?= /usr/local
BINDIR  := $(PREFIX)/bin

# --- Directories ---
SRCDIR   := src
INCDIR   := include
BUILDDIR := build
BINARY   := lmc

# --- Source files ---
SRCS := \
    $(SRCDIR)/main.c \
    $(SRCDIR)/lmc.c \
    $(SRCDIR)/arena.c \
    $(SRCDIR)/math_ops.c \
    $(SRCDIR)/quantization.c \
    $(SRCDIR)/sampling.c \
    $(SRCDIR)/tokenizer.c \
    $(SRCDIR)/models/gpt2_weights.c \
    $(SRCDIR)/models/gpt2_inference.c \
    $(SRCDIR)/backends/bin_loader.c \
    $(SRCDIR)/backends/gguf_loader.c

OBJS := $(patsubst $(SRCDIR)/%.c, $(BUILDDIR)/%.o, $(SRCS))

# --- Base flags ---
CFLAGS_BASE := -std=c99 -I$(INCDIR) -Wall -Wextra \
               -Wno-unused-parameter \
               -D_POSIX_C_SOURCE=200809L

# --- Optimized build (default) ---
# Detect host CPU for march=native; fall back to safe generic on cross-compile
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),x86_64)
    MARCH := -march=native
else ifeq ($(UNAME_M),aarch64)
    MARCH := -march=native
else ifeq ($(UNAME_M),armv7l)
    MARCH := -march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard
else
    MARCH :=
endif

CFLAGS_OPT  := $(CFLAGS_BASE) -O3 $(MARCH) -ffast-math -funroll-loops \
               -fomit-frame-pointer $(EXTRA_CFLAGS)

# --- Debug build ---
CFLAGS_DBG  := $(CFLAGS_BASE) -O0 -g3 -DLMC_DEBUG \
               -fsanitize=address,undefined \
               -fno-omit-frame-pointer $(EXTRA_CFLAGS)

# --- OpenMP build ---
CFLAGS_OMP  := $(CFLAGS_OPT) -fopenmp
LDFLAGS_OMP := -fopenmp

# --- Linker flags ---
LDFLAGS     := -lm

# =============================================================================
# DEFAULT TARGET: optimized build
# =============================================================================
.PHONY: all
all: $(BUILDDIR)/$(BINARY)
	@echo ""
	@echo "  ✓ Built: $(BUILDDIR)/$(BINARY)"
	@echo "  Run:     ./$(BUILDDIR)/$(BINARY) \"Your prompt here\""
	@echo ""

$(BUILDDIR)/$(BINARY): $(OBJS)
	$(CC) $(CFLAGS_OPT) $^ -o $@ $(LDFLAGS)

# Compile each source file
$(BUILDDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS_OPT) -c $< -o $@

# =============================================================================
# DEBUG BUILD
# =============================================================================
.PHONY: debug
debug: CFLAGS_ACTIVE := $(CFLAGS_DBG)
debug: $(BUILDDIR)/$(BINARY)_debug
	@echo "  ✓ Debug build: $(BUILDDIR)/$(BINARY)_debug"

OBJS_DBG := $(patsubst $(SRCDIR)/%.c, $(BUILDDIR)/debug/%.o, $(SRCS))

$(BUILDDIR)/$(BINARY)_debug: $(OBJS_DBG)
	$(CC) $(CFLAGS_DBG) $^ -o $@ $(LDFLAGS)

$(BUILDDIR)/debug/%.o: $(SRCDIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS_DBG) -c $< -o $@

# =============================================================================
# OPENMP MULTI-THREADED BUILD
# =============================================================================
.PHONY: openmp
openmp: CFLAGS_ACTIVE := $(CFLAGS_OMP)
openmp: $(BUILDDIR)/$(BINARY)_omp
	@echo "  ✓ OpenMP build: $(BUILDDIR)/$(BINARY)_omp"

OBJS_OMP := $(patsubst $(SRCDIR)/%.c, $(BUILDDIR)/omp/%.o, $(SRCS))

$(BUILDDIR)/$(BINARY)_omp: $(OBJS_OMP)
	$(CC) $(CFLAGS_OMP) $^ -o $@ $(LDFLAGS) $(LDFLAGS_OMP)

$(BUILDDIR)/omp/%.o: $(SRCDIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS_OMP) -c $< -o $@

# =============================================================================
# INSTALL
# =============================================================================
.PHONY: install
install: all
	install -d $(BINDIR)
	install -m 755 $(BUILDDIR)/$(BINARY) $(BINDIR)/$(BINARY)
	@echo "  ✓ Installed to $(BINDIR)/$(BINARY)"

.PHONY: uninstall
uninstall:
	rm -f $(BINDIR)/$(BINARY)
	@echo "  ✓ Removed $(BINDIR)/$(BINARY)"

# =============================================================================
# TESTS
# =============================================================================
.PHONY: test
test: all
	@echo "Running smoke tests..."
	@if [ -f gpt2_124m.bin ] || [ -f gpt2.f16.gguf ] || [ -f gpt2.Q8_0.gguf ]; then \
	    ./$(BUILDDIR)/$(BINARY) "Hello" --tokens 5 --seed 42 --quiet && \
	    echo "  ✓ Generation smoke test passed" ; \
	else \
	    echo "  ! No model file found — skipping generation test" ; \
	    echo "    Run: python3 scripts/converter.py  to get gpt2_124m.bin" ; \
	fi

# =============================================================================
# CLEAN
# =============================================================================
.PHONY: clean
clean:
	rm -rf $(BUILDDIR)
	@echo "  ✓ Cleaned"

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
	@echo "  make clean        Remove build/  directory"
	@echo "  make install      Install to PREFIX=$(PREFIX)"
	@echo "  make uninstall    Remove installed binary"
	@echo "  make test         Run smoke tests"
	@echo ""
	@echo "Variables:"
	@echo "  CC=gcc|clang      Compiler (default: gcc)"
	@echo "  PREFIX=/path      Install prefix (default: /usr/local)"
	@echo "  EXTRA_CFLAGS=...  Additional compiler flags"
	@echo ""
	@echo "Examples:"
	@echo "  make CC=clang"
	@echo "  make openmp"
	@echo "  make EXTRA_CFLAGS=\"-march=armv8-a+simd\""
	@echo "  make install PREFIX=\$$HOME/.local"
	@echo ""
