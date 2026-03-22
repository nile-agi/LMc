# ============================================================================
# lmc — Lightweight Model Engine
# Targets: Linux, macOS, Windows (MinGW/MSYS2)
# ============================================================================
CC      := gcc
SRCDIR  := src
SRCS    := $(SRCDIR)/utils.c \
           $(SRCDIR)/models.c \
           $(SRCDIR)/quant.c \
           $(SRCDIR)/gguf.c \
           $(SRCDIR)/ops.c \
           $(SRCDIR)/ops_llama.c \
           $(SRCDIR)/llama_tok.c \
           $(SRCDIR)/tokenizer.c \
           $(SRCDIR)/lmc.c
CFLAGS  := -std=c99 -O3 -march=native -ffast-math -funroll-loops \
           -Wall -Wextra -Wno-unused-parameter -I$(SRCDIR)
LIBS    := -lm
# ── Platform detection ────────────────────────────────────────────────────────
ifeq ($(OS), Windows_NT)
    TARGET      := lmc.exe
    TARGET_OMP  := lmc_omp.exe
    OMP_FLAGS   := -fopenmp
    OMP_LIBS    :=
    RM          := del /Q
else
    TARGET      := lmc
    TARGET_OMP  := lmc_omp
    RM          := rm -f
    UNAME := $(shell uname -s)
    ifeq ($(UNAME), Darwin)
        HOMEBREW := $(shell brew --prefix 2>/dev/null || echo /usr/local)
        LIBOMP   := $(HOMEBREW)/opt/libomp
        ifneq ($(wildcard $(LIBOMP)/lib/libomp.dylib),)
            OMP_FLAGS := -Xpreprocessor -fopenmp -I$(LIBOMP)/include
            OMP_LIBS  := -L$(LIBOMP)/lib -lomp
            CC_OMP    := $(CC)
        else
            LLVM     := $(HOMEBREW)/opt/llvm
            OMP_FLAGS := -Xpreprocessor -fopenmp
            OMP_LIBS  := -L$(LLVM)/lib -lomp
            CC_OMP    := $(LLVM)/bin/clang
        endif
    else
        OMP_FLAGS := -fopenmp
        OMP_LIBS  :=
        CC_OMP    := $(CC)
    endif
endif
CC_OMP ?= $(CC)
# ── Benchmark ─────────────────────────────────────────────────────────────────
BENCH_SRCS := $(SRCDIR)/utils.c $(SRCDIR)/models.c $(SRCDIR)/quant.c \
              $(SRCDIR)/gguf.c  $(SRCDIR)/ops.c    $(SRCDIR)/ops_llama.c \
              $(SRCDIR)/llama_tok.c $(SRCDIR)/tokenizer.c \
              benchmarks/bench.c
.PHONY: all omp bench bench-omp clean help
## all    — single-threaded build (default)
all: $(TARGET)
$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $^ -o $@ $(LIBS)
	@echo "Built: $@"
## omp    — OpenMP multi-threaded build (recommended for Large / XL)
omp: $(SRCS)
	$(CC_OMP) $(CFLAGS) $(OMP_FLAGS) $^ -o $(TARGET_OMP) $(LIBS) $(OMP_LIBS)
	@echo "Built: $(TARGET_OMP)"
## bench  — single-threaded benchmark binary
bench: $(BENCH_SRCS)
	$(CC) $(CFLAGS) $^ -o lmc_bench $(LIBS)
	@echo "Built: lmc_bench"
## bench-omp — OpenMP benchmark binary
bench-omp: $(BENCH_SRCS)
	$(CC_OMP) $(CFLAGS) $(OMP_FLAGS) $^ -o lmc_bench_omp $(LIBS) $(OMP_LIBS)
	@echo "Built: lmc_bench_omp"
## clean  — remove compiled binaries
clean:
ifeq ($(OS), Windows_NT)
	-$(RM) lmc.exe lmc_omp.exe lmc_bench.exe lmc_bench_omp.exe
else
	$(RM) lmc lmc_omp lmc_bench lmc_bench_omp
endif
## help   — show all targets and example usage
help:
	@echo ""
	@echo "  lmc — Lightweight Model Engine"
	@echo "  ─────────────────────────────────────────────────────"
	@echo "  make              Single-threaded build"
	@echo "  make omp          OpenMP multi-threaded build"
	@echo "  make bench        Benchmark binary (single-thread)"
	@echo "  make bench-omp    Benchmark binary (OpenMP)"
	@echo "  make clean        Remove compiled binaries"
	@echo ""
	@echo "  Example:"
	@echo "    ./lmc --model models/gpt2-xl.gguf \\"
	@echo "          --prompt \"Hello, how are you?\" \\"
	@echo "          --n-predict 128 --temp 0.7 --threads 4"
	@echo ""
	@echo "  macOS OpenMP: brew install libomp"
	@echo "  Windows:      use MinGW/MSYS2 terminal"
	@echo ""