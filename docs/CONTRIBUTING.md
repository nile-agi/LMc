# Contributing to LMc

Thank you for your interest in LMc — the AI inference engine built for the reality of computing
in Africa and the Global South. Every contribution, whether a bug fix, a new model backend, or
a documentation improvement, directly helps people who need it most.

---

## Table of contents

- [Getting started](#getting-started)
- [How to contribute](#how-to-contribute)
- [Branch and commit conventions](#branch-and-commit-conventions)
- [Pull request process](#pull-request-process)
- [Code style](#code-style)
- [Testing](#testing)
- [Areas that need help](#areas-that-need-help)
- [What not to do](#what-not-to-do)

---

## Getting started

### Prerequisites

- A C99-compatible compiler (GCC, Clang, or MSVC)
- `make` or `cmake` (check the README for your platform)
- No external libraries required — LMc has zero external dependencies by design

### Build from source

```bash
git clone https://github.com/nile-agi/LMc.git
cd LMc
make
```

For cross-compilation (e.g. ARM, Android):

```bash
make ARCH=arm TARGET=android
```

See `docs/building.md` for platform-specific instructions.

---

## How to contribute

### 1. Fork the repository

Click **Fork** on the GitHub page. This gives you your own copy to work in.

### 2. Create a feature branch

Never work directly on `main`. Create a branch from `main` with a descriptive name:

```bash
git checkout -b feat/add-llama3-backend
git checkout -b fix/memory-leak-on-arm
git checkout -b docs/improve-gguf-loading-notes
```

### 3. Make your changes

Keep changes focused. One branch = one logical change. If you find an unrelated bug while
working, open a separate issue rather than bundling it into your PR.

### 4. Test your changes

Run the full test suite before pushing:

```bash
make test
```

If you are adding a new feature, add a corresponding test in `tests/`. If the existing tests
do not cover your change, explain why in your PR description.

### 5. Push and open a pull request

```bash
git push origin feat/add-llama3-backend
```

Then go to GitHub and open a Pull Request against the `main` branch. Fill in the PR template fully.

---

## Branch and commit conventions

### Branch names

| Prefix | Use for |
|--------|---------|
| `feat/` | New features or model/hardware backends |
| `fix/` | Bug fixes |
| `docs/` | Documentation only |
| `refactor/` | Code restructuring without behaviour change |
| `perf/` | Performance improvements |
| `test/` | Adding or fixing tests |
| `ci/` | CI/CD pipeline changes |

### Commit messages

Use short, imperative present-tense messages:

```
Add GGUF v3 tensor loading
Fix null pointer on empty model path
Reduce peak memory on ARM Cortex-A53
```

Do not write:
```
Fixed a bug
WIP
changes
```

---

## Pull request process

1. Fill in the PR template completely
2. Ensure all CI checks pass (build, tests, lint)
3. Request review — one approval from a maintainer is required before merging
4. Respond to review comments within 7 days or the PR may be closed
5. A maintainer will squash-merge your PR and tag a release if appropriate

### PR title format

```
[feat] Add Mistral 7B backend
[fix] Correct token sampling on edge case
[docs] Add Android build instructions
```

---

## Code style

LMc is written in pure C99. The goal is code that is readable on an aging laptop with a small
screen — not clever code.

- **Indentation**: 4 spaces, no tabs
- **Line length**: 100 characters maximum
- **Function names**: `snake_case`
- **Struct names**: `PascalCase`
- **Constants and macros**: `UPPER_SNAKE_CASE`
- **Comments**: use `/* */` for block comments; `//` is acceptable for short inline notes
- **No dynamic memory in hot paths** — allocate up front, free at teardown
- **No global mutable state** unless absolutely unavoidable
- **Every public function must have a comment** explaining its purpose, parameters, and return value

Run the formatter before pushing:

```bash
make lint
```

---

## Testing

Tests live in `tests/`. Each test file corresponds to a module. To run a specific test:

```bash
make test TEST=tests/test_tokenizer.c
```

When adding a new backend or hardware target:

- Add a test that loads a known GGUF model and runs a forward pass
- Include expected output checksums for deterministic operations
- Test on the lowest-spec target you can access — an old phone or Raspberry Pi is ideal

---

## Areas that need help

These are the highest-priority areas right now:

- **Model backends** — Mistral, Phi, Gemma, and other GGUF-compatible architectures
- **Hardware backends** — NEON (ARM), RVV (RISC-V), and low-power Android optimisations
- **Windows support** — improving the MSVC build path
- **Quantisation** — Q4_K_M and Q5_K_S support
- **Documentation** — especially build guides for African Android devices and low-RAM laptops
- **Testing on real hardware** — if you have access to low-spec devices, running and reporting
  results is extremely valuable

If you are new to the project, look for issues labelled `good first issue`.

---

## What not to do

- **Do not add external dependencies.** LMc's zero-dependency design is a core feature — it is
  what makes deployment possible in offline and low-bandwidth environments.
- **Do not break the C99 standard.** We must compile on GCC 7 and older toolchains.
- **Do not add GPU-only code paths** without a CPU fallback. Every feature must work on a
  device with no GPU.
- **Do not commit model weights or large binary files.**

---

## Questions?

Open a GitHub Discussion or tag a maintainer in an issue. We are happy to help you get started.