#!/usr/bin/env python3
"""
scripts/converter.py — Convert HuggingFace GPT-2 weights to LMc binary format

Generates: gpt2_124m.bin (float32, ~500 MB)

Usage:
    pip install torch transformers
    python3 scripts/converter.py
    python3 scripts/converter.py --output /path/to/gpt2_124m.bin

Binary format (little-endian):
    Header: [magic:u32][version:u32][vocab_size:u32][seq_len:u32]
            [n_layers:u32][n_heads:u32][embed_dim:u32]
    Weights: float32 arrays in the order defined in src/models/gpt2_weights.c
"""

import struct
import sys
import argparse

def convert(output_path: str) -> None:
    try:
        import torch
        from transformers import GPT2LMHeadModel
    except ImportError:
        print("ERROR: Install dependencies first:")
        print("  pip install torch transformers")
        sys.exit(1)

    print(f"Loading GPT-2 124M from HuggingFace...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    sd = model.state_dict()

    MAGIC   = 0x47505432   # "GPT2"
    VERSION = 1
    VOCAB   = 50257
    SEQLEN  = 1024
    LAYERS  = 12
    HEADS   = 12
    DIM     = 768

    print(f"Writing to: {output_path}")

    with open(output_path, "wb") as f:
        # Header
        f.write(struct.pack("<7I", MAGIC, VERSION, VOCAB, SEQLEN, LAYERS, HEADS, DIM))

        def write_tensor(key, transpose=False):
            t = sd[key].float()
            if transpose:
                t = t.t().contiguous()
            f.write(t.numpy().tobytes())
            print(f"  {key:55s} {tuple(t.shape)}")

        # Embeddings
        write_tensor("transformer.wte.weight")
        write_tensor("transformer.wpe.weight")

        # Transformer layers
        for i in range(LAYERS):
            write_tensor(f"transformer.h.{i}.ln_1.weight")
            write_tensor(f"transformer.h.{i}.ln_1.bias")
            # GPT-2 stores Q, K, V concatenated in c_attn
            # c_attn.weight is [768, 2304] (in HF format [in, out])
            # We need [3072, 768] = [3*DIM, DIM] row-major
            write_tensor(f"transformer.h.{i}.attn.c_attn.weight", transpose=True)
            write_tensor(f"transformer.h.{i}.attn.c_attn.bias")
            write_tensor(f"transformer.h.{i}.attn.c_proj.weight", transpose=True)
            write_tensor(f"transformer.h.{i}.attn.c_proj.bias")
            write_tensor(f"transformer.h.{i}.ln_2.weight")
            write_tensor(f"transformer.h.{i}.ln_2.bias")
            write_tensor(f"transformer.h.{i}.mlp.c_fc.weight",    transpose=True)
            write_tensor(f"transformer.h.{i}.mlp.c_fc.bias")
            write_tensor(f"transformer.h.{i}.mlp.c_proj.weight",  transpose=True)
            write_tensor(f"transformer.h.{i}.mlp.c_proj.bias")

        # Final layer norm
        write_tensor("transformer.ln_f.weight")
        write_tensor("transformer.ln_f.bias")
        # Note: lm_head is tied to wte, not written separately

    size_mb = __import__("os").path.getsize(output_path) / (1024**2)
    print(f"\nDone! {output_path}  ({size_mb:.0f} MB)")
    print("Run: ./build/lmc \"Your prompt here\"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GPT-2 HF weights to LMc .bin format")
    parser.add_argument("--output", default="gpt2_124m.bin", help="Output path")
    args = parser.parse_args()
    convert(args.output)
