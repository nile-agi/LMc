#!/usr/bin/env python3
"""
scripts/get_tokenizer.py — Download GPT-2 tokenizer files

Downloads encoder.json and vocab.bpe to the current directory.
These files are required by LMc at runtime.

Usage:
    python3 scripts/get_tokenizer.py
    python3 scripts/get_tokenizer.py --outdir /path/to/dir
"""

import argparse
import os
import sys
import urllib.request


TOKENIZER_FILES = {
    "encoder.json": "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json",
    "vocab.bpe":    "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe",
}


def download(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)

    for filename, url in TOKENIZER_FILES.items():
        out_path = os.path.join(outdir, filename)
        if os.path.exists(out_path):
            print(f"  ✓ {filename} already exists, skipping")
            continue

        print(f"  Downloading {filename} ...")
        try:
            urllib.request.urlretrieve(url, out_path)
            size_kb = os.path.getsize(out_path) / 1024
            print(f"  ✓ {filename}  ({size_kb:.0f} KB)")
        except Exception as e:
            print(f"  ✗ Failed to download {filename}: {e}")
            sys.exit(1)

    print(f"\nTokenizer files ready in: {outdir}")
    print("Run: ./build/lmc \"Your prompt here\"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download GPT-2 tokenizer files")
    parser.add_argument("--outdir", default=".", help="Output directory (default: .)")
    args = parser.parse_args()
    download(args.outdir)
