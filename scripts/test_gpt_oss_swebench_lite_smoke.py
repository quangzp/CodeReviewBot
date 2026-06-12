#!/usr/bin/env python3
"""
Smoke test: load openai/gpt-oss-20b (Transformers) and princeton-nlp/SWE-bench_Lite.

Install (separate venv recommended; Python 3.10+ is safest for current wheels):

  python -m pip install --upgrade pip
  pip install -U torch transformers accelerate datasets huggingface_hub

Or:

  pip install -r scripts/requirements-smoke.txt

If pip is old, you may see "No matching distribution found for transformers"
(from versions: none) — upgrading pip fixes that.

Optional for some MXFP4 setups (see OpenAI gpt-oss Transformers cookbook):
  pip install -U triton==3.4 kernels

Run:
  python scripts/test_gpt_oss_swebench_lite_smoke.py
  python scripts/test_gpt_oss_swebench_lite_smoke.py --dataset-only
  python scripts/test_gpt_oss_swebench_lite_smoke.py --model-only

If the dataset or model is gated, run: huggingface-cli login
"""

from __future__ import annotations

import argparse
import sys

MODEL_NAME = "openai/gpt-oss-20b"
DATASET_NAME = "princeton-nlp/SWE-bench_Lite"


def run_model_smoke(max_new_tokens: int = 40) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer: {MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(
        f"Loading model: {MODEL_NAME} (may take a while; needs sufficient RAM/VRAM)",
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )

    messages = [
        {"role": "user", "content": "Who are you?"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    print("Generating...", flush=True)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    new_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(new_tokens)
    print("--- model output (new tokens only) ---")
    print(text)
    print("--- end ---")


def run_dataset_smoke() -> None:
    from datasets import load_dataset

    print(f"Loading dataset: {DATASET_NAME}", flush=True)
    ds = load_dataset(DATASET_NAME)
    print("Splits:", list(ds.keys()))
    first_split = next(iter(ds.keys()))
    n = len(ds[first_split])
    print(f"Rows in '{first_split}': {n}")
    sample = ds[first_split][0]
    print("First row keys:", sorted(sample.keys()))


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test gpt-oss-20b + SWE-bench_Lite")
    parser.add_argument(
        "--dataset-only",
        action="store_true",
        help="Only load SWE-bench_Lite (skip model).",
    )
    parser.add_argument(
        "--model-only",
        action="store_true",
        help="Only run a short gpt-oss generation (skip dataset).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=40,
        help="max_new_tokens for model.generate (default: 40).",
    )
    args = parser.parse_args()

    if args.dataset_only and args.model_only:
        print("Choose at most one of --dataset-only and --model-only.", file=sys.stderr)
        return 2

    run_model = not args.dataset_only
    run_ds = not args.model_only

    if run_ds:
        run_dataset_smoke()
    if run_model:
        run_model_smoke(max_new_tokens=args.max_new_tokens)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
