"""
eval1oth.py — Logits-based benchmark evaluation for reference models
Supports: GPT-2 small, OPT-125M, Pythia-160M

Same logit-scoring method as eval1.py so results are directly comparable
to CosmicFish scores.

Usage:
  python eval1oth.py --model gpt2
  python eval1oth.py --model opt-125m
  python eval1oth.py --model pythia-160m
  python eval1oth.py --model all                          # run all three
  python eval1oth.py --model gpt2 --quick                 # 100 examples
  python eval1oth.py --model all --benchmarks hellaswag piqa
  python eval1oth.py --model gpt2 --device cpu

Requires:
  pip install transformers torch tiktoken
  pip install datasets   (optional, only needed if cache is missing and
                          direct downloads fail)
"""

import os
import sys
import json
import re
import time
import argparse
import logging
import urllib.request
import urllib.error
import zipfile
from dataclasses import dataclass
from typing import List, Dict, Tuple, Callable

import torch
import torch.nn.functional as F

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Model registry
# ══════════════════════════════════════════════════════════════════════════════

# Maps CLI name → HuggingFace model ID
MODEL_IDS = {
    "gpt2":       "gpt2",
    "opt-125m":   "facebook/opt-125m",
    "pythia-160m": "EleutherAI/pythia-160m",
}

# Random-chance baselines
RANDOM_BASELINE = {
    "hellaswag":  25.0,
    "piqa":       50.0,
    "winogrande": 50.0,
}


# ══════════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MCExample:
    context: str
    choices: List[str]
    label: int


# ══════════════════════════════════════════════════════════════════════════════
# Dataset loaders  (identical logic to eval1.py — same cache files reused)
# ══════════════════════════════════════════════════════════════════════════════

def _download(url: str, dest: str) -> None:
    log.info(f"  Downloading {url}")
    try:
        urllib.request.urlretrieve(url, dest)
    except urllib.error.URLError as e:
        raise RuntimeError(f"Download failed: {e}") from e


def _hellaswag_preprocess(text: str) -> str:
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = text.replace("[header]", "")
    text = text.replace("[step]", "")
    text = text.replace("[substeps]", "")
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r" +", " ", text).strip()
    return text


def load_hellaswag(quick: bool = False, cache_dir: str = ".benchmark_cache") -> List[MCExample]:
    log.info("Loading HellaSwag …")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "hellaswag_val.jsonl")

    if not os.path.exists(cache_path):
        url = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
        try:
            _download(url, cache_path)
        except Exception as e:
            log.warning(f"  GitHub download failed ({e}), trying parquet …")
            try:
                import pandas as pd
                pq_url = "https://huggingface.co/datasets/Rowan/hellaswag/resolve/main/data/validation-00000-of-00001.parquet"
                pq_path = cache_path + ".parquet"
                _download(pq_url, pq_path)
                df = pd.read_parquet(pq_path)
                with open(cache_path, "w") as f:
                    for _, row in df.iterrows():
                        f.write(json.dumps(row.to_dict()) + "\n")
                os.remove(pq_path)
            except Exception as e2:
                log.error(f"  All downloads failed: {e2}")
                sys.exit(1)

    examples: List[MCExample] = []
    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            ctx = _hellaswag_preprocess(item["ctx"])
            choices = [_hellaswag_preprocess(e) for e in item["endings"]]
            label = int(item["label"])
            examples.append(MCExample(context=ctx, choices=choices, label=label))

    if quick:
        examples = examples[:100]
    log.info(f"  HellaSwag: {len(examples)} examples")
    return examples


def load_piqa(quick: bool = False, cache_dir: str = ".benchmark_cache") -> List[MCExample]:
    log.info("Loading PIQA …")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "piqa_val.jsonl")

    if not os.path.exists(cache_path):
        goals_url  = "https://raw.githubusercontent.com/ybisk/ybisk.github.io/master/piqa/data/valid.jsonl"
        labels_url = "https://raw.githubusercontent.com/ybisk/ybisk.github.io/master/piqa/data/valid-labels.lst"
        goals_path  = os.path.join(cache_dir, "piqa_valid.jsonl")
        labels_path = os.path.join(cache_dir, "piqa_valid_labels.lst")
        try:
            _download(goals_url, goals_path)
            _download(labels_url, labels_path)
            with open(goals_path) as gf, open(labels_path) as lf, open(cache_path, "w") as out:
                for goal_line, label_line in zip(gf, lf):
                    item = json.loads(goal_line)
                    item["label"] = int(label_line.strip())
                    out.write(json.dumps(item) + "\n")
        except Exception as e:
            log.warning(f"  GitHub download failed ({e}), trying parquet …")
            try:
                import pandas as pd
                pq_url = "https://huggingface.co/datasets/ybisk/piqa/resolve/main/data/validation-00000-of-00001.parquet"
                pq_path = cache_path + ".parquet"
                _download(pq_url, pq_path)
                df = pd.read_parquet(pq_path)
                with open(cache_path, "w") as f:
                    for _, row in df.iterrows():
                        f.write(json.dumps(row.to_dict()) + "\n")
                os.remove(pq_path)
            except Exception as e2:
                log.error(f"  All downloads failed: {e2}")
                sys.exit(1)

    examples: List[MCExample] = []
    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            goal = item.get("goal", item.get("ctx", "")).strip()
            sol1 = item.get("sol1", item.get("choice1", "")).strip()
            sol2 = item.get("sol2", item.get("choice2", "")).strip()
            label = int(item["label"])
            if not goal or not sol1 or not sol2:
                continue
            examples.append(MCExample(context=goal, choices=[sol1, sol2], label=label))

    if quick:
        examples = examples[:100]
    log.info(f"  PIQA: {len(examples)} examples")
    return examples


def load_winogrande(quick: bool = False, cache_dir: str = ".benchmark_cache") -> List[MCExample]:
    log.info("Loading WinoGrande …")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "winogrande_val.jsonl")

    if not os.path.exists(cache_path):
        downloaded = False
        try:
            import pandas as pd
            pq_url = (
                "https://huggingface.co/datasets/allenai/winogrande/resolve/main/"
                "data/validation-00000-of-00001.parquet"
            )
            pq_path = cache_path + ".parquet"
            _download(pq_url, pq_path)
            df = pd.read_parquet(pq_path)
            with open(cache_path, "w") as f:
                for _, row in df.iterrows():
                    f.write(json.dumps(row.to_dict()) + "\n")
            os.remove(pq_path)
            downloaded = True
        except Exception as e:
            log.warning(f"  Parquet download failed ({e}), trying zip …")

        if not downloaded:
            try:
                zip_path = os.path.join(cache_dir, "winogrande.zip")
                _download("https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1.zip", zip_path)
                with zipfile.ZipFile(zip_path) as zf:
                    candidates = [n for n in zf.namelist() if "dev" in n and n.endswith(".jsonl")]
                    if not candidates:
                        candidates = [n for n in zf.namelist() if n.endswith(".jsonl")]
                    chosen = next((c for c in candidates if "xl" in c and "dev" in c), candidates[0])
                    with zf.open(chosen) as src, open(cache_path, "wb") as dst:
                        dst.write(src.read())
                os.remove(zip_path)
            except Exception as e2:
                log.error(f"  All downloads failed: {e2}")
                sys.exit(1)

    examples: List[MCExample] = []
    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            sentence = item["sentence"]
            opt1 = item["option1"].strip()
            opt2 = item["option2"].strip()
            answer = str(item.get("answer", item.get("label", "1")))
            label = int(answer) - 1
            parts = sentence.split("_", 1)
            if len(parts) == 2:
                prefix, suffix = parts[0], parts[1]
                ctx = prefix.rstrip()
                choices = [opt1 + suffix, opt2 + suffix]
            else:
                ctx = ""
                choices = [sentence.replace("_", opt1), sentence.replace("_", opt2)]
            examples.append(MCExample(context=ctx, choices=choices, label=label))

    if quick:
        examples = examples[:100]
    log.info(f"  WinoGrande: {len(examples)} examples")
    return examples


BENCHMARK_MAP: Dict[str, Callable] = {
    "hellaswag":  load_hellaswag,
    "piqa":       load_piqa,
    "winogrande": load_winogrande,
}


# ══════════════════════════════════════════════════════════════════════════════
# HuggingFace model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_hf_model(model_key: str, device: str):
    """
    Load a HuggingFace causal LM and its tokenizer.
    Returns (model, tokenizer, max_length).

    Each model has quirks:
      GPT-2      — standard CausalLM, BOS token used as padding
      OPT-125M   — prepends a BOS token automatically; we must account for it
      Pythia-160M — EleutherAI GPT-NeoX family, very standard
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_id = MODEL_IDS[model_key]
    log.info(f"Loading {model_key} ({model_id}) …")

    # ── tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Ensure pad token exists (GPT-2 has none by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── model ─────────────────────────────────────────────────────────────────
    # OPT-125M's main branch only has pytorch_model.bin which requires torch>=2.6
    # due to CVE-2025-32434. Load from the community safetensors PR instead.
    revision = "refs/pr/51" if model_key == "opt-125m" else None
    log.info(f"  revision={revision or 'main'}")

    load_kwargs = dict(dtype=torch.float32)
    if revision:
        load_kwargs["revision"] = revision

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.to(device)
    model.eval()

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"  {model_key} ready — {params_m:.1f}M parameters")

    # Max context length
    max_len = getattr(model.config, "max_position_embeddings",
               getattr(model.config, "n_positions",
               getattr(model.config, "max_seq_len", 1024)))

    # OPT prepends a BOS token — flag this so the scorer can compensate
    adds_bos = model_key == "opt-125m"

    return model, tokenizer, max_len, adds_bos


# ══════════════════════════════════════════════════════════════════════════════
# Logit-based scoring  (HF version)
# ══════════════════════════════════════════════════════════════════════════════

def score_sequence_hf(
    model,
    tokenizer,
    max_len: int,
    adds_bos: bool,
    context_ids: List[int],
    continuation_ids: List[int],
    device: str,
) -> float:
    """
    Compute mean per-token log-likelihood of continuation_ids given context_ids.

    HF CausalLM models always return full-sequence logits, so this is simpler
    than the CosmicFish version — no targets trick needed.

    OPT note: OPT's tokenizer prepends a BOS token to every encoded string,
    which shifts all positions by 1. We detect this via `adds_bos` and strip
    the implicit BOS from context_ids so the continuation slice is correct.
    """
    # OPT tokenizer silently prepends BOS — strip it to avoid double-counting
    if adds_bos and len(context_ids) > 0 and context_ids[0] == tokenizer.bos_token_id:
        context_ids = context_ids[1:]
    if adds_bos and len(continuation_ids) > 0 and continuation_ids[0] == tokenizer.bos_token_id:
        continuation_ids = continuation_ids[1:]

    all_ids = context_ids + continuation_ids

    # Truncate from the left, preserving the full continuation
    if len(all_ids) > max_len:
        overflow = len(all_ids) - max_len
        context_ids = context_ids[overflow:]
        all_ids = context_ids + continuation_ids

    if len(all_ids) == 0 or len(continuation_ids) == 0:
        return -1e9

    input_ids = torch.tensor([all_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_ids)
        # HF always returns full [B, T, vocab] logits regardless of targets
        logits = outputs.logits  # [1, T, vocab]

    # logit[i] predicts token[i+1]
    # Continuation spans positions len(context_ids) … len(all_ids)-1
    # Predicting logits are at positions len(context_ids)-1 … len(all_ids)-2
    cont_start = len(context_ids) - 1
    cont_end   = len(all_ids) - 1

    logits_cont  = logits[0, cont_start:cont_end, :]           # [cont_len, vocab]
    targets_cont = torch.tensor(continuation_ids, dtype=torch.long, device=device)

    log_probs = F.log_softmax(logits_cont, dim=-1)
    token_log_probs = log_probs[
        torch.arange(len(continuation_ids), device=device),
        targets_cont,
    ]

    return token_log_probs.mean().item()


def evaluate_benchmark_hf(
    model,
    tokenizer,
    max_len: int,
    adds_bos: bool,
    examples: List[MCExample],
    benchmark_name: str,
    model_key: str,
    device: str,
    batch_log_every: int = 50,
) -> Dict:
    correct = 0
    total = len(examples)
    t0 = time.time()

    for i, ex in enumerate(examples):
        context_ids = tokenizer.encode(ex.context)

        scores = []
        for choice in ex.choices:
            # Add a leading space so the tokenizer treats the continuation as
            # a word boundary — same trick used in lm-harness
            cont = " " + choice if not choice.startswith(" ") else choice
            continuation_ids = tokenizer.encode(cont)

            if len(continuation_ids) == 0:
                scores.append(-1e9)
                continue

            s = score_sequence_hf(
                model, tokenizer, max_len, adds_bos,
                context_ids, continuation_ids, device,
            )
            scores.append(s)

        pred = scores.index(max(scores))
        if pred == ex.label:
            correct += 1

        if (i + 1) % batch_log_every == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            acc = correct / (i + 1) * 100
            speed = (i + 1) / elapsed
            log.info(
                f"  [{model_key}/{benchmark_name}] {i+1}/{total}  "
                f"acc={acc:.1f}%  {speed:.1f} ex/s"
            )

    accuracy = correct / total
    elapsed = time.time() - t0
    return {
        "model": model_key,
        "benchmark": benchmark_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "elapsed_s": round(elapsed, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(all_results: List[Dict], quick: bool) -> None:
    benchmarks = list(dict.fromkeys(r["benchmark"] for r in all_results))
    models     = list(dict.fromkeys(r["model"]     for r in all_results))

    # Build lookup
    lookup = {(r["model"], r["benchmark"]): r for r in all_results}

    col_w = 10
    bm_w  = 14

    print("\n" + "=" * 70)
    print(f"{'RESULTS SUMMARY':^70}")
    print("=" * 70)

    # Header
    header = f"  {'Model':<14}"
    for bm in benchmarks:
        header += f"  {bm.capitalize():>{col_w}}"
    header += f"  {'Overall':>{col_w}}"
    print(header)
    print(f"  {'-'*14}" + f"  {'-'*col_w}" * (len(benchmarks) + 1))

    for m in models:
        row = f"  {m:<14}"
        total_correct = 0
        total_examples = 0
        for bm in benchmarks:
            r = lookup.get((m, bm))
            if r:
                row += f"  {r['accuracy']*100:>{col_w-1}.1f}%"
                total_correct  += r["correct"]
                total_examples += r["total"]
            else:
                row += f"  {'—':>{col_w}}"
        if total_examples > 0:
            overall = total_correct / total_examples * 100
            row += f"  {overall:>{col_w-1}.1f}%"
        print(row)

    # Random baseline row
    print(f"  {'-'*14}" + f"  {'-'*col_w}" * (len(benchmarks) + 1))
    rand_row = f"  {'[random]':<14}"
    rand_sum = 0
    for bm in benchmarks:
        b = RANDOM_BASELINE.get(bm, 0)
        rand_row += f"  {b:>{col_w-1}.1f}%"
        rand_sum += b
    rand_row += f"  {rand_sum/len(benchmarks):>{col_w-1}.1f}%"
    print(rand_row)

    print("=" * 70)
    if quick:
        print("  ⚡ Quick mode was ON — results are from 100-example subsets")
    print()

    # Per-model delta table
    print("  vs. Random baseline (pp above random):")
    print(f"  {'Model':<14}" + "".join(f"  {bm.capitalize():>{col_w}}" for bm in benchmarks))
    print(f"  {'-'*14}" + f"  {'-'*col_w}" * len(benchmarks))
    for m in models:
        row = f"  {m:<14}"
        for bm in benchmarks:
            r = lookup.get((m, bm))
            if r:
                delta = r["accuracy"] * 100 - RANDOM_BASELINE.get(bm, 0)
                row += f"  {delta:>+{col_w}.1f}"
            else:
                row += f"  {'—':>{col_w}}"
        print(row)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Logit-based benchmark eval for GPT-2, OPT-125M, Pythia-160M"
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=list(MODEL_IDS.keys()) + ["all"],
        help="Which model to evaluate (default: all)",
    )
    parser.add_argument(
        "--benchmarks", nargs="+",
        default=["hellaswag", "piqa", "winogrande"],
        choices=list(BENCHMARK_MAP.keys()),
        help="Which benchmarks to run",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Device: cpu / cuda / mps / auto  (default: auto)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run only 100 examples per benchmark",
    )
    parser.add_argument(
        "--cache_dir", default=".benchmark_cache",
        help="Dataset cache directory (default: .benchmark_cache)",
    )
    args = parser.parse_args()

    # ── device ────────────────────────────────────────────────────────────────
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    log.info(f"Device: {device}")

    if args.quick:
        log.info("⚡ Quick mode: 100 examples per benchmark")

    # ── which models to run ───────────────────────────────────────────────────
    models_to_run = list(MODEL_IDS.keys()) if args.model == "all" else [args.model]

    # ── pre-load datasets (once, shared across models) ────────────────────────
    log.info("\nPre-loading datasets …")
    datasets: Dict[str, List[MCExample]] = {}
    for bm_name in args.benchmarks:
        datasets[bm_name] = BENCHMARK_MAP[bm_name](quick=args.quick, cache_dir=args.cache_dir)

    # ── evaluate each model ───────────────────────────────────────────────────
    all_results: List[Dict] = []

    for model_key in models_to_run:
        log.info(f"\n{'='*60}")
        log.info(f"Model: {model_key.upper()}")
        log.info(f"{'='*60}")

        try:
            model, tokenizer, max_len, adds_bos = load_hf_model(model_key, device)
        except Exception as e:
            log.error(f"  Failed to load {model_key}: {e}")
            log.error("  Make sure transformers is installed: pip install transformers")
            continue

        for bm_name in args.benchmarks:
            log.info(f"\n  --- {bm_name.upper()} ---")
            examples = datasets[bm_name]

            result = evaluate_benchmark_hf(
                model=model,
                tokenizer=tokenizer,
                max_len=max_len,
                adds_bos=adds_bos,
                examples=examples,
                benchmark_name=bm_name,
                model_key=model_key,
                device=device,
            )
            all_results.append(result)

        # Free VRAM before loading the next model
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ── summary ───────────────────────────────────────────────────────────────
    if all_results:
        print_summary(all_results, args.quick)


if __name__ == "__main__":
    main()