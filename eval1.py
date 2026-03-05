"""
eval1.py — Logits-based benchmark evaluation for CosmicFish
Supports: HellaSwag, PIQA, WinoGrande

How it works (logit scoring):
  For each multiple-choice example, we feed [context + each candidate] through
  the model and compute the average per-token log-likelihood for the candidate
  tokens only. The candidate with the highest score wins.
  This is the same method used by the original LM-Harness and avoids any
  generation quirks that make the model look worse than it is.

Usage:
  python eval1.py --checkpoint path/to/ckpt.pt
  python eval1.py --checkpoint path/to/ckpt.pt --quick          # 100 examples per benchmark
  python eval1.py --checkpoint path/to/ckpt.pt --benchmarks hellaswag piqa
  python eval1.py --checkpoint path/to/ckpt.pt --device cpu
"""

import os
import sys
import json
import math
import time
import argparse
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn.functional as F
import tiktoken
from torch.serialization import add_safe_globals

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import CosmicFish, CosmicConfig

add_safe_globals([CosmicConfig])

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset helpers
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MCExample:
    """A single multiple-choice example with one correct answer."""
    context: str                   # prompt / context
    choices: List[str]             # list of candidate continuations
    label: int                     # index of the correct choice


# ── shared download helper ────────────────────────────────────────────────────

import re
import urllib.request
import urllib.error


def _download(url: str, dest: str) -> None:
    """Download a file with a progress indicator."""
    log.info(f"  Downloading {url}")
    try:
        urllib.request.urlretrieve(url, dest)
    except urllib.error.URLError as e:
        raise RuntimeError(f"Download failed: {e}") from e


# ── HellaSwag ─────────────────────────────────────────────────────────────────

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
    """
    Load HellaSwag validation split.
    Downloads the raw JSONL directly from the official GitHub source —
    no datasets library script required.
    """
    log.info("Loading HellaSwag …")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "hellaswag_val.jsonl")

    if not os.path.exists(cache_path):
        # Official raw JSONL from the HellaSwag GitHub repo
        url = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
        try:
            _download(url, cache_path)
            log.info(f"  Saved to {cache_path}")
        except Exception as e:
            # Fallback: try HuggingFace parquet via pandas
            log.warning(f"  GitHub download failed ({e}), trying HuggingFace parquet …")
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
                log.info(f"  Saved {len(df)} examples to {cache_path}")
            except Exception as e2:
                log.error(f"  All download methods failed: {e2}")
                log.error("  Install pandas + pyarrow: pip install pandas pyarrow")
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


# ── PIQA ──────────────────────────────────────────────────────────────────────

def load_piqa(quick: bool = False, cache_dir: str = ".benchmark_cache") -> List[MCExample]:
    """
    Load PIQA validation split.
    Downloads raw files directly — avoids the deprecated loading script issue.
    PIQA hosts its data as separate goal/solution/label text files.
    """
    log.info("Loading PIQA …")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "piqa_val.jsonl")

    if not os.path.exists(cache_path):
        base = "https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa.zip"
        # The canonical raw files are hosted on yonatanbisk's GitHub
        goals_url  = "https://raw.githubusercontent.com/ybisk/ybisk.github.io/master/piqa/data/valid.jsonl"
        labels_url = "https://raw.githubusercontent.com/ybisk/ybisk.github.io/master/piqa/data/valid-labels.lst"

        goals_path  = os.path.join(cache_dir, "piqa_valid.jsonl")
        labels_path = os.path.join(cache_dir, "piqa_valid_labels.lst")

        try:
            _download(goals_url, goals_path)
            _download(labels_url, labels_path)
        except Exception as e:
            # Fallback: HuggingFace parquet (no script, just data files)
            log.warning(f"  GitHub download failed ({e}), trying HuggingFace parquet …")
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
                log.info(f"  Saved {len(df)} examples via parquet to {cache_path}")
                goals_path = None  # signal to use cache_path directly below
            except Exception as e2:
                log.error(f"  All download methods failed: {e2}")
                sys.exit(1)

        # Merge goals + labels into cache_path if we used the raw files
        if goals_path is not None and os.path.exists(goals_path):
            with open(goals_path) as gf, open(labels_path) as lf, open(cache_path, "w") as out:
                for goal_line, label_line in zip(gf, lf):
                    item = json.loads(goal_line)
                    item["label"] = int(label_line.strip())
                    out.write(json.dumps(item) + "\n")
            log.info(f"  Saved to {cache_path}")

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


# ── WinoGrande ────────────────────────────────────────────────────────────────

def load_winogrande(quick: bool = False, cache_dir: str = ".benchmark_cache") -> List[MCExample]:
    """
    Load WinoGrande validation split.
    Downloads the official data directly from AI2's release.
    """
    log.info("Loading WinoGrande …")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "winogrande_val.jsonl")

    if not os.path.exists(cache_path):
        # AI2 hosts the official WinoGrande files directly
        data_url   = "https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1.zip"
        # Fallback: AllenAI HuggingFace parquet (no loading script, pure data)
        parquet_url = (
            "https://huggingface.co/datasets/allenai/winogrande/resolve/main/"
            "data/validation-00000-of-00001.parquet"
        )

        downloaded = False

        # Try parquet first (simpler, no zip extraction needed)
        try:
            import pandas as pd
            pq_path = cache_path + ".parquet"
            _download(parquet_url, pq_path)
            df = pd.read_parquet(pq_path)
            with open(cache_path, "w") as f:
                for _, row in df.iterrows():
                    f.write(json.dumps(row.to_dict()) + "\n")
            os.remove(pq_path)
            log.info(f"  Saved {len(df)} examples to {cache_path}")
            downloaded = True
        except Exception as e:
            log.warning(f"  Parquet download failed ({e}), trying zip …")

        if not downloaded:
            # Try zip from AI2
            try:
                import zipfile, io
                zip_path = os.path.join(cache_dir, "winogrande.zip")
                _download(data_url, zip_path)
                with zipfile.ZipFile(zip_path) as zf:
                    # Find the dev/validation jsonl inside
                    candidates = [n for n in zf.namelist() if "dev" in n and n.endswith(".jsonl")]
                    if not candidates:
                        candidates = [n for n in zf.namelist() if n.endswith(".jsonl")]
                    if not candidates:
                        raise RuntimeError("No JSONL found in WinoGrande zip")
                    # Prefer xl split
                    chosen = next((c for c in candidates if "xl" in c and "dev" in c), candidates[0])
                    log.info(f"  Extracting {chosen}")
                    with zf.open(chosen) as src, open(cache_path, "wb") as dst:
                        dst.write(src.read())
                os.remove(zip_path)
                downloaded = True
            except Exception as e2:
                log.error(f"  All download methods failed: {e2}")
                log.error("  Try: pip install pandas pyarrow")
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
            label = int(answer) - 1  # "1"/"2" → 0/1

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


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, device: str) -> Tuple[CosmicFish, CosmicConfig]:
    log.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # ── resolve config ────────────────────────────────────────────────────────
    if "cosmicconf" in checkpoint:
        config = checkpoint["cosmicconf"]
    elif "config" in checkpoint:
        config = checkpoint["config"]
    else:
        log.warning("No config found in checkpoint — using defaults")
        config = CosmicConfig()

    log.info(
        f"  Config: {config.n_layer}L  {config.n_head}H  {config.n_embd}D  "
        f"block={config.block_size}  RoPE={config.use_rotary}  "
        f"GQA={config.use_gqa}  SwiGLU={config.use_swiglu}"
    )

    # ── build model ───────────────────────────────────────────────────────────
    # Silence the param-count print during eval
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model = CosmicFish(config)

    # ── load weights ──────────────────────────────────────────────────────────
    if "model_state_dict" in checkpoint:
        raw_sd = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        raw_sd = checkpoint["model"]
    else:
        raise ValueError("Cannot find model weights in checkpoint")

    # Strip compile / DDP prefixes
    sd = {}
    for k, v in raw_sd.items():
        k = k.removeprefix("_orig_mod.").removeprefix("module.")
        sd[k] = v

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        log.warning(f"  Missing keys: {len(missing)}")
    if unexpected:
        log.warning(f"  Unexpected keys: {len(unexpected)}")

    model.to(device)
    model.eval()
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"  Model ready — {params_m:.1f}M parameters")
    return model, config


# ══════════════════════════════════════════════════════════════════════════════
# Logit-based scoring
# ══════════════════════════════════════════════════════════════════════════════

def score_sequence(
    model: CosmicFish,
    config: CosmicConfig,
    tokenizer,
    context_ids: List[int],
    continuation_ids: List[int],
    device: str,
) -> float:
    """
    Compute the average per-token log-likelihood of `continuation_ids`
    given `context_ids` as context.

    We feed [context + continuation] through the model in a single forward
    pass and read off the logits at positions len(context)-1 … len(all)-2
    (i.e. the positions whose *next-token prediction* covers the continuation).

    Returns the mean log-prob (higher = more likely).
    """
    all_ids = context_ids + continuation_ids
    max_len = config.block_size

    # Truncate from the left if needed, always keeping at least 1 continuation token
    if len(all_ids) > max_len:
        overflow = len(all_ids) - max_len
        # trim context, not continuation
        context_ids = context_ids[overflow:]
        all_ids = context_ids + continuation_ids

    input_ids = torch.tensor([all_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        # full sequence logits — pass targets=None to get full sequence
        # We need full-sequence logits, not just the last token.
        # CosmicFish returns only the last-token logit when targets=None.
        # So we pass targets to trigger full-sequence mode, then discard the loss.
        targets = torch.full_like(input_ids, -1)
        logits, _ = model(input_ids, targets=targets)
        # logits: [1, T, vocab]

    # The logit at position i predicts token i+1.
    # Continuation tokens occupy positions len(context_ids) … len(all_ids)-1.
    # Their predicting logits are at positions len(context_ids)-1 … len(all_ids)-2.
    cont_start = len(context_ids) - 1          # first logit predicting a continuation token
    cont_end   = len(all_ids) - 1              # one past the last

    logits_cont = logits[0, cont_start:cont_end, :]        # [cont_len, vocab]
    targets_cont = torch.tensor(continuation_ids, dtype=torch.long, device=device)

    log_probs = F.log_softmax(logits_cont, dim=-1)         # [cont_len, vocab]
    token_log_probs = log_probs[
        torch.arange(len(continuation_ids), device=device),
        targets_cont
    ]                                                       # [cont_len]

    # Average log-prob (length-normalised so long choices don't get penalised)
    return token_log_probs.mean().item()


def evaluate_benchmark(
    model: CosmicFish,
    config: CosmicConfig,
    tokenizer,
    examples: List[MCExample],
    benchmark_name: str,
    device: str,
    batch_log_every: int = 50,
) -> Dict:
    correct = 0
    total = len(examples)
    t0 = time.time()

    for i, ex in enumerate(examples):
        # Tokenize context once
        context_ids = tokenizer.encode(ex.context)

        scores = []
        for choice in ex.choices:
            # Important: add a leading space to the continuation so the
            # tokenizer treats it as a word boundary (same trick lm-harness uses)
            cont = " " + choice if not choice.startswith(" ") else choice
            continuation_ids = tokenizer.encode(cont)

            if len(continuation_ids) == 0:
                scores.append(-1e9)
                continue

            s = score_sequence(model, config, tokenizer, context_ids, continuation_ids, device)
            scores.append(s)

        pred = scores.index(max(scores))
        if pred == ex.label:
            correct += 1

        if (i + 1) % batch_log_every == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            acc = correct / (i + 1) * 100
            speed = (i + 1) / elapsed
            log.info(
                f"  [{benchmark_name}] {i+1}/{total}  "
                f"acc={acc:.1f}%  {speed:.1f} ex/s"
            )

    accuracy = correct / total
    elapsed = time.time() - t0
    return {
        "benchmark": benchmark_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "elapsed_s": round(elapsed, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

BENCHMARK_MAP = {
    "hellaswag":   load_hellaswag,
    "piqa":        load_piqa,
    "winogrande":  load_winogrande,
}

# Random-chance baselines for reference
RANDOM_BASELINE = {
    "hellaswag":  25.0,   # 4 choices
    "piqa":       50.0,   # 2 choices
    "winogrande": 50.0,   # 2 choices
}


def main():
    parser = argparse.ArgumentParser(description="Logit-based benchmark eval for CosmicFish")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
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
        help="Run only 100 examples per benchmark (fast sanity-check)",
    )
    parser.add_argument(
        "--cache_dir", default=".benchmark_cache",
        help="Where to cache downloaded datasets (default: .benchmark_cache)",
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

    # ── model ─────────────────────────────────────────────────────────────────
    model, config = load_model(args.checkpoint, device)

    # ── tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = tiktoken.get_encoding("gpt2")
    log.info("Tokenizer: gpt2 (tiktoken)")

    if args.quick:
        log.info("⚡ Quick mode: 100 examples per benchmark")

    # ── run benchmarks ────────────────────────────────────────────────────────
    results = []
    for name in args.benchmarks:
        log.info(f"\n{'='*60}")
        log.info(f"Benchmark: {name.upper()}")
        log.info(f"{'='*60}")

        loader = BENCHMARK_MAP[name]
        examples = loader(quick=args.quick, cache_dir=args.cache_dir)

        result = evaluate_benchmark(
            model=model,
            config=config,
            tokenizer=tokenizer,
            examples=examples,
            benchmark_name=name,
            device=device,
        )
        results.append(result)

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'RESULTS SUMMARY':^60}")
    print("=" * 60)
    print(f"  {'Benchmark':<14}  {'Accuracy':>8}  {'Correct':>8}  {'Total':>6}  {'vs. Random':>10}")
    print(f"  {'-'*14}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*10}")

    total_correct = 0
    total_examples = 0
    for r in results:
        baseline = RANDOM_BASELINE.get(r["benchmark"], 0)
        delta = r["accuracy"] * 100 - baseline
        sign = "+" if delta >= 0 else ""
        print(
            f"  {r['benchmark']:<14}  "
            f"{r['accuracy']*100:>7.1f}%  "
            f"{r['correct']:>8}  "
            f"{r['total']:>6}  "
            f"{sign}{delta:>+.1f}pp"
        )
        total_correct += r["correct"]
        total_examples += r["total"]

    if len(results) > 1:
        overall = total_correct / total_examples * 100
        print(f"  {'-'*14}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*10}")
        print(f"  {'OVERALL':<14}  {overall:>7.1f}%  {total_correct:>8}  {total_examples:>6}")

    print("=" * 60)

    if args.quick:
        print("  ⚡ Quick mode was ON — results are from 100-example subsets")

    print()


if __name__ == "__main__":
    main()