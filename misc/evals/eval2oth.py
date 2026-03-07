"""
eval2oth.py — Logit-based QA evaluation for reference HuggingFace models
Benchmarks: TriviaQA, Natural Questions (NQ), ARC-Easy

Models: GPT-2 small, OPT-125M, Pythia-160M

Identical scoring to eval2.py so results are directly comparable to CosmicFish.
  TriviaQA/NQ : gold answers ranked against hard distractors
  ARC-Easy    : 4-choice MCQ, per-token log-likelihood ranking

Usage:
  python eval2oth.py --model gpt2
  python eval2oth.py --model opt-125m
  python eval2oth.py --model pythia-160m
  python eval2oth.py --model all                    # run all three
  python eval2oth.py --model all --quick            # 100 examples per benchmark
  python eval2oth.py --model gpt2 --benchmarks triviaqa arceasy
  python eval2oth.py --model all --device cpu

Requires:
  pip install transformers torch
  pip install datasets      (fallback if parquet downloads fail)
  pip install pandas pyarrow  (for parquet downloads)
"""

import os
import sys
import json
import re
import time
import string
import argparse
import logging
import urllib.request
import urllib.error
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

MODEL_IDS = {
    "gpt2":        "gpt2",
    "opt-125m":    "facebook/opt-125m",
    "pythia-160m": "EleutherAI/pythia-160m",
}

RANDOM_BASELINE = {
    "triviaqa": 0.0,   # open-ended
    "nq":       0.0,
    "arceasy":  25.0,  # 4 choices
}


# ══════════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MCExample:
    """Multiple-choice example (ARC-Easy)."""
    context: str
    choices: List[str]
    label: int


@dataclass
class QAExample:
    """Open-ended QA example (TriviaQA, NQ)."""
    question: str
    answers: List[str]
    is_unanswerable: bool = False
    passage: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# Answer normalisation
# ══════════════════════════════════════════════════════════════════════════════

def _normalise_answer(s: str) -> str:
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ══════════════════════════════════════════════════════════════════════════════
# Download helper
# ══════════════════════════════════════════════════════════════════════════════

def _download(url: str, dest: str) -> None:
    log.info(f"  Downloading {url}")
    try:
        urllib.request.urlretrieve(url, dest)
    except urllib.error.URLError as e:
        raise RuntimeError(f"Download failed: {e}") from e


# ══════════════════════════════════════════════════════════════════════════════
# Dataset loaders  (identical to eval2.py — reuses same cache files)
# ══════════════════════════════════════════════════════════════════════════════

def load_triviaqa(quick: bool = False, cache_dir: str = ".benchmark_cache") -> List[QAExample]:
    log.info("Loading TriviaQA …")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "triviaqa_val.jsonl")

    if not os.path.exists(cache_path):
        pq_candidates = [
            "https://huggingface.co/datasets/trivia_qa/resolve/main/data/rc.wikipedia/validation-00000-of-00001.parquet",
            "https://huggingface.co/datasets/trivia_qa/resolve/main/data/rc/validation-00000-of-00001.parquet",
            "https://huggingface.co/datasets/trivia_qa/resolve/main/data/unfiltered.nocontext/validation-00000-of-00001.parquet",
            "https://huggingface.co/datasets/trivia_qa/resolve/main/data/unfiltered/validation-00000-of-00001.parquet",
        ]
        pq_path = cache_path + ".parquet"
        parquet_ok = False
        try:
            import pandas as pd
            for pq_url in pq_candidates:
                try:
                    _download(pq_url, pq_path)
                    df = pd.read_parquet(pq_path)
                    with open(cache_path, "w") as f:
                        for _, row in df.iterrows():
                            f.write(json.dumps(row.to_dict()) + "\n")
                    os.remove(pq_path)
                    log.info(f"  Saved {len(df)} TriviaQA examples via parquet")
                    parquet_ok = True
                    break
                except Exception as e:
                    log.warning(f"  {pq_url.split('/')[-1]} failed: {e}")
                    if os.path.exists(pq_path):
                        os.remove(pq_path)
        except ImportError:
            log.warning("  pandas not installed, skipping parquet")

        if not parquet_ok:
            try:
                from datasets import load_dataset
                log.info("  Trying datasets library …")
                ds = load_dataset("trivia_qa", "rc.wikipedia", split="validation")
                with open(cache_path, "w") as f:
                    for item in ds:
                        f.write(json.dumps(dict(item)) + "\n")
                log.info(f"  Saved {len(ds)} TriviaQA examples")
            except Exception as e2:
                log.error(f"  All TriviaQA downloads failed: {e2}")
                sys.exit(1)

    examples: List[QAExample] = []
    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            question = item.get("question", item.get("Question", "")).strip()
            if not question:
                continue
            ans_field = item.get("answer", item.get("Answer", {}))
            if isinstance(ans_field, dict):
                aliases = ans_field.get("aliases", ans_field.get("Aliases", []))
                value   = ans_field.get("value",   ans_field.get("Value", ""))
                answers = list({value} | set(aliases)) if value else list(aliases)
            elif isinstance(ans_field, str):
                answers = [ans_field]
            else:
                answers = []
            answers = [a for a in answers if a.strip()]
            if not answers:
                continue
            examples.append(QAExample(question=question, answers=answers))

    if quick:
        examples = examples[:100]
    log.info(f"  TriviaQA: {len(examples)} examples")
    return examples


def load_nq(quick: bool = False, cache_dir: str = ".benchmark_cache") -> List[QAExample]:
    log.info("Loading Natural Questions …")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "nq_val.jsonl")

    if not os.path.exists(cache_path):
        pq_candidates = [
            "https://huggingface.co/datasets/google-research-datasets/nq_open/resolve/main/data/validation-00000-of-00001.parquet",
            "https://huggingface.co/datasets/nq_open/resolve/main/data/validation-00000-of-00001.parquet",
        ]
        pq_path = cache_path + ".parquet"
        parquet_ok = False
        try:
            import pandas as pd
            for pq_url in pq_candidates:
                try:
                    _download(pq_url, pq_path)
                    df = pd.read_parquet(pq_path)
                    with open(cache_path, "w") as f:
                        for _, row in df.iterrows():
                            f.write(json.dumps(row.to_dict()) + "\n")
                    os.remove(pq_path)
                    log.info(f"  Saved {len(df)} NQ examples via parquet")
                    parquet_ok = True
                    break
                except Exception as e:
                    log.warning(f"  NQ parquet {pq_url.split('/')[-1]} failed: {e}")
                    if os.path.exists(pq_path):
                        os.remove(pq_path)
        except ImportError:
            log.warning("  pandas not installed, skipping parquet")

        if not parquet_ok:
            try:
                from datasets import load_dataset
                log.info("  Trying datasets library for nq_open …")
                ds = load_dataset("google-research-datasets/nq_open", split="validation")
                with open(cache_path, "w") as f:
                    for item in ds:
                        f.write(json.dumps(dict(item)) + "\n")
                log.info(f"  Saved {len(ds)} NQ examples")
            except Exception as e2:
                log.error(f"  All NQ downloads failed: {e2}")
                sys.exit(1)

    examples: List[QAExample] = []
    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            question = item.get("question", item.get("Question", "")).strip()
            if not question:
                continue
            if not question.endswith("?"):
                question = question + "?"
            answers = []
            for key in ("answer", "answers", "short_answers", "Answer"):
                val = item.get(key)
                if val is None:
                    continue
                if isinstance(val, list):
                    answers = [str(a).strip() for a in val if str(a).strip()]
                elif isinstance(val, str) and val.strip():
                    answers = [val.strip()]
                if answers:
                    break
            if not answers:
                continue
            examples.append(QAExample(question=question, answers=answers))

    if quick:
        examples = examples[:100]
    log.info(f"  Natural Questions: {len(examples)} examples")
    return examples


def load_arceasy(quick: bool = False, cache_dir: str = ".benchmark_cache") -> List[MCExample]:
    log.info("Loading ARC-Easy …")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "arceasy_val.jsonl")

    if not os.path.exists(cache_path):
        pq_candidates = [
            "https://huggingface.co/datasets/allenai/ai2_arc/resolve/main/data/ARC-Easy/validation-00000-of-00001.parquet",
            "https://huggingface.co/datasets/ai2_arc/resolve/main/data/ARC-Easy/validation-00000-of-00001.parquet",
        ]
        pq_path = cache_path + ".parquet"
        parquet_ok = False
        try:
            import pandas as pd
            for pq_url in pq_candidates:
                try:
                    _download(pq_url, pq_path)
                    df = pd.read_parquet(pq_path)
                    with open(cache_path, "w") as f:
                        for _, row in df.iterrows():
                            f.write(json.dumps(row.to_dict()) + "\n")
                    os.remove(pq_path)
                    log.info(f"  Saved {len(df)} ARC-Easy examples via parquet")
                    parquet_ok = True
                    break
                except Exception as e:
                    log.warning(f"  ARC-Easy parquet failed: {e}")
                    if os.path.exists(pq_path):
                        os.remove(pq_path)
        except ImportError:
            pass

        if not parquet_ok:
            try:
                from datasets import load_dataset
                log.info("  Trying datasets library …")
                ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")
                with open(cache_path, "w") as f:
                    for item in ds:
                        f.write(json.dumps(dict(item)) + "\n")
                log.info(f"  Saved {len(ds)} ARC-Easy examples")
            except Exception as e2:
                log.error(f"  All ARC-Easy downloads failed: {e2}")
                sys.exit(1)

    examples: List[MCExample] = []
    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            question = item.get("question", "").strip()
            if not question:
                continue
            choices_field = item.get("choices", {})
            if isinstance(choices_field, dict):
                texts  = choices_field.get("text", [])
                labels = choices_field.get("label", [])
            elif isinstance(choices_field, list):
                texts  = [c["text"]  for c in choices_field]
                labels = [c["label"] for c in choices_field]
            else:
                continue
            if len(texts) < 2:
                continue
            answer_key = str(item.get("answerKey", item.get("answer_key", "A"))).strip()
            if answer_key in labels:
                label_idx = labels.index(answer_key)
            elif answer_key.isdigit():
                label_idx = int(answer_key) - 1
            else:
                label_idx = ord(answer_key.upper()) - ord("A")
            if not (0 <= label_idx < len(texts)):
                continue
            examples.append(MCExample(context=question, choices=list(texts), label=label_idx))

    if quick:
        examples = examples[:100]
    log.info(f"  ARC-Easy: {len(examples)} examples")
    return examples


BENCHMARK_MAP: Dict[str, Callable] = {
    "triviaqa": load_triviaqa,
    "nq":       load_nq,
    "arceasy":  load_arceasy,
}


# ══════════════════════════════════════════════════════════════════════════════
# HuggingFace model loading  (same approach as eval1oth.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_hf_model(model_key: str, device: str):
    """
    Load a HuggingFace causal LM and its tokenizer.
    Returns (model, tokenizer, max_len, adds_bos).

    OPT-125M quirk: tokenizer prepends BOS to every encoded string.
    OPT-125M weight quirk: main branch only has pytorch_model.bin (blocked by
      torch CVE-2025-32434). Load from safetensors PR refs/pr/51 instead.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_id = MODEL_IDS[model_key]
    log.info(f"Loading {model_key} ({model_id}) …")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # OPT-125M: load from community safetensors PR to avoid torch CVE block
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

    max_len = getattr(model.config, "max_position_embeddings",
               getattr(model.config, "n_positions",
               getattr(model.config, "max_seq_len", 1024)))

    # OPT tokenizer silently prepends BOS to every encode() call
    adds_bos = (model_key == "opt-125m")

    return model, tokenizer, max_len, adds_bos


# ══════════════════════════════════════════════════════════════════════════════
# Logit-based scoring  (HF version — same logic as eval1oth.py)
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
    Mean per-token log-likelihood of continuation_ids given context_ids.
    HF models always return full [B, T, vocab] logits — no targets trick needed.
    OPT BOS stripping: OPT tokenizer prepends BOS to every encode(); we strip
    it so the continuation slice is correct.
    """
    if adds_bos:
        bos = tokenizer.bos_token_id
        if context_ids and context_ids[0] == bos:
            context_ids = context_ids[1:]
        if continuation_ids and continuation_ids[0] == bos:
            continuation_ids = continuation_ids[1:]

    all_ids = context_ids + continuation_ids

    if len(all_ids) > max_len:
        overflow = len(all_ids) - max_len
        context_ids = context_ids[overflow:]
        all_ids = context_ids + continuation_ids

    if not all_ids or not continuation_ids:
        return -1e9

    input_ids = torch.tensor([all_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_ids).logits  # [1, T, vocab]

    cont_start = len(context_ids) - 1
    cont_end   = len(all_ids) - 1

    logits_cont  = logits[0, cont_start:cont_end, :]
    targets_cont = torch.tensor(continuation_ids, dtype=torch.long, device=device)

    log_probs = F.log_softmax(logits_cont, dim=-1)
    token_log_probs = log_probs[
        torch.arange(len(continuation_ids), device=device),
        targets_cont,
    ]

    return token_log_probs.mean().item()


# ══════════════════════════════════════════════════════════════════════════════
# Context formatting
# ══════════════════════════════════════════════════════════════════════════════

def _make_context(question: str) -> str:
    return f"Question: {question}\nAnswer:"


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation functions
# ══════════════════════════════════════════════════════════════════════════════

OPEN_QA_DISTRACTORS = [
    "United States", "United Kingdom", "France", "Germany", "China",
    "World War II", "William Shakespeare", "Albert Einstein",
    "1945", "1969", "2001", "1776",
    "London", "New York", "Paris", "Rome",
    "the President", "the government", "Congress",
    "oxygen", "carbon dioxide", "hydrogen",
    "red", "blue", "green",
    "yes", "no",
]


def evaluate_mc(
    model,
    tokenizer,
    max_len: int,
    adds_bos: bool,
    examples: List[MCExample],
    benchmark_name: str,
    model_key: str,
    device: str,
    batch_log_every: int = 100,
) -> Dict:
    """
    Multiple-choice evaluation for ARC-Easy.
    Identical method to eval1oth.py evaluate_benchmark_hf.
    Context = "Question: ...\nAnswer:", candidates = choice texts.
    """
    correct = 0
    total   = len(examples)
    t0      = time.time()

    for i, ex in enumerate(examples):
        # Format as Q/A prompt — same as eval2.py does for CosmicFish
        ctx = "Question: " + ex.context + "\nAnswer:"
        context_ids = tokenizer.encode(ctx)

        scores = []
        for choice in ex.choices:
            cont = " " + choice if not choice.startswith(" ") else choice
            cont_ids = tokenizer.encode(cont)
            if not cont_ids:
                scores.append(-1e9)
                continue
            s = score_sequence_hf(
                model, tokenizer, max_len, adds_bos,
                context_ids, cont_ids, device
            )
            scores.append(s)

        pred = scores.index(max(scores))
        if pred == ex.label:
            correct += 1

        if (i + 1) % batch_log_every == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            acc   = correct / (i + 1) * 100
            speed = (i + 1) / elapsed
            log.info(
                f"  [{model_key}/{benchmark_name}] {i+1}/{total}  "
                f"acc={acc:.1f}%  {speed:.1f} ex/s"
            )

    accuracy = correct / total if total > 0 else 0.0
    elapsed  = time.time() - t0
    return {
        "model":     model_key,
        "benchmark": benchmark_name,
        "accuracy":  accuracy,
        "correct":   correct,
        "total":     total,
        "elapsed_s": round(elapsed, 1),
    }


def evaluate_qa(
    model,
    tokenizer,
    max_len: int,
    adds_bos: bool,
    examples: List[QAExample],
    benchmark_name: str,
    model_key: str,
    device: str,
    batch_log_every: int = 100,
) -> Dict:
    """
    Open QA evaluation for TriviaQA and NQ.
    Gold answers are scored against a fixed hard-distractor set.
    Correct if any gold answer scores higher than all distractors.
    Identical method to eval2.py.
    """
    correct = 0
    total   = len(examples)
    t0      = time.time()

    for i, ex in enumerate(examples):
        if ex.is_unanswerable:
            total -= 1
            continue

        context_ids = tokenizer.encode(_make_context(ex.question))

        # Deduplicate gold answers
        seen_norm = set()
        gold_answers = []
        for a in ex.answers:
            n = _normalise_answer(a)
            if n and n not in seen_norm:
                seen_norm.add(n)
                gold_answers.append(a)

        if not gold_answers:
            total -= 1
            continue

        gold_norms  = {_normalise_answer(a) for a in gold_answers}
        distractors = [d for d in OPEN_QA_DISTRACTORS
                       if _normalise_answer(d) not in gold_norms]

        best_score      = -1e9
        best_is_correct = False

        for ans in gold_answers:
            cont = " " + ans if not ans.startswith(" ") else ans
            cont_ids = tokenizer.encode(cont)
            if not cont_ids:
                continue
            s = score_sequence_hf(
                model, tokenizer, max_len, adds_bos,
                context_ids, cont_ids, device
            )
            if s > best_score:
                best_score = s
                best_is_correct = True

        for d in distractors:
            cont_ids = tokenizer.encode(" " + d)
            if not cont_ids:
                continue
            s = score_sequence_hf(
                model, tokenizer, max_len, adds_bos,
                context_ids, cont_ids, device
            )
            if s > best_score:
                best_score = s
                best_is_correct = False

        if best_is_correct:
            correct += 1

        if (i + 1) % batch_log_every == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            acc   = correct / (i + 1) * 100
            speed = (i + 1) / elapsed
            log.info(
                f"  [{model_key}/{benchmark_name}] {i+1}/{total}  "
                f"acc={acc:.1f}%  {speed:.1f} ex/s"
            )

    accuracy = correct / total if total > 0 else 0.0
    elapsed  = time.time() - t0
    return {
        "model":     model_key,
        "benchmark": benchmark_name,
        "accuracy":  accuracy,
        "correct":   correct,
        "total":     total,
        "elapsed_s": round(elapsed, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Summary printer
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(all_results: List[Dict], quick: bool) -> None:
    benchmarks = list(dict.fromkeys(r["benchmark"] for r in all_results))
    models     = list(dict.fromkeys(r["model"]     for r in all_results))
    lookup     = {(r["model"], r["benchmark"]): r for r in all_results}

    col_w = 10

    print("\n" + "=" * 72)
    print(f"{'RESULTS SUMMARY':^72}")
    print("=" * 72)

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
    rand_vals = []
    for bm in benchmarks:
        b = RANDOM_BASELINE.get(bm, 0.0)
        rand_vals.append(b)
        label = f"{b:.0f}%" if b > 0 else "n/a"
        rand_row += f"  {label:>{col_w}}"
    rand_row += f"  {'—':>{col_w}}"
    print(rand_row)

    print("=" * 72)
    if quick:
        print("  ⚡ Quick mode was ON — results are from 100-example subsets")
    print()

    # vs random delta table (only for MC benchmarks)
    mc_benchmarks = [bm for bm in benchmarks if RANDOM_BASELINE.get(bm, 0) > 0]
    if mc_benchmarks:
        print("  vs. Random baseline (pp above random) — MC benchmarks only:")
        print(f"  {'Model':<14}" + "".join(f"  {bm.capitalize():>{col_w}}" for bm in mc_benchmarks))
        print(f"  {'-'*14}" + f"  {'-'*col_w}" * len(mc_benchmarks))
        for m in models:
            row = f"  {m:<14}"
            for bm in mc_benchmarks:
                r = lookup.get((m, bm))
                if r:
                    delta = r["accuracy"] * 100 - RANDOM_BASELINE[bm]
                    row += f"  {delta:>+{col_w}.1f}"
                else:
                    row += f"  {'—':>{col_w}}"
            print(row)
        print()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Logit-based QA eval for GPT-2, OPT-125M, Pythia-160M"
    )
    parser.add_argument(
        "--model", default="all",
        choices=list(MODEL_IDS.keys()) + ["all"],
        help="Which model to evaluate (default: all)",
    )
    parser.add_argument(
        "--benchmarks", nargs="+",
        default=["triviaqa", "nq", "arceasy"],
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

    models_to_run = list(MODEL_IDS.keys()) if args.model == "all" else [args.model]

    # ── pre-load all datasets once ────────────────────────────────────────────
    log.info("\nPre-loading datasets …")
    datasets: Dict[str, list] = {}
    for bm_name in args.benchmarks:
        datasets[bm_name] = BENCHMARK_MAP[bm_name](
            quick=args.quick, cache_dir=args.cache_dir
        )

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

            if bm_name == "arceasy":
                result = evaluate_mc(
                    model=model,
                    tokenizer=tokenizer,
                    max_len=max_len,
                    adds_bos=adds_bos,
                    examples=examples,
                    benchmark_name=bm_name,
                    model_key=model_key,
                    device=device,
                )
            else:
                result = evaluate_qa(
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

    print("  Metric: logit-based candidate ranking (same as eval2.py)")
    print("  TriviaQA/NQ : correct if gold answer ranks above hard distractors")
    print("  ARC-Easy    : 4-choice MCQ, per-token log-likelihood ranking")
    print()


if __name__ == "__main__":
    main()
