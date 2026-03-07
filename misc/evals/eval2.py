"""
eval2.py — Logit-based QA evaluation for CosmicFish
Benchmarks: TriviaQA, Natural Questions (NQ), ARC-Easy

How it works:
  These are open-ended QA benchmarks with one or more valid answers per question.
  TriviaQA/NQ: gold answers scored against hard distractors — correct if gold ranks #1.
  ARC-Easy: grade school science MCQ (4 choices). Score each choice with log-likelihood,
    pick the highest — identical method to HellaSwag/PIQA in eval1.py.
  Same logit scoring as eval1.py throughout.

Usage:
  python eval2.py --checkpoint CF300M.pt
  python eval2.py --checkpoint CF300M.pt --quick          # 100 examples per benchmark
  python eval2.py --checkpoint CF300M.pt --benchmarks triviaqa nq
  python eval2.py --checkpoint CF300M.pt --device cpu
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
import zipfile
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable

import torch
import torch.nn.functional as F
import tiktoken
from torch.serialization import add_safe_globals

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
# Data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MCExample:
    """Multiple-choice example (used for ARC-Easy)."""
    context: str
    choices: List[str]
    label: int


@dataclass
class QAExample:
    """A single QA example."""
    question: str                    # the question
    answers: List[str]               # all valid answer strings
    is_unanswerable: bool = False    # SQuAD v2 unanswerable questions
    passage: str = ""                # SQuAD v2 reading passage (empty for TriviaQA/NQ)


# ══════════════════════════════════════════════════════════════════════════════
# Answer normalisation  (standard EM/F1 normalisation used by all three datasets)
# ══════════════════════════════════════════════════════════════════════════════

def _normalise_answer(s: str) -> str:
    """Lower, strip punctuation, collapse whitespace, remove articles."""
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _answers_match(pred: str, gold_answers: List[str]) -> bool:
    """Return True if normalised pred matches any normalised gold answer."""
    pred_n = _normalise_answer(pred)
    return any(_normalise_answer(a) == pred_n for a in gold_answers)


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
# Dataset loaders
# ══════════════════════════════════════════════════════════════════════════════

# ── TriviaQA ──────────────────────────────────────────────────────────────────

def load_triviaqa(quick: bool = False, cache_dir: str = ".benchmark_cache") -> List[QAExample]:
    """
    Load TriviaQA validation split.
    Uses nq_open-style HuggingFace parquet (confirmed working URLs),
    falling back to the official datasets library if parquet fails.
    """
    log.info("Loading TriviaQA …")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "triviaqa_val.jsonl")

    if not os.path.exists(cache_path):
        # Confirmed working HuggingFace parquet endpoints (checked 2025)
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
            # Final fallback: HuggingFace datasets library
            try:
                from datasets import load_dataset
                log.info("  Trying datasets library …")
                ds = load_dataset("trivia_qa", "rc.wikipedia", split="validation")
                with open(cache_path, "w") as f:
                    for item in ds:
                        f.write(json.dumps(dict(item)) + "\n")
                log.info(f"  Saved {len(ds)} TriviaQA examples via datasets library")
            except Exception as e2:
                log.error(f"  All TriviaQA download methods failed: {e2}")
                log.error("  Try: pip install datasets   or   pip install pandas pyarrow")
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

            # Handle both parquet format and official JSON format
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


# ── Natural Questions ──────────────────────────────────────────────────────────

def load_nq(quick: bool = False, cache_dir: str = ".benchmark_cache") -> List[QAExample]:
    """
    Load Natural Questions validation split (short answers only).
    Uses the HuggingFace simplified NQ dataset.
    Examples with no short answer are skipped (they are effectively unanswerable
    at this granularity).
    """
    log.info("Loading Natural Questions …")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "nq_val.jsonl")

    if not os.path.exists(cache_path):
        # nq_open is the clean short-answer-only NQ split — perfect for evaluation
        # It has confirmed working parquet files on HuggingFace
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
            # Fallback: HuggingFace datasets library
            try:
                from datasets import load_dataset
                log.info("  Trying datasets library for nq_open …")
                ds = load_dataset("google-research-datasets/nq_open", split="validation")
                with open(cache_path, "w") as f:
                    for item in ds:
                        f.write(json.dumps(dict(item)) + "\n")
                log.info(f"  Saved {len(ds)} NQ examples via datasets library")
            except Exception as e2:
                log.error(f"  All NQ download methods failed: {e2}")
                log.error("  Try: pip install datasets   or   pip install pandas pyarrow")
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
            # Ensure question ends with '?'
            if not question.endswith("?"):
                question = question + "?"

            # Various field names depending on the source
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




# ── ARC-Easy ──────────────────────────────────────────────────────────────────

def load_arceasy(quick: bool = False, cache_dir: str = ".benchmark_cache") -> List[MCExample]:
    """
    Load ARC-Easy validation split.
    ARC (AI2 Reasoning Challenge) Easy subset — grade school science questions
    with 4 answer choices. Scored identically to HellaSwag in eval1.py.
    """
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
                log.error("  Try: pip install datasets  or  pip install pandas pyarrow")
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

            # choices field: either {"text": [...], "label": [...]} dict
            # or already a list depending on source
            choices_field = item.get("choices", {})
            if isinstance(choices_field, dict):
                texts  = choices_field.get("text", [])
                labels = choices_field.get("label", [])
            elif isinstance(choices_field, list):
                # list of {"text": ..., "label": ...} dicts
                texts  = [c["text"]  for c in choices_field]
                labels = [c["label"] for c in choices_field]
            else:
                continue

            if len(texts) < 2:
                continue

            # answerKey is "A"/"B"/"C"/"D" or "1"/"2"/"3"/"4"
            answer_key = str(item.get("answerKey", item.get("answer_key", "A"))).strip()

            # Map answer key to index
            if answer_key in labels:
                label_idx = labels.index(answer_key)
            elif answer_key.isdigit():
                label_idx = int(answer_key) - 1
            else:
                # fallback: try A=0, B=1, C=2, D=3
                label_idx = ord(answer_key.upper()) - ord("A")

            if not (0 <= label_idx < len(texts)):
                continue

            examples.append(MCExample(
                context=question,
                choices=list(texts),
                label=label_idx,
            ))

    if quick:
        examples = examples[:100]
    log.info(f"  ARC-Easy: {len(examples)} examples")
    return examples


# ══════════════════════════════════════════════════════════════════════════════
# Model loading  (identical to eval1.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, device: str) -> Tuple[CosmicFish, CosmicConfig]:
    log.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "cosmicconf" in checkpoint:
        config = checkpoint["cosmicconf"]
    elif "config" in checkpoint:
        config = checkpoint["config"]
    else:
        log.warning("No config in checkpoint — using defaults")
        config = CosmicConfig()

    log.info(
        f"  Config: {config.n_layer}L  {config.n_head}H  {config.n_embd}D  "
        f"block={config.block_size}  RoPE={config.use_rotary}  "
        f"GQA={config.use_gqa}  SwiGLU={config.use_swiglu}"
    )

    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model = CosmicFish(config)

    if "model_state_dict" in checkpoint:
        raw_sd = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        raw_sd = checkpoint["model"]
    else:
        raise ValueError("Cannot find model weights in checkpoint")

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
# Logit-based scoring  (same as eval1.py — exact same function)
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
    Mean per-token log-likelihood of continuation_ids given context_ids.
    Identical to eval1.py — uses the targets trick to get full-sequence logits.
    """
    all_ids = context_ids + continuation_ids
    max_len = config.block_size

    if len(all_ids) > max_len:
        overflow = len(all_ids) - max_len
        context_ids = context_ids[overflow:]
        all_ids = context_ids + continuation_ids

    if len(continuation_ids) == 0:
        return -1e9

    input_ids = torch.tensor([all_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        targets = torch.full_like(input_ids, -1)
        logits, _ = model(input_ids, targets=targets)

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
# QA evaluation
# ══════════════════════════════════════════════════════════════════════════════

# The "no answer" string used as a candidate for SQuAD v2 unanswerable questions
NO_ANSWER_STRING = "unanswerable"

def _make_context(question: str, passage: str = "") -> str:
    """Build the context string fed to the model before scoring the answer."""
    if passage:
        # Truncate passage — keep it short so the question stays near "Answer:"
        # The question being close to the answer token matters for attention
        if len(passage) > 500:
            passage = passage[:500].rsplit(" ", 1)[0] + " …"
        # Format: passage first, then question immediately before Answer:
        return f"{passage}\n\nQuestion: {question}\nAnswer:"
    return f"Question: {question}\nAnswer:"


def evaluate_benchmark(
    model: CosmicFish,
    config: CosmicConfig,
    tokenizer,
    examples: List[MCExample],
    benchmark_name: str,
    device: str,
    batch_log_every: int = 50,
) -> Dict:
    """
    Multiple-choice evaluation — identical to eval1.py.
    Used for ARC-Easy (4 choices). Scores each choice, picks the highest.
    """
    correct = 0
    total = len(examples)
    t0 = time.time()

    for i, ex in enumerate(examples):
        context_ids = tokenizer.encode(ex.context)
        scores = []
        for choice in ex.choices:
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


def evaluate_qa(
    model: CosmicFish,
    config: CosmicConfig,
    tokenizer,
    examples: List[QAExample],
    benchmark_name: str,
    device: str,
    batch_log_every: int = 50,
) -> Dict:
    """
    Candidate-ranking evaluation. Same method as eval1.py.

    BoolQ     — two candidates: "yes" / "no". Whichever scores higher wins.
    TriviaQA/NQ — gold answers vs. a fixed hard-distractor set. Correct if
                  any gold answer scores higher than all distractors.
    """
    correct = 0
    total   = len(examples)
    t0      = time.time()

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

    for i, ex in enumerate(examples):
        # Skip unanswerable examples (none in current benchmarks)
        if ex.is_unanswerable:
            total -= 1
            continue

        context_ids = tokenizer.encode(_make_context(ex.question, ex.passage))

        if True:  # TriviaQA / NQ
            # TriviaQA / NQ: score gold answers + distractors, argmax wins
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

            gold_norms   = {_normalise_answer(a) for a in gold_answers}
            distractors  = [d for d in OPEN_QA_DISTRACTORS
                            if _normalise_answer(d) not in gold_norms]

            best_score      = -1e9
            best_is_correct = False

            for ans in gold_answers:
                cont = " " + ans if not ans.startswith(" ") else ans
                cont_ids = tokenizer.encode(cont)
                if not cont_ids:
                    continue
                s = score_sequence(model, config, tokenizer, context_ids, cont_ids, device)
                if s > best_score:
                    best_score = s
                    best_is_correct = True

            for d in distractors:
                cont_ids = tokenizer.encode(" " + d)
                if not cont_ids:
                    continue
                s = score_sequence(model, config, tokenizer, context_ids, cont_ids, device)
                if s > best_score:
                    best_score = s
                    best_is_correct = False

            if best_is_correct:
                correct += 1

        if (i + 1) % batch_log_every == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            acc     = correct / (i + 1) * 100
            speed   = (i + 1) / elapsed
            log.info(
                f"  [{benchmark_name}] {i+1}/{total}  "
                f"acc={acc:.1f}%  {speed:.1f} ex/s"
            )

    accuracy = correct / total if total > 0 else 0.0
    elapsed  = time.time() - t0
    return {
        "benchmark": benchmark_name,
        "accuracy":  accuracy,
        "correct":   correct,
        "total":     total,
        "elapsed_s": round(elapsed, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

BENCHMARK_MAP: Dict[str, Callable] = {
    "triviaqa":  load_triviaqa,
    "nq":        load_nq,
    "arceasy":   load_arceasy,
}

RANDOM_BASELINE = {
    "triviaqa": 0.0,   # open-ended, no random baseline
    "nq":       0.0,
    "arceasy":  25.0,  # 4 choices
}


def main():
    parser = argparse.ArgumentParser(
        description="Logit-based QA evaluation for CosmicFish (TriviaQA, NQ, SQuAD v2)"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
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

        examples = BENCHMARK_MAP[name](quick=args.quick, cache_dir=args.cache_dir)

        if name == "arceasy":
            # Reformat to Q/A style so each choice is scored as a natural
            # answer continuation — same approach lm-harness uses for ARC
            nl = "\n"
            arc_examples = [
                MCExample(
                    context="Question: " + ex.context + nl + "Answer:",
                    choices=ex.choices,
                    label=ex.label,
                )
                for ex in examples
            ]
            result = evaluate_benchmark(
                model=model,
                config=config,
                tokenizer=tokenizer,
                examples=arc_examples,
                benchmark_name=name,
                device=device,
            )
        else:
            result = evaluate_qa(
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
    print(f"  {'Benchmark':<12}  {'Accuracy':>8}  {'Correct':>8}  {'Total':>7}  {'Note':>10}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*10}")

    for r in results:
        baseline = RANDOM_BASELINE.get(r["benchmark"], 0.0)
        note = f"vs {baseline:.0f}% rnd" if baseline > 0 else "open QA"
        print(
            f"  {r['benchmark']:<12}  "
            f"{r['accuracy']*100:>7.1f}%  "
            f"{r['correct']:>8}  "
            f"{r['total']:>7}  "
            f"{note:>10}"
        )

    print("=" * 60)
    print()
    print("  Metric: logit-based candidate ranking (same as eval1.py)")
    print("  TriviaQA/NQ : correct if gold answer ranks above hard distractors")
    print("  ARC-Easy    : 4-choice MCQ, same log-likelihood ranking as HellaSwag/PIQA")
  
    if args.quick:
        print("  ⚡ Quick mode was ON — results are from 100-example subsets")
    print()


if __name__ == "__main__":
    main()