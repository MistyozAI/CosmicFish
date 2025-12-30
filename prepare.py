#!/usr/bin/env python3
"""
Complete 60B Token Dataset Preparation Script for CosmicFish
Processes datasets ONE AT A TIME sequentially with HuggingFace authentication.
TOKENIZES ON-THE-FLY and creates train.bin/val.bin files directly.

Final Dataset Plan:
- Core Web & Text (51B): FineWeb(32B) + C4(8B) + OpenWebText(6B) + Wikipedia(5B)
- Technical (9B): CodeParrot(4B) + OpenWebMath(3B) + ArXiv(2B)
- Total: 60B tokens across 7 datasets + examples.txt
"""

import os
import sys
import time
import json
import pickle
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator, Any
import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm
import tiktoken
from termcolor import colored
import multiprocessing
from dataclasses import dataclass
from huggingface_hub import login

# HuggingFace Authentication - Hardcoded as requested
HF_TOKEN = "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"

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
EOT_TOKEN = tokenizer.eot_token

@dataclass
class DatasetConfig:
    """Configuration for each dataset"""
    name: str
    dataset_id: str
    config: Optional[str] = None
    target_tokens: int = 0
    use_streaming: bool = False
    columns: List[str] = None
    min_tokens: int = 50  # Minimum tokens per chunk
    test_size: float = 0.0005  # Validation split size

def print_status(message: str, status: str = "info") -> None:
    """Print colored status messages"""
    colors = {"info": "cyan", "success": "green", "error": "red", "warning": "yellow", "progress": "magenta"}
    print(colored(f"[{status.upper()}] {message}", colors.get(status, "white")))

def clean_text(text: str) -> str:
    """Clean text by removing special tokens and normalizing whitespace"""
    if not isinstance(text, str):
        return ""

    # Remove EOT tokens and other special patterns
    text = text.replace("<|endoftext|>", "")
    text = text.replace(chr(50256), "")  # EOT token character
    text = re.sub(r'<\|.*?\|>', '', text)  # Remove <|special|> patterns

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text

def count_tokens(text: str) -> int:
    """Count tokens in text"""
    try:
        return len(tokenizer.encode(text))
    except:
        return 0

def tokenize_text(text: str) -> List[int]:
    """Tokenize text and return token IDs"""
    try:
        return tokenizer.encode(text)
    except:
        return []

def chunk_and_tokenize_text(text: str, max_tokens: int = 2048, min_chunk_tokens: int = 50) -> List[List[int]]:
    """Split text into chunks and return tokenized chunks"""
    if not text.strip():
        return []

    try:
        tokens = tokenizer.encode(text)
    except:
        return []

    if len(tokens) <= max_tokens:
        return [tokens] if len(tokens) >= min_chunk_tokens else []

    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        if len(chunk_tokens) >= min_chunk_tokens:
            chunks.append(chunk_tokens)

    return chunks

class DatasetProcessor:
    """Main class for processing datasets ONE AT A TIME with on-the-fly tokenization"""

    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Authenticate with HuggingFace
        self.authenticate_huggingface()

        # Dataset configurations - IN PROCESSING ORDER
        self.dataset_configs = [
            # Core Web & Text (51B tokens) - Start with smaller ones first
            DatasetConfig(
                name="wikipedia",
                dataset_id="wikipedia",
                config="20220301.en",
                target_tokens=5_000_000_000,  # INCREASED TO 5B
                use_streaming=False,
                columns=["title", "text"],
                min_tokens=50,
                test_size=0.0005
            ),
            DatasetConfig(
                name="openwebtext",
                dataset_id="Skylion007/openwebtext",
                target_tokens=6_000_000_000,
                use_streaming=False,
                columns=["text"],
                min_tokens=50,
                test_size=0.0005
            ),
            DatasetConfig(
                name="c4",
                dataset_id="allenai/c4",
                config="en",
                target_tokens=8_000_000_000,
                use_streaming=True,
                columns=["text"],
                min_tokens=50,
                test_size=0.0005
            ),
            DatasetConfig(
                name="fineweb",
                dataset_id="HuggingFaceFW/fineweb",
                target_tokens=32_000_000_000,
                use_streaming=True,
                columns=["text"],
                min_tokens=50,
                test_size=0.0005
            ),

            # Technical (9B tokens)
            DatasetConfig(
                name="arxiv",
                dataset_id="allenai/peS2o",
                target_tokens=2_000_000_000,
                use_streaming=False,
                columns=["title", "abstract", "text"],
                min_tokens=50,
                test_size=0.0005
            ),
            DatasetConfig(
                name="openwebmath",
                dataset_id="open-web-math/open-web-math",
                target_tokens=3_000_000_000,
                use_streaming=False,
                columns=["text"],
                min_tokens=50,
                test_size=0.0005
            ),
            DatasetConfig(
                name="codeparrot",
                dataset_id="codeparrot/codeparrot-clean",
                target_tokens=4_000_000_000,
                use_streaming=True,
                columns=["content"],
                min_tokens=50,
                test_size=0.0005
            )
        ]

        self.total_tokens_processed = 0
        self.dataset_stats = {}
        self.examples_for_display = {}

    def authenticate_huggingface(self):
        """Authenticate with HuggingFace Hub"""
        try:
            login(token=HF_TOKEN)
            print_status("✓ HuggingFace authentication successful", "success")
        except Exception as e:
            print_status(f"⚠ HuggingFace authentication failed: {e}", "warning")
            print_status("Continuing without authentication - may hit rate limits", "warning")

    def extract_text_from_item(self, item: Dict, config: DatasetConfig) -> Optional[str]:
        """Extract and combine text from dataset item"""
        try:
            texts = []
            for column in config.columns:
                if column in item and item[column]:
                    if isinstance(item[column], str):
                        texts.append(item[column])
                    else:
                        texts.append(str(item[column]))

            if not texts:
                return None

            # Join with newlines and clean
            combined_text = "\n".join(texts)
            return clean_text(combined_text)

        except Exception as e:
            return None

    def flush_tokens_to_temp_file(self, tokens: List[int], temp_dir: Path, file_counter: int) -> str:
        """Flush accumulated tokens to a temporary binary file"""
        temp_file = temp_dir / f"temp_{file_counter:06d}.bin"
        np.array(tokens, dtype=np.uint16).tofile(temp_file)
        return str(temp_file)

    def combine_temp_files_to_final(self, temp_files: List[str], output_file: Path):
        """Combine temporary binary files into final train.bin or val.bin"""
        print_status(f"🔗 Combining {len(temp_files)} temporary files into {output_file.name}...", "info")

        with open(output_file, 'wb') as outf:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    with open(temp_file, 'rb') as inf:
                        outf.write(inf.read())
                    # Clean up temp file
                    os.remove(temp_file)

    def process_single_dataset(self, config: DatasetConfig, dataset_number: int, total_datasets: int) -> Dict:
        """Process a single dataset completely before moving to next - WITH ON-THE-FLY TOKENIZATION"""
        print_status("="*80, "info")
        print_status(f"📦 DATASET {dataset_number}/{total_datasets}: {config.name.upper()}", "info")
        print_status(f"🎯 Target: {config.target_tokens/1e9:.1f}B tokens", "info")
        print_status(f"📡 Streaming: {'Yes' if config.use_streaming else 'No'}", "info")
        print_status(f"🔢 Min Tokens: {config.min_tokens}", "info")
        print_status(f"🎲 Val Split: {config.test_size*100:.2f}%", "info")
        print_status("="*80, "info")

        dataset_dir = self.output_dir / config.name
        dataset_dir.mkdir(exist_ok=True)

        # Create temp directory for intermediate files
        temp_dir = dataset_dir / "temp"
        temp_dir.mkdir(exist_ok=True)

        # Load dataset
        print_status(f"📥 Loading {config.name} from {config.dataset_id}...", "info")
        try:
            if config.use_streaming:
                dataset = load_dataset(
                    config.dataset_id,
                    config.config,
                    split="train",
                    streaming=True,
                    trust_remote_code=True
                )
            else:
                dataset = load_dataset(
                    config.dataset_id,
                    config.config,
                    split="train",
                    trust_remote_code=True
                )
            print_status(f"✓ Successfully loaded {config.name}", "success")
        except Exception as e:
            print_status(f"❌ Failed to load {config.name}: {e}", "error")
            return {"tokens": 0, "chunks": 0, "examples": [], "error": str(e)}

        # Processing variables
        tokens_written = 0
        chunks_written = 0
        items_processed = 0
        examples_collected = []

        # Token accumulation for binary files
        accumulated_tokens = []
        temp_files = []
        file_counter = 0
        max_tokens_in_memory = 50_000_000  # 50M tokens max in memory (~200MB)

        start_time = time.time()
        print_status(f"🔄 Processing {config.name} items with on-the-fly tokenization...", "info")

        if config.use_streaming:
            # Streaming processing with progress bar
            pbar = tqdm(desc=f"Processing {config.name}", unit="items",
                       bar_format='{l_bar}{bar}| {n_fmt} items [{elapsed}, {rate_fmt}] Tokens: {postfix}')

            for item in dataset:
                if tokens_written >= config.target_tokens:
                    break

                text = self.extract_text_from_item(item, config)
                if not text:
                    continue

                # Tokenize chunks directly
                token_chunks = chunk_and_tokenize_text(text, min_chunk_tokens=config.min_tokens)

                for chunk_tokens in token_chunks:
                    if len(chunk_tokens) > 0:
                        # Add EOT token between chunks
                        chunk_tokens_with_eot = chunk_tokens + [EOT_TOKEN]
                        accumulated_tokens.extend(chunk_tokens_with_eot)

                        tokens_written += len(chunk_tokens_with_eot)
                        chunks_written += 1

                        # Collect examples for display (decode back to text for examples)
                        if len(examples_collected) < 3:
                            try:
                                example_text = tokenizer.decode(chunk_tokens[:100])  # First 100 tokens
                                examples_collected.append(example_text + "..." if len(chunk_tokens) > 100 else example_text)
                            except:
                                pass

                        # Flush to temp file if we have too many tokens in memory
                        if len(accumulated_tokens) >= max_tokens_in_memory:
                            temp_file = self.flush_tokens_to_temp_file(accumulated_tokens, temp_dir, file_counter)
                            temp_files.append(temp_file)
                            file_counter += 1
                            accumulated_tokens = []  # Reset accumulator

                        if tokens_written >= config.target_tokens:
                            break

                items_processed += 1
                if items_processed % 100 == 0:  # Update every 100 items
                    pbar.update(100)
                    pbar.set_postfix_str(f"{tokens_written/1e6:.1f}M/{config.target_tokens/1e9:.1f}B")

            pbar.close()

        else:
            # Non-streaming processing
            total_items = len(dataset)
            pbar = tqdm(dataset, desc=f"Processing {config.name}", total=total_items,
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Tokens: {postfix}')

            for item in pbar:
                if config.target_tokens > 0 and tokens_written >= config.target_tokens:
                    break

                text = self.extract_text_from_item(item, config)
                if not text:
                    continue

                # Tokenize chunks directly
                token_chunks = chunk_and_tokenize_text(text, min_chunk_tokens=config.min_tokens)

                for chunk_tokens in token_chunks:
                    if len(chunk_tokens) > 0:
                        # Add EOT token between chunks
                        chunk_tokens_with_eot = chunk_tokens + [EOT_TOKEN]
                        accumulated_tokens.extend(chunk_tokens_with_eot)

                        tokens_written += len(chunk_tokens_with_eot)
                        chunks_written += 1

                        # Collect examples for display (decode back to text for examples)
                        if len(examples_collected) < 3:
                            try:
                                example_text = tokenizer.decode(chunk_tokens[:100])  # First 100 tokens
                                examples_collected.append(example_text + "..." if len(chunk_tokens) > 100 else example_text)
                            except:
                                pass

                        # Flush to temp file if we have too many tokens in memory
                        if len(accumulated_tokens) >= max_tokens_in_memory:
                            temp_file = self.flush_tokens_to_temp_file(accumulated_tokens, temp_dir, file_counter)
                            temp_files.append(temp_file)
                            file_counter += 1
                            accumulated_tokens = []  # Reset accumulator

                        if config.target_tokens > 0 and tokens_written >= config.target_tokens:
                            break

                items_processed += 1
                pbar.set_postfix_str(f"{tokens_written/1e6:.1f}M")

            pbar.close()

        # Flush any remaining tokens
        if accumulated_tokens:
            temp_file = self.flush_tokens_to_temp_file(accumulated_tokens, temp_dir, file_counter)
            temp_files.append(temp_file)

        processing_time = time.time() - start_time

        # Create train/val split and final binary files
        print_status(f"💾 Creating train/val split for {config.name}...", "info")

        # Calculate split
        val_size = int(tokens_written * config.test_size)
        train_size = tokens_written - val_size

        # For simplicity, we'll put first files in val and rest in train
        val_tokens_needed = val_size
        train_temp_files = []
        val_temp_files = []

        current_val_tokens = 0
        for temp_file in temp_files:
            if current_val_tokens < val_tokens_needed:
                val_temp_files.append(temp_file)
                # Estimate tokens in this file (rough)
                file_size = os.path.getsize(temp_file)
                estimated_tokens = file_size // 2  # 2 bytes per uint16 token
                current_val_tokens += estimated_tokens
            else:
                train_temp_files.append(temp_file)

        # Combine temp files into final train.bin and val.bin
        train_path = dataset_dir / "train.bin"
        val_path = dataset_dir / "val.bin"

        if val_temp_files:
            self.combine_temp_files_to_final(val_temp_files, val_path)
        if train_temp_files:
            self.combine_temp_files_to_final(train_temp_files, train_path)

        # Clean up temp directory
        try:
            temp_dir.rmdir()
        except:
            pass  # Directory might not be empty, that's ok

        # Save metadata
        metadata = {
            "dataset_name": config.name,
            "dataset_id": config.dataset_id,
            "config": config.config,
            "target_tokens": config.target_tokens,
            "actual_tokens": tokens_written,
            "train_tokens": train_size,
            "val_tokens": val_size,
            "chunks_written": chunks_written,
            "items_processed": items_processed,
            "processing_time_seconds": processing_time,
            "processing_time_human": f"{processing_time/60:.1f} minutes",
            "tokens_per_second": tokens_written / processing_time if processing_time > 0 else 0,
            "completed_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "use_streaming": config.use_streaming,
            "min_tokens": config.min_tokens,
            "test_size": config.test_size,
            "files_created": ["train.bin", "val.bin"]
        }

        with open(dataset_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Success summary
        print_status(f"✅ {config.name} COMPLETED!", "success")
        print_status(f"   📊 Tokens: {tokens_written/1e6:.1f}M / {config.target_tokens/1e9:.1f}B ({(tokens_written/config.target_tokens)*100:.1f}%)", "success")
        print_status(f"   📝 Chunks: {chunks_written:,}", "success")
        print_status(f"   🎯 Train: {train_size/1e6:.1f}M tokens → train.bin", "success")
        print_status(f"   🎲 Val: {val_size/1e6:.1f}M tokens → val.bin", "success")
        print_status(f"   ⏱️ Time: {processing_time/60:.1f} minutes", "success")
        print_status(f"   🚀 Speed: {(tokens_written/1e6)/(processing_time/60):.1f}M tokens/min", "success")

        return {
            "tokens": tokens_written,
            "train_tokens": train_size,
            "val_tokens": val_size,
            "chunks": chunks_written,
            "examples": examples_collected,
            "metadata": metadata,
            "items_processed": items_processed,
            "processing_time": processing_time
        }

    def wait_between_datasets(self, seconds: int = 10):
        """Wait between datasets to avoid rate limiting"""
        print_status(f"⏸️ Waiting {seconds} seconds before next dataset...", "warning")
        for i in range(seconds, 0, -1):
            print(f"\rCountdown: {i} seconds", end="", flush=True)
            time.sleep(1)
        print("\r" + " " * 20 + "\r", end="")  # Clear countdown

    def generate_examples_file(self):
        """Generate examples.txt with samples from each dataset"""
        examples_file = self.output_dir / "examples.txt"

        with open(examples_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("EXAMPLES FROM 60B TOKEN DATASET - TOKENIZED BINARY FORMAT\n")
            f.write("="*80 + "\n\n")

            for dataset_name, stats in self.dataset_stats.items():
                f.write(f"=== {dataset_name.upper()} ===\n")
                f.write(f"Tokens: {stats['tokens']/1e6:.1f}M | Chunks: {stats['chunks']:,} | Time: {stats.get('processing_time', 0)/60:.1f}min\n")
                f.write(f"Train: {stats.get('train_tokens', 0)/1e6:.1f}M | Val: {stats.get('val_tokens', 0)/1e6:.1f}M\n")
                f.write("-" * 60 + "\n")

                examples = stats.get('examples', [])
                for i, example in enumerate(examples[:3], 1):
                    f.write(f"Example {i}:\n{example}\n\n")

                f.write("="*80 + "\n\n")

        print_status(f"✓ Generated examples.txt with samples from all {len(self.dataset_stats)} datasets", "success")

    def generate_summary_report(self):
        """Generate final summary report"""
        total_tokens = sum(stats['tokens'] for stats in self.dataset_stats.values())
        total_train_tokens = sum(stats.get('train_tokens', 0) for stats in self.dataset_stats.values())
        total_val_tokens = sum(stats.get('val_tokens', 0) for stats in self.dataset_stats.values())
        total_chunks = sum(stats['chunks'] for stats in self.dataset_stats.values())
        total_time = sum(stats.get('processing_time', 0) for stats in self.dataset_stats.values())

        report_file = self.output_dir / "summary_report.txt"

        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("60B TOKEN DATASET PREPARATION SUMMARY - TOKENIZED BINARY FORMAT\n")
            f.write("="*80 + "\n\n")

            f.write(f"TOTAL TOKENS PROCESSED: {total_tokens/1e9:.2f}B\n")
            f.write(f"TRAIN TOKENS: {total_train_tokens/1e9:.2f}B\n")
            f.write(f"VALIDATION TOKENS: {total_val_tokens/1e9:.2f}B\n")
            f.write(f"TOTAL CHUNKS CREATED: {total_chunks:,}\n")
            f.write(f"TARGET ACHIEVEMENT: {(total_tokens/60e9)*100:.1f}%\n")
            f.write(f"TOTAL PROCESSING TIME: {total_time/3600:.1f} hours\n")
            f.write(f"AVERAGE SPEED: {(total_tokens/1e6)/(total_time/60):.1f}M tokens/minute\n\n")

            f.write("DATASET PROCESSING ORDER & RESULTS:\n")
            f.write("-" * 70 + "\n")

            for i, (dataset_name, stats) in enumerate(self.dataset_stats.items(), 1):
                tokens = stats['tokens']
                train_tokens = stats.get('train_tokens', 0)
                val_tokens = stats.get('val_tokens', 0)
                chunks = stats['chunks']
                proc_time = stats.get('processing_time', 0)
                f.write(f"{i:2d}. {dataset_name:15s}: {tokens/1e9:6.2f}B total ({train_tokens/1e6:6.1f}M train, {val_tokens/1e3:6.1f}K val) [{proc_time/60:5.1f}min]\n")

            # Category breakdown
            f.write(f"\nCATEGORY BREAKDOWN:\n")
            f.write("-" * 30 + "\n")

            categories = {
                "Core Web & Text": ["wikipedia", "openwebtext", "c4", "fineweb"],
                "Technical": ["arxiv", "openwebmath", "codeparrot"]
            }

            for category, datasets in categories.items():
                f.write(f"\n{category}:\n")
                category_tokens = 0
                for dataset in datasets:
                    if dataset in self.dataset_stats:
                        tokens = self.dataset_stats[dataset]['tokens']
                        chunks = self.dataset_stats[dataset]['chunks']
                        category_tokens += tokens
                        f.write(f"  {dataset}: {tokens/1e9:.2f}B tokens ({chunks:,} chunks)\n")
                f.write(f"  Category Total: {category_tokens/1e9:.2f}B tokens\n")

            f.write(f"\nFORMAT: Tokenized binary files (train.bin + val.bin per dataset)\n")
            f.write(f"TOKENIZER: GPT-2 encoding (vocab_size: {tokenizer.n_vocab})\n")
            f.write(f"Processing completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Authentication: HuggingFace token used\n")
            f.write(f"Processing strategy: Sequential with on-the-fly tokenization\n")

        print_status(f"✓ Generated summary report: {total_tokens/1e9:.2f}B tokens total", "success")

        # Also print to console
        print_status("="*80, "info")
        print_status("🎉 ALL DATASETS PROCESSED WITH TOKENIZATION! 🎉", "success")
        print_status(f"📊 Total tokens: {total_tokens/1e9:.2f}B / 60B ({(total_tokens/60e9)*100:.1f}%)", "info")
        print_status(f"🎯 Train: {total_train_tokens/1e9:.2f}B tokens", "info")
        print_status(f"🎲 Val: {total_val_tokens/1e6:.1f}M tokens", "info")
        print_status(f"⏱️ Total time: {total_time/3600:.1f} hours", "info")
        print_status(f"📁 Output directory: {self.output_dir}", "info")
        print_status(f"📝 Summary: {report_file}", "info")
        print_status(f"🔍 Examples: {self.output_dir}/examples.txt", "info")
        print_status(f"🎯 Ready for training! Each dataset has train.bin + val.bin", "success")
        print_status("="*80, "info")

    def run(self):
        """Run the complete dataset preparation pipeline - ONE DATASET AT A TIME WITH TOKENIZATION"""
        print_status("🚀 Starting 60B Token Dataset Preparation - WITH ON-THE-FLY TOKENIZATION", "info")
        print_status(f"📂 Output directory: {self.output_dir}", "info")
        print_status(f"🎯 Target: 60B tokens across {len(self.dataset_configs)} datasets", "info")
        print_status(f"🔐 Authentication: HuggingFace token configured", "info")
        print_status(f"📋 Processing order: One dataset at a time (sequential)", "info")
        print_status(f"🔢 Output format: train.bin + val.bin per dataset", "info")
        print_status("="*80, "info")

        overall_start_time = time.time()

        # Process each dataset ONE AT A TIME
        for i, config in enumerate(self.dataset_configs, 1):
            try:
                # Wait between datasets to avoid rate limiting (except first one)
                if i > 1:
                    self.wait_between_datasets(10)

                # Process this single dataset completely
                stats = self.process_single_dataset(config, i, len(self.dataset_configs))

                if stats['tokens'] > 0:  # Only count successful datasets
                    self.dataset_stats[config.name] = stats
                    self.total_tokens_processed += stats['tokens']

                    print_status(f"📈 RUNNING TOTAL: {self.total_tokens_processed/1e9:.2f}B tokens ({(self.total_tokens_processed/60e9)*100:.1f}% of 60B)", "progress")
                else:
                    print_status(f"⚠️ {config.name} produced 0 tokens - skipping", "warning")

                print()  # Blank line between datasets

            except Exception as e:
                print_status(f"❌ Error processing {config.name}: {e}", "error")
                print_status("🔄 Continuing with next dataset...", "warning")
                continue

        # Generate final outputs
        print_status("📝 Generating final reports...", "info")
        self.generate_examples_file()
        self.generate_summary_report()

        overall_elapsed_time = time.time() - overall_start_time
        print_status(f"⏱️ TOTAL PROCESSING TIME: {overall_elapsed_time/3600:.1f} hours", "info")
        print_status(f"🚀 AVERAGE PROCESSING SPEED: {(self.total_tokens_processed/1e6)/(overall_elapsed_time/60):.1f}M tokens/minute", "info")

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Prepare 60B token dataset for CosmicFish training - Sequential Processing with Tokenization")
    parser.add_argument("--output_dir", type=str, default="data",
                       help="Output directory for processed datasets")
    parser.add_argument("--test_mode", action="store_true",
                       help="Run in test mode with reduced token targets")
    parser.add_argument("--start_from", type=str, default=None,
                       help="Start processing from specific dataset (useful for resuming)")

    args = parser.parse_args()

    # Initialize processor
    processor = DatasetProcessor(args.output_dir)

    # Test mode: reduce all targets by 1000x for quick testing
    if args.test_mode:
        print_status("🧪 Running in TEST MODE - reduced targets", "warning")
        for config in processor.dataset_configs:
            config.target_tokens = max(1_000_000, config.target_tokens // 1000)  # Min 1M tokens

    # Resume from specific dataset if requested
    if args.start_from:
        dataset_names = [config.name for config in processor.dataset_configs]
        if args.start_from in dataset_names:
            start_index = dataset_names.index(args.start_from)
            processor.dataset_configs = processor.dataset_configs[start_index:]
            print_status(f"🔄 Resuming from dataset: {args.start_from}", "warning")
        else:
            print_status(f"❌ Dataset '{args.start_from}' not found. Available: {dataset_names}", "error")
            sys.exit(1)

    # Run the sequential processing with tokenization
    processor.run()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_status("\n⏹️ Processing interrupted by user", "warning")
        print_status("💡 You can resume with: --start_from DATASET_NAME", "info")
        sys.exit(1)
    except Exception as e:
        print_status(f"\n❌ Fatal error: {e}", "error")
        import traceback
        traceback.print_exc()
        sys.exit(1)
