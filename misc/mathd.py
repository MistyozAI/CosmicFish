#!/usr/bin/env python3
"""
Math Dataset Preparation Script for CosmicFish Fine-tuning
Prepares GSM8K and MetaMathQA datasets in instruction-following format (similar to Alpaca)
Creates train.bin and val.bin files for fine-tuning on math reasoning.
"""

import os
import sys
import json
import random
import numpy as np
from pathlib import Path
from datasets import load_dataset
from tqdm.auto import tqdm
import tiktoken
from termcolor import colored

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
EOT_TOKEN = tokenizer.eot_token


def print_status(message: str, status: str = "info") -> None:
    """Print colored status messages"""
    colors = {"info": "cyan", "success": "green", "error": "red", "warning": "yellow"}
    print(colored(f"[{status.upper()}] {message}", colors.get(status, "white")))


class MathDatasetProcessor:
    """Processor for math datasets in instruction-following format"""

    def __init__(self, output_dir: str = "data/math"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Instruction template (similar to Alpaca format)
        self.instruction_template = """Below is a math problem. Solve it step by step.

### Problem:
{problem}

### Solution:
{solution}<|endoftext|>"""

        self.stats = {
            'total_examples': 0,
            'total_tokens': 0,
            'train_tokens': 0,
            'val_tokens': 0,
            'train_examples': 0,
            'val_examples': 0
        }

    def format_gsm8k_example(self, example):
        """Format a GSM8K example into instruction-following format"""
        problem = example['question'].strip()
        solution = example['answer'].strip()

        # Create formatted text
        text = self.instruction_template.format(
            problem=problem,
            solution=solution
        )

        return text

    def format_metamath_example(self, example):
        """Format a MetaMathQA example into instruction-following format"""
        # MetaMathQA has 'query' and 'response' fields
        problem = example['query'].strip()
        solution = example['response'].strip()

        # Create formatted text
        text = self.instruction_template.format(
            problem=problem,
            solution=solution
        )

        return text

    def process_gsm8k(self, max_examples=None):
        """Process GSM8K dataset"""
        print_status("📚 Loading GSM8K dataset...", "info")

        try:
            # Load the dataset
            dataset = load_dataset("openai/gsm8k", "main", trust_remote_code=True)
            train_data = dataset['train']
            test_data = dataset['test']

            print_status(f"✅ Loaded GSM8K: {len(train_data)} train, {len(test_data)} test", "success")

            # Combine train and test, then split for fine-tuning
            all_examples = []

            # Process training examples
            print_status("Processing GSM8K training examples...", "info")
            for example in tqdm(train_data, desc="GSM8K Train"):
                formatted_text = self.format_gsm8k_example(example)
                all_examples.append(formatted_text)

                if max_examples and len(all_examples) >= max_examples:
                    break

            # Process test examples (use as additional training data)
            if not max_examples or len(all_examples) < max_examples:
                print_status("Processing GSM8K test examples...", "info")
                remaining = max_examples - len(all_examples) if max_examples else None
                for i, example in enumerate(tqdm(test_data, desc="GSM8K Test")):
                    if remaining and i >= remaining:
                        break
                    formatted_text = self.format_gsm8k_example(example)
                    all_examples.append(formatted_text)

            print_status(f"✅ Processed {len(all_examples)} GSM8K examples", "success")
            return all_examples

        except Exception as e:
            print_status(f"❌ Error processing GSM8K: {e}", "error")
            return []

    def process_metamath(self, max_examples=50000):
        """Process MetaMathQA dataset (sampling subset)"""
        print_status("📚 Loading MetaMathQA dataset...", "info")

        try:
            # Load the dataset (it's large, so we'll sample)
            dataset = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)

            all_examples = []
            print_status(f"Processing {max_examples} MetaMathQA examples...", "info")

            # Stream and process examples
            for i, example in enumerate(tqdm(dataset, total=max_examples, desc="MetaMath")):
                if i >= max_examples:
                    break

                formatted_text = self.format_metamath_example(example)
                all_examples.append(formatted_text)

            print_status(f"✅ Processed {len(all_examples)} MetaMathQA examples", "success")
            return all_examples

        except Exception as e:
            print_status(f"❌ Error processing MetaMathQA: {e}", "error")
            return []

    def create_binary_files(self, examples, val_split=0.05):
        """Create train.bin and val.bin files from examples"""
        print_status(f"📊 Creating binary files from {len(examples)} examples...", "info")

        # Shuffle examples
        random.shuffle(examples)

        # Split into train and validation
        val_size = int(len(examples) * val_split)
        train_examples = examples[val_size:]
        val_examples = examples[:val_size]

        print_status(f"Split: {len(train_examples)} train, {len(val_examples)} val", "info")

        # Tokenize and create train.bin
        print_status("Tokenizing training examples...", "info")
        train_tokens = []
        for text in tqdm(train_examples, desc="Train tokenization"):
            tokens = tokenizer.encode(text)
            train_tokens.extend(tokens)

        # Tokenize and create val.bin
        print_status("Tokenizing validation examples...", "info")
        val_tokens = []
        for text in tqdm(val_examples, desc="Val tokenization"):
            tokens = tokenizer.encode(text)
            val_tokens.extend(tokens)

        # Convert to numpy arrays
        train_arr = np.array(train_tokens, dtype=np.uint16)
        val_arr = np.array(val_tokens, dtype=np.uint16)

        # Save binary files
        train_path = self.output_dir / "train.bin"
        val_path = self.output_dir / "val.bin"

        train_arr.tofile(train_path)
        val_arr.tofile(val_path)

        # Update stats
        self.stats['train_examples'] = len(train_examples)
        self.stats['val_examples'] = len(val_examples)
        self.stats['total_examples'] = len(examples)
        self.stats['train_tokens'] = len(train_tokens)
        self.stats['val_tokens'] = len(val_tokens)
        self.stats['total_tokens'] = len(train_tokens) + len(val_tokens)

        print_status(f"✅ Saved train.bin: {len(train_tokens):,} tokens ({len(train_tokens) / 1e6:.2f}M)", "success")
        print_status(f"✅ Saved val.bin: {len(val_tokens):,} tokens ({len(val_tokens) / 1e6:.2f}M)", "success")

        # Save some example texts for reference
        self.save_examples(train_examples[:5], val_examples[:5])

    def save_examples(self, train_samples, val_samples):
        """Save example texts for inspection"""
        examples_file = self.output_dir / "examples.txt"

        with open(examples_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("MATH DATASET EXAMPLES (GSM8K + MetaMathQA)\n")
            f.write("=" * 80 + "\n\n")

            f.write("=== TRAINING EXAMPLES ===\n\n")
            for i, example in enumerate(train_samples, 1):
                f.write(f"--- Example {i} ---\n")
                f.write(example)
                f.write("\n\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("=== VALIDATION EXAMPLES ===\n\n")
            for i, example in enumerate(val_samples, 1):
                f.write(f"--- Example {i} ---\n")
                f.write(example)
                f.write("\n\n")

        print_status(f"✅ Saved examples to {examples_file}", "success")

    def generate_summary(self):
        """Generate summary report"""
        summary_file = self.output_dir / "summary.txt"

        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MATH DATASET PREPARATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Total Examples: {self.stats['total_examples']:,}\n")
            f.write(f"Train Examples: {self.stats['train_examples']:,}\n")
            f.write(f"Val Examples: {self.stats['val_examples']:,}\n\n")

            f.write(f"Total Tokens: {self.stats['total_tokens']:,} ({self.stats['total_tokens'] / 1e6:.2f}M)\n")
            f.write(f"Train Tokens: {self.stats['train_tokens']:,} ({self.stats['train_tokens'] / 1e6:.2f}M)\n")
            f.write(f"Val Tokens: {self.stats['val_tokens']:,} ({self.stats['val_tokens'] / 1e6:.2f}M)\n\n")

            f.write(f"Datasets: GSM8K + MetaMathQA\n")
            f.write(f"Format: Instruction-following (similar to Alpaca)\n")
            f.write(f"Output: {self.output_dir}\n")

        print_status(f"✅ Summary saved to {summary_file}", "success")

    def run(self, use_gsm8k=True, use_metamath=True, gsm8k_max=None, metamath_max=50000):
        """Run the complete preparation pipeline"""
        print_status("🚀 Starting Math Dataset Preparation", "info")
        print_status(f"📂 Output directory: {self.output_dir}", "info")
        print_status("=" * 80, "info")

        all_examples = []

        # Process GSM8K
        if use_gsm8k:
            gsm8k_examples = self.process_gsm8k(max_examples=gsm8k_max)
            all_examples.extend(gsm8k_examples)
            print_status(f"📊 GSM8K contribution: {len(gsm8k_examples):,} examples", "info")

        # Process MetaMathQA
        if use_metamath:
            metamath_examples = self.process_metamath(max_examples=metamath_max)
            all_examples.extend(metamath_examples)
            print_status(f"📊 MetaMath contribution: {len(metamath_examples):,} examples", "info")

        if not all_examples:
            print_status("❌ No examples processed!", "error")
            return

        print_status(f"📊 Total examples collected: {len(all_examples):,}", "success")

        # Create binary files
        self.create_binary_files(all_examples)

        # Generate summary
        self.generate_summary()

        # Final report
        print_status("=" * 80, "info")
        print_status("🎉 MATH DATASET PREPARATION COMPLETE! 🎉", "success")
        print_status(f"📊 Total: {self.stats['total_tokens'] / 1e6:.2f}M tokens", "info")
        print_status(
            f"🎯 Train: {self.stats['train_tokens'] / 1e6:.2f}M tokens ({self.stats['train_examples']:,} examples)",
            "info")
        print_status(f"🎲 Val: {self.stats['val_tokens'] / 1e3:.1f}K tokens ({self.stats['val_examples']:,} examples)",
                     "info")
        print_status(f"📁 Files: {self.output_dir}/train.bin, {self.output_dir}/val.bin", "info")
        print_status(f"📄 Examples: {self.output_dir}/examples.txt", "info")
        print_status("🎯 Ready for fine-tuning on your Alpaca model!", "success")
        print_status("=" * 80, "info")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Prepare math datasets for CosmicFish fine-tuning")
    parser.add_argument("--output_dir", type=str, default="data/math",
                        help="Output directory for processed datasets")
    parser.add_argument("--gsm8k_max", type=int, default=None,
                        help="Maximum GSM8K examples (None = all ~8.5K)")
    parser.add_argument("--metamath_max", type=int, default=50000,
                        help="Maximum MetaMathQA examples to sample")
    parser.add_argument("--no_gsm8k", action="store_true",
                        help="Skip GSM8K dataset")
    parser.add_argument("--no_metamath", action="store_true",
                        help="Skip MetaMathQA dataset")
    parser.add_argument("--val_split", type=float, default=0.05,
                        help="Validation split ratio (default: 0.05)")

    args = parser.parse_args()

    # Initialize processor
    processor = MathDatasetProcessor(args.output_dir)

    # Run preparation
    processor.run(
        use_gsm8k=not args.no_gsm8k,
        use_metamath=not args.no_metamath,
        gsm8k_max=args.gsm8k_max,
        metamath_max=args.metamath_max
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_status("\n⏸️ Processing interrupted by user", "warning")
        sys.exit(1)
    except Exception as e:
        print_status(f"\n❌ Fatal error: {e}", "error")
        import traceback

        traceback.print_exc()
        sys.exit(1)