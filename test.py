"""
Test script for evaluating prompts with the pretrained CosmicFish model.
This allows for quick testing of different prompts and generation parameters.
"""

import argparse
import os
import sys
import time
import torch
import tiktoken
import re
import textwrap
from termcolor import colored

from model import CosmicFish, CosmicConfig


def clean_text(text):
    """Clean generated text by removing unwanted patterns and normalizing spacing."""
    # Fix encoding issues
    text = text.replace('�', "'")

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    # Normalize punctuation spacing
    text = re.sub(r'\s*([.,!?;:])\s*', r'\1 ', text)

    # Remove leading/trailing spaces
    text = text.strip()

    return text


def get_repetition_penalty_logits(input_ids, logits, penalty=1.2):
    """Apply repetition penalty to logits based on input_ids"""
    for input_ids_slice in input_ids:
        for token_id in set(input_ids_slice.tolist()):
            logits[:, token_id] /= penalty
    return logits


def detect_main_topic(prompt, tokenizer):
    """Extract potential main topics from the prompt to maintain focus"""
    # Simple keyword extraction
    words = prompt.lower().split()
    stopwords = {'a', 'an', 'the', 'in', 'on', 'at', 'by', 'for', 'with', 'about', 'to', 'and', 'or', 'of', 'is', 'are',
                 'was', 'were'}

    # Extract potential keywords (non-stopwords)
    keywords = [word for word in words if word not in stopwords and len(word) > 3]

    # If no keywords found, use the last noun or important word
    if not keywords and words:
        keywords = [words[-1]]

    return keywords


def extract_ngrams(text, n=3):
    """Extract n-grams from text for repetition detection"""
    words = text.split()
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


def apply_diversity_penalty(logits, recent_tokens, tokenizer, penalty=1.5):
    """Apply diversity penalty to avoid repetitive n-grams"""
    # Check recent tokens for repetition patterns
    if len(recent_tokens) > 10:
        # Look for repeating token pairs
        pairs = [(recent_tokens[i], recent_tokens[i + 1]) for i in range(len(recent_tokens) - 1)]
        pair_counts = {}
        for pair in pairs:
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

        # Penalize frequent pairs
        for pair, count in pair_counts.items():
            if count > 2:  # If pair appears more than twice
                # Find tokens that would extend this repetitive pair
                recent_context = (recent_tokens[-2], recent_tokens[-1])
                if recent_context == pair:
                    # Apply penalty to the token that would continue the pattern
                    third_token = recent_tokens[pairs.index(pair) + 2]
                    logits[:, third_token] /= penalty

    return logits


def apply_topic_focus(logits, topic_tokens, boost_factor=1.3):
    """Boost tokens related to the main topic to maintain focus"""
    # Boost topic tokens to keep generation on topic
    for token in topic_tokens:
        logits[:, token] *= boost_factor

    return logits


def apply_local_ngram_penalty(generated_text, logits, tokenizer, window_size=100, penalty=1.3):
    """Penalize recently used n-grams to prevent local repetition"""
    # Get recent text
    recent_text = generated_text[-window_size:] if len(generated_text) > window_size else generated_text

    # Extract 2-grams and 3-grams from recent text
    words = recent_text.split()
    if len(words) >= 3:
        two_grams = [' '.join(words[i:i + 2]) for i in range(len(words) - 1)]
        three_grams = [' '.join(words[i:i + 3]) for i in range(len(words) - 2)]

        # Penalize tokens that would create repeated n-grams
        recent_two_gram = ' '.join(words[-2:]) if len(words) >= 2 else ''

        for gram in two_grams:
            if gram == recent_two_gram:
                # Find tokens that might continue this pattern
                for token_id in range(logits.shape[1]):
                    token_text = tokenizer.decode([token_id])
                    if token_text.strip() and (recent_two_gram + ' ' + token_text.strip()) in three_grams:
                        logits[:, token_id] /= penalty

    return logits


def generate_text(model, tokenizer, prompt, device, args):
    """Generate text completion using the model with the specified parameters"""

    # Clean prompt
    cleaned_prompt = clean_text(prompt)
    tokens = tokenizer.encode(cleaned_prompt)
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    # Extract main topics for focus maintenance
    topic_keywords = detect_main_topic(cleaned_prompt, tokenizer)
    topic_tokens = []
    for keyword in topic_keywords:
        topic_tokens.extend(tokenizer.encode(keyword))

    # Setup for generation
    generated_text = cleaned_prompt
    recent_tokens = []
    used_ngrams = set()

    # Generation variables
    focus_start = True  # Strong focus at start, gradually relaxing
    prev_bigrams = []  # For repetition detection
    prev_output_len = 0  # For tracking progress

    print(f"\n{('=' * 80)}")
    print(f"Generating with:")
    print(f"  Temperature: {args.temperature}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Top-k: {args.top_k}")
    print(f"  Top-p: {args.top_p}")
    print(f"{('=' * 80)}\n")

    print(colored("PROMPT:", "green"))
    for line in textwrap.wrap(cleaned_prompt, width=80):
        print(line)
    print("\n" + "-" * 80 + "\n")

    print(colored("GENERATION:", "blue"))
    print(cleaned_prompt, end="", flush=True)

    # Track generation start time
    start_time = time.time()

    # Check if model uses rotary embeddings
    has_rotary = hasattr(model.config, 'use_rotary') and model.config.use_rotary

    # IMPORTANT FIX: Get freqs_cis from model if it exists, but pass it correctly during generation
    if has_rotary and hasattr(model, 'freqs_cis'):
        freqs_cis = model.freqs_cis.to(device)
    else:
        freqs_cis = None

    with torch.no_grad():
        for _ in range(args.max_tokens):
            # Get context window
            if input_ids.size(1) > model.config.block_size:
                context = input_ids[:, -model.config.block_size:]
            else:
                context = input_ids

            # Get predictions - FIXED: don't pass freqs_cis directly to model.forward()
            logits, _ = model(context)  # We don't pass freqs_cis here, model handles it internally

            logits = logits[:, -1, :] / args.temperature

            # Apply repetition penalty
            logits = get_repetition_penalty_logits(context, logits, 1.2)

            # Apply topic focus (stronger at the beginning, gradually reducing)
            if focus_start and topic_tokens:
                current_focus = 1.2 * (1 - len(recent_tokens) / (args.max_tokens * 2))
                if current_focus > 1.0:
                    logits = apply_topic_focus(logits, topic_tokens, current_focus)

            # Apply diversity penalty
            logits = apply_diversity_penalty(logits, recent_tokens, tokenizer)

            # Apply local n-gram penalty to avoid repeating phrases
            logits = apply_local_ngram_penalty(generated_text, logits, tokenizer)

            # Special handling for repeating words and stutters
            if len(recent_tokens) >= 2:
                # Check for repeating words (same token twice in a row)
                if recent_tokens[-1] == recent_tokens[-2]:
                    logits[:, recent_tokens[-1]] /= 2.0  # Reduce probability of three repeats

                # Check for alternating repetitions (e.g., "the of the of")
                if len(recent_tokens) >= 4 and recent_tokens[-1] == recent_tokens[-3] and recent_tokens[-2] == \
                        recent_tokens[-4]:
                    logits[:, recent_tokens[-1]] /= 2.0
                    logits[:, recent_tokens[-2]] /= 2.0

            # Apply top-k
            if args.top_k > 0:
                values, _ = torch.topk(logits, min(args.top_k, logits.shape[-1]))
                logits[logits < values[:, [-1]]] = float('-inf')

            # Apply top-p (nucleus) sampling
            if args.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > args.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample next token
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Update tracking variables
            recent_tokens.append(next_token.item())
            if len(recent_tokens) > 50:  # Keep a reasonable history
                recent_tokens.pop(0)

            # Add token to context
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Decode token
            new_text = tokenizer.decode([next_token.item()])

            # Check for end of text token or problematic loops
            if next_token.item() == 50256 or (len(recent_tokens) > 20 and len(set(recent_tokens[-10:])) < 3):
                break

            # Clean and add new text
            cleaned_text = clean_text(new_text)

            # Add space if needed
            if cleaned_text and not (
                    cleaned_text.startswith(('.', ',', '!', '?', ';', ':')) or generated_text.endswith(' ')):
                cleaned_text = ' ' + cleaned_text

            print(cleaned_text, end="", flush=True)
            generated_text += cleaned_text

            # Adaptive focus adjustment based on progress
            if len(generated_text) - prev_output_len > 50:
                # Every 50 characters, relax focus slightly
                focus_start = False
                prev_output_len = len(generated_text)

                # Update bigrams for repetition detection
                words = generated_text.split()
                if len(words) >= 2:
                    curr_bigrams = [' '.join(words[i:i + 2]) for i in range(len(words) - 1)]
                    # Count repeating bigrams
                    bigram_counts = {}
                    for bg in curr_bigrams:
                        bigram_counts[bg] = bigram_counts.get(bg, 0) + 1

                    # If we detect too much repetition, break
                    if any(count > 3 for count in bigram_counts.values()):
                        if len(generated_text) > len(cleaned_prompt) + 100:  # Only if we've generated enough text
                            break

    # Final cleaning
    final_text = clean_text(generated_text)

    # Calculate generation time and token stats
    end_time = time.time()
    generation_time = end_time - start_time
    total_tokens = len(input_ids[0]) - len(tokens)
    tokens_per_sec = total_tokens / generation_time if generation_time > 0 else 0

    # Print generation stats
    print("\n\n" + "-" * 80)
    print(f"Generated {total_tokens} tokens in {generation_time:.2f}s ({tokens_per_sec:.2f} tokens/sec)")
    print("=" * 80)

    return final_text


def load_model(model_path, device):
    """Load the model from checkpoint"""
    print(f"Loading model from {model_path}...")

    try:
        # checkpoint = torch.load(model_path, map_location=device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # Extract config from checkpoint
        if 'cosmicconf' in checkpoint:
            config = checkpoint['cosmicconf']
        elif 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # Try to extract configuration parameters from state dict
            model_args = {
                'n_layer': 10,
                'n_head': 16,
                'n_embd': 640,
                'block_size': 512,
                'bias': True,
                'vocab_size': 50257,
                'dropout': 0,  # Use 0 for evaluation
                'use_rotary': True,
                'use_swiglu': True,
                'use_gqa': True,
                'n_query_groups': 4
            }
            config = CosmicConfig(**model_args)

        # Create the model
        model = CosmicFish(config)

        # Load state dict - FIXED: Handle torch.compile _orig_mod prefixes
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # Remove any 'module.' prefix (from DDP) or '_orig_mod.' prefix (from torch.compile)
            state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
        elif 'model' in checkpoint:
            # Handle different checkpoint formats
            state_dict = checkpoint['model']
            # Remove any 'module.' prefix (from DDP) or '_orig_mod.' prefix (from torch.compile)
            state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)

        model.to(device)
        model.eval()  # Set to evaluation mode

        print(f"Model loaded successfully: {model.get_num_params() / 1e6:.2f}M parameters")
        return model, config

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)


def get_device():
    """Determine the device to use"""
    if torch.cuda.is_available():
        print("Using CUDA acceleration")
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("Using MPS acceleration")
        return 'mps'
    print("Using CPU")
    return 'cpu'


def load_prompt_from_file(file_path):
    """Load prompt from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error loading prompt from file: {str(e)}")
        sys.exit(1)


def input_ready():
    """Check if input is available without blocking."""
    import select
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])


def main():
    parser = argparse.ArgumentParser(description="Test prompts with a pretrained CosmicFish model")

    # Model parameters
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pretrained model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda or cpu)")

    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Temperature for sampling (default: 0.8)")
    parser.add_argument("--max_tokens", type=int, default=200,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling (0 to disable)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p (nucleus) sampling (1.0 to disable)")

    # Prompt handling
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt to complete")
    parser.add_argument("--file", type=str, default=None,
                        help="Load prompt from file")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode to test multiple prompts")
    parser.add_argument("--stop_key", type=str, default="q",
                        help="Key to press to stop generation (default: q)")

    args = parser.parse_args()

    # Configure device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU")
        device = "cpu"

    # Load the model
    model, config = load_model(args.model_path, device)

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Handle prompt
    if args.interactive:
        print("\nInteractive mode. Enter prompts and press Enter to generate.")
        print("Enter 'quit', 'exit', or Ctrl+C to exit.\n")

        while True:
            try:
                prompt = input(colored("Enter prompt: ", "yellow"))
                if prompt.lower() in ['quit', 'exit']:
                    break
                if not prompt.strip():
                    continue

                # Generate text for this prompt
                generate_text(model, tokenizer, prompt, device, args)
                print("\n")

            except KeyboardInterrupt:
                print("\nExiting interactive mode.")
                break
    else:
        # Get prompt from arguments or file
        if args.file:
            prompt = load_prompt_from_file(args.file)
        elif args.prompt:
            prompt = args.prompt
        else:
            print("Please provide a prompt with --prompt or --file, or use --interactive mode.")
            return

        # Generate text for the prompt
        generate_text(model, tokenizer, prompt, device, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)