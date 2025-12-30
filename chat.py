"""
Chat interface for interacting with the CosmicFish fine-tuned model.
Includes enhanced repetition penalty, live generation, and improved text quality.
"""

import os
import sys
import time
import argparse
import torch
import numpy as np
from termcolor import colored
import logging
import readline  # Enables arrow key history in terminal input
import re
import textwrap
import random
from collections import defaultdict
import tiktoken  # Use tiktoken for CosmicFish

from model import CosmicFish, CosmicConfig

from torch.serialization import add_safe_globals

add_safe_globals([CosmicConfig])

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Default prompt template that exactly matches what was used in training
DEFAULT_PROMPT_TEMPLATE = "Below is a conversation between a helpful AI assistant and a human. The assistant is knowledgeable, friendly, and provides detailed and accurate responses.\n\n"


class RepetitionPenaltyLogitsProcessor:
    """Apply repetition penalty to prevent repeating tokens."""

    def __init__(self, penalty=1.2):
        self.penalty = penalty

    def __call__(self, input_ids, scores):
        """Apply repetition penalty to logits where input_ids is already seen."""
        score = torch.gather(scores, 1, input_ids)
        # If score > 0, penalize by dividing; if score < 0, penalize by multiplying
        score = torch.where(score > 0, score / self.penalty, score * self.penalty)
        scores.scatter_(1, input_ids, score)
        return scores


class ChatSession:
    """Manages a chat session with the model."""

    def __init__(self, model, tokenizer, config):
        """Initialize chat session with model and configuration."""
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = config.device
        self.history = []
        self.history_tokens = []
        self.max_history_tokens = config.max_history_tokens
        self.prompt_template = config.prompt_template
        self.human_prefix = config.human_prefix
        self.assistant_prefix = config.assistant_prefix
        self.end_of_turn = config.end_of_turn
        self.block_size = config.block_size
        self.debug_mode = config.debug_mode
        self.repetition_penalty = config.repetition_penalty
        self.min_tokens_to_generate = config.min_tokens_to_generate
        # Maximum number of retries for problematic generations
        self.max_retries = 20

        # Check if model uses rotary embeddings
        self.use_rotary = getattr(model.config, 'use_rotary', False)

        self.fallback_responses = [
            "I'd be happy to help with that. Could you provide more details about what specific information you're looking for?",
            "That's a topic I can provide information about. What specific aspects would you like to know?",
            "I understand your question. I can share factual information on this topic if you could specify what aspects you're interested in.",
            "I can help with your question. To give you the most relevant information, could you clarify what specific details you're looking for?",
            "I'd be glad to address your question. To provide the most helpful response, could you specify what particular aspects of this topic interest you?"
        ]

        # Special failure message after all retries
        self.generation_failure_message = "I'm sorry, but I'm having difficulty generating a response to that prompt. Could you try rephrasing your question or asking something else?"

        # For token counting
        self.total_prompt_tokens = 0
        self.total_generated_tokens = 0

        # End markers for live generation
        self.end_markers = [
            f"{self.human_prefix}",
            "Human:",
            "\nHuman:",
            "\nH:",
            "H:",
            "<|endoftext|>",
            "Below is a conversation",
            "\nA:",
            "A:",
            "</s>",
            "User:",
            "\nUser:"
        ]

        # Print welcome message
        if config.display_welcome:
            self._print_welcome_message()

    def _print_welcome_message(self):
        """Print a welcome message to the user."""
        welcome_text = f"""
{'=' * 80}
Welcome to CosmicFish chat interface

This is a {self.model.get_num_params() / 1e6:.1f}M parameter model.
CosmicFish is an efficient LLM with an advanced architecture.

Type your prompts and CosmicFish will respond.

Special commands:
- /help: Show this help message
- /clear: Clear the conversation history
- /exit or /quit: Exit the chat
- /stats: Show token usage statistics
- /save [filename]: Save the conversation
- /load [filename]: Load a conversation
- /temp [value]: Set temperature (between 0.1 and 2.0)
- /penalty [value]: Set repetition penalty (1.0-2.0)
- /debug: Toggle debug mode
{'=' * 80}
"""
        print(colored(welcome_text, 'cyan'))

    def _format_prompt(self, user_input):
        """Format the complete prompt with history and current input."""
        # Start with the template
        formatted_prompt = self.prompt_template

        # Add conversation history
        for entry in self.history:
            role, text = entry
            if role == "human":
                formatted_prompt += f"{self.human_prefix}{text}{self.end_of_turn}"
            else:  # assistant
                formatted_prompt += f"{self.assistant_prefix}{text}{self.end_of_turn}"

        # Add the current user input
        formatted_prompt += f"{self.human_prefix}{user_input}{self.end_of_turn}{self.assistant_prefix}"

        return formatted_prompt

    def _tokenize(self, text):
        """Tokenize text and return token IDs."""
        return self.tokenizer.encode(text)

    def _update_history(self, user_input, response):
        """Update conversation history."""
        # Add to text history
        self.history.append(("human", user_input))
        self.history.append(("assistant", response))

        # Update token history for context window management
        user_tokens = self._tokenize(f"{self.human_prefix}{user_input}{self.end_of_turn}")
        response_tokens = self._tokenize(f"{self.assistant_prefix}{response}{self.end_of_turn}")

        self.history_tokens.extend(user_tokens)
        self.history_tokens.extend(response_tokens)

        # Track token usage
        self.total_prompt_tokens += len(user_tokens)
        self.total_generated_tokens += len(response_tokens)

        # Trim history if it gets too long
        self._trim_history_if_needed()

    def _trim_history_if_needed(self):
        """Trim history to fit within the context window."""
        if len(self.history_tokens) > self.max_history_tokens:
            # Remove oldest turns until we're under the limit
            while len(self.history_tokens) > self.max_history_tokens and len(self.history) >= 2:
                # Remove oldest human and assistant turn
                self.history = self.history[2:]

                # Find token boundary for the removed turns
                user_turn = self.history[0][1]
                assistant_turn = self.history[1][1]
                user_tokens = len(self._tokenize(f"{self.human_prefix}{user_turn}{self.end_of_turn}"))
                assistant_tokens = len(self._tokenize(f"{self.assistant_prefix}{assistant_turn}{self.end_of_turn}"))

                # Update token history
                self.history_tokens = self.history_tokens[user_tokens + assistant_tokens:]

    def _should_stop_generation(self, text):
        """Check if generation should stop based on end markers."""
        for marker in self.end_markers:
            if marker in text:
                return True
        return False

    def _clean_token_text(self, text):
        """Clean token text by fixing encoding issues."""
        # Fix the specific issue with �� -> ' and single � -> '
        text = text.replace('��', "'")
        text = text.replace('�', "'")
        return text

    def generate_with_repetition_penalty(self, input_ids, max_new_tokens, temperature, top_k, penalty=1.2, live=False):
        """Custom generate function with repetition penalty and optional live generation."""
        model = self.model
        device = self.device

        # Ensure model is in eval mode
        model.eval()

        # Initialize sequence with input_ids
        generated = input_ids.clone()

        # Initialize live text buffer
        live_buffer = ""

        # Create repetition penalty processor
        rep_processor = RepetitionPenaltyLogitsProcessor(penalty=penalty)

        # Counter for generated tokens
        tokens_generated = 0
        min_tokens = self.min_tokens_to_generate  # Ensure we generate at least this many tokens

        # EOT token ID for immediate stopping
        eot_token_id = self.tokenizer.eot_token  # tiktoken uses eot_token

        # Get rotary frequencies if model uses them
        freqs_cis = None
        if self.use_rotary:
            if hasattr(model, 'freqs_cis'):
                freqs_cis = model.freqs_cis.to(device)
            elif hasattr(model.transformer, 'freqs_cis'):
                freqs_cis = model.transformer.freqs_cis.to(device)

        # Generate tokens one at a time
        for _ in range(max_new_tokens):
            # Get only the last block_size tokens if context is too long
            if generated.size(1) > self.block_size:
                context = generated[:, -self.block_size:]
            else:
                context = generated

            # Forward pass for next token prediction
            with torch.no_grad():
                # Handle the forward pass based on model architecture
                try:
                    # First try without freqs_cis
                    logits, _ = model(context)
                except TypeError:
                    # If that fails and we have rotary embeddings, try with freqs_cis
                    if self.use_rotary and freqs_cis is not None:
                        logits, _ = model(context, freqs_cis=freqs_cis)
                    else:
                        # If still failing, raise the error
                        raise

            # Get logits for the next token (last position)
            next_token_logits = logits[:, -1, :]

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Apply repetition penalty
            if penalty > 1.0:
                next_token_logits = rep_processor(context, next_token_logits)

            # Optional top-k sampling
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Convert logits to probabilities
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)

            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)

            # Check if the next token is EOT and break immediately if so
            if next_token.item() == eot_token_id:
                if live:
                    # For live generation, yield a signal to stop without adding the EOT token
                    yield "", live_buffer, True
                break

            # Append next token to generated sequence
            generated = torch.cat((generated, next_token), dim=1)
            tokens_generated += 1

            # If live generation, decode and yield the token
            if live:
                # Decode the next token
                next_token_text = self.tokenizer.decode([next_token.item()])
                # Clean the token text to fix encoding issues
                next_token_text = self._clean_token_text(next_token_text)
                live_buffer += next_token_text

                # Check if we've hit an end marker in the buffer
                eot_marker_pos = live_buffer.find("<|endoftext|>")
                if eot_marker_pos != -1:
                    # Cut off at the EOT marker
                    live_buffer = live_buffer[:eot_marker_pos]
                    yield "", live_buffer, True
                    break

                # Check other end markers
                should_stop = tokens_generated >= min_tokens and self._should_stop_generation(live_buffer)
                yield next_token_text, live_buffer, should_stop

                if should_stop:
                    break

            # For non-live generation, check if we should stop
            elif tokens_generated >= min_tokens:
                # Check for end markers in the recent generated tokens
                recent_text = self.tokenizer.decode(generated[0, -20:].tolist())
                if self._should_stop_generation(recent_text):
                    break

        # Check if we generated any tokens at all
        if tokens_generated == 0 and not live:
            if self.debug_mode:
                print(colored("\n[No tokens generated in this attempt]", "red"))
            return None

        if not live:
            return generated

    def generate_response(self, user_input):
        """Generate a response to the user input."""
        # Format the complete prompt
        prompt = self._format_prompt(user_input)

        # Tokenize the prompt
        input_ids = torch.tensor(self._tokenize(prompt), dtype=torch.long).unsqueeze(0).to(self.device)

        # Ensure we don't exceed the model's context length
        if input_ids.size(1) > self.block_size:
            # If too long, keep the beginning part with the instruction template and trim the middle
            instruction_tokens = self._tokenize(self.prompt_template)
            # Keep the instruction and the most recent conversation that will fit
            keep_from_beginning = len(instruction_tokens)
            keep_from_end = self.block_size - keep_from_beginning

            # Combine beginning and end, ensuring we don't exceed array bounds
            if keep_from_end < 0:
                # If instruction alone is too long, trim it (shouldn't happen with reasonable templates)
                input_ids = input_ids[:, :self.block_size]
            else:
                # Keep instruction and most recent conversation
                input_ids = torch.cat([
                    input_ids[:, :keep_from_beginning],
                    input_ids[:, -(keep_from_end):]
                ], dim=1)

        # Track generation start time
        start_time = time.time()

        # Always use live generation
        return self._generate_live_response(input_ids, user_input, start_time)

    def _generate_live_response(self, input_ids, user_input, start_time):
        """Generate response with live token-by-token output."""
        # Initialize for live generation
        live_text = ""
        tokens_generated = 0
        retry_count = 0

        # Keep trying until we get a valid response or exhaust retries
        while retry_count <= self.max_retries:
            if retry_count > 0:
                # Calculate temperature for this retry
                if retry_count % 2 == 0:
                    # Even retries: increase temperature
                    temp_adjustment = min(0.2 * (retry_count // 2), 0.8)
                    current_temp = min(self.config.temperature + temp_adjustment, 1.8)
                else:
                    # Odd retries: decrease temperature
                    temp_adjustment = min(0.2 * ((retry_count + 1) // 2), 0.4)
                    current_temp = max(self.config.temperature - temp_adjustment, 0.2)

                if self.debug_mode:
                    print(colored(f"\n[Live retry {retry_count}: Using temperature {current_temp:.2f}]", "yellow"))
            else:
                current_temp = self.config.temperature

            # Reset for this attempt
            live_text = ""
            tokens_generated = 0
            generation_failed = False

            # Try to generate with current settings
            try:
                # Generate with live output
                for token_text, live_buffer, should_stop in self.generate_with_repetition_penalty(
                        input_ids,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=current_temp,
                        top_k=self.config.top_k,
                        penalty=self.repetition_penalty,
                        live=True
                ):
                    # If we should stop but there's a token, this is the last one
                    if should_stop:
                        # Update with the final clean buffer (will have EOT removed if present)
                        live_text = live_buffer
                        break

                    # Otherwise add the token and continue
                    if token_text:
                        live_text += token_text
                        tokens_generated += 1
                        yield token_text, live_text, False

                # Check if we got a valid response
                if not live_text or len(live_text.strip()) < 10:
                    if self.debug_mode:
                        print(colored("\n[Live generation produced empty or too short response, retrying]", "yellow"))
                    generation_failed = True
                    retry_count += 1
                    # Clear any partial output
                    if retry_count <= self.max_retries:
                        print("\r" + " " * 80 + "\r", end="")  # Clear the line
                else:
                    # We got a valid response, stop retrying
                    break

            except Exception as e:
                if self.debug_mode:
                    print(colored(f"\n[Live generation error: {str(e)}, retrying]", "red"))
                generation_failed = True
                retry_count += 1

        # If we still failed after all retries, use the failure message
        if generation_failed or not live_text or len(live_text.strip()) < 10:
            live_text = self.generation_failure_message
            if self.debug_mode:
                print(colored(f"\n[Returning failure message after {retry_count} live retries]", "red"))

        # Calculate time taken and metrics
        time_taken = time.time() - start_time
        tokens_per_second = tokens_generated / time_taken if time_taken > 0 else 0

        # Update history
        self._update_history(user_input, live_text)

        # Log generation stats
        logger.debug(f"Generated {tokens_generated} tokens in {time_taken:.2f}s ({tokens_per_second:.2f} tokens/s)")

        # Final yield of the complete response
        yield "", live_text, True

    def execute_command(self, command):
        """Execute a special command prefixed with /."""
        command = command.strip()

        if command == '/help':
            self._print_welcome_message()
            return True

        elif command == '/clear':
            self.history = []
            self.history_tokens = []
            print(colored("Conversation history cleared.", 'yellow'))
            return True

        elif command in ['/exit', '/quit']:
            print(colored("Goodbye!", 'cyan'))
            return False  # Signal to exit the chat loop

        elif command == '/stats':
            prompt_tokens = self.total_prompt_tokens
            generated_tokens = self.total_generated_tokens
            total_tokens = prompt_tokens + generated_tokens

            stats = f"""
Token usage statistics:
- Prompt tokens: {prompt_tokens}
- Generated tokens: {generated_tokens}
- Total tokens: {total_tokens}
- Current history length: {len(self.history_tokens)} tokens
- Current repetition penalty: {self.repetition_penalty}
- Current temperature: {self.config.temperature}
"""
            print(colored(stats, 'yellow'))
            return True

        elif command == '/debug':
            self.debug_mode = not self.debug_mode
            self.config.debug_mode = self.debug_mode  # Sync with config
            mode = "enabled" if self.debug_mode else "disabled"
            print(colored(f"Debug mode {mode}", 'yellow'))
            return True

        elif command.startswith('/penalty '):
            try:
                penalty = float(command[9:].strip())
                if 1.0 <= penalty <= 2.0:
                    self.repetition_penalty = penalty
                    print(colored(f"Repetition penalty set to {penalty}", 'yellow'))
                else:
                    print(colored("Repetition penalty should be between 1.0 and 2.0", 'red'))
            except ValueError:
                print(colored("Invalid repetition penalty value. Please use a number between 1.0 and 2.0", 'red'))
            return True

        elif command.startswith('/save '):
            filename = command[6:].strip()
            if not filename:
                print(colored("Please specify a filename: /save <filename>", 'red'))
                return True

            try:
                # Create conversations directory if it doesn't exist
                os.makedirs('conversations', exist_ok=True)

                # Add .txt extension if not present
                if not filename.endswith('.txt'):
                    filename += '.txt'

                filepath = os.path.join('conversations', filename)

                with open(filepath, 'w', encoding='utf-8') as f:
                    for entry in self.history:
                        role, text = entry
                        prefix = self.human_prefix if role == "human" else self.assistant_prefix
                        f.write(f"{prefix}{text}{self.end_of_turn}")

                print(colored(f"Conversation saved to {filepath}", 'green'))

            except Exception as e:
                print(colored(f"Error saving conversation: {str(e)}", 'red'))

            return True

        elif command.startswith('/load '):
            filename = command[6:].strip()
            if not filename:
                print(colored("Please specify a filename: /load <filename>", 'red'))
                return True

            try:
                # Add .txt extension if not present
                if not filename.endswith('.txt'):
                    filename += '.txt'

                filepath = os.path.join('conversations', filename)

                if not os.path.exists(filepath):
                    print(colored(f"File not found: {filepath}", 'red'))
                    return True

                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Parse conversation turns
                self.history = []
                self.history_tokens = []

                # Split by end of turn marker
                turns = content.split(self.end_of_turn)
                for turn in turns:
                    turn = turn.strip()
                    if not turn:
                        continue

                    if turn.startswith(self.human_prefix):
                        text = turn[len(self.human_prefix):].strip()
                        self.history.append(("human", text))
                    elif turn.startswith(self.assistant_prefix):
                        text = turn[len(self.assistant_prefix):].strip()
                        self.history.append(("assistant", text))

                # Recalculate token counts
                self.history_tokens = []
                for entry in self.history:
                    role, text = entry
                    if role == "human":
                        self.history_tokens.extend(self._tokenize(f"{self.human_prefix}{text}{self.end_of_turn}"))
                    else:
                        self.history_tokens.extend(self._tokenize(f"{self.assistant_prefix}{text}{self.end_of_turn}"))

                print(colored(f"Loaded conversation from {filepath} ({len(self.history) // 2} turns)", 'green'))

                # Print the conversation
                for i in range(0, len(self.history), 2):
                    if i < len(self.history):
                        user_text = self.history[i][1]
                        print(colored(f"\nYou: {user_text}", 'green'))

                    if i + 1 < len(self.history):
                        assistant_text = self.history[i + 1][1]
                        print(colored("CosmicFish: ", 'blue'), end="")
                        for line in assistant_text.split('\n'):
                            wrapped_lines = textwrap.wrap(line, width=100) if line.strip() else ['']
                            for wrapped_line in wrapped_lines:
                                print(wrapped_line)

            except Exception as e:
                print(colored(f"Error loading conversation: {str(e)}", 'red'))

            return True

        elif command.startswith('/temp '):
            try:
                temp = float(command[6:].strip())
                if 0.1 <= temp <= 2.0:
                    self.config.temperature = temp
                    print(colored(f"Temperature set to {temp}", 'yellow'))
                else:
                    print(colored("Temperature should be between 0.1 and 2.0", 'red'))
            except ValueError:
                print(colored("Invalid temperature value. Please use a number between 0.1 and 2.0", 'red'))
            return True

        else:
            print(colored(f"Unknown command: {command}. Type /help for available commands.", 'red'))
            return True


def main():
    parser = argparse.ArgumentParser(description="Chat with the CosmicFish model")

    # Model parameters
    parser.add_argument("--model_path", type=str, default="/Users/akhil/Documents/Mistyoz_AI/CosmicFish/Models/CF120.pt",
                        help="Path to the fine-tuned model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda or cpu)")

    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Temperature for sampling (default: 0.7)")
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Maximum number of tokens to generate per response")
    parser.add_argument("--min_tokens", type=int, default=10,
                        help="Minimum number of tokens to generate per response")
    parser.add_argument("--top_k", type=int, default=40,
                        help="Top-k sampling (0 to disable)")
    parser.add_argument("--repetition_penalty", type=float, default=1.2,
                        help="Repetition penalty (1.0 = no penalty, 1.2 = mild, 1.5 = moderate)")

    # Chat parameters
    parser.add_argument("--human_prefix", type=str, default="Human: ",
                        help="Prefix for human messages")
    parser.add_argument("--assistant_prefix", type=str, default="Assistant: ",
                        help="Prefix for assistant messages")
    parser.add_argument("--end_of_turn", type=str, default="\n\n",
                        help="Delimiter between conversation turns")
    parser.add_argument("--instruction", type=str,
                        default=DEFAULT_PROMPT_TEMPLATE,
                        help="Instruction prompt to prepend to the conversation")
    parser.add_argument("--max_history", type=int, default=2048,
                        help="Maximum number of tokens to keep in history")

    # UI parameters
    parser.add_argument("--no_welcome", action="store_true",
                        help="Don't display the welcome message")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")

    args = parser.parse_args()

    # Configure device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU")
        device = "cpu"

    # Load the model
    print(f"Loading model from {args.model_path}...")
    try:
        # FIXED: Handle torch.compile checkpoints with proper fallback
        try:
            checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
            logger.info("Loaded checkpoint with weights_only=True (secure mode)")
        except Exception as e:
            logger.warning(f"Failed to load with weights_only=True: {e}")
            logger.info("Falling back to weights_only=False (trusted checkpoint)")
            checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)

        # Get model configuration
        if 'cosmicconf' in checkpoint:
            logger.info("Using configuration from checkpoint (cosmicconf)")
            config = checkpoint['cosmicconf']
        elif 'config' in checkpoint:
            logger.info("Using configuration from checkpoint (config)")
            config = checkpoint['config']
        else:
            # Use default parameters if not found in checkpoint
            logger.warning("No configuration found in checkpoint, using default values")
            config = CosmicConfig(
                vocab_size=50257,
                block_size=2048,
                n_layer=24,
                n_head=24,
                n_embd=960,
                bias=True,
                dropout=0.1,
                use_rotary=True,
                use_swiglu=True,
                use_gqa=True,
                n_query_groups=4
            )

        # Create the model
        model = CosmicFish(config)

        # FIXED: Handle torch.compile state dict prefixes properly
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            raise ValueError("Could not find model weights in checkpoint")

        # Clean state dict keys to handle torch.compile prefixes
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            clean_key = key
            # Remove torch.compile prefix (_orig_mod.)
            if clean_key.startswith('_orig_mod.'):
                clean_key = clean_key[10:]  # len('_orig_mod.') = 10
            # Remove DDP prefix (module.)
            if clean_key.startswith('module.'):
                clean_key = clean_key[7:]   # len('module.') = 7
            
            cleaned_state_dict[clean_key] = value

        # Load the cleaned state dict
        try:
            model.load_state_dict(cleaned_state_dict)
            logger.info("✅ Successfully loaded model weights")
        except RuntimeError as e:
            logger.warning(f"⚠️ Failed strict loading: {e}")
            logger.info("🔄 Attempting flexible loading...")
            missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
            if missing_keys:
                logger.warning(f"⚠️ Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                logger.warning(f"⚠️ Unexpected keys: {len(unexpected_keys)}")

        model.to(device)
        model.eval()  # Set to evaluation mode

        # Get block size from model
        block_size = config.block_size

        print(f"Model loaded with {model.get_num_params() / 1e6:.2f}M parameters")

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    # Initialize the tokenizer
    try:
        tokenizer = tiktoken.get_encoding("gpt2")
    except Exception as e:
        print(f"Error loading tokenizer: {str(e)}")
        print("Please make sure you have installed tiktoken: pip install tiktoken")
        return

    # Create a config object with all the necessary parameters
    class ChatConfig:
        def __init__(self, args, block_size):
            self.device = args.device
            self.temperature = args.temperature
            self.max_new_tokens = args.max_tokens
            self.min_tokens_to_generate = args.min_tokens
            self.top_k = args.top_k
            self.human_prefix = args.human_prefix
            self.assistant_prefix = args.assistant_prefix
            self.end_of_turn = args.end_of_turn
            self.prompt_template = args.instruction
            self.max_history_tokens = args.max_history
            self.display_welcome = not args.no_welcome
            self.block_size = block_size
            self.debug_mode = args.debug
            self.repetition_penalty = args.repetition_penalty

    config = ChatConfig(args, block_size)

    # Initialize chat session
    chat = ChatSession(model, tokenizer, config)

    # Main chat loop
    print(colored("\nCosmicFish initialized. Type your message (or /help for commands).\n", 'cyan'))

    while True:
        try:
            # Get user input
            user_input = input(colored("You: ", 'green'))

            # Check if it's a command
            if user_input.startswith('/'):
                # Execute command, continue loop if True, exit if False
                if not chat.execute_command(user_input):
                    break
                continue

            # Skip if empty input
            if not user_input.strip():
                continue

            # Generate response using live generation
            live_buffer = ""
            final_response = None

            # Use the generator version
            response_generator = chat.generate_response(user_input)

            try:
                # First print the assistant prefix with model name
                print(colored("CosmicFish: ", 'blue'), end="")
                sys.stdout.flush()

                for token, live_text, is_done in response_generator:
                    # If this is the final clean response
                    if is_done:
                        final_response = live_text
                        # Print the final response directly if we didn't get any tokens yet
                        if not live_buffer:
                            print(final_response, end="")
                        break

                    # If we have a token to display
                    if token:
                        # Check if token contains <|endoftext|> and remove it if present
                        if "<|endoftext|>" in token:
                            token = token.replace("<|endoftext|>", "")
                            if token:  # Only print if there's anything left
                                print(token, end="", flush=True)
                            break

                        # Display it
                        print(token, end="", flush=True)
                        live_buffer += token

            except KeyboardInterrupt:
                # Allow user to interrupt generation
                print("\n[Generation interrupted]")
                # We still need a reasonable response for history
                final_response = "I was going to respond, but I'll stop here since you interrupted."

            # Add an extra line for readability
            print()

        except KeyboardInterrupt:
            print("\n\nKeyboard interrupt detected. Type /exit to quit or continue chatting.")

        except Exception as e:
            print(colored(f"\nError: {str(e)}", 'red'))
            logger.error(f"Error in chat loop: {str(e)}", exc_info=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)
