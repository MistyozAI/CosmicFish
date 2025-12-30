"""
Prepare single-turn conversational dataset for fine-tuning CosmicFish.
This script processes instruction-response datasets like Alpaca, Dolly, or their mixture.
ENHANCED: Now includes automatic cleaning of Alpaca GPT-4 dataset to remove AI disclaimers, 
incorrect factual information, AND conflicting AI identity claims.
"""

import os
import sys
import argparse
import json
import numpy as np
import tiktoken
import torch
from tqdm.auto import tqdm
from datasets import load_dataset
import logging
import time
from dataclasses import dataclass
import pickle
import random
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    dataset_name: str = "tatsu-lab/alpaca"  # Default to Alpaca dataset
    output_dir: str = "data/singleturn"
    test_size: float = 0.05  # 5% for validation
    seed: int = 42
    max_seq_length: int = 2048  # Maximum sequence length (must match model's block_size)
    human_prefix: str = "Human: "
    assistant_prefix: str = "Assistant: "
    end_of_turn: str = "\n\n"  # Delimiter between conversation turns
    instruction_prefix: str = "Below is a conversation between a helpful AI assistant and a human. The assistant is knowledgeable, friendly, and provides detailed, helpful responses.\n\n"
    encoding_name: str = "gpt2"  # GPT-2 tokenizer (same as used in original model)
    overwrite: bool = False
    language: str = "en"  # Filter for English data if dataset has language markers
    # New parameters for mixing datasets
    alpaca_ratio: float = 0.7  # 70% Alpaca, 30% Dolly when mixing


def should_filter_response(response: str, instruction: str = "") -> tuple[bool, str, str]:
    """
    Check if a response should be filtered out for containing AI disclaimers, incorrect facts, or conflicting AI identities
    Returns: (should_filter: bool, reason: str, category: str)
    """
    response_lower = response.lower().strip()
    instruction_lower = instruction.lower().strip()
    combined_text = (response_lower + " " + instruction_lower).strip()

    # AI identity disclaimers
    ai_phrases = [
        "i'm an ai", "i am an ai", "as an ai", "i'm a language model",
        "i am a language model", "as a language model", "i'm artificial intelligence",
        "i am artificial intelligence", "as an artificial intelligence"
    ]

    # Access limitations
    access_phrases = [
        "i don't have access to real-time", "i cannot access real-time",
        "i don't have access to current", "i cannot access current",
        "i don't have the ability to browse", "i cannot browse",
        "i don't have internet access", "i cannot access the internet"
    ]

    # Knowledge cutoff mentions
    cutoff_phrases = [
        "my knowledge cutoff", "knowledge cutoff", "my training data",
        "as of my last update", "my last update was"
    ]

    # Overly cautious responses
    cautious_phrases = [
        "i cannot provide real-time", "i'm unable to provide current",
        "i don't have information about recent", "i cannot give you the most current",
        "for the most up-to-date", "please check the latest"
    ]

    # NEW: AI Identity Conflicts - Filter out mentions of other AI systems
    ai_identity_patterns = [
        # Direct AI names
        (r'chatgpt', "ChatGPT mention"),
        (r'gpt-[0-9]', "GPT model mention"),
        (r'gpt [0-9]', "GPT model mention"),
        (r'openai', "OpenAI mention"),
        (r'google assistant', "Google Assistant mention"),
        (r'alexa', "Alexa mention"),
        (r'siri', "Siri mention"),
        (r'cortana', "Cortana mention"),
        (r'bard', "Bard mention"),
        (r'claude', "Claude mention"),
        
        # Company creation claims
        (r'created by google', "Google creation claim"),
        (r'made by google', "Google creation claim"),
        (r'developed by google', "Google creation claim"),
        (r'built by google', "Google creation claim"),
        (r'google created', "Google creation claim"),
        (r'created by openai', "OpenAI creation claim"),
        (r'made by openai', "OpenAI creation claim"),
        (r'developed by openai', "OpenAI creation claim"),
        (r'openai created', "OpenAI creation claim"),
        (r'created by microsoft', "Microsoft creation claim"),
        (r'made by microsoft', "Microsoft creation claim"),
        (r'developed by microsoft', "Microsoft creation claim"),
        (r'created by amazon', "Amazon creation claim"),
        (r'made by amazon', "Amazon creation claim"),
        (r'developed by amazon', "Amazon creation claim"),
        (r'created by apple', "Apple creation claim"),
        (r'made by apple', "Apple creation claim"),
        (r'developed by apple', "Apple creation claim"),
        
        # Generic AI identity statements that conflict
        (r'i am chatgpt', "ChatGPT identity claim"),
        (r'i\'m chatgpt', "ChatGPT identity claim"),
        (r'my name is chatgpt', "ChatGPT name claim"),
        (r'i am gpt', "GPT identity claim"),
        (r'i\'m gpt', "GPT identity claim"),
        (r'i am google assistant', "Google Assistant identity claim"),
        (r'i\'m google assistant', "Google Assistant identity claim"),
        (r'i am alexa', "Alexa identity claim"),
        (r'i\'m alexa', "Alexa identity claim"),
        (r'i am siri', "Siri identity claim"),
        (r'i\'m siri', "Siri identity claim"),
        
        # Training/model references that could conflict
        (r'trained by openai', "OpenAI training claim"),
        (r'trained by google', "Google training claim"),
        (r'developed at openai', "OpenAI development claim"),
        (r'developed at google', "Google development claim"),
        (r'built at openai', "OpenAI building claim"),
        (r'built at google', "Google building claim"),
        
        # Version/model specific references
        (r'gpt-3\.5', "GPT-3.5 mention"),
        (r'gpt-4', "GPT-4 mention"),
        (r'davinci', "Davinci model mention"),
        (r'text-davinci', "Text-Davinci mention"),
        
        # Generic conflicting statements
        (r'i\'m a product of', "Generic product claim"),
        (r'i am a product of', "Generic product claim"),
        (r'developed by the team at', "Generic team development claim"),
        (r'created by the team at', "Generic team creation claim"),
    ]

    # Incorrect factual information patterns
    incorrect_fact_patterns = [
        # Mumbai as capital of India patterns
        (r'mumbai.*capital.*india', "Mumbai as India capital"),
        (r'capital.*india.*mumbai', "Mumbai as India capital"),
        (r'india.*capital.*mumbai', "Mumbai as India capital"),
        (r'mumbai.*capital.*of.*india', "Mumbai as India capital"),
        (r'capital.*of.*india.*mumbai', "Mumbai as India capital"),
        (r'india\'s.*capital.*mumbai', "Mumbai as India capital"),
        (r'mumbai.*india\'s.*capital', "Mumbai as India capital"),
        (r'the.*capital.*india.*mumbai', "Mumbai as India capital"),
        (r'mumbai.*the.*capital.*india', "Mumbai as India capital"),
        (r'mumbai.*is.*the.*capital.*of.*india', "Mumbai as India capital"),
        (r'mumbai.*capital.*city.*india', "Mumbai as India capital"),

        # Other common geographical errors you might want to catch
        (r'sydney.*capital.*australia', "Sydney as Australia capital"),
        (r'new.*york.*capital.*us', "NYC as US capital"),
        (r'new.*york.*capital.*united.*states', "NYC as US capital"),
        (r'toronto.*capital.*canada', "Toronto as Canada capital"),
        (r'istanbul.*capital.*turkey', "Istanbul as Turkey capital"),

        # Some other common errors
        (r'rio.*de.*janeiro.*capital.*brazil', "Rio as Brazil capital"),
        (r'lagos.*capital.*nigeria', "Lagos as Nigeria capital"),
        (r'karachi.*capital.*pakistan', "Karachi as Pakistan capital"),
    ]

    # Check AI identity conflicts first (high priority)
    for pattern, description in ai_identity_patterns:
        if re.search(pattern, combined_text):
            return True, f"AI identity conflict: {description}", "AI_IDENTITY_CONFLICT"

    # Check incorrect factual patterns
    for pattern, description in incorrect_fact_patterns:
        if re.search(pattern, combined_text):
            return True, f"Incorrect fact: {description}", "FACTUAL_ERROR"

    # Check AI disclaimer phrase categories
    for phrase in ai_phrases:
        if phrase in response_lower:
            return True, f"AI disclaimer: '{phrase}'", "AI_DISCLAIMER"

    for phrase in access_phrases:
        if phrase in response_lower:
            return True, f"Access limitation: '{phrase}'", "ACCESS_LIMITATION"

    for phrase in cutoff_phrases:
        if phrase in response_lower:
            return True, f"Knowledge cutoff: '{phrase}'", "KNOWLEDGE_CUTOFF"

    for phrase in cautious_phrases:
        if phrase in response_lower:
            return True, f"Overly cautious: '{phrase}'", "OVERLY_CAUTIOUS"

    # Check for overly long responses (>500 words)
    word_count = len(response.split())
    if word_count > 500:
        return True, f"Too long: {word_count} words", "TOO_LONG"

    # Check for responses that are just disclaimers
    if len(response.strip()) < 10:
        return True, "Too short", "TOO_SHORT"

    return False, "Clean", "CLEAN"


def print_detailed_stats(filter_stats, dataset_name="Dataset"):
    """Print detailed filtering statistics with categories"""
    logger.info("\n" + "=" * 60)
    logger.info(f"📊 {dataset_name.upper()} CLEANING STATISTICS")
    logger.info("=" * 60)
    logger.info(f"📥 Total examples:     {filter_stats['total_examples']:,}")
    logger.info(
        f"✅ Kept examples:      {filter_stats['kept']:,} ({filter_stats['kept'] / filter_stats['total_examples'] * 100:.1f}%)")
    logger.info(
        f"🗑️  Filtered out:       {filter_stats['filtered_out']:,} ({filter_stats['filtered_out'] / filter_stats['total_examples'] * 100:.1f}%)")

    if filter_stats['filtered_out'] > 0:
        # Group by category
        categories = {}
        factual_errors = {}

        for reason, count in filter_stats['filter_reasons'].items():
            if reason.startswith("Incorrect fact:"):
                factual_errors[reason] = count
            else:
                # Extract category from reason
                if "AI identity conflict:" in reason:
                    cat = "AI Identity Conflicts"
                elif "AI disclaimer:" in reason:
                    cat = "AI Disclaimers"
                elif "Access limitation:" in reason:
                    cat = "Access Limitations"
                elif "Knowledge cutoff:" in reason:
                    cat = "Knowledge Cutoffs"
                elif "Overly cautious:" in reason:
                    cat = "Overly Cautious"
                elif reason == "Too short":
                    cat = "Too Short"
                elif reason.startswith("Too long:"):
                    cat = "Too Long"
                else:
                    cat = "Other"

                if cat not in categories:
                    categories[cat] = 0
                categories[cat] += count

        # Print category breakdown
        logger.info(f"\n📋 BREAKDOWN BY CATEGORY:")
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            percentage = count / filter_stats['filtered_out'] * 100
            logger.info(f"   • {category}: {count:,} ({percentage:.1f}%)")

        # Print AI identity conflicts specifically
        if any("AI identity conflict:" in reason for reason in filter_stats['filter_reasons'].keys()):
            ai_conflicts = {reason: count for reason, count in filter_stats['filter_reasons'].items() 
                          if reason.startswith("AI identity conflict:")}
            logger.info(f"\n🚨 AI IDENTITY CONFLICTS FOUND:")
            for conflict, count in sorted(ai_conflicts.items(), key=lambda x: x[1], reverse=True):
                percentage = count / filter_stats['filtered_out'] * 100
                logger.info(f"   • {conflict}: {count:,} ({percentage:.1f}%)")
        else:
            logger.info(f"\n✅ NO AI IDENTITY CONFLICTS FOUND!")
            logger.info(f"   • No mentions of ChatGPT, Google Assistant, etc.")
            logger.info(f"   • No conflicting AI creation claims")

        # Print factual errors specifically
        if factual_errors:
            logger.info(f"\n🚨 FACTUAL ERRORS FOUND:")
            for error, count in sorted(factual_errors.items(), key=lambda x: x[1], reverse=True):
                percentage = count / filter_stats['filtered_out'] * 100
                logger.info(f"   • {error}: {count:,} ({percentage:.1f}%)")
        else:
            logger.info(f"\n✅ NO FACTUAL ERRORS FOUND!")
            logger.info(f"   • No Mumbai capital errors")
            logger.info(f"   • No other geographical errors detected")

        # Print top individual reasons
        logger.info(f"\n🔍 TOP INDIVIDUAL FILTER REASONS:")
        for reason, count in sorted(filter_stats['filter_reasons'].items(), key=lambda x: x[1], reverse=True)[:7]:
            percentage = count / filter_stats['filtered_out'] * 100
            logger.info(f"   • {reason}: {count:,} ({percentage:.1f}%)")

    logger.info("=" * 60)


def format_conversation(question, answer, config):
    """Format a question and answer into a standardized format."""
    formatted_text = config.instruction_prefix
    formatted_text += f"{config.human_prefix}{question.strip()}{config.end_of_turn}"
    formatted_text += f"{config.assistant_prefix}{answer.strip()}{config.end_of_turn}"
    return formatted_text


def process_alpaca_dataset(config):
    """Process the Stanford Alpaca dataset (original, not GPT-4)."""
    logger.info(f"Loading Original Alpaca dataset...")

    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    # Format into single-turn conversations
    conversations = []
    filter_stats = {
        "total_examples": len(dataset),
        "filtered_out": 0,
        "kept": 0,
        "filter_reasons": {}
    }

    logger.info("🧹 Cleaning Original Alpaca dataset...")

    for item in tqdm(dataset, desc="Processing Alpaca examples"):
        instruction = item["instruction"].strip()
        response = item["output"].strip()

        # If there's input, add it to the instruction
        if item["input"] and item["input"].strip():
            instruction += f"\n{item['input'].strip()}"

        # Check if response should be filtered
        should_filter, reason, category = should_filter_response(response, instruction)

        if should_filter:
            filter_stats["filtered_out"] += 1
            if reason not in filter_stats["filter_reasons"]:
                filter_stats["filter_reasons"][reason] = 0
            filter_stats["filter_reasons"][reason] += 1
        else:
            conversations.append({
                "question": instruction,
                "answer": response,
                "source": "alpaca"
            })
            filter_stats["kept"] += 1

    # Print cleaning statistics
    print_detailed_stats(filter_stats, "Alpaca")
    logger.info(f"Processed {len(conversations)} cleaned Alpaca conversations")
    return conversations


def process_alpaca_gpt4_cleaned(config):
    """Process the Alpaca GPT-4 dataset with automatic cleaning of AI disclaimers, incorrect facts, and AI identity conflicts."""
    logger.info(f"Loading Alpaca GPT-4 dataset for cleaning...")

    # Try multiple sources for Alpaca GPT-4
    dataset = None
    dataset_sources = [
        "vicgalle/alpaca-gpt4",
        "tatsu-lab/alpaca",
        "yahma/alpaca-cleaned"
    ]

    for source in dataset_sources:
        try:
            dataset = load_dataset(source, split="train")
            logger.info(f"✅ Successfully loaded {len(dataset)} examples from {source}")
            break
        except Exception as e:
            logger.warning(f"Failed to load {source}: {e}")
            continue

    if dataset is None:
        logger.error("❌ Could not load any Alpaca dataset. Exiting.")
        sys.exit(1)

    # Clean and format into single-turn conversations
    conversations = []
    filter_stats = {
        "total_examples": len(dataset),
        "filtered_out": 0,
        "kept": 0,
        "filter_reasons": {}
    }

    logger.info("🧹 Cleaning dataset to remove AI disclaimers, incorrect facts, and AI identity conflicts...")

    for item in tqdm(dataset, desc="Processing and cleaning Alpaca GPT-4 examples"):
        instruction = item["instruction"].strip()
        response = item["output"].strip()

        # If there's input, add it to the instruction
        if item["input"] and item["input"].strip():
            instruction += f"\n{item['input'].strip()}"

        # Check if response should be filtered
        should_filter, reason, category = should_filter_response(response, instruction)

        if should_filter:
            filter_stats["filtered_out"] += 1
            if reason not in filter_stats["filter_reasons"]:
                filter_stats["filter_reasons"][reason] = 0
            filter_stats["filter_reasons"][reason] += 1
        else:
            conversations.append({
                "question": instruction,
                "answer": response,
                "source": "alpaca-gpt4-cleaned"
            })
            filter_stats["kept"] += 1

    # Print detailed cleaning statistics
    print_detailed_stats(filter_stats, "Alpaca GPT-4")
    logger.info(f"Processed {len(conversations)} cleaned Alpaca GPT-4 conversations")
    return conversations


def process_dolly_dataset(config):
    """Process the Dolly dataset with filtering."""
    logger.info(f"Loading Dolly15K dataset...")

    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

    # Format into single-turn conversations
    conversations = []
    filter_stats = {
        "total_examples": len(dataset),
        "filtered_out": 0,
        "kept": 0,
        "filter_reasons": {}
    }

    logger.info("🧹 Cleaning Dolly dataset...")

    for item in tqdm(dataset, desc="Processing Dolly examples"):
        instruction = item["instruction"].strip()
        response = item["response"].strip()

        # If there's context, add it to the instruction
        if "context" in item and item["context"] and item["context"].strip():
            instruction += f"\nContext: {item['context'].strip()}"

        # Check if response should be filtered
        should_filter, reason, category = should_filter_response(response, instruction)

        if should_filter:
            filter_stats["filtered_out"] += 1
            if reason not in filter_stats["filter_reasons"]:
                filter_stats["filter_reasons"][reason] = 0
            filter_stats["filter_reasons"][reason] += 1
        else:
            conversations.append({
                "question": instruction,
                "answer": response,
                "source": "dolly"
            })
            filter_stats["kept"] += 1

    # Print cleaning statistics if any were filtered
    if filter_stats["filtered_out"] > 0:
        print_detailed_stats(filter_stats, "Dolly")

    logger.info(f"Processed {len(conversations)} Dolly conversations")
    return conversations


def process_mixed_dataset(config):
    """Process a mixture of Alpaca and Dolly datasets."""
    logger.info(f"Loading Mixed Dataset: Alpaca + Dolly15K")

    # Load both datasets
    alpaca_conversations = process_alpaca_dataset(config)
    dolly_conversations = process_dolly_dataset(config)

    # Calculate how many examples to take from each dataset
    total_alpaca = len(alpaca_conversations)
    total_dolly = len(dolly_conversations)

    logger.info(f"Available: {total_alpaca} Alpaca + {total_dolly} Dolly = {total_alpaca + total_dolly} total")

    # Use all available data, but shuffle them together
    all_conversations = alpaca_conversations + dolly_conversations

    # Shuffle to mix the sources
    random.seed(config.seed)
    random.shuffle(all_conversations)

    # Log the mixture
    alpaca_count = len([c for c in all_conversations if c["source"] == "alpaca"])
    dolly_count = len([c for c in all_conversations if c["source"] == "dolly"])

    logger.info(
        f"Final mixture: {alpaca_count} Alpaca ({alpaca_count / len(all_conversations) * 100:.1f}%) + {dolly_count} Dolly ({dolly_count / len(all_conversations) * 100:.1f}%)")

    return all_conversations


def process_lima_dataset(config):
    """Process the LIMA dataset with filtering."""
    logger.info(f"Loading LIMA dataset...")

    try:
        dataset = load_dataset("GAIR/lima", split="train")

        # Format into single-turn conversations
        conversations = []
        filter_stats = {
            "total_examples": len(dataset),
            "filtered_out": 0,
            "kept": 0,
            "filter_reasons": {}
        }

        logger.info("🧹 Cleaning LIMA dataset...")

        for item in tqdm(dataset, desc="Processing LIMA examples"):
            # Extract just the first exchange
            if len(item["conversations"]) >= 2:  # Need at least one human and one assistant message
                human_msg = item["conversations"][0]["value"].strip()
                assistant_msg = item["conversations"][1]["value"].strip()

                # Check if response should be filtered
                should_filter, reason, category = should_filter_response(assistant_msg, human_msg)

                if should_filter:
                    filter_stats["filtered_out"] += 1
                    if reason not in filter_stats["filter_reasons"]:
                        filter_stats["filter_reasons"][reason] = 0
                    filter_stats["filter_reasons"][reason] += 1
                else:
                    conversations.append({
                        "question": human_msg,
                        "answer": assistant_msg,
                        "source": "lima"
                    })
                    filter_stats["kept"] += 1

        # Print cleaning statistics if any were filtered
        if filter_stats["filtered_out"] > 0:
            print_detailed_stats(filter_stats, "LIMA")

        logger.info(f"Processed {len(conversations)} LIMA conversations")
        return conversations
    except Exception as e:
        logger.error(f"Error loading LIMA dataset: {e}")
        return []


def process_oasst1_single_turns(config):
    """Process Open Assistant dataset, but extract only single turns with filtering."""
    logger.info(f"Loading OpenAssistant/oasst1 dataset...")

    # Specifically load the English subset if possible
    try:
        dataset = load_dataset("OpenAssistant/oasst1", "en", split="train")
        logger.info(f"Successfully loaded English OASST1 dataset")
    except Exception:
        # Fall back to loading the whole dataset and filtering
        dataset = load_dataset("OpenAssistant/oasst1", split="train")
        if config.language:
            dataset = dataset.filter(lambda example: example.get("lang") == config.language)
            logger.info(f"Filtered for {config.language} language: {len(dataset)} examples")

    # Group messages to find question-answer pairs
    conversations = []
    filter_stats = {
        "total_examples": 0,
        "filtered_out": 0,
        "kept": 0,
        "filter_reasons": {}
    }

    message_dict = {}
    parent_to_children = {}

    # First pass: build the message dictionary and parent-child relationships
    for item in tqdm(dataset, desc="Processing messages"):
        message_id = item["message_id"]
        parent_id = item["parent_id"]
        role = "human" if item["role"] == "prompter" else "assistant"

        # Store this message
        message_dict[message_id] = {
            "role": role,
            "content": item["text"],
            "parent_id": parent_id
        }

        # Add to parent-child mapping
        if parent_id not in parent_to_children:
            parent_to_children[parent_id] = []
        parent_to_children[parent_id].append(message_id)

    logger.info("🧹 Extracting and cleaning OASST1 conversations...")

    # Second pass: find human questions with exactly one assistant response
    for message_id, message in tqdm(message_dict.items(), desc="Extracting conversations"):
        # Only consider human messages
        if message["role"] != "human":
            continue

        # Check if this message has children
        if message_id in parent_to_children and len(parent_to_children[message_id]) == 1:
            child_id = parent_to_children[message_id][0]
            child = message_dict.get(child_id)

            # Make sure the child is an assistant message
            if child and child["role"] == "assistant":
                filter_stats["total_examples"] += 1

                # Check if response should be filtered
                should_filter, reason, category = should_filter_response(child["content"], message["content"])

                if should_filter:
                    filter_stats["filtered_out"] += 1
                    if reason not in filter_stats["filter_reasons"]:
                        filter_stats["filter_reasons"][reason] = 0
                    filter_stats["filter_reasons"][reason] += 1
                else:
                    conversations.append({
                        "question": message["content"].strip(),
                        "answer": child["content"].strip(),
                        "source": "oasst1"
                    })
                    filter_stats["kept"] += 1

    # Print cleaning statistics if any were filtered
    if filter_stats["filtered_out"] > 0:
        print_detailed_stats(filter_stats, "OASST1")

    logger.info(f"Extracted {len(conversations)} single-turn conversations from OASST1")
    return conversations


def prepare_dataset(config):
    """Prepare the specified single-turn dataset."""
    os.makedirs(config.output_dir, exist_ok=True)

    # Check if processed data already exists
    train_path = os.path.join(config.output_dir, 'train.bin')
    val_path = os.path.join(config.output_dir, 'val.bin')

    if os.path.exists(train_path) and os.path.exists(val_path) and not config.overwrite:
        logger.info(f"Processed data already exists at {config.output_dir}. Use --overwrite to reprocess.")
        return

    # Load the tokenizer
    enc = tiktoken.get_encoding(config.encoding_name)

    # Process the dataset based on which one was specified
    dataset_name_lower = config.dataset_name.lower()

    if "alpaca-gpt4-cleaned" in dataset_name_lower or "alpaca_gpt4_cleaned" in dataset_name_lower:
        conversations = process_alpaca_gpt4_cleaned(config)
    elif "mixed" in dataset_name_lower or ("alpaca" in dataset_name_lower and "dolly" in dataset_name_lower):
        conversations = process_mixed_dataset(config)
    elif "alpaca" in dataset_name_lower:
        conversations = process_alpaca_dataset(config)
    elif "dolly" in dataset_name_lower:
        conversations = process_dolly_dataset(config)
    elif "lima" in dataset_name_lower:
        conversations = process_lima_dataset(config)
    elif "oasst" in dataset_name_lower:
        conversations = process_oasst1_single_turns(config)
    else:
        logger.error(f"Unknown dataset: {config.dataset_name}")
        logger.info("Available options: alpaca, alpaca-gpt4-cleaned, dolly, mixed, lima, oasst1")
        sys.exit(1)

    if not conversations:
        logger.error(f"No conversations extracted from dataset. Exiting.")
        sys.exit(1)

    # Shuffle conversations one more time
    random.seed(config.seed)
    random.shuffle(conversations)

    # Split into train and validation sets
    val_size = int(len(conversations) * config.test_size)
    train_conversations = conversations[val_size:]
    val_conversations = conversations[:val_size]

    logger.info(f"Train: {len(train_conversations)} conversations")
    logger.info(f"Validation: {len(val_conversations)} conversations")

    # Show source distribution
    if any('source' in conv for conv in conversations):
        train_sources = {}
        val_sources = {}

        for conv in train_conversations:
            source = conv.get('source', 'unknown')
            train_sources[source] = train_sources.get(source, 0) + 1

        for conv in val_conversations:
            source = conv.get('source', 'unknown')
            val_sources[source] = val_sources.get(source, 0) + 1

        logger.info(f"Train distribution: {train_sources}")
        logger.info(f"Validation distribution: {val_sources}")

    # Format and tokenize the conversations
    def process_conversations(conversation_list):
        all_tokens = []
        for conv in tqdm(conversation_list, desc="Formatting and tokenizing"):
            # Format the conversation
            formatted_text = format_conversation(conv["question"], conv["answer"], config)

            # Tokenize
            tokens = enc.encode(formatted_text)
            if len(tokens) > config.max_seq_length:
                tokens = tokens[:config.max_seq_length]

            all_tokens.extend(tokens)
            # Add an extra token to separate conversations
            all_tokens.append(enc.eot_token)

        return all_tokens

    logger.info("Processing training conversations...")
    train_tokens = process_conversations(train_conversations)

    logger.info("Processing validation conversations...")
    val_tokens = process_conversations(val_conversations)

    logger.info(f"Train tokens: {len(train_tokens)}")
    logger.info(f"Validation tokens: {len(val_tokens)}")

    # Save as binary files
    def save_to_binary(tokens, filename):
        arr = np.array(tokens, dtype=np.uint16)
        arr.tofile(filename)
        logger.info(f"Saved {len(tokens)} tokens to {filename}")

    save_to_binary(train_tokens, train_path)
    save_to_binary(val_tokens, val_path)

    # Save metadata
    meta = {
        'vocab_size': enc.n_vocab,
        'total_tokens': {
            'train': len(train_tokens),
            'val': len(val_tokens)
        },
        'dataset_name': config.dataset_name,
        'creation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': {k: v for k, v in vars(config).items()},
        'num_conversations': len(conversations),
        'cleaning_enabled': True,  # Mark that cleaning was performed
        'filters_applied': [
            'AI disclaimers',
            'Access limitations', 
            'Knowledge cutoffs',
            'Overly cautious responses',
            'AI identity conflicts',
            'Incorrect factual information',
            'Too long/short responses'
        ]
    }

    with open(os.path.join(config.output_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    # Also save a few examples as text for inspection
    with open(os.path.join(config.output_dir, 'examples.txt'), 'w', encoding='utf-8') as f:
        for i, conv in enumerate(val_conversations[:5]):
            source = conv.get('source', 'unknown')
            f.write(f"Example {i + 1} (Source: {source}):\n")
            f.write("-" * 50 + "\n")
            f.write(format_conversation(conv["question"], conv["answer"], config))
            f.write("\n\n" + "=" * 50 + "\n\n")

    logger.info("Dataset preparation completed!")


def main():
    parser = argparse.ArgumentParser(description="Prepare single-turn dataset for fine-tuning with advanced cleaning")
    parser.add_argument("--dataset", type=str, default="alpaca-gpt4-cleaned",
                        help="Dataset to use (options: alpaca, alpaca-gpt4-cleaned, dolly, mixed, lima, oasst1)")
    parser.add_argument("--output_dir", type=str, default="data/alpaca_gpt4_cleaned_pure",
                        help="Output directory for processed data")
    parser.add_argument("--test_size", type=float, default=0.05,
                        help="Validation split size (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length (default: 512)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing files")
    parser.add_argument("--language", type=str, default="en",
                        help="Filter for language (default: en for English)")
    parser.add_argument("--encoding", type=str, default="gpt2",
                        help="Tokenizer encoding (default: gpt2)")

    args = parser.parse_args()

    config = DatasetConfig(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        test_size=args.test_size,
        seed=args.seed,
        max_seq_length=args.max_length,
        overwrite=args.overwrite,
        language=args.language,
        encoding_name=args.encoding
    )

    logger.info(f"Configuration:")
    logger.info(f"  Dataset: {config.dataset_name}")
    logger.info(f"  Output: {config.output_dir}")
    logger.info(f"  Max length: {config.max_seq_length}")
    logger.info(f"  Validation split: {config.test_size}")
    logger.info(f"")
    logger.info(f"Enhanced Cleaning Features:")
    logger.info(f"  ✅ Removes AI disclaimers and limitations")
    logger.info(f"  ✅ Filters out incorrect factual information")
    logger.info(f"  ✅ Removes conflicting AI identity claims")
    logger.info(f"  ✅ Eliminates mentions of other AI systems")
    logger.info(f"  ✅ Cleans creation/development claims")

    prepare_dataset(config)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error during dataset preparation: {str(e)}", exc_info=True)
        sys.exit(1)