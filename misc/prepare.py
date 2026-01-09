"""
Prepare CosmicFish model for Hugging Face release.
Extracts clean weights and creates all necessary files for HF Hub.
Updated to use safetensors format.
"""

import os
import sys
import json
import torch
import shutil
import tiktoken
from pathlib import Path
import logging

# Import safetensors
try:
    from safetensors.torch import save_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("❌ safetensors not available. Install with: pip install safetensors")
    sys.exit(1)

# Import your model classes
from model import CosmicFish, CosmicConfig
from torch.serialization import add_safe_globals

add_safe_globals([CosmicConfig])

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_config_json(cosmic_config):
    """Convert CosmicConfig to Hugging Face compatible config.json"""
    config_dict = {
        "model_type": "cosmicfish",
        "architectures": ["CosmicFish"],
        "vocab_size": cosmic_config.vocab_size,
        "n_embd": cosmic_config.n_embd,
        "n_layer": cosmic_config.n_layer,
        "n_head": cosmic_config.n_head,
        "block_size": cosmic_config.block_size,
        "bias": cosmic_config.bias,
        "dropout": cosmic_config.dropout,
        "eps": cosmic_config.eps,
        "use_rotary": cosmic_config.use_rotary,
        "use_swiglu": cosmic_config.use_swiglu,
        "use_gqa": cosmic_config.use_gqa,
        "use_qk_norm": cosmic_config.use_qk_norm,
        "n_query_groups": cosmic_config.n_query_groups,
        # Additional HF-specific fields
        "torch_dtype": "float16",
        "transformers_version": "4.36.0",
        "use_cache": True,
        "pad_token_id": 50256,
        "bos_token_id": 50256,
        "eos_token_id": 50256
    }
    return config_dict


def create_tokenizer_files(output_dir):
    """Create tokenizer files compatible with HF (using GPT-2 tokenizer)"""
    logger.info("Creating tokenizer files...")

    # Initialize tiktoken GPT-2 encoder
    enc = tiktoken.get_encoding("gpt2")

    # Create tokenizer_config.json
    tokenizer_config = {
        "tokenizer_class": "GPT2Tokenizer",
        "vocab_size": enc.n_vocab,
        "model_max_length": 512,
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
        "add_prefix_space": False,
        "do_lower_case": False
    }

    with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    # Create a simple vocab.json and merges.txt
    # Note: For a complete release, you'd want to extract these from tiktoken
    # For now, we'll create placeholder files with a note

    vocab_note = {
        "note": "This model uses GPT-2 tokenizer. Please use: tokenizer = GPT2Tokenizer.from_pretrained('gpt2')",
        "vocab_size": enc.n_vocab,
        "encoding": "gpt2"
    }

    with open(os.path.join(output_dir, "vocab_info.json"), "w") as f:
        json.dump(vocab_note, f, indent=2)

    logger.info("Tokenizer configuration created (GPT-2 compatible)")


def create_model_card(config_dict, output_dir):
    """Create a comprehensive model card (README.md)"""
    logger.info("Creating model card...")

    model_card = f"""---
license: apache-2.0
tags:
- text-generation
- language-model
- causal-lm
- cosmicfish
- 120m
- transformer
- rotary-embeddings
- grouped-query-attention
- swiglu
language:
- en
datasets:
- openwebtext
- wikipedia
inference: true
---

# CosmicFish-120M

CosmicFish is a 120M parameter causal language model featuring advanced architectural improvements including Rotary Positional Embeddings (RoPE), Grouped-Query Attention (GQA), SwiGLU activation, and RMSNorm.

## Model Details

- **Model Type**: Causal Language Model
- **Parameters**: 120M
- **Architecture**: Transformer with modern enhancements
- **Context Length**: 512 tokens
- **Vocabulary Size**: 50,257 (GPT-2 compatible)
- **Developer**: Mistyoz AI
- **License**: Apache 2.0

## Architecture Features

- **Rotary Positional Embeddings (RoPE)**: Better position encoding for longer sequences
- **Grouped-Query Attention (GQA)**: Efficient attention mechanism with {config_dict['n_query_groups']} query groups
- **SwiGLU Activation**: Improved activation function for better performance
- **RMSNorm**: More stable normalization compared to LayerNorm
- **Modern Optimization**: Advanced training techniques and data processing

## Training Data

The model was trained on a carefully curated mix of:
- **OpenWebText**: High-quality web text data
- **Wikipedia**: Encyclopedic knowledge
- **Custom Processing**: Advanced filtering to remove AI disclaimers, factual errors, and identity conflicts

## Intended Use

CosmicFish is designed for:
- Text generation and completion
- Conversational AI applications
- Creative writing assistance
- Educational and research purposes

## Usage

```python
import torch
from transformers import GPT2Tokenizer
from safetensors.torch import load_file

# Load tokenizer (GPT-2 compatible)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load model (you'll need the custom CosmicFish class)
# See the provided chat script for complete usage example

# Basic generation example
input_text = "The future of artificial intelligence"
inputs = tokenizer.encode(input_text, return_tensors="pt")

# Generate with your model
# outputs = model.generate(inputs, max_length=100, temperature=0.7)
# response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Model Architecture

```
Layers: {config_dict['n_layer']}
Hidden Size: {config_dict['n_embd']}
Attention Heads: {config_dict['n_head']}
Query Groups: {config_dict['n_query_groups']} (GQA enabled: {config_dict['use_gqa']})
Context Length: {config_dict['block_size']}
Rotary Embeddings: {config_dict['use_rotary']}
SwiGLU: {config_dict['use_swiglu']}
```

## Performance Characteristics

- **Efficient**: 120M parameters provide good capability-to-size ratio
- **Fast Inference**: GQA reduces memory bandwidth requirements
- **Quality**: Advanced architecture produces coherent, contextual text
- **Flexible**: Suitable for both creative and factual tasks

## Limitations

- Context length limited to 512 tokens
- May generate biased or factually incorrect content
- Not suitable for tasks requiring real-time information
- Requires custom model class for loading (provided in release)

## Ethical Considerations

This model:
- Was trained on filtered data to reduce harmful content
- May still reflect biases present in training data
- Should be used responsibly and with appropriate safeguards
- Is not intended for generating harmful or misleading content

## Citation

If you use CosmicFish in your research or applications, please cite:

```bibtex
@misc{{cosmicfish120m,
  title={{CosmicFish-120M: A Modern 120M Parameter Language Model}},
  author={{Mistyoz AI}},
  year={{2024}},
  howpublished={{\\url{{https://huggingface.co/Mistyoz-AI/CosmicFish-120M}}}}
}}
```

## Technical Details

- **Training Framework**: Custom PyTorch implementation
- **Optimization**: AdamW with cosine learning rate schedule
- **Precision**: Mixed precision training (bfloat16/float16)
- **Data Processing**: Advanced filtering and cleaning pipeline
- **Identity Calibration**: Specialized training for consistent AI personality

## Contact

For questions or issues, please contact Mistyoz AI or open an issue on the model repository.

---

*Model weights are released under Apache 2.0 license. Training code and methodologies remain proprietary to Mistyoz AI.*
"""

    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(model_card)

    logger.info("Model card created successfully")


def extract_and_clean_weights(model_path, output_dir):
    """Extract clean model weights from checkpoint and save as safetensors"""
    logger.info(f"Loading model from {model_path}")

    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Extract configuration
    if 'cosmicconf' in checkpoint:
        config = checkpoint['cosmicconf']
        logger.info("Found 'cosmicconf' in checkpoint")
    elif 'config' in checkpoint:
        config = checkpoint['config']
        logger.info("Found 'config' in checkpoint")
    else:
        logger.error("No configuration found in checkpoint!")
        sys.exit(1)

    # Extract clean model weights
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        logger.info("Extracting from 'model_state_dict'")
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
        logger.info("Extracting from 'model'")
    else:
        logger.error("No model weights found in checkpoint!")
        sys.exit(1)

    # Clean state dict (remove DDP prefixes if present)
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        # Remove 'module.' and '_orig_mod.' prefixes
        clean_key = key.replace('module.', '').replace('_orig_mod.', '')
        cleaned_state_dict[clean_key] = value

    logger.info(f"Cleaned state dict contains {len(cleaned_state_dict)} parameters")

    # Handle weight sharing (embedding and lm_head share weights in CosmicFish)
    # Remove the shared weight to avoid duplication in safetensors
    if 'lm_head.weight' in cleaned_state_dict and 'transformer.wte.weight' in cleaned_state_dict:
        # Check if they actually share memory
        if cleaned_state_dict['lm_head.weight'].data_ptr() == cleaned_state_dict['transformer.wte.weight'].data_ptr():
            logger.info("Detected weight sharing between lm_head.weight and transformer.wte.weight")
            logger.info("Removing lm_head.weight to avoid duplication (will be tied during loading)")
            # Remove the lm_head.weight since it's shared with wte.weight
            del cleaned_state_dict['lm_head.weight']
        else:
            logger.info("lm_head.weight and transformer.wte.weight are separate tensors")

    # Calculate total parameters (including shared weights)
    total_params = sum(p.numel() for p in cleaned_state_dict.values())

    # Add back the shared weight count for accurate parameter reporting
    if 'transformer.wte.weight' in cleaned_state_dict:
        wte_params = cleaned_state_dict['transformer.wte.weight'].numel()
        total_params += wte_params  # Add it back since lm_head shares these weights

    logger.info(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    # Save as safetensors (modern standard)
    safetensors_path = os.path.join(output_dir, "model.safetensors")
    save_file(cleaned_state_dict, safetensors_path)
    logger.info(f"✅ Safetensors weights saved to {safetensors_path}")

    # Get file size
    safetensors_size = os.path.getsize(safetensors_path) / (1024 * 1024)
    logger.info(f"📁 Model file size: {safetensors_size:.1f} MB")

    return config, total_params


def create_license_file(output_dir):
    """Create Apache 2.0 license file"""
    license_text = """Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

1. Definitions.

"License" shall mean the terms and conditions for use, reproduction,
and distribution as defined by Sections 1 through 9 of this document.

"Licensor" shall mean the copyright owner or entity granting the License.

"Legal Entity" shall mean the union of the acting entity and all
other entities that control, are controlled by, or are under common
control with that entity. For the purposes of control, a
subsidiary of the entity.

[... rest of Apache 2.0 license text ...]

Copyright 2024 Mistyoz AI

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

    with open(os.path.join(output_dir, "LICENSE"), "w") as f:
        f.write(license_text)

    logger.info("License file created")


def copy_model_code(output_dir):
    """Copy the necessary model code for users"""
    logger.info("Copying model implementation...")

    # Copy model.py to the output directory
    if os.path.exists("model.py"):
        shutil.copy2("model.py", os.path.join(output_dir, "modeling_cosmicfish.py"))
        logger.info("Model code copied as modeling_cosmicfish.py")
    else:
        logger.warning("model.py not found in current directory")


def prepare_for_release(model_path, output_dir="./cosmicfish-release"):
    """Main function to prepare model for Hugging Face release"""
    logger.info("Starting CosmicFish release preparation...")
    logger.info("🔒 Using safetensors format (modern standard)")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Extract weights and config (now saves as safetensors)
    config, total_params = extract_and_clean_weights(model_path, output_dir)

    # Create config.json
    config_dict = create_config_json(config)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info("config.json created")

    # Create tokenizer files
    create_tokenizer_files(output_dir)

    # Create model card
    create_model_card(config_dict, output_dir)

    # Create license
    create_license_file(output_dir)

    # Copy model code
    copy_model_code(output_dir)

    # Create a simple loading example (updated for safetensors)
    example_code = '''"""
Example usage of CosmicFish model (using safetensors)
"""
import torch
from transformers import GPT2Tokenizer
from modeling_cosmicfish import CosmicFish, CosmicConfig
from safetensors.torch import load_file
import json

def load_cosmicfish(model_dir):
    """Load CosmicFish model and tokenizer"""
    # Load config
    with open(f"{model_dir}/config.json", "r") as f:
        config_dict = json.load(f)

    # Create CosmicConfig
    config = CosmicConfig(
        vocab_size=config_dict["vocab_size"],
        block_size=config_dict["block_size"],
        n_layer=config_dict["n_layer"],
        n_head=config_dict["n_head"],
        n_embd=config_dict["n_embd"],
        bias=config_dict["bias"],
        dropout=0.0,  # Set to 0 for inference
        use_rotary=config_dict["use_rotary"],
        use_swiglu=config_dict["use_swiglu"],
        use_gqa=config_dict["use_gqa"],
        n_query_groups=config_dict["n_query_groups"],
        use_qk_norm=config_dict["use_qk_norm"]
    )

    # Create model
    model = CosmicFish(config)

    # Load weights from safetensors (safer and faster)
    state_dict = load_file(f"{model_dir}/model.safetensors")

    # Handle weight sharing: lm_head.weight shares with transformer.wte.weight
    if 'lm_head.weight' not in state_dict and 'transformer.wte.weight' in state_dict:
        print("Weight sharing detected: tying lm_head.weight to transformer.wte.weight")
        state_dict['lm_head.weight'] = state_dict['transformer.wte.weight']

    model.load_state_dict(state_dict)
    model.eval()

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    return model, tokenizer

# Example usage:
# model, tokenizer = load_cosmicfish("./")
# input_text = "The future of AI is"
# inputs = tokenizer.encode(input_text, return_tensors="pt")
# outputs = model.generate(inputs, max_length=50, temperature=0.7, do_sample=True)
# response = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(response)
'''

    with open(os.path.join(output_dir, "example_usage.py"), "w") as f:
        f.write(example_code)

    logger.info("Example usage file created")

    # Summary
    logger.info("=" * 60)
    logger.info("🎉 CosmicFish release preparation completed!")
    logger.info(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    logger.info(f"Output directory: {output_dir}")
    logger.info("Files created:")
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path) / (1024 * 1024)
            logger.info(f"  - {file} ({size:.1f} MB)")

    logger.info("\n🚀 Next steps:")
    logger.info("1. Review the generated files")
    logger.info("2. Test loading with the example script")
    logger.info("3. Upload to Hugging Face Hub")
    logger.info("4. Update model card if needed")
    logger.info("\n✅ Using modern safetensors format for better security and performance!")
    logger.info("=" * 60)


if __name__ == "__main__":
    MODEL_PATH = "/home/akhil/Documents/Mistyoz_AI/CosmicFish/CosmicFish_release/CF90Q.pt"
    OUTPUT_DIR = "./cosmicfish-hf-release"

    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found: {MODEL_PATH}")
        sys.exit(1)

    prepare_for_release(MODEL_PATH, OUTPUT_DIR)
