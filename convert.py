 #!/usr/bin/env python3
"""
Clean CosmicFish to PyTorch Mobile converter with inference testing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.mobile_optimizer as mobile_optimizer
import sys
import os
import logging
import tiktoken

# Import your model
from model import CosmicFish, CosmicConfig
from torch.serialization import add_safe_globals

add_safe_globals([CosmicConfig])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MobileWrapper(nn.Module):
    """Wrapper optimized for mobile deployment."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, input_ids):
        logits, _ = self.model(input_ids, targets=None)
        return logits


def convert_to_mobile(input_path, output_dir=None):
    """Convert CosmicFish to mobile-optimized format."""
    
    if output_dir is None:
        output_dir = os.path.dirname(input_path)
    
    base_name = os.path.basename(input_path).replace('.pt', '')
    
    logger.info(f"Loading model from {input_path}...")
    
    # Load model
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
    config = checkpoint.get('cosmicconf') or checkpoint.get('config')
    
    model = CosmicFish(config)
    state_dict = checkpoint.get('model_state_dict') or checkpoint.get('model')
    if isinstance(state_dict, dict):
        state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v
                     for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    
    logger.info(f"Model loaded: {model.get_num_params() / 1e6:.1f}M parameters")
    logger.info(f"Block size: {config.block_size}, Vocab size: {config.vocab_size}")
    
    # Wrap model
    wrapped = MobileWrapper(model)
    wrapped.eval()
    
    # Create example input for tracing
    example = torch.randint(0, config.vocab_size, (1, 128), dtype=torch.long)
    
    # 1. Create TorchScript version
    logger.info("Creating TorchScript model...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, example, strict=False)
    
    # Save regular TorchScript
    script_path = os.path.join(output_dir, f"{base_name}_script.pt")
    traced.save(script_path)
    logger.info(f"✅ TorchScript saved: {script_path}")
    
    # 2. Create mobile-optimized version
    logger.info("Optimizing for mobile...")
    optimized = mobile_optimizer.optimize_for_mobile(traced)
    
    # Save mobile version
    mobile_path = os.path.join(output_dir, f"{base_name}_mobile.ptl")
    optimized._save_for_lite_interpreter(mobile_path)
    logger.info(f"✅ Mobile model saved: {mobile_path}")
    
    # Report sizes
    original_size = os.path.getsize(input_path) / (1024 * 1024)
    script_size = os.path.getsize(script_path) / (1024 * 1024)
    mobile_size = os.path.getsize(mobile_path) / (1024 * 1024)
    
    logger.info("\n📊 Model sizes:")
    logger.info(f"Original: {original_size:.1f} MB")
    logger.info(f"TorchScript: {script_size:.1f} MB")
    logger.info(f"Mobile: {mobile_size:.1f} MB ({(1-mobile_size/original_size)*100:.1f}% reduction)")
    
    logger.info("\n✅ Conversion complete!")
    
    return script_path, mobile_path, config


def generate_text(model, tokenizer, prompt, max_tokens=100, temperature=0.7, top_k=50):
    """Generate text from a prompt using the model."""
    
    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    
    generated_ids = input_ids.copy()
    
    # Generate tokens
    with torch.no_grad():
        for _ in range(max_tokens):
            # Get model predictions
            outputs = model(input_tensor)
            logits = outputs[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).squeeze()
            
            # Stop if we hit the end token
            if next_token_id.item() == tokenizer.eot_token:
                break
            
            # Append to sequence
            generated_ids.append(next_token_id.item())
            input_tensor = torch.cat([input_tensor, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)
            
            # Truncate if we exceed block size (use 512 as default max)
            if input_tensor.shape[1] > 512:
                input_tensor = input_tensor[:, -512:]
    
    # Decode the generated text
    generated_text = tokenizer.decode(generated_ids)
    return generated_text


def test_model_with_prompts(model_path, config):
    """Test the model with various prompts."""
    
    logger.info(f"\n🧪 Testing model: {model_path}")
    
    # Load the model
    model = torch.jit.load(model_path, map_location='cpu')
    model.eval()
    
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Test prompts
    test_prompts = [
        "What is AI?",
        "What is climate change?",
        "What is the capital of France?"
    ]
    
    logger.info("\n" + "="*80)
    logger.info("Running inference tests...")
    logger.info("="*80)
    
    for prompt in test_prompts:
        logger.info(f"\n📝 Prompt: {prompt}")
        logger.info("-" * 40)
        
        try:
            # Generate response
            response = generate_text(
                model,
                tokenizer,
                prompt,
                max_tokens=50,  # Keep responses short for testing
                temperature=0.7,
                top_k=40
            )
            
            # Clean up the response
            response = response.strip()
            
            logger.info(f"🤖 Response: {response}")
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
    
    logger.info("\n" + "="*80)
    logger.info("✅ Inference testing complete!")
    
    # Also do a basic shape test
    logger.info("\n🔍 Basic shape test:")
    test_input = torch.randint(0, config.vocab_size, (1, 10), dtype=torch.long)
    with torch.no_grad():
        output = model(test_input)
        logger.info(f"Input shape: {test_input.shape} → Output shape: {output.shape}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python clean_converter_with_inference.py <input_model.pt> [output_dir]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        # Convert the model
        script_path, mobile_path, config = convert_to_mobile(input_path, output_dir)
        
        # Test the converted model with prompts
        logger.info("\n" + "🚀 Testing the converted model with your prompts..." + "\n")
        
        # Test the TorchScript version (it's faster for testing)
        test_model_with_prompts(script_path, config)
        
        logger.info("\n✨ All done! Your models are ready for deployment.")
        logger.info(f"\n📱 Mobile model: {mobile_path}")
        logger.info(f"💻 TorchScript model: {script_path}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
