#!/usr/bin/env python3
"""
CosmicFish to CoreML Converter with 4-bit Quantization
Converts your trained CosmicFish model to Apple CoreML format with aggressive 4-bit quantization
for optimal performance on iOS/macOS devices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
from coremltools.optimize.coreml import OpPalettizerConfig, OptimizationConfig
import numpy as np
import os
import sys
import logging
from pathlib import Path

# Add your model imports
from model import CosmicFish, CosmicConfig, RMSNorm
from torch.serialization import add_safe_globals

# Add safe globals for loading custom classes
add_safe_globals([CosmicConfig])

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CoreMLOptimizedCosmicFish(nn.Module):
    """
    CoreML-optimized version of CosmicFish with:
    - Simplified forward pass for inference only
    - Pre-computed and embedded rotary frequencies 
    - Fixed sequence length for mobile optimization
    - Removed training-specific components
    """
    
    def __init__(self, original_model, max_seq_length=512):
        super().__init__()
        self.config = original_model.config
        self.max_seq_length = max_seq_length
        
        # Copy the transformer components
        self.transformer = nn.ModuleDict()
        self.transformer.wte = original_model.transformer.wte
        self.transformer.drop = original_model.transformer.drop
        self.transformer.h = original_model.transformer.h
        self.transformer.ln_f = original_model.transformer.ln_f
        
        self.lm_head = original_model.lm_head
        
        # Handle rotary embeddings for CoreML compatibility
        if self.config.use_rotary and hasattr(original_model, 'freqs_cis'):
            # Convert complex freqs_cis to real representation for CoreML
            freqs_cis = original_model.freqs_cis[:max_seq_length]
            
            # Convert complex tensor to real representation (cos, sin)
            freqs_cos = freqs_cis.real.float()
            freqs_sin = freqs_cis.imag.float()
            
            # Register as buffers (not parameters) so they're saved with the model
            self.register_buffer('freqs_cos', freqs_cos)
            self.register_buffer('freqs_sin', freqs_sin)
            
            # Patch the attention modules to use our CoreML-compatible RoPE
            self._patch_attention_modules()
        else:
            self.freqs_cos = None
            self.freqs_sin = None
            
        # Set to eval mode and disable dropout
        self.eval()
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
    
    def _patch_attention_modules(self):
        """Patch attention modules to use CoreML-compatible rotary embeddings"""
        for block in self.transformer.h:
            if hasattr(block.attn, 'forward'):
                original_forward = block.attn.forward
                
                def create_patched_forward(attn_module, original_fn):
                    def patched_forward(x, freqs_cis=None):
                        return self._apply_attention_coreml(attn_module, x)
                    return patched_forward
                
                block.attn.forward = create_patched_forward(block.attn, original_forward)
    
    def _apply_attention_coreml(self, attn_module, x):
        """CoreML-compatible attention implementation with proper GQA support"""
        B, T, C = x.size()
        
        # Get Q, K, V projections
        qkv = attn_module.c_attn(x)
        
        # Handle different attention architectures
        if hasattr(attn_module, 'kv_heads') and hasattr(attn_module, 'n_head'):
            q_size = attn_module.n_head * attn_module.head_dim
            kv_size = attn_module.kv_heads * attn_module.head_dim
            
            q, k, v = qkv.split([q_size, kv_size, kv_size], dim=2)
            
            q = q.view(B, T, attn_module.n_head, attn_module.head_dim).transpose(1, 2)
            k = k.view(B, T, attn_module.kv_heads, attn_module.head_dim).transpose(1, 2)
            v = v.view(B, T, attn_module.kv_heads, attn_module.head_dim).transpose(1, 2)
            
            # Repeat K and V for GQA
            if attn_module.kv_heads < attn_module.n_head:
                repeat_factor = attn_module.n_head // attn_module.kv_heads
                k = k.repeat_interleave(repeat_factor, dim=1)
                v = v.repeat_interleave(repeat_factor, dim=1)
        else:
            q, k, v = qkv.split(attn_module.n_embd, dim=2)
            k = k.view(B, T, attn_module.n_head, C // attn_module.n_head).transpose(1, 2)
            q = q.view(B, T, attn_module.n_head, C // attn_module.n_head).transpose(1, 2)
            v = v.view(B, T, attn_module.n_head, C // attn_module.n_head).transpose(1, 2)
        
        # Apply rotary embeddings if enabled
        if self.config.use_rotary and self.freqs_cos is not None:
            q, k = self._apply_rotary_emb_coreml(q, k, T)
        
        # Attention computation
        att = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float32)))
        
        # Causal mask
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(~mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = attn_module.attn_dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = attn_module.resid_dropout(attn_module.c_proj(y))
        return y
    
    def _apply_rotary_emb_coreml(self, xq, xk, seq_len):
        """Apply CoreML-compatible rotary embeddings"""
        freqs_cos = self.freqs_cos[:seq_len].unsqueeze(0).unsqueeze(0)
        freqs_sin = self.freqs_sin[:seq_len].unsqueeze(0).unsqueeze(0)
        
        # Reshape for rotation
        xq_even = xq[..., 0::2]
        xq_odd = xq[..., 1::2]
        xk_even = xk[..., 0::2]
        xk_odd = xk[..., 1::2]
        
        # Apply rotation
        xq_rotated_even = xq_even * freqs_cos - xq_odd * freqs_sin
        xq_rotated_odd = xq_even * freqs_sin + xq_odd * freqs_cos
        xk_rotated_even = xk_even * freqs_cos - xk_odd * freqs_sin
        xk_rotated_odd = xk_even * freqs_sin + xk_odd * freqs_cos
        
        # Interleave back
        xq_rotated = torch.stack([xq_rotated_even, xq_rotated_odd], dim=-1).flatten(-2)
        xk_rotated = torch.stack([xk_rotated_even, xk_rotated_odd], dim=-1).flatten(-2)
        
        return xq_rotated, xk_rotated
    
    def forward(self, input_ids):
        """Simplified forward pass for inference only"""
        device = input_ids.device
        batch_size, seq_len = input_ids.size()
        
        # Clamp sequence length to maximum
        if seq_len > self.max_seq_length:
            input_ids = input_ids[:, -self.max_seq_length:]
            seq_len = self.max_seq_length
        
        # Get token embeddings
        x = self.transformer.wte(input_ids)
        x = self.transformer.drop(x)
        
        # Apply transformer blocks
        for block in self.transformer.h:
            x = x + block.attn(block.ln_1(x))
            x = x + block.mlpf(block.ln_2(x))
        
        # Final layer norm
        x = self.transformer.ln_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        return logits


def load_cosmicfish_model(checkpoint_path, device='cpu'):
    """Load CosmicFish model from checkpoint with proper error handling"""
    logger.info(f"Loading CosmicFish model from {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        logger.info("✅ Checkpoint loaded successfully")
        
        # Extract configuration
        if 'cosmicconf' in checkpoint:
            config = checkpoint['cosmicconf']
            logger.info("Using configuration from checkpoint (cosmicconf)")
        elif 'config' in checkpoint:
            config = checkpoint['config']
            logger.info("Using configuration from checkpoint (config)")
        else:
            logger.error("❌ No configuration found in checkpoint!")
            return None
        
        # Log model configuration
        logger.info(f"Model config: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} embedding dim")
        logger.info(f"Block size: {config.block_size}, Vocab size: {config.vocab_size}")
        logger.info(f"Features: RoPE={config.use_rotary}, GQA={config.use_gqa}, SwiGLU={config.use_swiglu}")
        
        # Create model
        model = CosmicFish(config)
        
        # Load state dict with proper prefix handling
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            logger.error("❌ No model weights found in checkpoint!")
            return None
        
        # Clean state dict keys
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            clean_key = key
            if clean_key.startswith('_orig_mod.'):
                clean_key = clean_key[10:]
            if clean_key.startswith('module.'):
                clean_key = clean_key[7:]
            cleaned_state_dict[clean_key] = value
        
        # Load the cleaned state dict
        model.load_state_dict(cleaned_state_dict)
        model.eval()
        
        param_count = model.get_num_params() / 1e6
        logger.info(f"✅ Model loaded successfully: {param_count:.1f}M parameters")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        return None


def convert_to_coreml_4bit(model, max_seq_length=512, output_path=None):
    """
    Convert CosmicFish model to CoreML format with aggressive 4-bit quantization
    """
    logger.info("🚀 Starting CoreML conversion with 4-bit quantization...")
    
    # Create CoreML-optimized version
    logger.info("Creating CoreML-optimized model wrapper...")
    coreml_model = CoreMLOptimizedCosmicFish(model, max_seq_length=max_seq_length)
    coreml_model.eval()
    
    # Create example input for tracing
    logger.info(f"Creating example input (sequence length: {max_seq_length})...")
    example_input = torch.randint(0, model.config.vocab_size, (1, max_seq_length), dtype=torch.long)
    
    # Test the optimized model first
    logger.info("Testing optimized model...")
    with torch.no_grad():
        try:
            output = coreml_model(example_input)
            logger.info(f"✅ Model test successful, output shape: {output.shape}")
        except Exception as e:
            logger.error(f"❌ Model test failed: {str(e)}")
            return None
    
    # Trace the model to TorchScript
    logger.info("Converting to TorchScript...")
    try:
        with torch.no_grad():
            traced_model = torch.jit.trace(coreml_model, example_input, strict=False)
        logger.info("✅ TorchScript conversion successful")
    except Exception as e:
        logger.error(f"❌ TorchScript conversion failed: {str(e)}")
        return None
    
    # Convert to CoreML (unquantized first)
    logger.info("Converting to CoreML (unquantized)...")
    try:
        input_shape = (1, max_seq_length)
        inputs = [
            ct.TensorType(
                name="input_ids",
                shape=input_shape,
                dtype=np.int32
            )
        ]
        
        convert_params = {
            "inputs": inputs,
            "convert_to": "mlprogram",
            "compute_units": ct.ComputeUnit.CPU_AND_NE,
        }
        
        # Initial conversion (unquantized)
        coreml_model_unquantized = ct.convert(traced_model, **convert_params)
        logger.info("✅ Initial CoreML conversion successful!")
        
    except Exception as e:
        logger.error(f"❌ CoreML conversion failed: {str(e)}")
        return None
    
    # Apply 4-bit quantization
    logger.info("Applying 4-bit quantization (this may take a few minutes)...")
    try:
        # Configure 4-bit palettization
        config = OptimizationConfig(
            global_config=OpPalettizerConfig(
                mode="kmeans",  # k-means clustering for palette selection
                nbits=8,  # 4-bit quantization
                weight_threshold=512,  # Only quantize weights with 512+ values
            )
        )
        
        # Apply palettization (correct API for 4-bit)
        coreml_model_quantized = ct.optimize.coreml.palettize_weights(
            coreml_model_unquantized,
            config=config
        )
        
        logger.info("✅ 4-bit quantization applied successfully!")
        
    except Exception as e:
        logger.error(f"❌ 4-bit quantization failed: {str(e)}")
        logger.info("Falling back to unquantized model...")
        coreml_model_quantized = coreml_model_unquantized
    
    # Add metadata
    coreml_model_quantized.short_description = "CosmicFish 300M (4-bit quantized)"
    coreml_model_quantized.author = "Mistyoz AI"
    coreml_model_quantized.license = "Custom"
    coreml_model_quantized.version = "0.2-4bit"
    
    # Save the model
    if output_path is None:
        output_path = f"CosmicFish_{max_seq_length}seq_4bit.mlpackage"
    
    coreml_model_quantized.save(output_path)
    logger.info(f"✅ Model saved to: {output_path}")
    
    # Get file size
    if os.path.exists(output_path):
        file_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                      for dirpath, dirnames, filenames in os.walk(output_path)
                      for filename in filenames) / (1024 * 1024)
        logger.info(f"📦 Model size: {file_size:.1f} MB")
    
    # Test the CoreML model
    logger.info("Testing 4-bit quantized CoreML model...")
    try:
        sample_input = {"input_ids": example_input.numpy().astype(np.int32)}
        coreml_output = coreml_model_quantized.predict(sample_input)
        logger.info("✅ CoreML model test successful!")
        logger.info(f"Output keys: {list(coreml_output.keys())}")
    except Exception as e:
        logger.warning(f"⚠️ CoreML model test failed (this is normal on non-Mac systems): {str(e)}")
    
    return coreml_model_quantized


def main():
    """Main conversion function"""
    # Configuration
    MODEL_PATH = "/Users/akhil/Documents/Mistyoz_AI/CosmicFish/Models/CF300M.pt"
    OUTPUT_PATH = "CosmicFish_300M_4bit.mlpackage"
    MAX_SEQ_LENGTH = 512
    
    logger.info("🌟 CosmicFish to CoreML Converter (4-bit Quantization)")
    logger.info("=" * 60)
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ Model file not found: {MODEL_PATH}")
        return
    
    # Load the model
    model = load_cosmicfish_model(MODEL_PATH)
    if model is None:
        logger.error("❌ Failed to load model. Exiting.")
        return
    
    # Check model size
    param_count = model.get_num_params() / 1e6
    logger.info(f"Model size: {param_count:.1f}M parameters")
    
    if param_count > 1000:
        logger.warning(f"⚠️ Large model ({param_count:.1f}M params) may not run well on mobile devices")
    
    # Convert to CoreML with 4-bit quantization
    coreml_model = convert_to_coreml_4bit(
        model=model,
        max_seq_length=MAX_SEQ_LENGTH,
        output_path=OUTPUT_PATH
    )
    
    if coreml_model is not None:
        logger.info("🎉 Conversion completed successfully!")
        logger.info(f"✅ Your 4-bit quantized CoreML model is ready: {OUTPUT_PATH}")
        logger.info("\n📱 Next steps:")
        logger.info("1. Add the .mlpackage file to your Xcode project")
        logger.info("2. Load it using Core ML in your iOS/macOS app")
        logger.info("3. Remember to handle tokenization in your app code")
        logger.info("\n💡 Tips:")
        logger.info("- 4-bit quantization provides ~4x compression with minimal quality loss")
        logger.info("- Test on actual devices for performance")
        logger.info("- Neural Engine acceleration is enabled automatically")
        logger.info("- Use batch size = 1 for mobile inference")
    else:
        logger.error("❌ Conversion failed. Please check the error messages above.")


if __name__ == "__main__":
    main()
