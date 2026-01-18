"""
CosmicFish to CoreML Converter
Converts your trained CosmicFish model to Apple CoreML format for deployment on iOS/macOS devices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
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
            freqs_cis = original_model.freqs_cis[:max_seq_length]  # Truncate to max length

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
        """
        Patch attention modules to use CoreML-compatible rotary embeddings
        """
        for block in self.transformer.h:
            if hasattr(block.attn, 'forward'):
                # Store original forward method
                original_forward = block.attn.forward

                # Create new forward method that uses our CoreML-compatible RoPE
                def create_patched_forward(attn_module, original_fn):
                    def patched_forward(x, freqs_cis=None):
                        # Override the freqs_cis with our CoreML-compatible version
                        return self._apply_attention_coreml(attn_module, x)

                    return patched_forward

                # Apply the patch
                block.attn.forward = create_patched_forward(block.attn, original_forward)

    def _apply_attention_coreml(self, attn_module, x):
        """
        CoreML-compatible attention implementation with proper GQA support
        """
        B, T, C = x.size()

        # Get Q, K, V projections
        qkv = attn_module.c_attn(x)

        # Handle different attention architectures
        if hasattr(attn_module, 'kv_heads') and hasattr(attn_module, 'n_head'):
            # GQA case - different head counts for Q vs K,V
            q_size = attn_module.n_head * attn_module.head_dim
            kv_size = attn_module.kv_heads * attn_module.head_dim

            q = qkv[:, :, :q_size]
            k = qkv[:, :, q_size:q_size + kv_size]
            v = qkv[:, :, q_size + kv_size:q_size + 2 * kv_size]

            # Reshape for attention
            q = q.view(B, T, attn_module.n_head, attn_module.head_dim).transpose(1, 2)  # [B, n_head, T, head_dim]
            k = k.view(B, T, attn_module.kv_heads, attn_module.head_dim).transpose(1, 2)  # [B, kv_heads, T, head_dim]
            v = v.view(B, T, attn_module.kv_heads, attn_module.head_dim).transpose(1, 2)  # [B, kv_heads, T, head_dim]

            # For GQA, we need to repeat K and V to match Q's head count
            # Each KV head serves multiple Q heads
            heads_per_kv = attn_module.n_head // attn_module.kv_heads
            k = k.repeat_interleave(heads_per_kv, dim=1)  # [B, n_head, T, head_dim]
            v = v.repeat_interleave(heads_per_kv, dim=1)  # [B, n_head, T, head_dim]

        else:
            # Regular multi-head attention
            head_dim = C // 3  # Assuming equal split for Q, K, V
            q, k, v = qkv.chunk(3, dim=2)
            n_head = getattr(attn_module, 'n_head', 12)  # fallback
            head_dim = C // n_head

            q = q.view(B, T, n_head, head_dim).transpose(1, 2)
            k = k.view(B, T, n_head, head_dim).transpose(1, 2)
            v = v.view(B, T, n_head, head_dim).transpose(1, 2)

        # Apply rotary embeddings if enabled
        if self.config.use_rotary and self.freqs_cos is not None:
            # Create position IDs
            position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
            q, k = self.apply_rotary_emb_coreml(q, k, position_ids)

        # Scaled dot-product attention with causal mask
        scale = 1.0 / (q.size(-1) ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float('-inf'))

        # Apply softmax and attention
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)

        # Reshape and apply output projection
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = attn_module.c_proj(out)

        return out

    def apply_rotary_emb_coreml(self, xq, xk, position_ids):
        """
        CoreML-compatible rotary embedding application using real cos/sin tensors
        """
        if self.freqs_cos is None or self.freqs_sin is None:
            return xq, xk

        # Get the frequency values for the current positions
        cos = self.freqs_cos[position_ids]  # [batch, seq_len, head_dim//2]
        sin = self.freqs_sin[position_ids]  # [batch, seq_len, head_dim//2]

        # Reshape queries and keys for rotary embedding
        # xq, xk shape: [batch, n_heads, seq_len, head_dim]
        batch_size, n_heads, seq_len, head_dim = xq.shape

        # Reshape to separate even/odd dimensions
        xq_even = xq[..., 0::2]  # [batch, n_heads, seq_len, head_dim//2]
        xq_odd = xq[..., 1::2]  # [batch, n_heads, seq_len, head_dim//2]
        xk_even = xk[..., 0::2]  # [batch, n_heads, seq_len, head_dim//2]
        xk_odd = xk[..., 1::2]  # [batch, n_heads, seq_len, head_dim//2]

        # Expand cos, sin to match the head dimension
        cos = cos.unsqueeze(1).expand_as(xq_even)  # [batch, n_heads, seq_len, head_dim//2]
        sin = sin.unsqueeze(1).expand_as(xq_even)  # [batch, n_heads, seq_len, head_dim//2]

        # Apply rotary embedding
        xq_rotated_even = xq_even * cos - xq_odd * sin
        xq_rotated_odd = xq_even * sin + xq_odd * cos
        xk_rotated_even = xk_even * cos - xk_odd * sin
        xk_rotated_odd = xk_even * sin + xk_odd * cos

        # Interleave the results back
        xq_rotated = torch.stack([xq_rotated_even, xq_rotated_odd], dim=-1).flatten(-2)
        xk_rotated = torch.stack([xk_rotated_even, xk_rotated_odd], dim=-1).flatten(-2)

        return xq_rotated, xk_rotated

    def forward(self, input_ids):
        """
        Simplified forward pass for inference only
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.size()

        # Clamp sequence length to maximum
        if seq_len > self.max_seq_length:
            input_ids = input_ids[:, -self.max_seq_length:]
            seq_len = self.max_seq_length

        # Get token embeddings
        x = self.transformer.wte(input_ids)  # [batch, seq_len, n_embd]
        x = self.transformer.drop(x)

        # Apply transformer blocks (now with patched attention)
        for block in self.transformer.h:
            # Pre-norm architecture
            x = x + block.attn(block.ln_1(x))  # Attention with residual
            x = x + block.mlpf(block.ln_2(x))  # MLP with residual

        # Final layer norm
        x = self.transformer.ln_f(x)

        # Language modeling head
        logits = self.lm_head(x)

        return logits


def load_cosmicfish_model(checkpoint_path, device='cpu'):
    """
    Load CosmicFish model from checkpoint with proper error handling
    """
    logger.info(f"Loading CosmicFish model from {checkpoint_path}")

    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        logger.info("âœ… Checkpoint loaded successfully")

        # Extract configuration
        if 'cosmicconf' in checkpoint:
            config = checkpoint['cosmicconf']
            logger.info("Using configuration from checkpoint (cosmicconf)")
        elif 'config' in checkpoint:
            config = checkpoint['config']
            logger.info("Using configuration from checkpoint (config)")
        else:
            logger.error("âŒ No configuration found in checkpoint!")
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
            logger.error("âŒ No model weights found in checkpoint!")
            return None

        # Clean state dict keys (remove DDP/compile prefixes)
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            clean_key = key
            if clean_key.startswith('_orig_mod.'):
                clean_key = clean_key[10:]  # Remove torch.compile prefix
            if clean_key.startswith('module.'):
                clean_key = clean_key[7:]  # Remove DDP prefix
            cleaned_state_dict[clean_key] = value

        # Load the cleaned state dict
        model.load_state_dict(cleaned_state_dict)
        model.eval()

        param_count = model.get_num_params() / 1e6
        logger.info(f"âœ… Model loaded successfully: {param_count:.1f}M parameters")

        return model

    except Exception as e:
        logger.error(f"âŒ Error loading model: {str(e)}")
        return None


def convert_to_coreml(model, max_seq_length=512, output_path=None, quantize=True):
    """
    Convert CosmicFish model to CoreML format
    """
    logger.info("ðŸš€ Starting CoreML conversion...")

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
            logger.info(f"âœ… Model test successful, output shape: {output.shape}")
        except Exception as e:
            logger.error(f"âŒ Model test failed: {str(e)}")
            return None

    # Trace the model to TorchScript
    logger.info("Converting to TorchScript...")
    try:
        with torch.no_grad():
            traced_model = torch.jit.trace(coreml_model, example_input, strict=False)
        logger.info("âœ… TorchScript conversion successful")
    except Exception as e:
        logger.error(f"âŒ TorchScript conversion failed: {str(e)}")
        return None

    # Convert to CoreML
    logger.info("Converting to CoreML...")
    try:
        # Define input specification
        input_shape = (1, max_seq_length)
        inputs = [
            ct.TensorType(
                name="input_ids",
                shape=input_shape,
                dtype=np.int32
            )
        ]

        # Set conversion parameters
        convert_params = {
            "inputs": inputs,
            "convert_to": "mlprogram",  # Use ML Program format for latest features
            "compute_units": ct.ComputeUnit.CPU_AND_NE,  # Target CPU and Neural Engine
        }

        # Add quantization if requested
        if quantize:
            convert_params["pass_pipeline"] = ct.PassPipeline.DEFAULT_PALETTIZATION
            logger.info("Applying quantization for smaller model size...")

        # Perform conversion
        coreml_model_converted = ct.convert(traced_model, **convert_params)

        # Add metadata
        coreml_model_converted.short_description = "CosmicFish 300M"
        coreml_model_converted.author = "Mistyoz AI"
        coreml_model_converted.license = "Custom"
        coreml_model_converted.version = "0.1"

        logger.info("âœ… CoreML conversion successful!")

        # Save the model
        if output_path is None:
            output_path = f"CosmicFish_{max_seq_length}seq_{'quantized' if quantize else 'full'}.mlpackage"

        coreml_model_converted.save(output_path)
        logger.info(f"âœ… Model saved to: {output_path}")

        # Get file size
        if os.path.exists(output_path):
            file_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                            for dirpath, dirnames, filenames in os.walk(output_path)
                            for filename in filenames) / (1024 * 1024)  # MB
            logger.info(f"ðŸ“¦ Model size: {file_size:.1f} MB")

        # Test the CoreML model
        logger.info("Testing CoreML model...")
        try:
            sample_input = {"input_ids": example_input.numpy().astype(np.int32)}
            coreml_output = coreml_model_converted.predict(sample_input)
            logger.info("âœ… CoreML model test successful!")
            logger.info(f"Output keys: {list(coreml_output.keys())}")
        except Exception as e:
            logger.warning(f"âš ï¸ CoreML model test failed (this is normal on non-Mac systems): {str(e)}")

        return coreml_model_converted

    except Exception as e:
        logger.error(f"âŒ CoreML conversion failed: {str(e)}")
        logger.error("This might be due to unsupported operations. Try reducing model complexity or sequence length.")
        return None


def main():
    """
    Main conversion function
    """
    # Configuration
    MODEL_PATH = "/Users/akhil/Documents/Mistyoz_AI/CosmicFish/CosmicFish-Aug/Models/CF90V4Q.pt"
    OUTPUT_PATH = "CosmicFish_90M.mlpackage"
    MAX_SEQ_LENGTH = 512  # Reduced for mobile compatibility
    QUANTIZE = True  # Enable quantization for smaller model size

    logger.info("ðŸŒŸ CosmicFish to CoreML Converter")
    logger.info("=" * 50)

    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        logger.error(f"âŒ Model file not found: {MODEL_PATH}")
        return

    # Load the model
    model = load_cosmicfish_model(MODEL_PATH)
    if model is None:
        logger.error("âŒ Failed to load model. Exiting.")
        return

    # Check model size for mobile compatibility
    param_count = model.get_num_params() / 1e6
    if param_count > 1000:  # > 1B parameters
        logger.warning(f"âš ï¸ Large model ({param_count:.1f}M params) may not run well on mobile devices")
        logger.warning("Consider using a smaller model or sequence length")

    # Convert to CoreML
    coreml_model = convert_to_coreml(
        model=model,
        max_seq_length=MAX_SEQ_LENGTH,
        output_path=OUTPUT_PATH,
        quantize=QUANTIZE
    )

    if coreml_model is not None:
        logger.info("ðŸŽ‰ Conversion completed successfully!")
        logger.info(f"âœ… Your CoreML model is ready: {OUTPUT_PATH}")
        logger.info("\nðŸ“± Next steps:")
        logger.info("1. Add the .mlpackage file to your Xcode project")
        logger.info("2. Load it using Core ML in your iOS/macOS app")
        logger.info("3. Remember to handle tokenization in your app code")
        logger.info("\nðŸ’¡ Tips:")
        logger.info("- Test on actual devices for performance")
        logger.info("- Consider reducing sequence length if memory is an issue")
        logger.info("- Use batch size = 1 for mobile inference")
    else:
        logger.error("âŒ Conversion failed. Please check the error messages above.")


if __name__ == "__main__":
    main()