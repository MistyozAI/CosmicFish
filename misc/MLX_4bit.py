#!/usr/bin/env python3
"""
CosmicFish PyTorch to MLX Converter with 4-bit Quantization
Converts CosmicFish 300M model from PyTorch to MLX format and quantizes to 4-bit
"""

import os
import sys
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np

import torch
import mlx.core as mx
import mlx.nn as nn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class CosmicConfig:
    """Configuration for CosmicFish model."""
    vocab_size: int = 50257
    block_size: int = 2048
    n_layer: int = 24
    n_head: int = 24
    n_embd: int = 960
    bias: bool = True
    dropout: float = 0.1
    n_query_groups: int = 4
    eps: float = 1e-6
    use_rotary: bool = True
    use_swiglu: bool = True
    use_qk_norm: bool = False
    use_gqa: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.use_gqa:
            assert self.n_head % self.n_query_groups == 0, \
                f"n_head ({self.n_head}) must be divisible by n_query_groups ({self.n_query_groups})"

    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


# ============================================================================
# MLX Model Architecture
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention with RoPE support."""

    def __init__(self, config: CosmicConfig):
        super().__init__()

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.n_query_groups = config.n_query_groups
        self.dropout = config.dropout
        self.use_qk_norm = config.use_qk_norm

        # Calculate number of KV heads for GQA
        self.n_kv_heads = config.n_head // config.n_query_groups if config.use_gqa else config.n_head

        # Combined QKV projection
        self.qkv_proj_size = (self.n_head + 2 * self.n_kv_heads) * self.head_dim
        self.c_attn = nn.Linear(config.n_embd, self.qkv_proj_size, bias=config.bias)

        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # RoPE
        if config.use_rotary:
            self.rope = nn.RoPE(self.head_dim, traditional=True)

        # Query-Key normalization (optional)
        if config.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.eps)

    def __call__(self, x, mask=None):
        B, T, C = x.shape

        # QKV projection
        qkv = self.c_attn(x)

        # Split into Q, K, V
        q_size = self.n_head * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim

        q = qkv[:, :, :q_size]
        k = qkv[:, :, q_size:q_size + kv_size]
        v = qkv[:, :, q_size + kv_size:]

        # Reshape for multi-head attention
        q = q.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE if enabled
        if hasattr(self, 'rope'):
            q = self.rope(q)
            k = self.rope(k)

        # Apply QK normalization if enabled
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Expand K, V for GQA
        if self.n_kv_heads != self.n_head:
            repeats = self.n_head // self.n_kv_heads
            k = mx.repeat(k, repeats, axis=1)
            v = mx.repeat(v, repeats, axis=1)

        # Scaled dot-product attention
        scale = 1.0 / mx.sqrt(mx.array(self.head_dim, dtype=mx.float32))
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale

        # Apply causal mask
        if mask is None:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
            mask = mask.astype(scores.dtype)

        scores = scores + mask
        attn_weights = mx.softmax(scores, axis=-1)

        # Compute attention output
        out = attn_weights @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)

        # Output projection
        out = self.c_proj(out)

        return out


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP implementation."""

    def __init__(self, config: CosmicConfig):
        super().__init__()
        hidden_dim = 4 * config.n_embd

        self.gate = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.up = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.down = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)

    def __call__(self, x):
        gate = self.gate(x)
        up = self.up(x)
        return self.down(nn.silu(up) * gate)


class TransformerBlock(nn.Module):
    """Transformer block with pre-normalization."""

    def __init__(self, config: CosmicConfig):
        super().__init__()

        self.ln_1 = RMSNorm(config.n_embd, eps=config.eps)
        self.attn = GroupedQueryAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, eps=config.eps)

        if config.use_swiglu:
            self.mlp = SwiGLUMLP(config)
        else:
            # Traditional MLP with GELU
            self.mlp = nn.Sequential(
                nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
                nn.GELU(),
                nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
            )

    def __call__(self, x, mask=None):
        # Pre-norm architecture with residual connections
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class CosmicFishMLX(nn.Module):
    """CosmicFish model in MLX."""

    def __init__(self, config: CosmicConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # Transformer blocks
        self.blocks = [TransformerBlock(config) for _ in range(config.n_layer)]

        # Final layer norm
        self.ln_f = RMSNorm(config.n_embd, eps=config.eps)

        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def __call__(self, x, mask=None):
        # Token embeddings
        x = self.wte(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Final layer norm
        x = self.ln_f(x)

        # Language modeling head
        logits = self.lm_head(x)

        return logits


# ============================================================================
# Weight Conversion Utilities
# ============================================================================

def clean_state_dict_keys(state_dict):
    """Remove common prefixes from PyTorch state dict keys."""
    cleaned = {}
    for key, value in state_dict.items():
        clean_key = key
        if clean_key.startswith('_orig_mod.'):
            clean_key = clean_key[10:]
        if clean_key.startswith('module.'):
            clean_key = clean_key[7:]
        cleaned[clean_key] = value
    return cleaned


def pytorch_to_mlx(tensor):
    """Convert PyTorch tensor to MLX array."""
    return mx.array(tensor.cpu().float().numpy())


def build_weight_mapping(config: CosmicConfig):
    """Build mapping from PyTorch weight names to MLX weight names."""
    mapping = {}

    # Embedding weights
    mapping['transformer.wte.weight'] = 'wte.weight'

    # Transformer blocks
    for i in range(config.n_layer):
        prefix_pt = f'transformer.h.{i}'
        prefix_mlx = f'blocks[{i}]'

        # Layer norms
        mapping[f'{prefix_pt}.ln_1.weight'] = f'{prefix_mlx}.ln_1.weight'
        mapping[f'{prefix_pt}.ln_2.weight'] = f'{prefix_mlx}.ln_2.weight'

        # Attention
        mapping[f'{prefix_pt}.attn.c_attn.weight'] = f'{prefix_mlx}.attn.c_attn.weight'
        mapping[f'{prefix_pt}.attn.c_attn.bias'] = f'{prefix_mlx}.attn.c_attn.bias'
        mapping[f'{prefix_pt}.attn.c_proj.weight'] = f'{prefix_mlx}.attn.c_proj.weight'
        mapping[f'{prefix_pt}.attn.c_proj.bias'] = f'{prefix_mlx}.attn.c_proj.bias'

        # QK norm (if used)
        if config.use_qk_norm:
            mapping[f'{prefix_pt}.attn.q_norm.weight'] = f'{prefix_mlx}.attn.q_norm.weight'
            mapping[f'{prefix_pt}.attn.k_norm.weight'] = f'{prefix_mlx}.attn.k_norm.weight'

        # MLP
        if config.use_swiglu:
            mapping[f'{prefix_pt}.mlp.gate.weight'] = f'{prefix_mlx}.mlp.gate.weight'
            mapping[f'{prefix_pt}.mlp.gate.bias'] = f'{prefix_mlx}.mlp.gate.bias'
            mapping[f'{prefix_pt}.mlp.up.weight'] = f'{prefix_mlx}.mlp.up.weight'
            mapping[f'{prefix_pt}.mlp.up.bias'] = f'{prefix_mlx}.mlp.up.bias'
            mapping[f'{prefix_pt}.mlp.down.weight'] = f'{prefix_mlx}.mlp.down.weight'
            mapping[f'{prefix_pt}.mlp.down.bias'] = f'{prefix_mlx}.mlp.down.bias'
        else:
            mapping[f'{prefix_pt}.mlp.c_fc.weight'] = f'{prefix_mlx}.mlp.layers[0].weight'
            mapping[f'{prefix_pt}.mlp.c_fc.bias'] = f'{prefix_mlx}.mlp.layers[0].bias'
            mapping[f'{prefix_pt}.mlp.c_proj.weight'] = f'{prefix_mlx}.mlp.layers[2].weight'
            mapping[f'{prefix_pt}.mlp.c_proj.bias'] = f'{prefix_mlx}.mlp.layers[2].bias'

    # Final layer norm
    mapping['transformer.ln_f.weight'] = 'ln_f.weight'

    return mapping


def convert_weights(pytorch_state_dict, mlx_model, weight_mapping, config):
    """Convert PyTorch weights and load them directly into MLX model."""
    logger.info("Converting and loading weights into MLX model...")

    converted_count = 0
    skipped_count = 0

    found_lm_head = False
    wte_weight = None

    for pt_name, mlx_path in weight_mapping.items():
        if pt_name in pytorch_state_dict:
            pt_tensor = pytorch_state_dict[pt_name]
            mlx_array = pytorch_to_mlx(pt_tensor)

            if pt_name == 'lm_head.weight':
                found_lm_head = True

            if pt_name == 'transformer.wte.weight':
                wte_weight = mlx_array

            # Parse the MLX path and navigate to set the weight
            parts = mlx_path.replace('[', '.').replace(']', '').split('.')

            current = mlx_model
            for i, part in enumerate(parts[:-1]):
                if part.isdigit():
                    current = current[int(part)]
                else:
                    current = getattr(current, part)

            final_attr = parts[-1]
            if hasattr(current, final_attr):
                setattr(current, final_attr, mlx_array)
                converted_count += 1

                if converted_count % 50 == 0:
                    logger.info(f"Converted {converted_count} weights...")
            else:
                logger.warning(f"Attribute {final_attr} not found in {type(current).__name__}")
        else:
            skipped_count += 1

    logger.info(f"✅ Converted {converted_count} weights, skipped {skipped_count}")

    # Handle weight tying
    if not found_lm_head and wte_weight is not None:
        logger.info("Tying lm_head.weight with wte.weight")
        mlx_model.lm_head.weight = wte_weight
    elif found_lm_head:
        logger.info("lm_head.weight loaded from checkpoint")

    return mlx_model


def verify_conversion(pytorch_model, mlx_model, pytorch_state_dict, mlx_weights):
    """Verify the conversion by comparing parameter counts and sample weights."""
    logger.info("Verifying conversion...")

    pt_params = sum(p.numel() for p in pytorch_model.parameters())

    def count_params(params_dict):
        total = 0
        if isinstance(params_dict, dict):
            for v in params_dict.values():
                total += count_params(v)
        elif isinstance(params_dict, list):
            for v in params_dict:
                total += count_params(v)
        elif isinstance(params_dict, mx.array):
            total += params_dict.size
        return total

    mlx_params = count_params(mlx_weights)

    logger.info(f"PyTorch parameters: {pt_params:,} ({pt_params / 1e6:.2f}M)")
    logger.info(f"MLX parameters: {mlx_params:,} ({mlx_params / 1e6:.2f}M)")

    if abs(pt_params - mlx_params) < pt_params * 0.01:
        logger.info("✅ Parameter counts match!")
    else:
        logger.warning(
            f"⚠️  Parameter count difference: {abs(pt_params - mlx_params)} ({abs(pt_params - mlx_params) / pt_params * 100:.2f}%)")
        logger.info("   This may be due to weight tying between wte and lm_head")


# ============================================================================
# Main Conversion Function
# ============================================================================

def convert_cosmicfish_to_mlx(
        pytorch_checkpoint_path: str,
        output_dir: str,
        quantize_model: bool = True,
        quantize_bits: int = 4,
        quantize_group_size: int = 64
):
    """
    Convert CosmicFish model from PyTorch to MLX with optional 4-bit quantization.

    Args:
        pytorch_checkpoint_path: Path to PyTorch .pt checkpoint
        output_dir: Directory to save MLX model
        quantize_model: Whether to apply 4-bit quantization
        quantize_bits: Number of bits for quantization (4 or 8)
        quantize_group_size: Group size for quantization
    """
    logger.info("=" * 70)
    logger.info("CosmicFish PyTorch → MLX Conversion with 4-bit Quantization")
    logger.info("=" * 70)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # Step 1: Load PyTorch checkpoint
    # ========================================================================
    logger.info(f"\n📂 Loading PyTorch checkpoint from: {pytorch_checkpoint_path}")

    checkpoint = torch.load(pytorch_checkpoint_path, map_location='cpu', weights_only=False)

    # Extract configuration
    if 'cosmicconf' in checkpoint:
        config_dict = checkpoint['cosmicconf'].__dict__
        logger.info("✅ Found 'cosmicconf' in checkpoint")
    elif 'config' in checkpoint:
        config_dict = checkpoint['config'].__dict__
        logger.info("✅ Found 'config' in checkpoint")
    else:
        raise ValueError("❌ No configuration found in checkpoint!")

    config = CosmicConfig.from_dict(config_dict)
    logger.info(f"\n📋 Model Configuration:")
    logger.info(f"  Layers: {config.n_layer}")
    logger.info(f"  Heads: {config.n_head}")
    logger.info(f"  Embedding dim: {config.n_embd}")
    logger.info(f"  Vocab size: {config.vocab_size}")
    logger.info(f"  Block size: {config.block_size}")
    logger.info(f"  Query groups: {config.n_query_groups}")

    # Extract state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        raise ValueError("❌ No model weights found in checkpoint!")

    state_dict = clean_state_dict_keys(state_dict)
    logger.info(f"✅ Loaded {len(state_dict)} weight tensors")

    # ========================================================================
    # Step 2: Create PyTorch model for verification
    # ========================================================================
    logger.info("\n🔧 Creating PyTorch model for verification...")

    try:
        from model import CosmicFish as CosmicFishPyTorch
        from model import CosmicConfig as CosmicConfigPyTorch

        pt_config = CosmicConfigPyTorch(**config_dict)
        pytorch_model = CosmicFishPyTorch(pt_config)
        pytorch_model.load_state_dict(state_dict, strict=False)
        pytorch_model.eval()
        logger.info(f"✅ PyTorch model loaded: {pytorch_model.get_num_params() / 1e6:.2f}M parameters")
    except ImportError:
        logger.warning("⚠️  Could not import PyTorch model for verification")
        pytorch_model = None

    # ========================================================================
    # Step 3: Create MLX model
    # ========================================================================
    logger.info("\n🎯 Creating MLX model...")

    mlx_model = CosmicFishMLX(config)
    logger.info("✅ MLX model architecture created")

    # ========================================================================
    # Step 4: Convert weights
    # ========================================================================
    logger.info("\n⚙️  Converting weights...")

    weight_mapping = build_weight_mapping(config)
    mlx_model = convert_weights(state_dict, mlx_model, weight_mapping, config)

    mlx_weights = mlx_model.parameters()

    # ========================================================================
    # Step 5: Apply 4-bit Quantization
    # ========================================================================
    if quantize_model:
        logger.info(f"\n🔥 Applying {quantize_bits}-bit quantization (group_size={quantize_group_size})...")
        logger.info("   This will significantly reduce model size with minimal quality loss")

        # Quantize the model
        nn.quantize(mlx_model, group_size=quantize_group_size, bits=quantize_bits)

        logger.info(f"✅ Model quantized to {quantize_bits}-bit!")

        # Re-get weights after quantization
        mlx_weights = mlx_model.parameters()

    # ========================================================================
    # Step 6: Verify conversion
    # ========================================================================
    if pytorch_model is not None:
        verify_conversion(pytorch_model, mlx_model, state_dict, mlx_weights)

    # ========================================================================
    # Step 7: Save MLX model
    # ========================================================================
    logger.info("\n💾 Saving MLX model...")

    # Save configuration with quantization info
    config_to_save = asdict(config)
    if quantize_model:
        config_to_save['quantization'] = {
            'group_size': quantize_group_size,
            'bits': quantize_bits
        }

    config_path = output_path / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config_to_save, f, indent=2)
    logger.info(f"✅ Saved config to: {config_path}")

    # Save weights (excluding lm_head for weight tying)
    logger.info("Saving weights...")

    all_weights = mlx_model.parameters()

    if 'lm_head' in all_weights and 'weight' in all_weights['lm_head']:
        logger.info("  Excluding lm_head.weight (will be tied to wte.weight on load)")
        del all_weights['lm_head']['weight']
        if not all_weights['lm_head']:
            del all_weights['lm_head']

    weights_path = output_path / "model.safetensors"

    from mlx.utils import tree_flatten
    flat_weights = dict(tree_flatten(all_weights))

    mx.save_safetensors(str(weights_path), flat_weights)

    logger.info(f"✅ Saved weights to: {weights_path}")

    # Calculate file sizes
    weights_size = weights_path.stat().st_size / (1024 * 1024)
    logger.info(f"📊 Model size: {weights_size:.2f} MB")

    if pytorch_model is not None:
        pt_params = sum(p.numel() for p in pytorch_model.parameters())
        original_size = (pt_params * 2) / (1024 * 1024)  # float16
        if quantize_model:
            expected_quantized = (pt_params * quantize_bits / 8) / (1024 * 1024)
            logger.info(f"   Original (fp16): ~{original_size:.2f} MB")
            logger.info(f"   Quantized ({quantize_bits}-bit): ~{expected_quantized:.2f} MB")
            logger.info(f"   Compression ratio: {original_size / weights_size:.2f}x")

    # ========================================================================
    # Step 8: Test inference
    # ========================================================================
    logger.info("\n🧪 Testing inference...")

    try:
        test_input = mx.array([[1, 2, 3, 4, 5]])

        with mx.stream(mx.cpu):
            logits = mlx_model(test_input)

        logger.info(f"✅ Inference test passed!")
        logger.info(f"   Input shape: {test_input.shape}")
        logger.info(f"   Output shape: {logits.shape}")
        logger.info(f"   Output dtype: {logits.dtype}")

    except Exception as e:
        logger.error(f"❌ Inference test failed: {e}")

    # ========================================================================
    # Done!
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("✅ CONVERSION COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"MLX model saved to: {output_path}")
    logger.info(f"  - config.json")
    logger.info(f"  - model.safetensors")
    if quantize_model:
        logger.info(f"\n🎉 Model quantized to {quantize_bits}-bit!")
        logger.info(f"   Size reduction: ~{(1 - weights_size / original_size) * 100:.1f}%")
    logger.info("=" * 70)


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert CosmicFish model from PyTorch to MLX with 4-bit quantization")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/Users/akhil/Documents/Mistyoz_AI/CosmicFish/Models/CF300M.pt",
        help="Path to PyTorch checkpoint (.pt file)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./CF300M_mlx",
        help="Output directory for MLX model"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        default=True,
        help="Apply quantization (default: True)"
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="Quantization bits: 4 or 8 (default: 4)"
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Quantization group size (default: 64)"
    )

    args = parser.parse_args()

    # Run conversion
    convert_cosmicfish_to_mlx(
        pytorch_checkpoint_path=args.model_path,
        output_dir=args.output_dir,
        quantize_model=args.quantize,
        quantize_bits=args.bits,
        quantize_group_size=args.group_size
    )
