#!/usr/bin/env python3
"""
CosmicFish PyTorch to MLX Converter
Converts CosmicFish 300M model from PyTorch to MLX format
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
        # Q: n_head * head_dim, K: n_kv_heads * head_dim, V: n_kv_heads * head_dim
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
        q = q.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)  # [B, n_head, T, head_dim]
        k = k.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)  # [B, n_kv_heads, T, head_dim]
        v = v.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)  # [B, n_kv_heads, T, head_dim]

        # Apply RoPE if enabled
        if hasattr(self, 'rope'):
            q = self.rope(q)
            k = self.rope(k)

        # Apply QK normalization if enabled
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Expand K, V for GQA (repeat each KV head for multiple Q heads)
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
        out = attn_weights @ v  # [B, n_head, T, head_dim]

        # Reshape back
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

        # Language modeling head - will share weights with wte in load_weights
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

    def sanitize(self):
        """Remove weight tying to make parameters independent."""
        # This is called after loading to ensure lm_head has its own copy
        # In case weight tying caused issues
        pass

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
        # Remove torch.compile prefix
        if clean_key.startswith('_orig_mod.'):
            clean_key = clean_key[10:]
        # Remove DDP prefix
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

    # Transformer blocks - use array notation for tree_unflatten
    for i in range(config.n_layer):
        prefix_pt = f'transformer.h.{i}'
        prefix_mlx = f'blocks[{i}]'  # Use array notation

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

    # LM head will be tied to wte later

    return mapping


def convert_weights(pytorch_state_dict, mlx_model, weight_mapping, config):
    """Convert PyTorch weights and load them directly into MLX model."""
    logger.info("Converting and loading weights into MLX model...")

    converted_count = 0
    skipped_count = 0

    # Track if we found the lm_head weight
    found_lm_head = False
    wte_weight = None

    # Convert and set weights directly
    for pt_name, mlx_path in weight_mapping.items():
        if pt_name in pytorch_state_dict:
            pt_tensor = pytorch_state_dict[pt_name]
            mlx_array = pytorch_to_mlx(pt_tensor)

            # Special handling for lm_head
            if pt_name == 'lm_head.weight':
                found_lm_head = True

            # Track wte weight for tying
            if pt_name == 'transformer.wte.weight':
                wte_weight = mlx_array

            # Parse the MLX path and navigate to set the weight
            # e.g., "blocks[0].attn.c_attn.weight"
            parts = mlx_path.replace('[', '.').replace(']', '').split('.')

            # Navigate to the target
            current = mlx_model
            for i, part in enumerate(parts[:-1]):
                if part.isdigit():
                    # It's an index into a list
                    current = current[int(part)]
                else:
                    current = getattr(current, part)

            # Set the final weight
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

    # Handle weight tying for lm_head with wte
    # If lm_head.weight wasn't in the state dict, tie it to wte.weight
    if not found_lm_head and wte_weight is not None:
        logger.info("Tying lm_head.weight with wte.weight (sharing the same array)")
        mlx_model.lm_head.weight = wte_weight
    elif found_lm_head:
        logger.info("lm_head.weight loaded from checkpoint")

    return mlx_model


def verify_conversion(pytorch_model, mlx_model, pytorch_state_dict, mlx_weights):
    """Verify the conversion by comparing parameter counts and sample weights."""
    logger.info("Verifying conversion...")

    # Count parameters
    pt_params = sum(p.numel() for p in pytorch_model.parameters())

    def count_params(params_dict):
        """Count parameters in nested dict/list structure."""
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

    # Allow small difference for weight tying
    if abs(pt_params - mlx_params) < pt_params * 0.01:  # Within 1%
        logger.info("✅ Parameter counts match!")
    else:
        logger.warning(
            f"⚠️  Parameter count difference: {abs(pt_params - mlx_params)} ({abs(pt_params - mlx_params) / pt_params * 100:.2f}%)")
        logger.info("   This may be due to weight tying between wte and lm_head")

    # Sample a few weights to verify values
    logger.info("\nChecking sample weights:")

    # Check embedding weight
    if 'transformer.wte.weight' in pytorch_state_dict and 'wte' in mlx_weights:
        pt_weight = pytorch_state_dict['transformer.wte.weight'].cpu().float().numpy()
        mlx_weight = np.array(mlx_weights['wte']['weight'])

        logger.info(f"  wte.weight:")
        logger.info(f"    PyTorch - shape: {pt_weight.shape}, mean: {pt_weight.mean():.6f}, std: {pt_weight.std():.6f}")
        logger.info(
            f"    MLX     - shape: {mlx_weight.shape}, mean: {mlx_weight.mean():.6f}, std: {mlx_weight.std():.6f}")

        if np.allclose(pt_weight, mlx_weight, rtol=1e-3, atol=1e-5):
            logger.info(f"    ✅ Values match!")
        else:
            max_diff = np.abs(pt_weight - mlx_weight).max()
            logger.warning(f"    ⚠️  Values differ (max diff: {max_diff:.6f})")

    # Check final layer norm
    if 'transformer.ln_f.weight' in pytorch_state_dict and 'ln_f' in mlx_weights:
        pt_weight = pytorch_state_dict['transformer.ln_f.weight'].cpu().float().numpy()
        mlx_weight = np.array(mlx_weights['ln_f']['weight'])

        logger.info(f"  ln_f.weight:")
        logger.info(f"    PyTorch - shape: {pt_weight.shape}, mean: {pt_weight.mean():.6f}")
        logger.info(f"    MLX     - shape: {mlx_weight.shape}, mean: {mlx_weight.mean():.6f}")

        if np.allclose(pt_weight, mlx_weight, rtol=1e-3, atol=1e-5):
            logger.info(f"    ✅ Values match!")
        else:
            max_diff = np.abs(pt_weight - mlx_weight).max()
            logger.warning(f"    ⚠️  Values differ (max diff: {max_diff:.6f})")


# ============================================================================
# Main Conversion Function
# ============================================================================

def convert_cosmicfish_to_mlx(
        pytorch_checkpoint_path: str,
        output_dir: str,
        float16: bool = True
):
    """
    Convert CosmicFish model from PyTorch to MLX.

    Args:
        pytorch_checkpoint_path: Path to PyTorch .pt checkpoint
        output_dir: Directory to save MLX model
        float16: Whether to convert weights to float16 for efficiency
    """
    logger.info("=" * 70)
    logger.info("CosmicFish PyTorch → MLX Conversion")
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
    logger.info(f"  Use RoPE: {config.use_rotary}")
    logger.info(f"  Use SwiGLU: {config.use_swiglu}")
    logger.info(f"  Use GQA: {config.use_gqa}")

    # Extract state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        raise ValueError("❌ No model weights found in checkpoint!")

    # Clean state dict keys
    state_dict = clean_state_dict_keys(state_dict)
    logger.info(f"✅ Loaded {len(state_dict)} weight tensors")

    # ========================================================================
    # Step 2: Create PyTorch model for verification
    # ========================================================================
    logger.info("\n🔧 Creating PyTorch model for verification...")

    # Import the PyTorch model (assuming model.py is in same directory)
    try:
        from model import CosmicFish as CosmicFishPyTorch
        from model import CosmicConfig as CosmicConfigPyTorch

        pt_config = CosmicConfigPyTorch(**config_dict)
        pytorch_model = CosmicFishPyTorch(pt_config)
        pytorch_model.load_state_dict(state_dict, strict=False)
        pytorch_model.eval()
        logger.info(f"✅ PyTorch model loaded: {pytorch_model.get_num_params() / 1e6:.2f}M parameters")
    except ImportError:
        logger.warning("⚠️  Could not import PyTorch model for verification (model.py not found)")
        pytorch_model = None

    # ========================================================================
    # Step 3: Create MLX model
    # ========================================================================
    logger.info("\n🍎 Creating MLX model...")

    mlx_model = CosmicFishMLX(config)
    logger.info("✅ MLX model architecture created")

    # ========================================================================
    # Step 4: Convert weights
    # ========================================================================
    logger.info("\n⚙️  Converting weights...")

    weight_mapping = build_weight_mapping(config)
    mlx_model = convert_weights(state_dict, mlx_model, weight_mapping, config)

    # Get parameters for verification and saving
    mlx_weights = mlx_model.parameters()

    # Convert to float16 if requested
    if float16:
        logger.info("Converting to float16 for efficiency...")

        # Import tree_map from mlx.utils
        from mlx.utils import tree_map

        def to_float16(x):
            if isinstance(x, mx.array) and x.dtype == mx.float32:
                return x.astype(mx.float16)
            return x

        # Convert all parameters
        fp16_weights = tree_map(to_float16, mlx_weights)

        # Update the model with fp16 weights
        mlx_model.update(fp16_weights)

        # Re-get weights after conversion
        mlx_weights = mlx_model.parameters()

    # ========================================================================
    # Step 5: Verify conversion
    # ========================================================================
    if pytorch_model is not None:
        verify_conversion(pytorch_model, mlx_model, state_dict, mlx_weights)

    # ========================================================================
    # Step 6: Save MLX model
    # ========================================================================
    logger.info("\n💾 Saving MLX model...")

    # Save configuration
    config_path = output_path / "config.json"
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    logger.info(f"✅ Saved config to: {config_path}")

    # Save weights - but EXCLUDE lm_head.weight to avoid duplication
    # We'll tie it to wte.weight when loading
    logger.info("Saving weights (excluding lm_head for weight tying)...")

    # Get all parameters
    all_weights = mlx_model.parameters()

    # Remove lm_head.weight if it exists
    if 'lm_head' in all_weights and 'weight' in all_weights['lm_head']:
        logger.info("  Excluding lm_head.weight (will be tied to wte.weight on load)")
        del all_weights['lm_head']['weight']
        # If lm_head is now empty, remove it entirely
        if not all_weights['lm_head']:
            del all_weights['lm_head']

    # Save weights
    weights_path = output_path / "model.safetensors"

    # Flatten the weights for saving
    from mlx.utils import tree_flatten
    flat_weights = dict(tree_flatten(all_weights))

    # Save using MLX's save_safetensors
    mx.save_safetensors(str(weights_path), flat_weights)

    logger.info(f"✅ Saved weights to: {weights_path}")

    # Calculate file sizes
    weights_size = weights_path.stat().st_size / (1024 * 1024)
    logger.info(f"📊 Model size: {weights_size:.2f} MB (with weight tying)")

    # Calculate expected size from PyTorch
    if pytorch_model is not None:
        pt_params = sum(p.numel() for p in pytorch_model.parameters())
        expected_size = (pt_params * 2) / (1024 * 1024)  # float16 = 2 bytes
        logger.info(f"   Expected: ~{expected_size:.2f} MB for {pt_params / 1e6:.1f}M parameters")

    # ========================================================================
    # Step 7: Test inference
    # ========================================================================
    logger.info("\n🧪 Testing inference...")

    try:
        # Create dummy input
        test_input = mx.array([[1, 2, 3, 4, 5]])  # Shape: [1, 5]

        # Run forward pass
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
    logger.info("\nTo load the model in MLX:")
    logger.info("  from mlx import nn")
    logger.info("  import mlx.core as mx")
    logger.info(f"  model = CosmicFishMLX(config)")
    logger.info(f"  model.load_weights('{weights_path}')")
    logger.info("=" * 70)


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert CosmicFish model from PyTorch to MLX")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/Users/akhil/Documents/Mistyoz_AI/CosmicFish/Models/CF90M.pt",
        help="Path to PyTorch checkpoint (.pt file)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./CF300M_mlx",
        help="Output directory for MLX model"
    )
    parser.add_argument(
        "--float16",
        action="store_true",
        default=True,
        help="Convert weights to float16 for efficiency"
    )

    args = parser.parse_args()

    # Run conversion
    convert_cosmicfish_to_mlx(
        pytorch_checkpoint_path=args.model_path,
        output_dir=args.output_dir,
        float16=args.float16
    )