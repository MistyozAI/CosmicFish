"""
Identity Calibration fine-tuning script for CosmicFish.
Mixes identity dataset with conversational data to prevent overfitting while teaching identity.
FIXED: PyTorch 2.6+ compatibility with proper safe globals and torch.compile prefix handling
"""

import os
import sys
import time
import math
import argparse
import logging
import pickle
import numpy as np
import torch
from contextlib import nullcontext
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Import the CosmicFish model
from model import CosmicFish, CosmicConfig

# FIXED: Add safe globals for PyTorch 2.6+ compatibility
from torch.serialization import add_safe_globals
add_safe_globals([CosmicConfig])

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def clean_state_dict_keys(state_dict):
    """Clean state dict keys by removing common prefixes from different training setups"""
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
    
    return cleaned_state_dict


def diagnose_state_dict_mismatch(checkpoint_state_dict, model_state_dict):
    """Diagnose mismatches between checkpoint and model state dicts"""
    checkpoint_keys = set(checkpoint_state_dict.keys())
    model_keys = set(model_state_dict.keys())
    
    missing_keys = model_keys - checkpoint_keys
    unexpected_keys = checkpoint_keys - model_keys
    
    logger.info("=== STATE DICT DIAGNOSIS ===")
    logger.info(f"Checkpoint has {len(checkpoint_keys)} keys")
    logger.info(f"Model expects {len(model_keys)} keys")
    
    if missing_keys:
        logger.warning(f"Missing {len(missing_keys)} keys in checkpoint:")
        for key in sorted(list(missing_keys)[:5]):
            logger.warning(f"  - {key}")
        if len(missing_keys) > 5:
            logger.warning(f"  ... and {len(missing_keys) - 5} more")
    
    if unexpected_keys:
        logger.warning(f"Unexpected {len(unexpected_keys)} keys in checkpoint:")
        for key in sorted(list(unexpected_keys)[:5]):
            logger.warning(f"  + {key}")
        if len(unexpected_keys) > 5:
            logger.warning(f"  ... and {len(unexpected_keys) - 5} more")
    
    # Check for common prefix patterns
    common_prefixes = ['_orig_mod.', 'module.', '_forward_module.']
    for prefix in common_prefixes:
        if any(key.startswith(prefix) for key in checkpoint_keys):
            logger.info(f"Found '{prefix}' prefix in checkpoint keys")


def get_lr(iter_num, warmup_iters, learning_rate, lr_decay_iters, min_lr, decay_type='cosine'):
    """Learning rate scheduler with warmup and decay."""
    
    # Warmup phase
    if iter_num < warmup_iters:
        return learning_rate * iter_num / warmup_iters
    
    # Decay phase
    if iter_num > lr_decay_iters:
        return min_lr
    
    # Calculate decay ratio
    decay_ratio = (iter_num - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    
    if decay_type == 'cosine':
        # Cosine decay
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    else:
        # Linear decay
        coeff = 1.0 - decay_ratio
    
    return min_lr + coeff * (learning_rate - min_lr)


def setup_model_and_optimizer(args, device, iter_num=0):
    """Set up the model and optimizer for calibration fine-tuning."""
    
    # Load the conversational fine-tuned model
    if not os.path.exists(args.model_path):
        raise ValueError(f"Model checkpoint not found at {args.model_path}")
    
    logger.info(f"Loading conversational model from {args.model_path}")
    
    # FIXED: Use weights_only=False for backwards compatibility with custom classes
    try:
        # First try the safe way (weights_only=True) in case the checkpoint was saved properly
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
        logger.info("Loaded checkpoint with weights_only=True (secure mode)")
    except Exception as e:
        # If that fails, fall back to weights_only=False since we trust our own checkpoint
        logger.warning(f"Failed to load with weights_only=True: {e}")
        logger.info("Falling back to weights_only=False (trusted checkpoint)")
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    
    # Extract config from checkpoint
    if 'cosmicconf' in checkpoint:
        logger.info("Using configuration from checkpoint (cosmicconf)")
        config = checkpoint['cosmicconf']
    elif 'config' in checkpoint:
        logger.info("Using configuration from checkpoint (config)")
        config = checkpoint['config']
    else:
        # Try to extract configuration parameters from state dict
        logger.warning("No configuration found in checkpoint, using default values for 300M model")
        model_args = {
            'n_layer': 24,
            'n_head': 24,
            'n_embd': 960,
            'block_size': 2048,
            'bias': True,
            'vocab_size': 50257,
            'dropout': 0.1,
            'use_rotary': True,
            'use_swiglu': True,
            'use_gqa': True,
            'n_query_groups': 4
        }
        config = CosmicConfig(**model_args)
    
    # Create the model
    model = CosmicFish(config)
    
    # Load state dict with improved prefix handling
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        raise ValueError("Could not find model weights in checkpoint")
    
    # Clean the state dict keys
    cleaned_state_dict = clean_state_dict_keys(state_dict)
    
    # Load the cleaned state dict
    try:
        model.load_state_dict(cleaned_state_dict)
        logger.info("✅ Successfully loaded model weights")
    except RuntimeError as e:
        logger.error(f"❌ Failed to load state dict: {e}")
        
        # Diagnose the mismatch
        diagnose_state_dict_mismatch(cleaned_state_dict, model.state_dict())
        
        # Try loading with strict=False as a fallback
        logger.warning("🔄 Attempting to load with strict=False...")
        missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"⚠️  Missing keys: {len(missing_keys)} (these will be randomly initialized)")
        if unexpected_keys:
            logger.warning(f"⚠️  Unexpected keys: {len(unexpected_keys)} (these will be ignored)")
            
        if len(missing_keys) > len(model.state_dict()) * 0.1:  # More than 10% missing
            logger.error("❌ Too many missing keys, this checkpoint is incompatible")
            raise RuntimeError("Checkpoint incompatible - too many missing parameters")
        else:
            logger.info("✅ Loaded checkpoint with some missing/unexpected keys")
    
    model.to(device)
    logger.info(f"Model loaded: {model.get_num_params() / 1e6:.1f}M parameters")
    
    # Create optimizer with very conservative learning rate for calibration
    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(args.beta1, args.beta2),
        device_type=device
    )
    
    # If resuming training, load optimizer state
    if args.resume and iter_num > 0 and 'optimizer_state_dict' in checkpoint:
        logger.info(f"Resuming calibration from iteration {iter_num}")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Update learning rate for all param groups
        lr = get_lr(iter_num, args.warmup_iters, args.learning_rate,
                   args.lr_decay_iters, args.min_lr, args.decay_type)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    return model, optimizer, config


def get_mixed_batch(identity_data_train, identity_data_val, conv_data_train, conv_data_val, 
                   block_size, batch_size, device, split='train', identity_ratio=0.3):
    """Get a mixed batch from both identity and conversational datasets."""
    
    # Choose datasets based on split
    if split == 'train':
        identity_data = identity_data_train
        conv_data = conv_data_train
    else:
        identity_data = identity_data_val
        conv_data = conv_data_val
    
    # Determine how many examples from each dataset
    identity_count = int(batch_size * identity_ratio)
    conv_count = batch_size - identity_count
    
    # Get identity examples
    if identity_count > 0 and len(identity_data) > block_size:
        identity_ix = torch.randint(len(identity_data) - block_size, (identity_count,))
        identity_x = torch.stack([torch.from_numpy((identity_data[i:i + block_size]).astype(np.int64)) for i in identity_ix])
        identity_y = torch.stack([torch.from_numpy((identity_data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in identity_ix])
    else:
        identity_x = torch.empty((0, block_size), dtype=torch.long)
        identity_y = torch.empty((0, block_size), dtype=torch.long)
    
    # Get conversational examples
    if conv_count > 0 and len(conv_data) > block_size:
        conv_ix = torch.randint(len(conv_data) - block_size, (conv_count,))
        conv_x = torch.stack([torch.from_numpy((conv_data[i:i + block_size]).astype(np.int64)) for i in conv_ix])
        conv_y = torch.stack([torch.from_numpy((conv_data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in conv_ix])
    else:
        conv_x = torch.empty((0, block_size), dtype=torch.long)
        conv_y = torch.empty((0, block_size), dtype=torch.long)
    
    # Combine batches
    if identity_x.size(0) > 0 and conv_x.size(0) > 0:
        x = torch.cat([identity_x, conv_x], dim=0)
        y = torch.cat([identity_y, conv_y], dim=0)
    elif identity_x.size(0) > 0:
        x, y = identity_x, identity_y
    else:
        x, y = conv_x, conv_y
    
    # Shuffle the combined batch
    perm = torch.randperm(x.size(0))
    x = x[perm]
    y = y[perm]
    
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def evaluate_mixed(model, identity_data_train, identity_data_val, conv_data_train, conv_data_val,
                  block_size, batch_size, device, eval_iters=20, ctx=nullcontext()):
    """Evaluate the model on both identity and conversational data."""
    model.eval()
    
    # Evaluate on identity data
    identity_losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_mixed_batch(identity_data_train, identity_data_val, conv_data_train, conv_data_val,
                              block_size, batch_size, device, 'val', identity_ratio=1.0)  # Pure identity
        with ctx:
            logits, loss = model(X, Y)
        identity_losses[k] = loss.item()
    
    # Evaluate on conversational data
    conv_losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_mixed_batch(identity_data_train, identity_data_val, conv_data_train, conv_data_val,
                              block_size, batch_size, device, 'val', identity_ratio=0.0)  # Pure conversational
        with ctx:
            logits, loss = model(X, Y)
        conv_losses[k] = loss.item()
    
    # Evaluate on mixed data
    mixed_losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_mixed_batch(identity_data_train, identity_data_val, conv_data_train, conv_data_val,
                              block_size, batch_size, device, 'val', identity_ratio=0.3)  # Mixed
        with ctx:
            logits, loss = model(X, Y)
        mixed_losses[k] = loss.item()
    
    model.train()
    return identity_losses.mean().item(), conv_losses.mean().item(), mixed_losses.mean().item()


def save_checkpoint(model, optimizer, iter_num, best_val_loss, args, config, is_best=False):
    """Save a checkpoint of the calibrated model."""
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine filename based on whether this is the best model
    filename = 'best_calibrated.pt' if is_best else f'calibrated_{iter_num:06d}.pt'
    filepath = os.path.join(args.output_dir, filename)
    
    # Get the model to save (unwrap any compile wrappers)
    model_to_save = model
    if hasattr(model, '_orig_mod'):
        model_to_save = model._orig_mod
    
    # Create the checkpoint dictionary
    checkpoint = {
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'cosmicconf': config,
        'calibration_completed': True,  # Mark as calibrated
        'pytorch_version': torch.__version__,
        'training_stage': 'identity_calibration'
    }
    
    # Save the checkpoint
    torch.save(checkpoint, filepath)
    logger.info(f"Saved calibrated checkpoint to {filepath}")
    
    # If we're saving a regular checkpoint, also save the latest
    if not is_best:
        latest_path = os.path.join(args.output_dir, 'latest_calibrated.pt')
        torch.save(checkpoint, latest_path)


def main():
    parser = argparse.ArgumentParser(description="Calibrate CosmicFish with identity dataset while preserving conversational abilities")
    
    # Model and data paths
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to conversational fine-tuned model checkpoint")
    parser.add_argument("--identity_dir", type=str, default="data/identity",
                       help="Directory containing the identity dataset")
    parser.add_argument("--conv_dir", type=str, default="data/alpaca_gpt4_cleaned_pure",
                       help="Directory containing the conversational dataset")
    parser.add_argument("--output_dir", type=str, default="out/calibrated",
                       help="Directory to save calibrated model")
    
    # Training parameters (conservative for calibration)
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for calibration (smaller than original training)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-7,
                       help="Very low learning rate for calibration")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9,
                       help="Beta1 for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.95,
                       help="Beta2 for Adam optimizer")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                       help="Gradient clipping value")
    parser.add_argument("--max_iters", type=int, default=1000,
                       help="Maximum number of calibration iterations")
    
    # Learning rate schedule
    parser.add_argument("--warmup_iters", type=int, default=50,
                       help="Number of iterations for learning rate warmup")
    parser.add_argument("--lr_decay_iters", type=int, default=1000,
                       help="Number of iterations for learning rate decay")
    parser.add_argument("--min_lr", type=float, default=1e-7,
                       help="Minimum learning rate")
    parser.add_argument("--decay_type", type=str, default="cosine", choices=["cosine", "linear"],
                       help="Type of learning rate decay")
    
    # Calibration-specific parameters
    parser.add_argument("--identity_ratio", type=float, default=0.4,
                       help="Ratio of identity examples in each batch (0.0-1.0)")
    parser.add_argument("--early_stop_threshold", type=float, default=0.15,
                       help="Stop if conversational loss increases by this much")
    
    # Evaluation and checkpointing
    parser.add_argument("--eval_interval", type=int, default=50,
                       help="Interval between evaluations")
    parser.add_argument("--eval_iters", type=int, default=20,
                       help="Number of iterations to evaluate on")
    parser.add_argument("--log_interval", type=int, default=10,
                       help="Interval between log messages")
    parser.add_argument("--save_interval", type=int, default=100,
                       help="Interval between saving checkpoints")
    
    # Device and mixed precision
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use (cuda or cpu)")
    parser.add_argument("--dtype", type=str,
                       default="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16",
                       help="Data type for mixed precision training")
    
    # Resuming from checkpoint
    parser.add_argument("--resume", action="store_true",
                       help="Resume from the latest calibration checkpoint if available")
    
    args = parser.parse_args()
    
    # Log important info
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
    
    # Check if datasets exist
    identity_train_path = os.path.join(args.identity_dir, 'train.bin')
    identity_val_path = os.path.join(args.identity_dir, 'val.bin')
    conv_train_path = os.path.join(args.conv_dir, 'train.bin')
    conv_val_path = os.path.join(args.conv_dir, 'val.bin')
    
    for path, name in [(identity_train_path, "Identity train"), 
                       (identity_val_path, "Identity val"),
                       (conv_train_path, "Conversational train"),
                       (conv_val_path, "Conversational val")]:
        if not os.path.exists(path):
            logger.error(f"{name} dataset file not found: {path}")
            logger.info(f"Expected: {path}")
            sys.exit(1)
    
    # Load datasets
    logger.info(f"Loading datasets...")
    identity_train_data = np.memmap(identity_train_path, dtype=np.uint16, mode='r')
    identity_val_data = np.memmap(identity_val_path, dtype=np.uint16, mode='r')
    conv_train_data = np.memmap(conv_train_path, dtype=np.uint16, mode='r')
    conv_val_data = np.memmap(conv_val_path, dtype=np.uint16, mode='r')
    
    logger.info(f"Identity dataset: {len(identity_train_data):,} train, {len(identity_val_data):,} val tokens")
    logger.info(f"Conversational dataset: {len(conv_train_data):,} train, {len(conv_val_data):,} val tokens")
    
    # Prepare for training
    device = args.device
    
    # Configure mixed precision
    if device == 'cuda':
        if args.dtype == 'bfloat16' and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16
        ctx = torch.amp.autocast(device_type='cuda', dtype=amp_dtype)
        logger.info(f"Using mixed precision training with {amp_dtype}")
    else:
        ctx = nullcontext()
    
    # Initialize or resume from checkpoint
    iter_num = 0
    best_val_loss = float('inf')
    initial_conv_loss = None
    
    # Check for latest checkpoint if resuming
    if args.resume:
        latest_path = os.path.join(args.output_dir, 'latest_calibrated.pt')
        if os.path.exists(latest_path):
            try:
                checkpoint = torch.load(latest_path, map_location=device, weights_only=True)
            except:
                checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
            iter_num = checkpoint.get('iter_num', 0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            logger.info(f"Resuming calibration from iteration {iter_num} with best val loss {best_val_loss:.4f}")
    
    # Set up model and optimizer
    model, optimizer, config = setup_model_and_optimizer(args, device, iter_num)
    
    # Get the block size from the model config
    block_size = config.block_size
    logger.info(f"Block size: {block_size}")
    
    # Initial evaluation to establish baseline
    logger.info("Evaluating baseline performance...")
    identity_loss, conv_loss, mixed_loss = evaluate_mixed(
        model, identity_train_data, identity_val_data, conv_train_data, conv_val_data,
        block_size, args.batch_size, device, args.eval_iters, ctx
    )
    initial_conv_loss = conv_loss
    logger.info(f"Baseline - Identity: {identity_loss:.4f}, Conversational: {conv_loss:.4f}, Mixed: {mixed_loss:.4f}")
    
    # Calibration loop
    logger.info(f"Starting identity calibration from iteration {iter_num} to {args.max_iters}")
    logger.info(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    logger.info(f"Identity ratio: {args.identity_ratio}")
    logger.info(f"Learning rate: {args.learning_rate}")
    
    t0 = time.time()
    while iter_num < args.max_iters:
        # Update learning rate
        lr = get_lr(iter_num, args.warmup_iters, args.learning_rate,
                   args.lr_decay_iters, args.min_lr, args.decay_type)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Gradient accumulation loop
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        
        for micro_step in range(args.gradient_accumulation_steps):
            X, Y = get_mixed_batch(identity_train_data, identity_val_data, conv_train_data, conv_val_data,
                                 block_size, args.batch_size, device, 'train', args.identity_ratio)
            
            with ctx:
                logits, loss = model(X, Y)
                # Scale loss for gradient accumulation
                loss = loss / args.gradient_accumulation_steps
            
            # Accumulate loss for logging
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        # Update weights
        optimizer.step()
        
        # Logging
        if iter_num % args.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            tokens_per_sec = args.batch_size * block_size * args.gradient_accumulation_steps / dt
            logger.info(f"Iter {iter_num}: loss {total_loss:.4f}, lr {lr:.8f}, {tokens_per_sec:.1f} tokens/sec")
            t0 = t1
        
        # Evaluation
        if iter_num % args.eval_interval == 0:
            identity_loss, conv_loss, mixed_loss = evaluate_mixed(
                model, identity_train_data, identity_val_data, conv_train_data, conv_val_data,
                block_size, args.batch_size, device, args.eval_iters, ctx
            )
            
            logger.info(f"Iter {iter_num}: Identity {identity_loss:.4f}, Conv {conv_loss:.4f}, Mixed {mixed_loss:.4f}")
            
            # Check for overfitting on conversational data
            if initial_conv_loss is not None and conv_loss > initial_conv_loss + args.early_stop_threshold:
                logger.warning(f"Conversational loss increased by {conv_loss - initial_conv_loss:.4f}, stopping early!")
                break
            
            # Save best model based on mixed loss
            if mixed_loss < best_val_loss:
                best_val_loss = mixed_loss
                save_checkpoint(model, optimizer, iter_num, best_val_loss, args, config, is_best=True)
                logger.info(f"New best mixed validation loss: {best_val_loss:.4f}")
        
        # Regular checkpointing
        if iter_num % args.save_interval == 0 or iter_num == args.max_iters - 1:
            save_checkpoint(model, optimizer, iter_num, best_val_loss, args, config)
        
        iter_num += 1
    
    logger.info(f"Identity calibration completed. Best mixed validation loss: {best_val_loss:.4f}")
    logger.info(f"Calibrated model saved to {os.path.join(args.output_dir, 'best_calibrated.pt')}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error during calibration: {str(e)}", exc_info=True)
        sys.exit(1)