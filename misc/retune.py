"""
Re-fine-tuning script for CosmicFish on math datasets.
Builds on an already fine-tuned Alpaca model and adds math reasoning capabilities.
Uses LOWER learning rate to preserve existing conversational skills while adding math.
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

# Import the improved CosmicFish model
from model import CosmicFish, CosmicConfig

# Add safe globals for PyTorch 2.6+ compatibility
from torch.serialization import add_safe_globals

add_safe_globals([CosmicConfig])

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


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
    """Set up the model and optimizer for re-fine-tuning."""

    # Load the Alpaca fine-tuned model
    if not os.path.exists(args.alpaca_ckpt):
        raise ValueError(f"Alpaca checkpoint not found at {args.alpaca_ckpt}")

    logger.info(f"Loading Alpaca fine-tuned model from {args.alpaca_ckpt}")
    logger.info("⚠️  Using LOWER learning rate to preserve conversational skills!")

    # Load checkpoint
    try:
        checkpoint = torch.load(args.alpaca_ckpt, map_location=device, weights_only=True)
        logger.info("Loaded checkpoint with weights_only=True (secure mode)")
    except Exception as e:
        logger.warning(f"Failed to load with weights_only=True: {e}")
        logger.info("Falling back to weights_only=False (trusted checkpoint)")
        checkpoint = torch.load(args.alpaca_ckpt, map_location=device, weights_only=False)

    # Extract config from checkpoint
    if 'cosmicconf' in checkpoint:
        logger.info("Using configuration from checkpoint (cosmicconf)")
        config = checkpoint['cosmicconf']
    elif 'config' in checkpoint:
        logger.info("Using configuration from checkpoint (config)")
        config = checkpoint['config']
    else:
        logger.warning("No configuration found in checkpoint, using default values")
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
            'n_query_groups': 4,
            'use_qk_norm': False
        }
        config = CosmicConfig(**model_args)

    # Create the model
    model = CosmicFish(config)

    # Load state dict with prefix handling
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        raise ValueError("Could not find model weights in checkpoint")

    # Clean state dict (remove prefixes)
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        clean_key = key
        # Remove torch.compile prefix
        if clean_key.startswith('_orig_mod.'):
            clean_key = clean_key[10:]
        # Remove DDP prefix
        if clean_key.startswith('module.'):
            clean_key = clean_key[7:]

        cleaned_state_dict[clean_key] = value

    # Load the cleaned state dict
    try:
        model.load_state_dict(cleaned_state_dict)
        logger.info("✅ Successfully loaded Alpaca model weights")
    except Exception as e:
        logger.error(f"❌ Failed to load state dict: {e}")
        raise

    model.to(device)

    # Create optimizer with LOWER learning rate for re-fine-tuning
    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(args.beta1, args.beta2),
        device_type=device
    )

    # If resuming training, load optimizer state
    if args.resume and iter_num > 0 and 'optimizer_state_dict' in checkpoint:
        logger.info(f"Resuming from iteration {iter_num}")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Update learning rate for all param groups
        lr = get_lr(iter_num, args.warmup_iters, args.learning_rate,
                    args.lr_decay_iters, args.min_lr, args.decay_type)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return model, optimizer, config


def get_batch(data, block_size, batch_size, device):
    """Get a batch of data for training or validation."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def evaluate(model, data, block_size, batch_size, device, eval_iters=20, ctx=nullcontext()):
    """Evaluate the model on validation data."""
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(data, block_size, batch_size, device)
        with ctx:
            logits, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def save_checkpoint(model, optimizer, iter_num, best_val_loss, args, config, is_best=False):
    """Save a checkpoint of the model."""
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'cosmicconf': config,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'args': vars(args)
    }

    # Save latest checkpoint
    latest_path = os.path.join(args.output_dir, 'latest.pt')
    torch.save(checkpoint, latest_path)
    logger.info(f"Saved checkpoint to {latest_path}")

    # Save best model if this is the best
    if is_best:
        best_path = os.path.join(args.output_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
        logger.info(f"💾 Saved BEST model to {best_path}")


def main():
    parser = argparse.ArgumentParser(description="Re-fine-tune CosmicFish on math datasets")

    # Model and dataset paths
    parser.add_argument("--alpaca_ckpt", type=str, required=True,
                        help="Path to your Alpaca fine-tuned checkpoint")
    parser.add_argument("--math_dataset_dir", type=str, default="data/math",
                        help="Directory containing math train.bin and val.bin")
    parser.add_argument("--output_dir", type=str, default="out/math_finetuned",
                        help="Directory to save math-enhanced model")

    # Training parameters - CONSERVATIVE for preserving existing skills
    parser.add_argument("--batch_size", type=int, default=12,
                        help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                        help="Learning rate (LOWER than initial fine-tuning to preserve skills)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="Beta1 for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.95,
                        help="Beta2 for Adam optimizer")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping value")
    parser.add_argument("--max_iters", type=int, default=3000,
                        help="Maximum iterations (1-2 epochs for math)")

    # Learning rate schedule
    parser.add_argument("--warmup_iters", type=int, default=50,
                        help="Number of iterations for learning rate warmup")
    parser.add_argument("--lr_decay_iters", type=int, default=3000,
                        help="Number of iterations for learning rate decay")
    parser.add_argument("--min_lr", type=float, default=5e-7,
                        help="Minimum learning rate")
    parser.add_argument("--decay_type", type=str, default="cosine", choices=["cosine", "linear"],
                        help="Type of learning rate decay")

    # Evaluation and checkpointing
    parser.add_argument("--eval_interval", type=int, default=200,
                        help="Interval between evaluations")
    parser.add_argument("--eval_iters", type=int, default=40,
                        help="Number of iterations to evaluate on")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Interval between log messages")
    parser.add_argument("--save_interval", type=int, default=500,
                        help="Interval between saving checkpoints")

    # Device and mixed precision
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--dtype", type=str,
                        default="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16",
                        help="Data type for mixed precision training")

    # Resuming from checkpoint
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest checkpoint if available")

    args = parser.parse_args()

    # Log important info
    logger.info("=" * 80)
    logger.info("🔢 MATH RE-FINE-TUNING FOR COSMICFISH")
    logger.info("=" * 80)
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
    logger.info(f"Base model: {args.alpaca_ckpt}")
    logger.info(f"Math dataset: {args.math_dataset_dir}")
    logger.info(f"Learning rate: {args.learning_rate} (LOWER to preserve Alpaca skills)")
    logger.info("=" * 80)

    # Check if math dataset exists
    train_path = os.path.join(args.math_dataset_dir, 'train.bin')
    val_path = os.path.join(args.math_dataset_dir, 'val.bin')

    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        logger.error(f"❌ Math dataset files not found in {args.math_dataset_dir}")
        logger.info(f"Expected files: {train_path}, {val_path}")
        logger.info("💡 Run prepare_math.py first to create the math dataset!")
        sys.exit(1)

    # Load math dataset
    logger.info(f"📚 Loading math dataset from {args.math_dataset_dir}")
    train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
    val_data = np.memmap(val_path, dtype=np.uint16, mode='r')

    logger.info(f"✅ Train data: {len(train_data):,} tokens ({len(train_data) / 1e6:.2f}M)")
    logger.info(f"✅ Val data: {len(val_data):,} tokens ({len(val_data) / 1e6:.2f}M)")

    # Prepare for training
    device = args.device

    # Configure mixed precision
    if device == 'cuda':
        if args.dtype == 'bfloat16' and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16
        ctx = torch.amp.autocast(device_type='cuda', dtype=amp_dtype)
        logger.info(f"✅ Using mixed precision training with {amp_dtype}")
    else:
        ctx = nullcontext()

    # Initialize or resume from checkpoint
    iter_num = 0
    best_val_loss = float('inf')

    # Check for latest checkpoint if resuming
    if args.resume:
        latest_path = os.path.join(args.output_dir, 'latest.pt')
        if os.path.exists(latest_path):
            try:
                checkpoint = torch.load(latest_path, map_location=device, weights_only=True)
            except:
                checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
            iter_num = checkpoint.get('iter_num', 0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            logger.info(f"📂 Resuming from iteration {iter_num} with best val loss {best_val_loss:.4f}")

    # Set up model and optimizer
    model, optimizer, config = setup_model_and_optimizer(args, device, iter_num)

    # Get the block size from the model config
    block_size = config.block_size
    logger.info(f"✅ Model loaded: {model.get_num_params() / 1e6:.1f}M parameters")
    logger.info(f"✅ Block size: {block_size}")

    # Re-fine-tuning loop
    logger.info("=" * 80)
    logger.info(f"🚀 Starting MATH re-fine-tuning from iteration {iter_num} to {args.max_iters}")
    logger.info(f"📊 Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    logger.info(f"📉 Learning rate: {args.learning_rate} (conservative to avoid catastrophic forgetting)")
    logger.info(f"⏱️  Expected epochs: ~1-2 (prevents overfitting on math)")
    logger.info("=" * 80)

    t0 = time.time()
    running_loss = 0.0

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
            X, Y = get_batch(train_data, block_size, args.batch_size, device)

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

        # Track running loss
        running_loss += total_loss

        # Logging
        if iter_num % args.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            tokens_per_sec = args.batch_size * block_size * args.gradient_accumulation_steps / dt
            avg_loss = running_loss / args.log_interval if iter_num > 0 else total_loss

            logger.info(
                f"Iter {iter_num}/{args.max_iters}: loss {avg_loss:.4f}, lr {lr:.8f}, {tokens_per_sec:.1f} tok/s")

            running_loss = 0.0
            t0 = t1

        # Evaluation
        if iter_num % args.eval_interval == 0:
            val_loss = evaluate(model, val_data, block_size, args.batch_size,
                                device, args.eval_iters, ctx)
            logger.info(f"📊 Iter {iter_num}: val_loss {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, iter_num, best_val_loss, args, config, is_best=True)
                logger.info(f"🌟 New best validation loss: {best_val_loss:.4f}")

        # Regular checkpointing
        if iter_num % args.save_interval == 0 or iter_num == args.max_iters - 1:
            save_checkpoint(model, optimizer, iter_num, best_val_loss, args, config)

        iter_num += 1

    logger.info("=" * 80)
    logger.info(f"🎉 MATH RE-FINE-TUNING COMPLETED!")
    logger.info(f"🏆 Best validation loss: {best_val_loss:.4f}")
    logger.info(f"💾 Best model saved to {os.path.join(args.output_dir, 'best_model.pt')}")
    logger.info(f"💡 Your model now has: Alpaca conversation skills + Math reasoning!")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"❌ Error during re-fine-tuning: {str(e)}", exc_info=True)
        sys.exit(1)