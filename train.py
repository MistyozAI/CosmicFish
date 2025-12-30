import os
import sys
import time
import math
import pickle
import random
import numpy as np
import logging
from contextlib import nullcontext
from ast import literal_eval
import matplotlib.pyplot as plt  # Added for plotting

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Add wandb for monitoring
import wandb

from model import CosmicFish, CosmicConfig

# Hardcoded W&B API key
WANDB_API_KEY = "1f0ae889f79e848c6b0ed9e2c88ceaa16fb771a1"

# ========== Utility functions ==========

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(name, log_level='INFO', log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level))

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{name}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level))
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

    return logger


def configure_logging(log_level='INFO'):
    logging.basicConfig(level=getattr(logging, log_level),
                        format='%(asctime)s - %(levelname)s - %(message)s')


def setup_wandb():
    """Initialize Weights & Biases for monitoring"""
    try:
        # Login with hardcoded API key
        wandb.login(key=WANDB_API_KEY)
        logger.info("✅ W&B authentication successful")
        return True
    except Exception as e:
        logger.warning(f"⚠️ W&B authentication failed: {e}")
        logger.warning("Continuing without W&B logging")
        return False


# ========== Training parameters ==========

out_dir = 'out'
eval_interval = 500
log_interval = 1  # Update on every iteration
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# Model parameters
n_layer = 24
n_head = 24
n_embd = 960
block_size = 2048 
bias = True
vocab_size = 50257
dropout = 0.1

# New model parameters for the architectural improvements
use_rotary = True  # Use rotary positional embeddings
use_swiglu = True  # Use SwiGLU activation
use_gqa = True  # Use Grouped-Query Attention
n_query_groups = 4  # Number of query groups for GQA
use_qk_norm = False  # Whether to use query-key normalization

# Optimizer parameters
batch_size = 16
learning_rate = 8e-5
max_iters = 250000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 3.0

# Learning rate decay parameters
lr_decay = True
warmup_iters = 3000
lr_decay_iters = 250000
min_lr = learning_rate / 10

# Curriculum learning parameters - NEW!
use_curriculum = True  # Enable curriculum learning
initial_web_ratio = 0.90  # Start with 90% web content
final_web_ratio = 0.70   # End with 70% web content
initial_tech_ratio = 0.10  # Start with 10% technical content
final_tech_ratio = 0.30   # End with 30% technical content

# Dataset hybrid sampling weights - NEW!
WEB_WEIGHTS = {
    'fineweb': 0.40,      # Less than natural 64%
    'c4': 0.25,           # More than natural 16%
    'openwebtext': 0.20,  # More than natural 12%
    'wikipedia': 0.15     # More than natural 8%
}

TECH_WEIGHTS = {
    'codeparrot': 0.40,   # Less than natural 44%
    'openwebmath': 0.35,  # More than natural 33%
    'arxiv': 0.25         # More than natural 23%
}

# W&B parameters
use_wandb = True  # Enable W&B logging
wandb_project = "CosmicFish-300M"  # W&B project name
wandb_run_name = None  # Will be auto-generated if None

# Data parameters
gradient_accumulation_steps = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

# Logging
run_name = 'CosmicFish'
logging_dir = 'logs'
log_level = 'INFO'

# ========== Process command line arguments ==========

# Process command line arguments for configuration overrides
for arg in sys.argv[1:]:
    if '=' not in arg:
        # assume it's the name of a config file
        assert not arg.startswith('--')
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
    else:
        # assume it's a --key=value argument
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]  # remove '--'
        if key in globals():
            try:
                # attempt to eval it (e.g. if bool, number, etc.)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempt = val
            # ensure the types match ok
            assert type(attempt) == type(globals()[key])
            # cross fingers
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")

# ========== Curriculum Learning Functions ==========

def get_curriculum_ratios(iter_num, total_iters):
    """Calculate current web/technical ratios based on training progress"""
    if not use_curriculum:
        return 0.85, 0.15  # Default ratios if curriculum disabled

    progress = iter_num / total_iters
    web_ratio = initial_web_ratio - (progress * (initial_web_ratio - final_web_ratio))
    tech_ratio = initial_tech_ratio + (progress * (final_tech_ratio - initial_tech_ratio))

    # Ensure ratios sum to 1.0
    total = web_ratio + tech_ratio
    web_ratio /= total
    tech_ratio /= total

    return web_ratio, tech_ratio


def load_all_datasets():
    """Load all available datasets for curriculum learning"""
    data_dir = 'data'
    datasets = {}

    # Define all possible datasets
    web_datasets = ['wikipedia', 'openwebtext', 'c4', 'fineweb']
    tech_datasets = ['arxiv', 'openwebmath', 'codeparrot']

    logger.info("Loading datasets for curriculum learning...")

    for dataset_name in web_datasets + tech_datasets:
        train_path = os.path.join(data_dir, dataset_name, 'train.bin')
        val_path = os.path.join(data_dir, dataset_name, 'val.bin')

        if os.path.exists(train_path) and os.path.exists(val_path):
            datasets[dataset_name] = {
                'train': np.memmap(train_path, dtype=np.uint16, mode='r'),
                'val': np.memmap(val_path, dtype=np.uint16, mode='r')
            }
            train_size = len(datasets[dataset_name]['train'])
            val_size = len(datasets[dataset_name]['val'])
            logger.info(f"✓ Loaded {dataset_name}: {train_size:,} train tokens, {val_size:,} val tokens")
        else:
            logger.warning(f"⚠ Dataset {dataset_name} not found, skipping")

    if not datasets:
        logger.error("❌ No datasets found! Please run prepare.py first")
        sys.exit(1)

    # Categorize datasets
    available_web = [name for name in web_datasets if name in datasets]
    available_tech = [name for name in tech_datasets if name in datasets]

    logger.info(f"Available web datasets: {available_web}")
    logger.info(f"Available technical datasets: {available_tech}")

    return datasets


def distribute_samples_among_datasets(total_samples, weights, available_datasets):
    """
    Distribute samples among datasets ensuring exact total
    """
    if total_samples == 0:
        return {}

    # Calculate initial samples for each dataset
    samples_per_dataset = {}
    allocated_samples = 0

    for dataset_name, weight in weights.items():
        if dataset_name in available_datasets:
            samples = int(total_samples * weight)
            samples_per_dataset[dataset_name] = samples
            allocated_samples += samples

    # Distribute remaining samples to datasets with highest weights
    remaining = total_samples - allocated_samples
    if remaining > 0:
        # Sort datasets by weight (descending) to prioritize higher-weighted datasets
        sorted_datasets = sorted(
            [(name, weight) for name, weight in weights.items() if name in available_datasets],
            key=lambda x: x[1], reverse=True
        )

        # Add one sample to highest weighted datasets until we reach exact total
        for i in range(remaining):
            dataset_name = sorted_datasets[i % len(sorted_datasets)][0]
            samples_per_dataset[dataset_name] += 1

    # Verify we have exactly the right number of samples
    actual_total = sum(samples_per_dataset.values())
    assert actual_total == total_samples, f"Sample distribution error: {actual_total} != {total_samples}"

    return samples_per_dataset


def get_curriculum_batch_distribution(iter_num):
    """FIXED: Calculate curriculum distribution for FULL effective batch size (256 samples)"""
    effective_batch_size = batch_size * gradient_accumulation_steps  # 16 * 16 = 256
    web_ratio, tech_ratio = get_curriculum_ratios(iter_num, max_iters)

    # Calculate total samples for each category over FULL 256 samples
    web_samples_total = int(effective_batch_size * web_ratio)
    tech_samples_total = effective_batch_size - web_samples_total

    # Get available datasets for each category
    available_web = [name for name in ['wikipedia', 'openwebtext', 'c4', 'fineweb'] if name in datasets]
    available_tech = [name for name in ['arxiv', 'openwebmath', 'codeparrot'] if name in datasets]

    # Distribute among datasets using the FULL sample counts
    web_distribution = distribute_samples_among_datasets(web_samples_total, WEB_WEIGHTS, available_web)
    tech_distribution = distribute_samples_among_datasets(tech_samples_total, TECH_WEIGHTS, available_tech)

    # Combine distributions
    total_distribution = {**web_distribution, **tech_distribution}

    return total_distribution, web_ratio, tech_ratio


def split_distribution_across_microbatches(total_distribution, num_microbatches, microbatch_size):
    """Split the total distribution across micro-batches as evenly as possible"""
    microbatch_distributions = []
    remaining_distribution = total_distribution.copy()

    for i in range(num_microbatches):
        microbatch_dist = {}
        assigned = 0

        # Try to distribute evenly across remaining micro-batches
        remaining_microbatches = num_microbatches - i

        for dataset_name in total_distribution.keys():
            remaining = remaining_distribution.get(dataset_name, 0)
            if remaining > 0:
                # Calculate how many to assign to this micro-batch
                target_for_this_batch = remaining // remaining_microbatches
                # Add one more if there's a remainder and we're early enough
                if remaining % remaining_microbatches > (remaining_microbatches - i - 1):
                    target_for_this_batch += 1

                # Don't exceed micro-batch size limit
                can_assign = min(target_for_this_batch, microbatch_size - assigned, remaining)

                if can_assign > 0:
                    microbatch_dist[dataset_name] = can_assign
                    remaining_distribution[dataset_name] -= can_assign
                    assigned += can_assign

        # If we haven't filled the micro-batch, add more from datasets with remaining samples
        while assigned < microbatch_size:
            # Find dataset with most remaining samples
            max_dataset = max(remaining_distribution, key=remaining_distribution.get, default=None)
            if max_dataset and remaining_distribution[max_dataset] > 0:
                microbatch_dist[max_dataset] = microbatch_dist.get(max_dataset, 0) + 1
                remaining_distribution[max_dataset] -= 1
                assigned += 1
            else:
                # If no remaining samples, pad with any available dataset
                first_dataset = next(iter(total_distribution.keys()))
                microbatch_dist[first_dataset] = microbatch_dist.get(first_dataset, 0) + 1
                assigned += 1

        microbatch_distributions.append(microbatch_dist)

    return microbatch_distributions


def get_microbatch_from_distribution(split, microbatch_distribution):
    """Get one micro-batch based on specified distribution"""
    all_x = []
    all_y = []

    for dataset_name, dataset_samples in microbatch_distribution.items():
        if dataset_samples > 0 and dataset_name in datasets:
            data = datasets[dataset_name][split]
            if len(data) > block_size:
                ix = torch.randint(len(data) - block_size, (dataset_samples,))
                x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
                y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
                all_x.append(x)
                all_y.append(y)

    # Combine and shuffle
    if all_x:
        x = torch.cat(all_x, dim=0)
        y = torch.cat(all_y, dim=0)

        # Shuffle the combined batch
        perm = torch.randperm(x.size(0))
        x = x[perm]
        y = y[perm]

        x, y = x.to(device), y.to(device)
    else:
        # Fallback if no valid samples
        x = torch.zeros((batch_size, block_size), dtype=torch.long, device=device)
        y = torch.zeros((batch_size, block_size), dtype=torch.long, device=device)

    return x, y


# ========== Plotting functions ==========

def create_plot_dir():
    """Create directory for saving plots"""
    plot_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir

def plot_metrics(train_losses, val_losses, perplexities, tokens_trained, iter_nums, plot_dir):
    """Create and save plots for loss, perplexity, and training progress"""
    # Plot loss vs tokens trained
    plt.figure(figsize=(10, 6))
    plt.plot(tokens_trained, train_losses, label='Train Loss')
    plt.plot(tokens_trained, val_losses, label='Validation Loss')
    plt.xlabel('Tokens Trained')
    plt.ylabel('Loss')
    plt.title('Loss vs Tokens Trained')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'loss_vs_tokens.png'))
    plt.close()

    # Plot perplexity vs tokens trained
    plt.figure(figsize=(10, 6))
    plt.plot(tokens_trained, perplexities, label='Validation Perplexity')
    plt.xlabel('Tokens Trained')
    plt.ylabel('Perplexity')
    plt.title('Perplexity vs Tokens Trained')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'perplexity_vs_tokens.png'))
    plt.close()

    # Training progress plot (percentage complete vs loss)
    plt.figure(figsize=(10, 6))
    percent_complete = [(i / max_iters) * 100 for i in iter_nums]
    plt.plot(percent_complete, train_losses, label='Train Loss')
    plt.xlabel('Progress (%)')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'training_progress.png'))
    plt.close()

# ========== Main training code ==========

# Set up logging
logger = get_logger(run_name, log_level, logging_dir)

# Set random seed
set_seed(42)

# Enable CUDA optimizations for maximum performance
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)  # Disable slower fallbacks
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    logger.info("✅ CUDA performance optimizations enabled")

# Distributed setup
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{ddp_local_rank}'

    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % torch.cuda.device_count() == 0
    gradient_accumulation_steps //= torch.cuda.device_count()
else:
    master_process = True
    seed_offset = 0
    ddp_rank = 0
    ddp_world_size = 1

# Model configuration
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=vocab_size,
    dropout=dropout,
    use_rotary=use_rotary,
    use_swiglu=use_swiglu,
    use_gqa=use_gqa,
    n_query_groups=n_query_groups,
    use_qk_norm=use_qk_norm
)

# Initialize model
config = CosmicConfig(**model_args)
model = CosmicFish(config)

# Load model checkpoint if resuming
if init_from == 'resume':
    logger.info("Resuming from checkpoint")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from == 'scratch':
    logger.info("Initializing from scratch")
    iter_num = 0
    best_val_loss = 1e9
else:
    raise ValueError(f"Unknown init_from: {init_from}")

# Move model to appropriate device
model.to(device)

# Apply torch.compile for maximum performance
if hasattr(torch, 'compile'):
    logger.info("🚀 Applying torch.compile with max-autotune...")
    model = torch.compile(model, mode='max-autotune')
    logger.info("✅ Torch compile enabled - expect 20-30% speedup!")
else:
    logger.warning("⚠️ torch.compile not available - update PyTorch for better performance")

# Wrap model in DDP if using distributed training
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank], output_device=ddp_local_rank)

# Initialize optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type=device)

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Load all datasets for curriculum learning
datasets = load_all_datasets()

# Initialize W&B if enabled and master process
wandb_enabled = False
if use_wandb and master_process:
    wandb_enabled = setup_wandb()
    if wandb_enabled:
        # Auto-generate run name if not provided
        if wandb_run_name is None:
            wandb_run_name = f"cosmicfish-{n_layer}L-{n_embd}d-{time.strftime('%Y%m%d-%H%M%S')}"

        # Initialize W&B run
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                # Model config
                "model": {
                    "n_layer": n_layer,
                    "n_head": n_head,
                    "n_embd": n_embd,
                    "block_size": block_size,
                    "vocab_size": vocab_size,
                    "dropout": dropout,
                    "use_rotary": use_rotary,
                    "use_swiglu": use_swiglu,
                    "use_gqa": use_gqa,
                    "n_query_groups": n_query_groups,
                    "use_qk_norm": use_qk_norm,
                    "total_params": model.get_num_params() if not ddp else model.module.get_num_params()
                },
                # Training config
                "training": {
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "max_iters": max_iters,
                    "weight_decay": weight_decay,
                    "grad_clip": grad_clip,
                    "warmup_iters": warmup_iters,
                    "lr_decay_iters": lr_decay_iters,
                    "min_lr": min_lr,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "effective_batch_size": batch_size * gradient_accumulation_steps
                },
                # Curriculum config
                "curriculum": {
                    "use_curriculum": use_curriculum,
                    "initial_web_ratio": initial_web_ratio,
                    "final_web_ratio": final_web_ratio,
                    "initial_tech_ratio": initial_tech_ratio,
                    "final_tech_ratio": final_tech_ratio,
                    "web_weights": WEB_WEIGHTS,
                    "tech_weights": TECH_WEIGHTS
                },
                # Dataset info
                "datasets": {
                    "available_datasets": list(datasets.keys()),
                    "total_datasets": len(datasets)
                },
                # Performance optimizations
                "optimizations": {
                    "torch_compile": hasattr(torch, 'compile'),
                    "cuda_optimizations": torch.cuda.is_available(),
                    "flash_attention": True,
                    "bf16": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
                }
            },
            tags=["curriculum", "cosmicfish", f"{n_layer}L", f"{n_embd}d", "torch-compile", "optimized"],
            notes=f"OPTIMIZED: Torch compile + CUDA opts + curriculum over full effective batch size ({batch_size * gradient_accumulation_steps} samples)"
        )
        logger.info(f"🚀 W&B run initialized: {wandb_run_name}")

# Set up automatic mixed precision (AMP) training
if device.startswith('cuda'):
    # Use appropriate dtype based on CUDA capabilities
    if dtype == 'bfloat16' and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    ctx = torch.amp.autocast(device_type='cuda', dtype=amp_dtype)
else:
    ctx = nullcontext()


# Evaluation
@torch.no_grad()
def evaluate(model, eval_iters):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        # For evaluation, use the same distribution approach
        total_distribution, _, _ = get_curriculum_batch_distribution(iter_num)
        microbatch_distributions = split_distribution_across_microbatches(
            total_distribution, gradient_accumulation_steps, batch_size
        )
        # Just use first micro-batch for evaluation
        X, Y = get_microbatch_from_distribution('val', microbatch_distributions[0])
        with ctx:
            logits, loss = model(X, Y)
            losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out.item()


# Learning rate decay scheduler
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# Progress tracking with curriculum info
def print_progress(iteration, max_iterations, start_time, loss, lr, total_distribution, web_ratio, tech_ratio, val_loss=None):
    elapsed_time = time.time() - start_time
    percent_done = (iteration / max_iterations) * 100

    # Prepare curriculum info
    curriculum_info = f"Web:{web_ratio:.1%}/Tech:{tech_ratio:.1%}"

    # Prepare total distribution info (full 256 samples)
    samples_info = " | ".join([f"{name}:{count}" for name, count in total_distribution.items()])
    total_samples = sum(total_distribution.values())

    # Prepare progress string
    progress_str = (
        f"\rIter {iteration}/{max_iterations} ({percent_done:.1f}%) | "
        f"Loss:{loss:.4f} | LR:{lr:.6f} | {curriculum_info} | "
        f"Total 256: {samples_info} | EffectiveBatch:{total_samples}"
    )

    # Add validation loss if provided
    if val_loss is not None:
        progress_str += f" | Val:{val_loss:.4f}"

    # Print and flush to ensure immediate display
    print(progress_str, end='', flush=True)


# Training loop
def train():
    global iter_num, best_val_loss

    model.train()

    # Create the output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    # Create plot directory
    plot_dir = create_plot_dir()

    # Initialize metrics for plotting
    train_losses = []
    val_losses = []
    perplexities = []
    tokens_trained = []
    iter_nums = []

    start_time = time.time()  # Start time for overall progress tracking
    running_loss = 0

    logger.info(f"🚀 Starting curriculum training:")
    logger.info(f"📊 Initial ratio: {initial_web_ratio:.1%} web, {initial_tech_ratio:.1%} technical")
    logger.info(f"📊 Final ratio: {final_web_ratio:.1%} web, {final_tech_ratio:.1%} technical")
    logger.info(f"📚 Available datasets: {list(datasets.keys())}")
    logger.info(f"🎯 FIXED: Distribution calculated over full effective batch size ({batch_size * gradient_accumulation_steps} samples)")

    while True:
        # Update learning rate according to schedule
        lr = get_lr(iter_num) if lr_decay else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # FIXED: Calculate distribution for FULL effective batch size once per iteration
        total_distribution, web_ratio, tech_ratio = get_curriculum_batch_distribution(iter_num)

        # Split distribution across micro-batches
        microbatch_distributions = split_distribution_across_microbatches(
            total_distribution, gradient_accumulation_steps, batch_size
        )

        # Gradient accumulation loop
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # Only sync gradients on the last micro step
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)

            # Get micro-batch based on pre-calculated distribution
            X, Y = get_microbatch_from_distribution('train', microbatch_distributions[micro_step])

            # Forward pass with mixed precision
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Track loss for logging
            running_loss += loss.item()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Update weights
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Record current loss for logging
        current_loss = running_loss
        running_loss = 0

        # Calculate curriculum progress
        curriculum_progress = iter_num / max_iters
        tokens_trained_so_far = iter_num * batch_size * block_size * gradient_accumulation_steps

        # Log to W&B every iteration (if enabled)
        if wandb_enabled and master_process:
            wandb_log_dict = {
                "train/loss": current_loss,
                "train/learning_rate": lr,
                "train/tokens": tokens_trained_so_far,
                "train/iteration": iter_num,
                "train/progress": curriculum_progress * 100,
                "curriculum/web_ratio": web_ratio,
                "curriculum/tech_ratio": tech_ratio,
                "curriculum/progress": curriculum_progress,
            }

            # Log total distribution (full 256 samples)
            for dataset_name, count in total_distribution.items():
                wandb_log_dict[f"distribution_total/{dataset_name}"] = count

            wandb.log(wandb_log_dict, step=iter_num)

        # Log progress for every iteration (master process only)
        if master_process and (iter_num % log_interval == 0):
            print_progress(iter_num, max_iters, start_time, current_loss, lr, total_distribution, web_ratio, tech_ratio)

        # Periodic validation and checkpoint saving
        if iter_num % eval_interval == 0 and master_process:
            val_loss = evaluate(model, eval_iters)

            # Calculate perplexity
            perplexity = math.exp(val_loss)

            # Update metrics for plotting
            train_losses.append(current_loss)
            val_losses.append(val_loss)
            perplexities.append(perplexity)
            tokens_trained.append(tokens_trained_so_far)
            iter_nums.append(iter_num)

            # Log to W&B (validation metrics)
            if wandb_enabled:
                wandb.log({
                    "val/loss": val_loss,
                    "val/perplexity": perplexity,
                    "val/best_loss": best_val_loss
                }, step=iter_num)

            # Generate and save plots
            plot_metrics(train_losses, val_losses, perplexities, tokens_trained, iter_nums, plot_dir)

            # Update progress with validation loss
            print_progress(iter_num, max_iters, start_time, current_loss, lr, total_distribution, web_ratio, tech_ratio, val_loss)
            logger.info(f"\nIter {iter_num}: val_loss = {val_loss:.4f}, perplexity = {perplexity:.2f}, tokens = {tokens_trained_so_far}")

            # Save checkpoint if validation loss improved or if always_save_checkpoint is True
            if val_loss < best_val_loss or always_save_checkpoint:
                best_val_loss = val_loss
                logger.info(f"Saving checkpoint to {out_dir}")

                # Get the model to save (unwrap DDP)
                if ddp:
                    model_to_save = model.module
                else:
                    model_to_save = model

                # Prepare and save checkpoint
                checkpoint = {
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                    'curriculum_progress': curriculum_progress,
                    'web_ratio': web_ratio,
                    'tech_ratio': tech_ratio
                }
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

                # Log checkpoint save to W&B
                if wandb_enabled:
                    wandb.log({"checkpoint/saved": True, "checkpoint/best_val_loss": best_val_loss}, step=iter_num)

        # Increment iteration counter and check for termination
        iter_num += 1
        if iter_num > max_iters:
            print()  # New line after progress
            break

    # Finish W&B run
    if wandb_enabled:
        wandb.finish()
        logger.info("🏁 W&B run completed")


if __name__ == '__main__':
    configure_logging(log_level)

    # Run evaluation only if specified
    if eval_only:
        val_loss = evaluate(model, eval_iters)
        logger.info(f"Validation loss: {val_loss:.4f}")
    else:
        # Otherwise run training
        train()