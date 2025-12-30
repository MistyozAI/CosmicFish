# CosmicFish

A modern, efficient language model built from scratch with production-ready deployment capabilities. Built at Mistyoz AI.

## Overview

CosmicFish is a family of language models featuring modern architectural components and comprehensive training infrastructure. Designed for both research experimentation and mobile deployment.

## Key Features

- **Modern Architecture**: Grouped-Query Attention (GQA), Rotary Position Embeddings (RoPE), SwiGLU activation, RMSNorm
- **Training Pipeline**: Curriculum learning, mixed-precision training, distributed training support, RLHF capabilities

## Quick Start

### Training
```bash
# Prepare datasets
python prepare.py

# Train base model with curriculum learning
python train.py

# Conversational fine-tuning
python convd.py

# Identity calibration
python calib.py
```

### Inference
```bash
# Interactive chat
python chat.py --model_path models/cosmicfish.pt
```

### Deployment
```bash
# Convert to CoreML (iOS)
python coreml.py

# Convert to MLX (Apple Silicon)
python convert.py --format mlx

# Quantize model
python quantize.py --bits 4
```

## Model Specs

| Variant | Parameters | Layers | Heads | Embedding | Context |
|---------|------------|--------|-------|-----------|---------|
| CosmicFish-300M | 369M | 24 | 16 | 960 | 2048 |
| CosmicFish-120M | 121M | 12 | 16 | 704 | 512 |
| CosmicFish-90M | 91M | 10 | 16 | 640 | 512 |

## Training Loss

![Training Loss Curve CosmicFish-300M](assets/300M_TL.png)

The model shows healthy convergence from ~8.0 to ~2.5 loss over 130k steps with stable training dynamics.

## Project Structure

```
├── model.py          # Core model architecture
├── train.py          # Main training loop with curriculum learning
├── prepare.py        # Dataset preparation and tokenization
├── convd.py          # Conversational fine-tuning
├── calib.py          # Identity calibration
├── finetune.py       # General fine-tuning utilities
├── chat.py           # Interactive inference interface
├── RLHF.py           # Human feedback collection
├── convert.py        # Multi-platform export (MLX, ONNX)
├── coreml.py         # CoreML conversion for iOS
├── quantize.py       # Model quantization
├── eval.py           # Benchmark evaluation
└── test.py           # Model testing utilities
```

## Technical Highlights

- **Curriculum Learning**: Dynamic dataset mixing ratios progressing from web content to technical materials
- **Mixed Precision**: Automatic mixed precision (AMP) with bfloat16 support
- **Flash Attention**: Optimized attention mechanism for faster training
- **Torch Compile**: JIT compilation for inference speedup
- **Comprehensive Logging**: W&B integration for experiment tracking

## Requirements

```
torch>=2.0
transformers
tiktoken
wandb
coremltools
numpy
```

## License

Apache 2.0 License

## Citation

```bibtex
@software{cosmicfish2025,
  title={CosmicFish: A Modern Efficient Language Model},
  author={Akhil},
  organization={Mistyoz AI},
  year={2025}
}
```

---

Mistyoz AI, Hyderabad
