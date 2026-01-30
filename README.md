# Skye's Pretraining Pipeline

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0-ee4c2c)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/cuda-12.x-76b900)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![DDP](https://img.shields.io/badge/DDP-torchrun-orange)](https://pytorch.org/docs/stable/elastic/run.html)
[![W&B](https://img.shields.io/badge/W%26B-enabled-yellow)](https://wandb.ai/)

An efficient, model-agnostic pretraining pipeline for language models. This repository provides a complete experimental workflow, and is highly inspired by the methodology described in [`Gated Delta Networks: Improving Mamba2 with Delta Rule`](https://arxiv.org/pdf/2412.06464)

## Key Features
- **Custom Model Support**: Seamlessly switch between local Python implementations and HuggingFace Hub models via sample_config.json.
- **Performance Optimized**: Built-in support for FlashAttention, BF16 mixed precision, and Fused AdamW.
- **Distributed by Design**: Full support for Multi-GPU training via torchrun and Distributed Data Parallel (DDP).
- **Atomic Checkpointing**: Rolling checkpoint system with atomic writes to prevent corruption during hardware failures.
- **Industry Standard Eval**: Integrated with lm-evaluation-harness for zero-shot benchmarking on HellaSwag, ARC, and more.

# Getting Started

## Requirements
```
pip install torch transformers datasets wandb lm-eval numpy 
```

## Data Preparation 
Use prepare_fineweb.py to tokenize and shard the FineWeb-EDU dataset. This script generates memory-mapped .bin files for high-speed streaming during training.
```
# Process 10B tokens with 32 CPU workers
python prepare_fineweb.py --tokens 10e9 --output ./data --workers 32
```

## Configuration
The sample_config.json acts as the single source of truth for your experiments. 

| Section | Key Parameters | Description |
| :--- | :--- | :--- |
| **Model** | `source`, `module`, `config_overrides` | Defines the model origin (local vs. HF) and architectural details like `hidden_size` or `intermediate_size`. |
| **Tokenizer** | `name_or_path`, `save_with_model` | Specifies the HuggingFace tokenizer to use and whether to bundle it with saved checkpoints. |
| **Data** | `data_dir`, `seq_length` | Points to the pre-tokenized binary shards and sets the training sequence length (e.g., 4096). |
| **Training** | `learning_rate`, `warmup_tokens`, `dtype` | Manages the optimization schedule, including precision settings like `bfloat16` and `fused_adamw`. |
| **Checkpointing**| `checkpoint_dir`, `checkpoint_interval` | Configures where and how often to save rolling and final model states. |
| **Logging** | `wandb_project`, `eval_interval` | Controls experiment tracking via Weights & Biases and the frequency of validation runs. |
| **System** | `num_workers`, `pin_memory` | Handles hardware-level optimizations for data loading efficiency and reproducibility. |

## Training
The pipeline supports two primary execution modes. All hyperparameters are managed via the configuration file.

```
# Single-GPU
python train.py --config sample_config.json
# Multi-GPU (with DDP)
torchrun --nproc_per_node=8 train.py --config sample_config.json
```

*Note: Effective Batch Size = batch_size × gradient_accumulation_steps × world_size.*

## Project Structure
```
├── train.py                # Main DDP training loop & logic
├── prepare_fineweb.py      # Data streaming and tokenization utility
├── run_eval.py             # lm-eval-harness integration script
├── sample_config.json      # Hyperparameter & architecture definitions
└── models                 # (Optional) Local model implementations
    └── transformer
        ├── modeling_transformer.py
        └── configuration_transformer.py
```
## Logging & Monitoring
This pipeline integrates with Weights & Biases (WandB). To track your runs, update the logging section in your config. `wandb_project: #Your project name`. `wandb_run_name: #Unique identifier for the experiment`.

## Citation
```
@software{,
  author = {Skye Gunasekaran},
  title = {Skye's Pretraining Pipeline},
  url = {github.com/SkyeGunasekaran/skyepretraining},
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=SkyeGunasekaran/skyepretraining&type=date&legend=top-left)](https://www.star-history.com/#SkyeGunasekaran/skyepretraining&type=date&legend=top-left)
