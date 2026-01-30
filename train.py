#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import math
import os
import shutil
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TrainingConfig:
    """
    Training hyperparameters.

    This follows the methodology proposed by Songlin Yang:
    - Cosine learning rate schedule
    - AdamW optimizer with standard betas
    - Gradient clipping
    - Warmup period proportional to training budget
    """

    # Data
    data_dir: str = "./data"
    seq_length: int = 2048
    
    # Batch size: effective_batch = batch_size * grad_accum * world_size
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    
    # Optimization
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # Schedule
    warmup_tokens: float = 0.5  # Billions of tokens for warmup
    max_tokens: float = 10.0    # Billions of tokens total
    lr_decay_style: str = "cosine"
    
    # Precision & Performance
    dtype: str = "bfloat16"  # bfloat16, float16, float32
    compile_model: bool = False
    gradient_checkpointing: bool = False
    
    # Checkpointing & Logging
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 10
    eval_interval: int = 500
    eval_iters: int = 50
    checkpoint_interval: int = 1000
    
    # WandB (optional)
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    
    # System
    num_workers: int = 4
    prefetch_factor: int = 4
    seed: int = 42
    pin_memory: bool = True
    persistent_workers: bool = True


# =============================================================================
# GPU Optimizations
# =============================================================================


def setup_gpu_optimizations():
    """Configure CUDA settings for optimal performance."""
    if not torch.cuda.is_available():
        return

    # Enable TF32 for faster matmuls on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # cuDNN optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Enable efficient attention backends
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    # NCCL optimizations for multi-GPU
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_P2P_LEVEL", "NVL")


# =============================================================================
# Data Loading
# =============================================================================


class ShardedTokenDataset(IterableDataset):
    """
    Streaming dataset for pre-tokenized binary shards.

    Expects data files named: {prefix}*.bin containing uint16 token IDs.
    Handles distributed training with proper worker sharding.
    """

    def __init__(
        self,
        data_dir: Path,
        prefix: str,
        seq_length: int,
        world_size: int = 1,
        rank: int = 0,
        seed: int = 42,
        epoch: int = 0,
    ):
        self.data_dir = Path(data_dir)
        self.prefix = prefix
        self.seq_length = seq_length
        self.world_size = world_size
        self.rank = rank
        self.seed = seed
        self.epoch = epoch

        self.shard_paths = sorted(self.data_dir.glob(f"{prefix}*.bin"))
        if not self.shard_paths:
            raise ValueError(f"No shards found: {self.data_dir}/{prefix}*.bin")

    def __iter__(self):
        # Handle DataLoader workers
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = self.rank
            total_workers = self.world_size
        else:
            worker_id = self.rank * worker_info.num_workers + worker_info.id
            total_workers = self.world_size * worker_info.num_workers

        rng = np.random.default_rng(self.seed + self.epoch)

        # Shuffle shards for better randomization
        shards = list(self.shard_paths)
        rng.shuffle(shards)

        for shard_path in shards:
            data = np.memmap(shard_path, dtype=np.uint16, mode="r")
            num_sequences = (len(data) - 1) // self.seq_length

            if num_sequences <= 0:
                continue

            # Shuffle and shard indices
            indices = np.arange(num_sequences)
            rng.shuffle(indices)
            my_indices = indices[worker_id::total_workers]

            for idx in my_indices:
                pos = idx * self.seq_length
                chunk = data[pos : pos + self.seq_length + 1].astype(np.int64)
                x = torch.from_numpy(chunk[:-1])
                yield x, x


def create_dataloader(
    data_dir: str,
    prefix: str,
    seq_length: int,
    batch_size: int,
    world_size: int = 1,
    rank: int = 0,
    seed: int = 42,
    epoch: int = 0,
    num_workers: int = 4,
    prefetch_factor: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
) -> DataLoader:
    """Create a DataLoader for sharded token data."""
    dataset = ShardedTokenDataset(
        data_dir=data_dir,
        prefix=prefix,
        seq_length=seq_length,
        world_size=world_size,
        rank=rank,
        seed=seed,
        epoch=epoch,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers and num_workers > 0,
    )


# =============================================================================
# Learning Rate Schedule
# =============================================================================


def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
    decay_style: str = "cosine",
) -> float:
    """Compute learning rate with warmup and decay."""
    # Linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # After training
    if step >= max_steps:
        return min_lr

    # Decay phase
    if max_steps <= warmup_steps:
        return min_lr

    progress = (step - warmup_steps) / (max_steps - warmup_steps)

    if decay_style == "cosine":
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    elif decay_style == "linear":
        coeff = 1.0 - progress
    else:
        coeff = 1.0

    return min_lr + coeff * (max_lr - min_lr)


# =============================================================================
# Checkpointing
# =============================================================================


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    step: int,
    epoch: int,
    tokens_processed: int,
    config: TrainingConfig,
    checkpoint_dir: Path,
    is_ddp: bool = False,
) -> None:
    """
    Save a rolling checkpoint with atomic write.
    
    Overwrites previous checkpoint to save disk space.
    Uses tmp directory + rename for crash safety.
    """
    checkpoint_dir = Path(checkpoint_dir)
    latest_dir = checkpoint_dir / "latest"
    tmp_dir = checkpoint_dir / "latest.tmp"

    # Clean up any failed previous save
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    raw_model = model.module if is_ddp else model

    # Save training state
    training_state = {
        "step": step,
        "epoch": epoch,
        "tokens_processed": tokens_processed,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "config": asdict(config),
        "rng_state": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    torch.save(training_state, tmp_dir / "training_state.pt")

    # Save model weights
    torch.save(raw_model.state_dict(), tmp_dir / "pytorch_model.bin")

    # Save model config for HF compatibility
    if hasattr(raw_model, "config"):
        raw_model.config.save_pretrained(tmp_dir)

    # Atomic swap
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    tmp_dir.rename(latest_dir)

    logger.info(f"Checkpoint saved: step {step}, {tokens_processed/1e9:.2f}B tokens")


def save_final_model(
    model: nn.Module,
    checkpoint_dir: Path,
    is_ddp: bool = False,
    tokenizer_config: dict | None = None,
) -> None:
    """Save final model in HuggingFace format with tokenizer."""
    final_dir = Path(checkpoint_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    raw_model = model.module if is_ddp else model
    raw_model.save_pretrained(final_dir, safe_serialization=True)

    # Save tokenizer if configured
    if tokenizer_config and tokenizer_config.get("save_with_model", True):
        try:
            from transformers import AutoTokenizer
            tokenizer_path = tokenizer_config.get("name_or_path")
            if tokenizer_path:
                logger.info(f"Saving tokenizer from {tokenizer_path}")
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                tokenizer.save_pretrained(final_dir)
        except Exception as e:
            logger.warning(f"Could not save tokenizer: {e}")

    logger.info(f"Final model saved to {final_dir}")


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    device: torch.device = torch.device("cpu"),
) -> dict[str, Any]:
    """
    Load checkpoint from directory or legacy .pt file.
    
    Returns dict with step, epoch, tokens_processed.
    """
    path = Path(path)
    raw_model = model.module if hasattr(model, "module") else model

    if path.is_dir():
        # Directory format (preferred)
        logger.info(f"Loading checkpoint from {path}")

        # Load model weights
        model_path = path / "pytorch_model.bin"
        if not model_path.exists():
            model_path = path / "model.safetensors"

        if model_path.suffix == ".bin":
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
        else:
            from safetensors.torch import load_file
            state_dict = load_file(model_path)

        raw_model.load_state_dict(state_dict)

        # Load training state
        state_path = path / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=device, weights_only=False)

            if optimizer and "optimizer_state_dict" in state:
                optimizer.load_state_dict(state["optimizer_state_dict"])

            if scaler and state.get("scaler_state_dict"):
                scaler.load_state_dict(state["scaler_state_dict"])

            # Restore RNG state
            if "rng_state" in state:
                rng = state["rng_state"]
                if "torch" in rng:
                    torch.set_rng_state(rng["torch"])
                if "numpy" in rng:
                    np.random.set_state(rng["numpy"])
                if rng.get("cuda") and torch.cuda.is_available():
                    try:
                        torch.cuda.set_rng_state_all(rng["cuda"])
                    except Exception as e:
                        logger.warning(f"Could not restore CUDA RNG: {e}")

            return {
                "step": state["step"],
                "epoch": state.get("epoch", 0),
                "tokens_processed": state.get("tokens_processed", 0),
            }

    else:
        # Legacy single-file format
        logger.info(f"Loading legacy checkpoint {path}")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model_state_dict"])

        if optimizer and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scaler and ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        return {
            "step": ckpt.get("step", 0),
            "epoch": ckpt.get("epoch", 0),
            "tokens_processed": ckpt.get("tokens_processed", 0),
        }

    return {"step": 0, "epoch": 0, "tokens_processed": 0}


# =============================================================================
# Evaluation
# =============================================================================


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    eval_iters: int,
    ctx,
    device: torch.device,
) -> dict[str, float]:
    """Run evaluation and return metrics."""
    model.eval()
    losses = []

    for i, (x, y) in enumerate(dataloader):
        if i >= eval_iters:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with ctx:
            outputs = model(input_ids=x, labels=y)
        losses.append(outputs.loss.item())

    model.train()

    avg_loss = np.mean(losses) if losses else float("inf")
    return {
        "loss": avg_loss,
        "perplexity": np.exp(avg_loss) if avg_loss < 100 else float("inf"),
    }


# =============================================================================
# Utility Functions
# =============================================================================


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_local_class(module_name: str, class_name: str):
    """Dynamically import a class from a local module."""
    import importlib.util
    import sys
    
    # Try importing from current directory first
    module_path = Path(module_name + ".py")
    if module_path.exists():
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    else:
        # Fall back to normal import
        module = importlib.import_module(module_name)
    
    return getattr(module, class_name)


def get_model_and_config(
    model_name: str | None = None,
    model_path: str | None = None,
    config_path: str | None = None,
    config_overrides: dict | None = None,
    model_config: dict | None = None,
):
    """
    Load or create a model and config.

    Supports loading strategies:
    1. model_config dict with 'source': 'local_class' - Load from local Python files
    2. model_config dict with 'source': 'huggingface' - Load from HF hub
    3. model_name: Load from HuggingFace hub (shorthand)
    4. model_path with config_path: Load config, create model
    5. model_path alone: Load pretrained from local path
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    # Strategy 1: Full model config dict (from JSON config file)
    if model_config:
        source = model_config.get("source", "huggingface")
        overrides = model_config.get("config_overrides", {})
        
        if source == "local_class":
            # Load model and config classes from local Python files
            config_module = model_config.get("config_module")
            config_class_name = model_config.get("config_class_name")
            model_module = model_config.get("module")
            model_class_name = model_config.get("class_name")
            
            logger.info(f"Loading local classes: {model_module}.{model_class_name}")
            
            # Import config class and create config
            ConfigClass = load_local_class(config_module, config_class_name)
            config = ConfigClass(**overrides)
            
            # Import model class
            ModelClass = load_local_class(model_module, model_class_name)
            
            # Create or load model
            if model_config.get("from_pretrained") and model_config.get("pretrained_path"):
                logger.info(f"Loading pretrained weights from {model_config['pretrained_path']}")
                model = ModelClass.from_pretrained(
                    model_config["pretrained_path"],
                    config=config,
                )
            else:
                logger.info("Creating fresh model from config")
                model = ModelClass(config)
            
            return model, config
        
        elif source == "huggingface":
            # Load from HuggingFace hub or local HF-format path
            name_or_path = model_config.get("name_or_path") or model_config.get("pretrained_path")
            
            if not name_or_path:
                raise ValueError("HuggingFace source requires 'name_or_path' or 'pretrained_path'")
            
            logger.info(f"Loading from HuggingFace: {name_or_path}")
            config = AutoConfig.from_pretrained(name_or_path)
            
            for k, v in overrides.items():
                setattr(config, k, v)
            
            if model_config.get("from_pretrained", True):
                model = AutoModelForCausalLM.from_pretrained(
                    name_or_path,
                    config=config,
                    torch_dtype=torch.bfloat16,
                )
            else:
                model = AutoModelForCausalLM.from_config(config)
            
            return model, config
        
        else:
            raise ValueError(f"Unknown model source: {source}")

    # Strategy 2: Simple model_name (HuggingFace hub)
    if model_name:
        logger.info(f"Loading model from hub: {model_name}")
        config = AutoConfig.from_pretrained(model_name)
        
        if config_overrides:
            for k, v in config_overrides.items():
                setattr(config, k, v)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.bfloat16,
        )
        return model, config

    # Strategy 3: config_path (HF-format config.json)
    if config_path:
        logger.info(f"Loading config from {config_path}")
        config = AutoConfig.from_pretrained(config_path)
        
        if config_overrides:
            for k, v in config_overrides.items():
                setattr(config, k, v)
        
        if model_path:
            logger.info(f"Creating model from config (class from {model_path})")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                ignore_mismatched_sizes=True,
            )
        else:
            logger.info("Creating model from config")
            model = AutoModelForCausalLM.from_config(config)
        
        return model, config

    # Strategy 4: model_path alone (pretrained local)
    if model_path:
        logger.info(f"Loading pretrained model from {model_path}")
        config = AutoConfig.from_pretrained(model_path)
        
        if config_overrides:
            for k, v in config_overrides.items():
                setattr(config, k, v)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
        )
        return model, config

    raise ValueError("Must provide model_name, model_path, config_path, or model_config")


# =============================================================================
# Main Training Loop
# =============================================================================


def train(
    config: TrainingConfig,
    model_name: str | None = None,
    model_path: str | None = None,
    config_path: str | None = None,
    config_overrides: dict | None = None,
    model_config: dict | None = None,
    tokenizer_config: dict | None = None,
    resume_path: str | None = None,
) -> None:
    """
    Main training function.

    Args:
        config: Training configuration
        model_name: HuggingFace model name/path (e.g., 'fla-hub/gla-1.3B-100B')
        model_path: Local path to model or model class
        config_path: Path to config.json for fresh model creation
        config_overrides: Dict of config values to override
        model_config: Full model config dict (from JSON config file)
        tokenizer_config: Tokenizer config dict (from JSON config file)
        resume_path: Path to checkpoint to resume from
    """
    setup_gpu_optimizations()

    # Distributed setup
    is_ddp = "RANK" in os.environ
    if is_ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        is_master = rank == 0
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_master = True

    # Suppress logging on non-master
    if not is_master:
        logging.getLogger().setLevel(logging.WARNING)

    if is_master:
        logger.info(f"Training on {world_size} GPU(s)")
        logger.info(f"Config: {asdict(config)}")

    # Set seeds
    torch.manual_seed(config.seed + rank)
    np.random.seed(config.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed + rank)

    # Load model
    model, model_cfg = get_model_and_config(
        model_name=model_name,
        model_path=model_path,
        config_path=config_path,
        config_overrides=config_overrides,
        model_config=model_config,
    )
    model = model.to(device)

    param_count = count_parameters(model)
    if is_master:
        logger.info(f"Model: {type(model).__name__}")
        logger.info(f"Parameters: {param_count / 1e6:.1f}M")

    # Gradient checkpointing
    if config.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            if is_master:
                logger.info("Gradient checkpointing enabled")
        else:
            logger.warning("Model doesn't support gradient checkpointing")

    # Compile (PyTorch 2.0+)
    if config.compile_model:
        if is_master:
            logger.info("Compiling model...")
        model = torch.compile(model)

    # Precision
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    dtype = dtype_map[config.dtype]
    ctx = torch.autocast(device_type="cuda", dtype=dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=(config.dtype == "float16"))

    # DDP
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
        fused=torch.cuda.is_available(),
    )

    # Compute schedule
    tokens_per_step = (
        config.batch_size
        * config.seq_length
        * config.gradient_accumulation_steps
        * world_size
    )
    max_steps = int(config.max_tokens * 1e9 / tokens_per_step)
    warmup_steps = int(config.warmup_tokens * 1e9 / tokens_per_step)
    target_tokens = int(config.max_tokens * 1e9)

    if is_master:
        logger.info(f"Tokens per step: {tokens_per_step:,}")
        logger.info(f"Warmup: {config.warmup_tokens}B tokens = {warmup_steps:,} steps")
        logger.info(f"Total: {config.max_tokens}B tokens = {max_steps:,} steps")

    # Resume from checkpoint
    start_step = 0
    epoch = 0
    tokens_processed = 0

    if resume_path:
        ckpt_info = load_checkpoint(resume_path, model, optimizer, scaler, device)
        start_step = ckpt_info["step"]
        epoch = ckpt_info["epoch"]
        tokens_processed = ckpt_info.get("tokens_processed", start_step * tokens_per_step)
        if is_master:
            logger.info(f"Resumed: step {start_step}, {tokens_processed/1e9:.2f}B tokens")

    # Data loaders
    train_loader = create_dataloader(
        config.data_dir,
        "train",
        config.seq_length,
        config.batch_size,
        world_size,
        rank,
        seed=config.seed,
        epoch=epoch,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
    )
    val_loader = create_dataloader(
        config.data_dir,
        "val",
        config.seq_length,
        config.batch_size,
        world_size=1,
        rank=0,
        num_workers=min(2, config.num_workers),
        pin_memory=config.pin_memory,
        persistent_workers=False,
    )

    # WandB
    wandb_run = None
    if is_master and config.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=asdict(config),
                resume="allow",
            )
        except Exception as e:
            logger.warning(f"Failed to init wandb: {e}")

    # Update checkpoint_dir path
    checkpoint_dir = Path(config.checkpoint_dir)
    
    if is_master:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving checkpoints to {checkpoint_dir}")

    # Training loop
    model.train()
    train_iter = iter(train_loader)
    running_loss = 0.0
    t0 = time.time()

    no_sync = model.no_sync if is_ddp else nullcontext
    data_exhausted = False

    for step in range(start_step, max_steps):
        # Check token budget
        if tokens_processed >= target_tokens:
            if is_master:
                logger.info(f"Token budget reached: {tokens_processed/1e9:.2f}B")
            break

        # Learning rate schedule
        lr = get_lr(
            step,
            warmup_steps,
            max_steps,
            config.learning_rate,
            config.min_lr,
            config.lr_decay_style,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        # Gradient accumulation
        for micro_step in range(config.gradient_accumulation_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                if is_master:
                    logger.info(f"Data exhausted at step {step}")
                data_exhausted = True
                break

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Sync gradients only on last micro-step
            sync_ctx = (
                no_sync()
                if micro_step < config.gradient_accumulation_steps - 1
                else nullcontext()
            )

            with sync_ctx:
                with ctx:
                    outputs = model(input_ids=x, labels=y)
                    loss = outputs.loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward()

            running_loss += loss.item()

        if data_exhausted:
            break

        # Gradient clipping and optimizer step
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        tokens_processed += tokens_per_step

        # Logging
        if step > 0 and step % config.log_interval == 0 and is_master:
            dt = time.time() - t0
            t0 = time.time()
            
            tok_sec = (config.log_interval * tokens_per_step) / dt
            loss_avg = running_loss / config.log_interval

            logger.info(
                f"step {step:>6d} | loss {loss_avg:.4f} | lr {lr:.2e} | "
                f"grad {grad_norm:.2f} | {tok_sec/1000:.1f}k tok/s | "
                f"{tokens_processed/1e9:.2f}B tokens"
            )

            if wandb_run:
                wandb_run.log(
                    {
                        "train/loss": loss_avg,
                        "train/lr": lr,
                        "train/grad_norm": grad_norm,
                        "perf/tokens_per_sec": tok_sec,
                        "progress/tokens_B": tokens_processed / 1e9,
                        "progress/step": step,
                    },
                    step=step,
                )

            running_loss = 0.0

        # Evaluation
        if step > 0 and step % config.eval_interval == 0 and is_master:
            metrics = evaluate(model, val_loader, config.eval_iters, ctx, device)
            logger.info(f"Eval step {step}: loss={metrics['loss']:.4f}, ppl={metrics['perplexity']:.2f}")

            if wandb_run:
                wandb_run.log(
                    {
                        "val/loss": metrics["loss"],
                        "val/perplexity": metrics["perplexity"],
                    },
                    step=step,
                )

        # Checkpoint
        if step > 0 and step % config.checkpoint_interval == 0 and is_master:
            save_checkpoint(
                model,
                optimizer,
                scaler,
                step,
                epoch,
                tokens_processed,
                config,
                checkpoint_dir,
                is_ddp,
            )

    # Final saves
    if is_master:
        logger.info("Training complete. Saving final checkpoint...")
        save_checkpoint(
            model,
            optimizer,
            scaler,
            step if "step" in locals() else max_steps,
            epoch,
            tokens_processed,
            config,
            checkpoint_dir,
            is_ddp,
        )
        save_final_model(model, checkpoint_dir, is_ddp, tokenizer_config)
        logger.info(f"Total tokens: {tokens_processed/1e9:.4f}B")

    if wandb_run:
        wandb_run.finish()

    if is_ddp:
        dist.destroy_process_group()


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a HuggingFace-compatible language model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file (takes precedence)
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON config file (overrides other args)",
    )

    # Model specification (mutually exclusive approaches)
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model_name",
        type=str,
        help="HuggingFace model name (e.g., 'fla-hub/gla-1.3B-100B')",
    )
    model_group.add_argument(
        "--model_path",
        type=str,
        help="Local path to model or model class",
    )
    model_group.add_argument(
        "--model_config_path",
        type=str,
        help="Path to HF config.json for creating fresh model",
    )

    # Data
    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--data_dir", type=str, default="/data")
    data_group.add_argument("--seq_length", type=int, default=4096)

    # Training
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--batch_size", type=int, default=16)
    train_group.add_argument("--gradient_accumulation_steps", type=int, default=4)
    train_group.add_argument("--learning_rate", type=float, default=1e-4)
    train_group.add_argument("--min_lr", type=float, default=1e-5)
    train_group.add_argument("--weight_decay", type=float, default=0.1)
    train_group.add_argument("--grad_clip", type=float, default=1.0)
    train_group.add_argument("--warmup_tokens", type=float, default=1.0)
    train_group.add_argument("--max_tokens", type=float, default=100.0)

    # Performance
    perf_group = parser.add_argument_group("Performance")
    perf_group.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
    perf_group.add_argument("--compile", action="store_true", help="Use torch.compile")
    perf_group.add_argument("--gradient_checkpointing", action="store_true")

    # Checkpointing
    ckpt_group = parser.add_argument_group("Checkpointing")
    ckpt_group.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    ckpt_group.add_argument("--resume", type=str, help="Resume from checkpoint")
    ckpt_group.add_argument("--checkpoint_interval", type=int, default=1000)

    # Logging
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument("--log_interval", type=int, default=10)
    log_group.add_argument("--eval_interval", type=int, default=500)
    log_group.add_argument("--eval_iters", type=int, default=50)
    log_group.add_argument("--wandb_project", type=str)
    log_group.add_argument("--wandb_run_name", type=str)

    # System
    sys_group = parser.add_argument_group("System")
    sys_group.add_argument("--num_workers", type=int, default=4)
    sys_group.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def load_config_file(config_path: str) -> dict:
    """Load and parse JSON config file."""
    with open(config_path, "r") as f:
        return json.load(f)


def main():
    args = parse_args()

    # Load from config file if provided
    if args.config:
        logger.info(f"Loading config from {args.config}")
        file_config = load_config_file(args.config)
        
        # Extract sections
        model_config = file_config.get("model", {})
        tokenizer_config = file_config.get("tokenizer", {})
        data_config = file_config.get("data", {})
        training_config = file_config.get("training", {})
        checkpointing_config = file_config.get("checkpointing", {})
        logging_config = file_config.get("logging", {})
        system_config = file_config.get("system", {})
        
        # Build TrainingConfig from file
        config = TrainingConfig(
            # Data
            data_dir=data_config.get("data_dir", "./data"),
            seq_length=data_config.get("seq_length", 2048),
            # Training
            batch_size=training_config.get("batch_size", 8),
            gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
            learning_rate=training_config.get("learning_rate", 3e-4),
            min_lr=training_config.get("min_lr", 3e-5),
            weight_decay=training_config.get("weight_decay", 0.1),
            beta1=training_config.get("beta1", 0.9),
            beta2=training_config.get("beta2", 0.95),
            grad_clip=training_config.get("grad_clip", 1.0),
            warmup_tokens=training_config.get("warmup_tokens", 0.5),
            max_tokens=training_config.get("max_tokens", 10.0),
            lr_decay_style=training_config.get("lr_decay_style", "cosine"),
            dtype=training_config.get("dtype", "bfloat16"),
            compile_model=training_config.get("compile_model", False),
            gradient_checkpointing=training_config.get("gradient_checkpointing", False),
            # Checkpointing
            checkpoint_dir=checkpointing_config.get("checkpoint_dir", "./checkpoints"),
            checkpoint_interval=checkpointing_config.get("checkpoint_interval", 1000),
            # Logging
            log_interval=logging_config.get("log_interval", 10),
            eval_interval=logging_config.get("eval_interval", 500),
            eval_iters=logging_config.get("eval_iters", 50),
            wandb_project=logging_config.get("wandb_project"),
            wandb_run_name=logging_config.get("wandb_run_name"),
            # System
            num_workers=system_config.get("num_workers", 4),
            prefetch_factor=system_config.get("prefetch_factor", 4),
            seed=system_config.get("seed", 42),
            pin_memory=system_config.get("pin_memory", True),
            persistent_workers=system_config.get("persistent_workers", True),
        )
        
        train(
            config=config,
            model_config=model_config,
            tokenizer_config=tokenizer_config,
            resume_path=args.resume,
        )
    
    else:
        # Build config from CLI args
        config = TrainingConfig(
            data_dir=args.data_dir,
            seq_length=args.seq_length,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            min_lr=args.min_lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            warmup_tokens=args.warmup_tokens,
            max_tokens=args.max_tokens,
            dtype=args.dtype,
            compile_model=args.compile,
            gradient_checkpointing=args.gradient_checkpointing,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_interval=args.checkpoint_interval,
            log_interval=args.log_interval,
            eval_interval=args.eval_interval,
            eval_iters=args.eval_iters,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            num_workers=args.num_workers,
            seed=args.seed,
        )

        # Validate model specification
        if not any([args.model_name, args.model_path, args.model_config_path]):
            raise ValueError("Must provide --config, --model_name, --model_path, or --model_config_path")

        train(
            config=config,
            model_name=args.model_name,
            model_path=args.model_path,
            config_path=args.model_config_path,
            resume_path=args.resume,
        )


if __name__ == "__main__":
    main()
