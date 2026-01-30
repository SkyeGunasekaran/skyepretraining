#!/usr/bin/env python3
"""
Fast LM Evaluation for Custom Transformers

Evaluates a HuggingFace-compatible model on standard benchmarks using 
EleutherAI's lm-evaluation-harness.

Example:
    python eval.py --checkpoint checkpoints/small --batch-size 64
    python eval.py --checkpoint ./model --tasks hellaswag arc_easy --device cpu
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

# Custom model imports (swap these for AutoModel if using standard HF models)
try:
    from models.transformer.configuration_transformer import TransformerConfig
    from models.transformer.modeling_transformer import TransformerForCausalLM
    CUSTOM_MODEL_AVAILABLE = True
except ImportError:
    CUSTOM_MODEL_AVAILABLE = False
    from transformers import AutoConfig, AutoModelForCausalLM

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.set_float32_matmul_precision('high')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("Eval")


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a language model on standard benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-c", "--checkpoint", 
        default="checkpoints/small",
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "-b", "--batch-size", 
        type=int, 
        default=128,
        help="Batch size for evaluation (reduce by 1/2 if CUDA OOM)"
    )
    parser.add_argument(
        "--max-length", 
        type=int, 
        default=4096,
        help="Sequence length model was trained on"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[
            "wikitext",
            "lambada_openai",
            "piqa",
            "hellaswag",
            "winogrande",
            "arc_easy",
            "arc_challenge",
            "boolq",
        ],
        help="Evaluation tasks from lm-eval"
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples (0 for zero-shot)"
    )
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Model precision"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on"
    )
    parser.add_argument(
        "-o", "--output",
        default="results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--use-auto-model",
        action="store_true",
        help="Use HuggingFace AutoModel instead of custom Transformer classes"
    )
    
    return parser.parse_args()


def load_model(checkpoint_path: str, dtype_str: str, device: str, use_auto: bool = False):
    """Load model from checkpoint with appropriate class."""
    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32
    }[dtype_str]
    
    logger.info(f"Loading model from {checkpoint_path}...")
    
    if use_auto or not CUSTOM_MODEL_AVAILABLE:
        if not use_auto and not CUSTOM_MODEL_AVAILABLE:
            logger.warning("Custom model files not found, falling back to AutoModel")
        
        config = AutoConfig.from_pretrained(checkpoint_path)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            config=config,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        if device != "cuda":
            model = model.to(device)
    else:
        config = TransformerConfig.from_pretrained(checkpoint_path)
        model = TransformerForCausalLM.from_pretrained(
            checkpoint_path,
            config=config,
            torch_dtype=torch_dtype
        ).to(device)
    
    logger.info(f"Model loaded: {model.config.model_type}")
    return model


def extract_metric(metrics: dict) -> str:
    """Extract the primary metric for a task."""
    # Priority order for common metrics
    for key in ["acc", "acc_norm", "exact_match", "f1", "perplexity", "bits_per_byte"]:
        if key in metrics:
            val = metrics[key]
            if isinstance(val, (int, float)):
                return f"{key}: {val:.4f}"
    # Fallback to first numeric metric
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not k.startswith("_"):
            return f"{k}: {v:.4f}"
    return "N/A"


def main():
    args = get_args()
    
    # Validate checkpoint exists
    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    logger.info(f"Device: {args.device} | Batch: {args.batch_size} | Dtype: {args.dtype}")
    
    try:
        # Load model
        model = load_model(args.checkpoint, args.dtype, args.device, args.use_auto_model)
        
        # Wrap for lm-eval
        lm = HFLM(
            pretrained=model,
            tokenizer=args.checkpoint,
            batch_size=args.batch_size,
            max_length=args.max_length,
            trust_remote_code=True
        )
        
        # Run evaluation
        logger.info(f"Starting evaluation on {len(args.tasks)} tasks...")
        start = time.time()
        
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            device=args.device
        )
        
        duration = (time.time() - start) / 60
        
        # Print results table
        print("\n" + "=" * 60)
        print(f"RESULTS ({duration:.1f} min)")
        print("=" * 60)
        print(f"{'Task':<20} {'Metric':<30}")
        print("-" * 60)
        
        for task in args.tasks:
            if task in results.get("results", {}):
                metrics = results["results"][task]
                metric_str = extract_metric(metrics)
                print(f"{task:<20} {metric_str:<30}")
        
        print("=" * 60)
        
        # Save full results
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Full results saved to {args.output}")
        
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA OOM! Reduce --batch-size (try 64, 32, or 16)")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
