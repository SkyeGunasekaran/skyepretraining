#!/usr/bin/env python3
import os
import argparse
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

# =============================================================================
# Global Config & Constants
# =============================================================================

FINEWEB_DATASET = "HuggingFaceFW/fineweb-edu"
FINEWEB_SUBSET = "sample-100BT"
TOKENIZER_ID = "meta-llama/Llama-2-7b-hf"

DTYPE = np.uint16
EOS_TOKEN = 2  # Standard Llama 2 EOS

_worker_tokenizer = None

# =============================================================================
# Worker Functions
# =============================================================================

def worker_init(tokenizer_id):
    """Initialize the tokenizer once per worker process."""
    global _worker_tokenizer
    _worker_tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

def process_batch(text_batch):
    """
    Tokenizes a list of strings and returns raw numpy arrays.
    """
    global _worker_tokenizer

    # Tokenize batch without padding/truncation
    encodings = _worker_tokenizer(
        text_batch,
        add_special_tokens=False,
        return_attention_mask=False
    )

    all_ids = []
    for ids in encodings['input_ids']:
        all_ids.extend(ids)
        all_ids.append(EOS_TOKEN)

    return np.array(all_ids, dtype=DTYPE)

# =============================================================================
# Main Processing Logic
# =============================================================================

def process_fineweb(args):
    # 1. Setup Output
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.bin")
    val_path = os.path.join(output_dir, "val.bin")

    print(f"\n{'='*50}")    
    print(f"fineweb-edu preprocessing")
    print(f"{'='*50}")
    print(f"Target:       {args.tokens / 1e9:.2f} B tokens")
    print(f"Workers:      {args.workers}")
    print(f"Output:       {train_path}, {val_path}")
    print(f"{'='*50}\n")

    # 2. Setup Dataset Stream
    dataset = load_dataset(
        FINEWEB_DATASET,
        name=FINEWEB_SUBSET,
        split="train",
        streaming=True
    ).shuffle(seed=args.seed, buffer_size=10_000)

    def batch_generator():
        batch = []
        for item in dataset:
            batch.append(item['text'])
            if len(batch) == args.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    # 3. Main Loop
    with open(train_path, "wb") as f_train, open(val_path, "wb") as f_val:

        with mp.Pool(args.workers, initializer=worker_init, initargs=(TOKENIZER_ID,)) as pool:
            
            iterator = pool.imap_unordered(process_batch, batch_generator())

            total_tokens = 0
            train_tokens = 0
            val_tokens = 0
            target = int(args.tokens)
            
            # Create a dedicated Random Generator for splitting
            rng = np.random.default_rng(args.seed)

            pbar = tqdm(total=target, unit="tok", desc="Writing")

            try:
                for token_array in iterator:
                    # Instead of masking tokens, we route the WHOLE chunk.
                    # This ensures sentences remain contiguous.
                    
                    # Roll a die (0.0 to 1.0)
                    if rng.random() < args.val_fraction:
                        # Write entire batch to Validation
                        f_val.write(token_array.tobytes())
                        val_tokens += len(token_array)
                    else:
                        # Write entire batch to Train
                        f_train.write(token_array.tobytes())
                        train_tokens += len(token_array)
                    
                    total_tokens += len(token_array)
                    pbar.update(len(token_array))

                    if total_tokens >= target:
                        print(f"\nTarget reached: {total_tokens} tokens.")
                        break

            except KeyboardInterrupt:
                print("\nInterrupted by user. Saving current progress...")
                pool.terminate()
            finally:
                pbar.close()

    # 7. Final Summary
    print(f"\nPreprocessing Complete")
    print(f"Train Tokens: {train_tokens / 1e9:.4f} B")
    print(f"Val Tokens:   {val_tokens / 1e9:.4f} B")
    print(f"Total Size:   {os.path.getsize(train_path) / (1024**3):.2f} GiB (Train)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=float, default=100e6, help="Target total tokens")
    parser.add_argument("--output", type=str, default="data_fineweb", help="Output directory")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="CPU workers")
    parser.add_argument("--batch_size", type=int, default=1000, help="Documents per batch sent to workers")
    parser.add_argument("--val_fraction", type=float, default=0.005, help="Validation split (0.005 = 0.5%)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    process_fineweb(args)
