"""
Quick evaluation script for all checkpoints in output_dir/
Evaluates NDCG@5, NDCG@10 on eval, val, and test sets
Outputs results to TSV file

Reuses evaluate.py - imports and calls its main() function directly (line 69)
Reuses calc.py logic - compute_ndcg_from_json() extracts the metric computation
Simpler flow:
For each checkpoint → call evaluate.py to generate predictions JSON
Parse JSON → compute NDCG metrics
Collect all results → write to TSV

Usage:
python quick_eval.py \
  --output_dir ./checkpoints \
  --category Industrial_and_Scientific \
  --output_tsv results.tsv
"""

import os
import csv
import math
import json
import tempfile
from glob import glob
from tqdm import tqdm
import fire
import numpy as np


def compute_ndcg_from_json(json_path, item_file):
    """Compute NDCG@5 and NDCG@10 from prediction JSON (reusing calc.py logic)"""
    with open(item_file, 'r') as f:
        items = f.readlines()
        item_names = [_.split('\t')[0].strip() for _ in items]

    item_dict = {}
    for i, name in enumerate(item_names):
        if name not in item_dict:
            item_dict[name] = [i]
        else:
            item_dict[name].append(i)

    with open(json_path, 'r') as f:
        test_data = json.load(f)

    predictions = [[_.strip("\"\n").strip() for _ in sample["predict"]] for sample in test_data]

    ndcg_5, ndcg_10 = 0.0, 0.0

    for idx, pred_list in enumerate(predictions):
        target = test_data[idx]['output']
        if isinstance(target, list):
            target = target[0].strip("\"").strip()
        else:
            target = target.strip(" \n\"")

        # Find position of target
        pos = len(pred_list)
        for i, pred in enumerate(pred_list):
            if pred == target:
                pos = i
                break

        # Compute NDCG
        if pos < 5:
            ndcg_5 += 1.0 / math.log2(pos + 2)
        if pos < 10:
            ndcg_10 += 1.0 / math.log2(pos + 2)

    ndcg_5 /= len(predictions)
    ndcg_10 /= len(predictions)

    return {"ndcg@5": ndcg_5, "ndcg@10": ndcg_10}


def evaluate_checkpoint(checkpoint, data_file, info_file, category,
                        batch_size=8, num_beams=20, max_new_tokens=128):
    """Evaluate a checkpoint using evaluate.py"""
    from evaluate import main as evaluate_main

    # Create temporary file for results
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        result_json = tmp.name

    try:
        # Call evaluate.py's main function
        evaluate_main(
            base_model=checkpoint,
            train_file="",  # Not used for evaluation
            info_file=info_file,
            category=category,
            test_data_path=data_file,
            result_json_data=result_json,
            batch_size=batch_size,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )

        # Compute metrics from output JSON
        metrics = compute_ndcg_from_json(result_json, info_file)
        return metrics

    finally:
        # Clean up temp file
        if os.path.exists(result_json):
            os.remove(result_json)


def main(
    output_dir: str = "./output_dir",
    category: str = "Industrial_and_Scientific",
    eval_file: str = None,
    val_file: str = None,
    test_file: str = None,
    info_file: str = None,
    output_tsv: str = "eval_results_rl_ep2.tsv",
    batch_size: int = 128,
    num_beams: int = 16,
):
    """
    Evaluate all checkpoints in output_dir and save results to TSV

    Args:
        output_dir: Directory containing model checkpoints
        category: Dataset category
        eval_file: Path to eval file (optional)
        val_file: Path to validation file (optional)
        test_file: Path to test file (optional)
        info_file: Path to info file (optional)
        output_tsv: Output TSV file path
        batch_size: Batch size for evaluation
        num_beams: Number of beams for beam search
    """
    # Auto-detect data files
    if val_file is None:
        val_files = glob(f"./data/Amazon/valid/{category}*.csv")
        val_file = val_files[0] if val_files else None

    if test_file is None:
        test_files = glob(f"./data/Amazon/test/{category}*.csv")
        test_file = test_files[0] if test_files else None

    if info_file is None:
        info_files = glob(f"./data/Amazon/info/{category}*.txt")
        info_file = info_files[0] if info_files else None

    if not info_file:
        raise ValueError(f"No info file found for category {category}")

    # Find all checkpoints (including nested in subdirectories)
    def checkpoint_sort_key(path):
        basename = os.path.basename(path)
        if basename == "final_checkpoint":
            return (float('inf'),)
        if basename.startswith("checkpoint-"):
            try:
                return (int(basename.split("-")[1]),)
            except (ValueError, IndexError):
                pass
        return (float('inf'),)
    
    checkpoints = sorted(
        glob(os.path.join(output_dir, "checkpoint-*")) +
        glob(os.path.join(output_dir, "final_checkpoint")) +
        glob(os.path.join(output_dir, "*/checkpoint-*")) +
        glob(os.path.join(output_dir, "*/final_checkpoint")),
        key=checkpoint_sort_key
    )
    if not checkpoints:
        print(f"No checkpoints found in {output_dir}")
        return

    print(f"Found {len(checkpoints)} checkpoints")
    print(f"Category: {category}")
    print(f"Val file: {val_file}")
    print(f"Test file: {test_file}")
    print(f"Info file: {info_file}\n")

    # Prepare TSV file and fieldnames
    fieldnames = ["model_path"]
    if eval_file:
        fieldnames.extend(["ndcg@5_eval", "ndcg@10_eval"])
    if val_file:
        fieldnames.extend(["ndcg@5_val", "ndcg@10_val"])
    if test_file:
        fieldnames.extend(["ndcg@5_test", "ndcg@10_test"])

    # Open TSV and write header
    tsv_file = open(output_tsv, 'w', newline='')
    writer = csv.DictWriter(tsv_file, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()

    # Evaluate each checkpoint
    for checkpoint in tqdm(checkpoints, desc="Evaluating checkpoints"):
        # print(f"\nEvaluating: {os.path.basename(checkpoint)}")
        row = {"model_path": checkpoint}

        # Evaluate on each dataset
        if eval_file:
            # print("  - Eval set...")
            metrics = evaluate_checkpoint(checkpoint, eval_file, info_file, category,
                                         batch_size, num_beams)
            metrics = {k: f"{v:.4f}" for k, v in metrics.items()}

            row["ndcg@5_eval"] = metrics["ndcg@5"]
            row["ndcg@10_eval"] = metrics["ndcg@10"]

        if val_file:
            # print("  - Val set...")
            metrics = evaluate_checkpoint(checkpoint, val_file, info_file, category,
                                         batch_size, num_beams)
            row["ndcg@5_val"] = metrics["ndcg@5"]
            row["ndcg@10_val"] = metrics["ndcg@10"]

        if test_file:
            # print("  - Test set...")
            metrics = evaluate_checkpoint(checkpoint, test_file, info_file, category,
                                         batch_size, num_beams)
            row["ndcg@5_test"] = metrics["ndcg@5"]
            row["ndcg@10_test"] = metrics["ndcg@10"]

        # Print results
        for key, value in row.items():
            if key != "model_path":
                print(f"    {key}: {value:.4f}")

        # Write row immediately
        writer.writerow(row)
        tsv_file.flush()

    tsv_file.close()

    print(f"\n{'='*60}")
    print(f"Results saved to {output_tsv}")
    print(f"{'='*60}\n")


if __name__ == "__main__":

    # output_dir = "./output_dir"
    output_dir = "./output_dir/Qwen2.5-1.5B_RL/ep4"
    output_tsv = "eval_results_rl_ep4.tsv"
    category = "Industrial_and_Scientific"
    batch_size = 128


    main(output_dir=output_dir, output_tsv=output_tsv, category=category, batch_size=batch_size)



    # fire.Fire(main)
