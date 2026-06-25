"""
qwen3_text2emb.py
=================

Encode the per-item text of a processed Amazon dataset with **Qwen3-Embedding**
(default Qwen/Qwen3-Embedding-8B) using HuggingFace `transformers`, producing the
item embedding matrix consumed by the m1Rec RQ stage (rqvae.py / rqkmeans_*.py) to
build the semantic IDs for genrec.

Pipeline fit (identical I/O contract to amazon_text2emb.py):
  INPUT : <root>/<dataset>.item.json    (item_id -> {title, description, brand, categories})
  OUTPUT: <root>/<dataset>.emb-<plm_name>-td.npy
          float32 array of shape [num_items, embedding_dim], **row i == item id i**
          (rows sorted by integer item id), exactly what rqvae.py / rqkmeans_*.py
          load via `np.load(...)`.

Why a new file instead of amazon_text2emb.py:
  amazon_text2emb.py uses MEAN pooling + right padding, which is NOT how the Qwen3
  Embedding models are meant to be used. Per the Qwen3-Embedding model card the
  correct recipe is:
      * left padding,
      * LAST-token pooling (take the hidden state of the final non-pad token),
      * L2-normalize the pooled vector.
  For embedding *documents* (our item texts) no task instruction is prepended
  (instructions are only recommended on the query side). An optional --instruct is
  exposed for completeness but defaults to empty (no instruction).

Multi-GPU: uses `accelerate` the same way as amazon_text2emb.py, so launch with
  `accelerate launch --num_processes N qwen3_text2emb.py ...`.
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
from accelerate import Accelerator
from accelerate.utils import gather_object

from utils import load_json, clean_text


# ----------------------------------------------------------------------------------
# Text construction (matches amazon_text2emb.py: title + description)
# ----------------------------------------------------------------------------------
def load_data(args):
    if args.root:
        print("args.root: ", args.root)
    item2feature_path = os.path.join(args.root, f'{args.dataset}.item.json')
    return load_json(item2feature_path)


def generate_text(item2feature, features):
    """Build (item_id, text) per item by concatenating the requested feature fields."""
    item_text_list = []
    for item in item2feature:
        data = item2feature[item]
        text = []
        for meta_key in features:
            if meta_key in data:
                cleaned = clean_text(data[meta_key]).strip()
                if cleaned != "":
                    text.append(cleaned)

        if len(text) == 0:
            text = ["unknown item"]

        try:
            item_id = int(item)
        except (TypeError, ValueError):
            item_id = item

        item_text_list.append((item_id, " ".join(text)))

    return item_text_list


def preprocess_text(args):
    print('Process text data: ')
    print('Dataset: ', args.dataset)
    item2feature = load_data(args)
    features = [f.strip() for f in args.features.split(',') if f.strip()]
    item_text_list = generate_text(item2feature, features)
    return item_text_list


# ----------------------------------------------------------------------------------
# Qwen3-Embedding recipe
# ----------------------------------------------------------------------------------
def last_token_pool(last_hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool the hidden state of the last non-pad token (Qwen3-Embedding recipe)."""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'


def load_qwen3_model(model_path, attn_impl=None):
    print("Loading Qwen3-Embedding model:", model_path)
    # Qwen3-Embedding requires left padding (last-token pooling).
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    model_kwargs = dict(torch_dtype=torch.float16, low_cpu_mem_usage=True)
    if attn_impl:
        model_kwargs['attn_implementation'] = attn_impl
    model = AutoModel.from_pretrained(model_path, **model_kwargs)
    return tokenizer, model


def generate_item_embedding(args, item_text_list, tokenizer, model, accelerator):
    all_ids, all_texts = zip(*item_text_list)
    total_items = len(all_texts)

    # Shard items across processes (each rank embeds its slice, gathered at the end).
    num_processes = accelerator.num_processes
    process_index = accelerator.process_index
    chunk_size = int(np.ceil(total_items / num_processes))
    start_idx = process_index * chunk_size
    end_idx = min(start_idx + chunk_size, total_items)
    local_ids = all_ids[start_idx:end_idx]
    local_texts = all_texts[start_idx:end_idx]

    if accelerator.is_main_process:
        print(f"Total items: {total_items}")
        print(f"Start generating embeddings with {num_processes} processes...")

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    local_results = []
    pbar = tqdm(total=len(local_texts), desc=f"Proc {process_index}",
                disable=not accelerator.is_local_main_process)

    with torch.no_grad():
        for i in range(0, len(local_texts), args.batch_size):
            batch_texts = list(local_texts[i: i + args.batch_size])
            batch_ids = local_ids[i: i + args.batch_size]

            # Optional word dropout (data augmentation), matching amazon_text2emb.py.
            if args.word_drop_ratio > 0:
                batch_texts = [
                    ' '.join(wd for wd in t.split(' ') if random.random() > args.word_drop_ratio)
                    for t in batch_texts
                ]

            # Optional task instruction (documents normally use none).
            if args.instruct:
                batch_texts = [get_detailed_instruct(args.instruct, t) for t in batch_texts]

            encoded = tokenizer(
                batch_texts,
                max_length=args.max_sent_len,
                truncation=True,
                padding=True,
                return_tensors='pt',
            ).to(accelerator.device)

            outputs = model(input_ids=encoded.input_ids, attention_mask=encoded.attention_mask)

            # Last-token pooling + L2 normalization (Qwen3-Embedding recipe).
            emb = last_token_pool(outputs.last_hidden_state, encoded.attention_mask)
            emb = F.normalize(emb, p=2, dim=1)

            emb = emb.float().cpu().numpy()  # save as float32 for the RQ stage
            for idx, vec in zip(batch_ids, emb):
                local_results.append((idx, vec))

            pbar.update(len(batch_texts))

    pbar.close()
    accelerator.wait_for_everyone()

    all_results_flat = gather_object(local_results)

    if accelerator.is_main_process:
        print("Gathering finished. Sorting by item id and saving...")
        # Sort by item id so row i corresponds to item id i (downstream alignment).
        all_results_flat.sort(key=lambda x: x[0])
        final_embeddings = np.stack([x[1] for x in all_results_flat], axis=0).astype(np.float32)
        print('Final Embeddings shape: ', final_embeddings.shape)

        os.makedirs(args.root, exist_ok=True)
        file_path = os.path.join(args.root, f"{args.dataset}.emb-{args.plm_name}-td.npy")
        np.save(file_path, final_embeddings)
        print(f"Saved to {file_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Industrial_and_Scientific',
                        help='dataset / category name (matches <dataset>.item.json)')
    parser.add_argument('--root', type=str, default="",
                        help='dir holding <dataset>.item.json; the .npy is written here too')
    # Default kept as "qwen" so the existing rqvae.sh / rqkmeans_*.py defaults
    # (which look for `*.emb-qwen-td.npy`) work without changes.
    parser.add_argument('--plm_name', type=str, default='qwen',
                        help='tag in the output filename: <dataset>.emb-<plm_name>-td.npy')
    parser.add_argument('--plm_checkpoint', type=str, default='Qwen/Qwen3-Embedding-8B',
                        help='HF model id or local path of the Qwen3-Embedding model')
    parser.add_argument('--features', type=str, default='title,description',
                        help='comma-separated item.json fields to concatenate into the text')
    parser.add_argument('--instruct', type=str, default='',
                        help='optional task instruction; empty = none (recommended for documents)')
    parser.add_argument('--max_sent_len', type=int, default=2048,
                        help='max tokens per item (Qwen3-Embedding supports up to 32K)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='per-process batch size (8B model is large; tune to GPU memory)')
    parser.add_argument('--attn_implementation', type=str, default=None,
                        help='e.g. flash_attention_2 for faster / lower-memory inference')
    parser.add_argument('--word_drop_ratio', type=float, default=-1, help='word drop ratio (augmentation)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    accelerator = Accelerator()
    if accelerator.is_main_process:
        print(f"Running with {accelerator.num_processes} processes.")

    item_text_list = preprocess_text(args)

    tokenizer, model = load_qwen3_model(args.plm_checkpoint, attn_impl=args.attn_implementation)
    model = model.to(accelerator.device)
    model.eval()

    generate_item_embedding(args, item_text_list, tokenizer, model, accelerator)
