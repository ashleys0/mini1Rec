#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OPQ (unordered) semantic-ID construction
========================================

Takes the item embedding matrix produced by text2emb/qwen3_text2emb.py
(`{dataset}.emb-qwen-td.npy`, row i == item id i) and tokenizes it into
fixed-length OPQ semantic IDs via FAISS.

Unlike the residual quantizers (rqvae.py / rqkmeans_*.py) the OPQ digits are
*unordered* (a flat product quantization in an OPQ-rotated space) rather than a
coarse-to-fine residual hierarchy. The FAISS recipe is:
  - optional PCA whitening of the embeddings,
  - train OPQ (index_factory "OPQ{M},IVF1,PQ{M}x{nbits}", METRIC_INNER_PRODUCT),
  - add every item, read the PQ byte codes back (IVF1 keeps insertion order, so
    row i == item id i),
  - emit one M-digit code per item.

Output (same contract as rqkmeans_faiss.py):
  {output_root}/{dataset}/{dataset}.opq.index.json
      {item_id: ["<a_N>", "<b_N>", ...]}   one letter-prefixed token per digit
"""

import argparse
import json
import os

import faiss
import numpy as np


# ---------------------------------------------------------------------------
# OPQ index training + code extraction
# ---------------------------------------------------------------------------
def _extract_pq_codes(index, n_items, n_digit, n_bits):
    """Read the PQ byte codes out of the single IVF list and unpack per-digit codes.

    With IVF1 there is exactly one inverted list holding every vector in insertion
    (== add) order, so row i corresponds to item id i.
    """
    ivf_index = faiss.downcast_index(index.index)
    invlists = faiss.extract_index_ivf(ivf_index).invlists
    ls = invlists.list_size(0)
    assert ls == n_items, f"IVF list holds {ls} vectors, expected {n_items}"
    raw = faiss.rev_swig_ptr(invlists.get_codes(0), ls * invlists.code_size)
    raw = raw.reshape(-1, invlists.code_size)

    codes = np.zeros((n_items, n_digit), dtype=np.int64)
    for row in range(n_items):
        br = faiss.BitstringReader(faiss.swig_ptr(raw[row]), invlists.code_size)
        for d in range(n_digit):
            codes[row, d] = br.read(n_bits)
    return codes


def generate_opq(embs, n_codebook, n_bits, omp_threads=32):
    faiss.omp_set_num_threads(omp_threads)

    index_factory = f"OPQ{n_codebook},IVF1,PQ{n_codebook}x{n_bits}"
    index = faiss.index_factory(
        embs.shape[1], index_factory, faiss.METRIC_INNER_PRODUCT
    )

    print(f"[OPQ] Training index '{index_factory}' on {embs.shape[0]} items...")
    index.train(embs)
    index.add(embs)

    return _extract_pq_codes(index, embs.shape[0], n_codebook, n_bits)


def analyze_codes(codes):
    n_items = codes.shape[0]
    combos = len(set(map(tuple, codes)))
    print(f"[OPQ] codes shape={codes.shape}  unique full-paths={combos}  "
          f"collision_rate={1 - combos / n_items:.4f}")


def save_indices_json(codes, path):
    """Write {item_id: ["<a_N>", "<b_N>", ...]} (one letter-prefixed token per digit)."""
    idx = {}
    for i, code in enumerate(codes):
        idx[i] = [f"<{chr(97 + j)}_{int(c)}>" for j, c in enumerate(code)]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(idx, f, indent=2)
    print("[OPQ] Saved indices:", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="OPQ semantic-ID construction")
    parser.add_argument("--dataset", default="Industrial_and_Scientific")
    parser.add_argument("--data_path", type=str, default=None,
                        help="path to the {dataset}.emb-qwen-td.npy embeddings")
    parser.add_argument("--output_root", default="../data")

    parser.add_argument("--n_codebook", type=int, default=16,
                        help="number of OPQ digits (M)")
    parser.add_argument("--n_bits", type=int, default=8,
                        help="bits per digit (codebook_size = 2**n_bits)")
    parser.add_argument("--pca", type=int, default=0,
                        help="PCA-whiten to this dim before OPQ; <=0 disables")
    parser.add_argument("--faiss_omp_num_threads", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    data_path = args.data_path or \
        f"../data/Amazon/index/{args.dataset}.emb-qwen-td.npy"

    print("[OPQ] loading:", data_path)
    embs = np.load(data_path)
    embs = np.ascontiguousarray(embs, dtype=np.float32)
    print("[OPQ] embeddings shape:", embs.shape)

    if args.pca and args.pca > 0:
        from sklearn.decomposition import PCA
        print(f"[OPQ] Applying PCA(whiten=True) -> {args.pca} dims")
        pca = PCA(n_components=args.pca, whiten=True, random_state=args.seed)
        embs = np.ascontiguousarray(pca.fit_transform(embs), dtype=np.float32)

    codes = generate_opq(embs, args.n_codebook, args.n_bits,
                         omp_threads=args.faiss_omp_num_threads)
    assert codes.min() >= 0 and codes.max() < (1 << args.n_bits)
    analyze_codes(codes)

    out_json = os.path.join(args.output_root, args.dataset,
                            f"{args.dataset}.opq.index.json")
    save_indices_json(codes, out_json)


if __name__ == "__main__":
    main()
