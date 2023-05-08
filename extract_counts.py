# has code copied from https://github.com/JSdoubleL/DISCO
import treeswift as ts
from typing import List, Dict
import torch
import numpy as np
import argparse
from itertools import combinations

def normalize_name(name : str, labels2idx: Dict[str, int], delimiter, nth_delimiter) -> int:
    gene_to_species = lambda x: delimiter.join(x.split(delimiter)[:nth_delimiter])
    return labels2idx[gene_to_species(name)]

def calculate_counts(trees : List[ts.Tree], labels2idx: Dict[str, int], delimiter, nth_delimiter) -> torch.Tensor:
    num_species = len(labels2idx)
    all_counts = []
    for tree in trees:
        counts = [0] * num_species
        for n in tree.traverse_leaves():
            i = normalize_name(n.label, labels2idx, delimiter, nth_delimiter)
            counts[i] += 1
        m = np.zeros((num_species, num_species))
        for u, v in combinations(range(num_species), 2):
            m[u, v] = min(counts[u], counts[v])
            m[v, u] = min(counts[u], counts[v])
        for i in range(num_species):
            m[i, i] = counts[i]
        all_counts.append(m)
    all_counts = np.array(all_counts)
    # average per axis 0
    all_counts = np.mean(all_counts, axis=0)
    return torch.tensor(all_counts).float()

from collections import defaultdict, Counter

def _calculate_pearson(trees : List[ts.Tree], labels2idx: Dict[str, int]) -> torch.Tensor:
    num_species = len(labels2idx)
    global_counts = [[] for _ in range(num_species)]
    for tree in trees:
        counts = Counter()
        for n in tree.traverse_leaves():
            i = normalize_name(n.label, labels2idx)
            counts[i] += 1
        for s in range(num_species):
            global_counts[s].append(counts[s])
    global_counts = np.array(global_counts)
    variance_counts = np.var(global_counts, axis=1)
    correlation_matrix = np.corrcoef(global_counts)
    for i in range(num_species):
        correlation_matrix[i, i] = variance_counts[i] if variance_counts[i] > 0 else 1e-6
    return torch.tensor(correlation_matrix).float()
    

def transform_to_log_counts(counts : torch.Tensor) -> torch.Tensor:
    # if count is 0, set it to 1
    counts = torch.where(counts == 0, torch.ones_like(counts) * 1e-6, counts)
    return torch.log(counts) + 1

def numeric_sort_key(s):
    if s.isdigit():
        return int(s)
    else:
        return s.lower()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract log counts from a tree')
    parser.add_argument('tree', help='tree file')
    parser.add_argument('out', help='output file')
    parser.add_argument('-s', "--species", help='species tree file')
    parser.add_argument('-d', "--delimiter", type=str, 
                        help="Delimiter separating species name from rest of leaf label", default='-')
    parser.add_argument('-n', '--nth-delimiter', type=int,
                        help="Split on nth delimiter (only works with -d)", default=1)
    parser.add_argument('--pearson', action='store_true', help='calculate pearson correlation matrix')
    parser.add_argument('--limit', type=int, default=None, help='limit number of trees to read')
    args = parser.parse_args()

    stree = ts.read_tree_newick(args.species)
    labels = sorted([l.label for l in stree.traverse_leaves()], key=numeric_sort_key)
    labels2idx = {l: i for i, l in enumerate(labels)}

    trees = []
    with open(args.tree, 'r') as f:
        for l in f:
            trees.append(ts.read_tree_newick(l))
            if args.limit is not None and len(trees) >= args.limit:
                break
    if not args.pearson:
        counts = calculate_counts(trees, labels2idx, args.delimiter, args.nth_delimiter)
        log_counts = counts
    else:
        log_counts = _calculate_pearson(trees, labels2idx)
    torch.save(log_counts, args.out)
    torch.save(labels2idx, args.out + '.labels2idx')