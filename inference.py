# infer a species tree from a co-occurence matrix
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedder(nn.Module):
  def __init__(self, input_dim = 25, hidden_dim = 100, output_dim = 3):
        super(Embedder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LeakyReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

  def forward(self, x):
      return F.normalize(self.encoder(x), p=2, dim=-1)
import numpy as np
import argparse
import re
from table_fifth import TreeSet
from random import sample, shuffle
from qr.deep_cost import predict_ils

ALL_ROOTED_QUINTETS = []
for n in ["caterpillar", "pseudo_caterpillar", "balanced"]:
    trees_path = f"qr/topologies/{n}.tre"
    with open(trees_path, 'r') as f:
        for l in f:
            ALL_ROOTED_QUINTETS.append(l.strip())

IX_MAPPING = torch.load("old_ix2new_ix.pt")

def infer_species_tree(trees, q) -> str:
    d = trees.coalesence_times_by_topology(q)
    y = predict_ils(d)
    q_raw = ALL_ROOTED_QUINTETS[IX_MAPPING[torch.argmax(y).item()]]
    return multireplace(q_raw, {
        "1": str(q[0]),
        "2": str(q[1]),
        "3": str(q[2]),
        "4": str(q[3]),
        "5": str(q[4]),
    })[5:]

def multireplace(string, replacements, ignore_case=False):
    """
    Given a string and a replacement map, it returns the replaced string.
    :param str string: string to execute replacements on
    :param dict replacements: replacement dictionary {value to find: value to replace}
    :param bool ignore_case: whether the match should be case insensitive
    :rtype: str
    """
    if not replacements:
        # Edge case that'd produce a funny regex and cause a KeyError
        return string

    # If case insensitive, we need to normalize the old string so that later a replacement
    # can be found. For instance with {"HEY": "lol"} we should match and find a replacement for "hey",
    # "HEY", "hEy", etc.
    if ignore_case:
        def normalize_old(s):
            return s.lower()

        re_mode = re.IGNORECASE
    else:
        def normalize_old(s):
            return s

        re_mode = 0

    replacements = {normalize_old(key): val for key, val in replacements.items()}

    # Place longer ones first to keep shorter substrings from matching where the longer ones should take place
    # For instance given the replacements {'ab': 'AB', 'abc': 'ABC'} against the string 'hey abc', it should produce
    # 'hey ABC' and not 'hey ABc'
    rep_sorted = sorted(replacements, key=len, reverse=True)
    rep_escaped = map(re.escape, rep_sorted)

    # Create a big OR regex that matches any of the substrings to replace
    pattern = re.compile("|".join(rep_escaped), re_mode)

    # For each match, look up the new string in the replacements, being the key the normalized old string
    return pattern.sub(lambda match: replacements[normalize_old(match.group(0))], string)


if __name__ == '__main__':
    import treeswift as ts
    parser = argparse.ArgumentParser(description='Infer species tree from co-occurence matrix')
    parser.add_argument('trees')
    parser.add_argument('-s', '--species', help='input file')
    parser.add_argument('-o', '--output', help='output file')
    args = parser.parse_args()
    taxa = [l.label for l in ts.read_tree_newick(args.species).traverse_leaves()]
    trees = TreeSet(args.trees)
    sampled_quintets = set()
    for _ in range(5000):
        five_taxa = sample(taxa, 5)
        shuffle(five_taxa)
        sampled_quintets.add(tuple(five_taxa))
    with open(args.output, 'w') as f:
        for q in sampled_quintets:
            f.write(infer_species_tree(trees, q) + "\n")