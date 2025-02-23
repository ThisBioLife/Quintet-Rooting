from typing import Tuple
import torch.nn as nn
import torch.nn.functional as F




import argparse
import time
import dendropy
import numpy as np
import json
import sys
from table_fifth import TreeSet

from qr.adr_theory import *
from qr.fitness_cost import *
from qr.quintet_sampling import *
from qr.utils import *
from qr.version import __version__
from warnings import warn





import qr.deep_cost as dc

def extract_subcounts(t : torch.Tensor, indices: Tuple[int, int, int, int, int]) -> np.ndarray:
    out = []
    for i in range(5):
        for j in range(i, 5):
            out.append(t[indices[i], indices[j]].item())
    return np.asarray(out, dtype=np.single)

def features_from_co_occurence(d, label2idx, q):
    q_int = [label2idx[x] for x in q]
    t = extract_subcounts(d, q_int)
    normalizer = t[0] + t[5] + t[9] + t[12] + t[14]
    t = t / normalizer
    return t

def main(args):
    st_time = time.time()
    script_path = os.path.realpath(__file__).rsplit("/", 1)[0]

    # input args
    species_tree_path = args.speciestree
    gene_tree_path = args.genetrees
    output_path = args.outputtree
    sampling_method = args.samplingmethod.lower()
    random.seed(args.seed)
    cost_func = args.cost.lower()
    shape_coef = args.coef
    mult_le = args.multiplicity
    abratio = args.abratio
    co_matrix = args.gdl
    if co_matrix is not None:
        label2idx = torch.load(co_matrix + '.labels2idx')
        co_matrix = torch.load(co_matrix)

    header = """*********************************
*     Quintet Rooting """ + __version__ + """    *
*********************************"""
    sys.stdout.write(header + '\n')

    sys.stdout.write('Species tree: ' + species_tree_path + '\n')


    tns = dendropy.TaxonNamespace()
    unrooted_species = dendropy.Tree.get(path=species_tree_path, schema='newick',
                                         taxon_namespace=tns, rooting="force-unrooted", suppress_edge_lengths=True, preserve_underscores=True)
    if len(tns) < 5:
        raise Exception("Species tree " + species_tree_path + " has less than 5 taxa!\n")
    gene_trees = TreeSet(gene_tree_path)

    # reading fixed quintet topology files
    tns_base = dendropy.TaxonNamespace()
    unrooted_quintets_base = dendropy.TreeList.get(path=script_path + '/qr/topologies/quintets.tre',
                                                   taxon_namespace=tns_base, schema='newick')
    rooted_quintets_base = dendropy.TreeList(taxon_namespace=tns_base)
    rooted_quintets_base.read(path=script_path + '/qr/topologies/caterpillar.tre', schema='newick',
                              rooting="default-rooted")
    rooted_quintets_base.read(path=script_path + '/qr/topologies/pseudo_caterpillar.tre', schema='newick',
                              rooting="default-rooted")
    rooted_quintets_base.read(path=script_path + '/qr/topologies/balanced.tre', schema='newick',
                              rooting="default-rooted")
    rooted_quintet_indices = np.load(script_path + '/qr/rooted_quintet_indices.npy')

    sys.stdout.write('Loading time: %.2f sec\n' % (time.time() - st_time))
    ss_time = time.time()

    # search space of rooted trees
    rooted_candidates = get_all_rooted_trees(unrooted_species)
    r_score = np.zeros(len(rooted_candidates))

    sys.stdout.write('Creating search space time: %.2f sec\n' % (time.time() - ss_time))
    sm_time = time.time()

    # set of sampled quintets
    taxon_set = [t.label for t in tns]
    sample_quintet_taxa = []
    if len(taxon_set) == 5 or sampling_method == 'exh':
        sample_quintet_taxa = list(itertools.combinations(taxon_set, 5))
    elif sampling_method == 'tc':
        sample_quintet_taxa = triplet_cover_sample(taxon_set)
    elif sampling_method == 'le':
        sample_quintet_taxa = linear_quintet_encoding_sample(unrooted_species, taxon_set, mult_le)
    elif sampling_method == 'rl':
        sample_quintet_taxa = random_linear_sample_lazy(taxon_set, mult_le)

    sys.stdout.write('Quintet sampling time: %.2f sec\n' % (time.time() - sm_time))
    proc_time = time.time()

    sys.stdout.write("Number of taxa (n): %d\n" % len(tns))
    sys.stdout.write("Number of gene trees (k): %d\n" % len(gene_trees))
    sys.stdout.write("Size of search space (|R|): %d\n" % len(rooted_candidates))
    sys.stdout.write("Size of sampled quintets set (|Q*|): %d\n" % len(sample_quintet_taxa))

    # preprocessing
    quintet_scores = np.zeros((len(sample_quintet_taxa), 7))
    quintet_unrooted_indices = np.zeros(len(sample_quintet_taxa), dtype=int)
    quintets_r_all = []

    for j in range(len(sample_quintet_taxa)):
        q_taxa = sample_quintet_taxa[j]
        quintets_u = [
            dendropy.Tree.get(data=map_taxon_namespace(str(q), q_taxa) + ';', schema='newick', rooting='force-unrooted',
                              taxon_namespace=tns) for q in unrooted_quintets_base]
        quintets_r = [
            dendropy.Tree.get(data=map_taxon_namespace(str(q), q_taxa) + ';', schema='newick', rooting='force-rooted',
                              taxon_namespace=tns) for q in rooted_quintets_base]
        subtree_u = unrooted_species.extract_tree_with_taxa_labels(labels=q_taxa, suppress_unifurcations=True)
        try:
            quintet_counts = np.asarray(gene_trees.coalesence_times_by_topology(q_taxa))
        except:
            raise Exception("Error in computing coalescence times for quintet " + str(q_taxa) + "\n")
        quintet_tree_dist = quintet_counts
        quintet_unrooted_indices[j] = get_quintet_unrooted_index(subtree_u, quintets_u)
        if co_matrix is not None:
            gdl_feature = features_from_co_occurence(co_matrix, label2idx, q_taxa)
        else:
            gdl_feature = None
        quintet_scores[j] = compute_cost_rooted_quintets(quintet_tree_dist, quintet_unrooted_indices[j],
                                                         rooted_quintet_indices, cost_func, len(gene_trees),
                                                         len(sample_quintet_taxa), shape_coef, abratio, gdl_feature)
        quintets_r_all.append(quintets_r)

    sys.stdout.write('Preprocessing time: %.2f sec\n' % (time.time() - proc_time))
    sc_time = time.time()

    # computing scores
    min_score = sys.maxsize
    for i in range(len(rooted_candidates)):
        r = rooted_candidates[i]
        for j in range(len(sample_quintet_taxa)):
            q_taxa = sample_quintet_taxa[j]
            subtree_r = r.extract_tree_with_taxa_labels(labels=q_taxa, suppress_unifurcations=True)
            r_idx = get_quintet_rooted_index(subtree_r, quintets_r_all[j], quintet_unrooted_indices[j])
            r_score[i] += quintet_scores[j][r_idx]
            if not args.confidencescore and r_score[i] > min_score:
                break
        if r_score[i] < min_score:
            min_score = r_score[i]

    min_idx = np.argmin(r_score)
    with open(output_path, 'w') as fp:
        fp.write(str(rooted_candidates[min_idx]) + ';\n')

    sys.stdout.write('Scoring time: %.2f sec\n' % (time.time() - sc_time))
    sys.stdout.write('Best rooting: \n%s \n' % str(rooted_candidates[min_idx]))

    # computing confidence scores
    if args.confidencescore:
        sys.stdout.write('Scores of all rooted trees:\n %s \n' % str(r_score))
        confidence_scores = (np.max(r_score) - r_score) / np.sum(np.max(r_score) - r_score)
        tree_ranking_indices = np.argsort(r_score)
        with open(output_path + ".rank.cfn", 'w') as fp:
            for i in tree_ranking_indices:
                fp.write(str(rooted_candidates[i]) + ';\n')
                fp.write(str(r_score[i]) + '\n')
    else:
        with open(output_path + ".score.json", 'w') as fp:
            json.dump({'score': r_score[min_idx], 'min_score': min_score}, fp)

    sys.stdout.write('Total execution time: %.2f sec\n' % (time.time() - st_time))


def compute_cost_rooted_quintets(u_distribution, u_idx, rooted_quintet_indices, cost_func, k, q_size, shape_coef, 
                                 abratio, 
                                 co_occurrence_matrix=None):
    """
    Scores the 7 possible rootings of an unrooted quintet
    :param np.ndarray u_distribution: unrooted quintet tree probability distribution
    :param int u_idx: index of unrooted binary tree
    :param np.ndarray rooted_quintet_indices: indices of partial orders for all rooted quintet trees
    :param str cost_func: type of the fitness function
    :rtype: np.ndarray
    """
    if cost_func == 'dl':
        return dc.ils_cost_between(u_idx, u_distribution)
    elif cost_func == 'gdl':
        return dc.gdl_cost_between(u_idx, co_occurrence_matrix)
    elif cost_func == 'joint':
        return dc.joint_cost_between(u_idx, u_distribution, co_occurrence_matrix)
    rooted_tree_indices = u2r_mapping[u_idx]
    costs = np.zeros(7)
    for i in range(7):
        idx = rooted_tree_indices[i]
        unlabeled_topology = idx_2_unlabeled_topology(idx)
        indices = rooted_quintet_indices[idx]
        costs[i] = cost(u_distribution, indices, unlabeled_topology, cost_func, k, q_size, shape_coef, abratio)
    return costs


def get_all_rooted_trees(unrooted_tree):
    """
    Generates all the possible rooted trees with a given unrooted topology
    :param dendropy.Tree unrooted_tree: an unrooted tree topology
    :rtype: list
    """
    rooted_candidates = []
    tree = dendropy.Tree(unrooted_tree)
    for edge in tree.preorder_edge_iter():
        try:
            tree.reroot_at_edge(edge, update_bipartitions=True)
            rooted_candidates.append(dendropy.Tree(tree))
        except:
            continue
    # removing duplicates
    rooted_candidates[0].resolve_polytomies(update_bipartitions=True)
    for i in range(1, len(rooted_candidates)):
        if dendropy.calculate.treecompare.symmetric_difference(rooted_candidates[0], rooted_candidates[i]) == 0:
            rooted_candidates.pop(0)
            break
    return rooted_candidates


def parse_args():
    parser = argparse.ArgumentParser(description=str('== Quintet Rooting ' + __version__ + ' =='))

    parser.add_argument("-t", "--speciestree", type=str,
                        help="input unrooted species tree in newick format",
                        required=True, default=None)

    parser.add_argument("-g", "--genetrees", type=str,
                        help="input gene trees in newick format",
                        required=True, default=None)

    parser.add_argument("-o", "--outputtree", type=str,
                        help="output file containing a rooted species tree",
                        required=True, default=None)

    parser.add_argument("-sm", "--samplingmethod", type=str,
                        help="quintet sampling method (LE for linear encoding (default), EXH for exhaustive",
                        required=False, default='LE')

    parser.add_argument("-c", "--cost", type=str,
                        help="cost function (STAR for running QR-STAR)",
                        required=False, default='d')

    parser.add_argument("-cfs", "--confidencescore", action='store_true',
                        help="output confidence scores for each possible rooted tree as well as a ranking")

    parser.add_argument("-mult", "--multiplicity", type=int,
                        help="multiplicity (number of quintets mapped to each edge) in QR-LE",
                        required=False, default=1)

    parser.add_argument("-norm", "--normalized", action='store_true',
                        help="normalization for unresolved gene trees or missing taxa",
                        required=False, default=False)

    parser.add_argument("-coef", "--coef", type=float,
                        help="coefficient for shape penalty term in QR-STAR", required=False, default=0)

    parser.add_argument("-abratio", "--abratio", type=float,
                        help="Ratio between invariant and inequality penalties used in QR-STAR", required=False, default=1)

    parser.add_argument("-rs", "--seed", type=int,
                        help="random seed", required=False, default=1234)

    parser.add_argument("-gdl", "--gdl", required=False, help="GDL signal, the co-occurence matrix")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_args())
