import numpy as np
from decorators import input_to_numpy, none_if_missing_arg
from utils import all_accuracies
from comb_modules.utils import edges_from_grid
from comb_modules.dijkstra import dijkstra
import itertools
import torch

@none_if_missing_arg
def perfect_match_accuracy(true_paths, suggested_paths):
    matching_correct = torch.sum(torch.abs(true_paths - suggested_paths), axis=-1)
    avg_matching_correct = (matching_correct < 0.5).float().mean()
    return avg_matching_correct.item()


@none_if_missing_arg
def cost_ratio(vertex_costs, true_paths, suggested_paths, e2i=None):
    sps = []
    tps = []
    for vc, tp, sp in zip(vertex_costs, true_paths, suggested_paths):
        vc = vc.flatten()
        sps += [vc[np.unique(e2i[sp.astype(bool)]) - 1].sum()]
        tps += [vc[np.unique(e2i[tp.astype(bool)]) - 1].sum()]
    #suggested_paths_costs = suggested_paths * vertex_costs
    #true_paths_costs = true_paths * vertex_costs
    #return (np.sum(suggested_paths_costs, axis=1) / np.sum(true_paths_costs, axis=1)).mean()
    suggested_paths_costs = np.stack(sps)
    true_paths_costs = np.stack(tps)
    return (suggested_paths_costs / true_paths_costs)


def compute_metrics(true_paths, suggested_paths, true_vertex_costs, e2i=None, valid_paths=None):
    batch_size = true_vertex_costs.shape[0]
    metrics = {
        "perfect_match_accuracy": perfect_match_accuracy(true_paths.reshape(batch_size,-1), suggested_paths.reshape(batch_size,-1)),
        **all_accuracies(true_vertex_costs, true_paths, suggested_paths, 6, e2i, valid_paths)
    }
    return metrics


def is_valid_label_fn(suggested_path):
    inverted_path = 1.-suggested_path
    shortest_path, _, _ = dijkstra(inverted_path)
    is_valid = (shortest_path * inverted_path).sum() == 0
    return is_valid

def get_neighbors(i,j,dim, dim2=None):
    if dim2 is None:
        dim2 = dim
    ret = []
    d = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for x,y in d:
        ii = i+x
        jj = j+y
        if ii >= 0 and jj >= 0 and ii < dim and jj < dim2:
            ret.append((ii,jj))
    return ret

def is_valid_label_fn_new(suggested_path):
    # starts from top left and searches until reach bottom right
    # expects there to always be 1 unvisited neighbor to continue the path
    # expects all 1s to be visited
    if suggested_path[0][0] != 1:
        return False
    ones_count = suggested_path.sum()
    prev = (-1,-1)
    cur = (0,0)
    ones_visited_count = 0
    dim = suggested_path.shape[0]
    while cur is not None:
        ones_visited_count += 1
        if cur == (dim-1,dim-1):
            break
        nbrs = get_neighbors(cur[0], cur[1], dim)
        nxt = None
        for nbr in nbrs:
            if nbr == prev:
                continue
            if suggested_path[nbr[0]][nbr[1]] == 1:
                if nxt is not None:
                    return False
                nxt = nbr
        prev = cur
        cur = nxt
    if ones_visited_count < ones_count:
        return False
    return True
