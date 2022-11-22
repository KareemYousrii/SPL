import numpy as np
import heapq
import torch
from functools import partial
from comb_modules.utils import get_neighbourhood_func
from collections import namedtuple
from utils import maybe_parallelize

DijkstraOutput = namedtuple("DijkstraOutput", ["shortest_path", "is_unique", "transitions", "edges"])


def dijkstra(matrix, neighbourhood_fn="8-grid", request_transitions=False, graph=None):

    x_max, y_max = matrix.shape
    neighbors_func = partial(get_neighbourhood_func(neighbourhood_fn), x_max=x_max, y_max=y_max)

    costs = np.full_like(matrix, 1.0e10)
    costs[0][0] = matrix[0][0]
    num_path = np.zeros_like(matrix)
    num_path[0][0] = 1
    priority_queue = [(matrix[0][0], (0, 0))]
    certain = set()
    transitions = dict()

    while priority_queue:
        cur_cost, (cur_x, cur_y) = heapq.heappop(priority_queue)
        if (cur_x, cur_y) in certain:
            pass

        for x, y in neighbors_func(cur_x, cur_y):
            if (x, y) not in certain:
                if matrix[x][y] + costs[cur_x][cur_y] < costs[x][y]:
                    costs[x][y] = matrix[x][y] + costs[cur_x][cur_y]
                    heapq.heappush(priority_queue, (costs[x][y], (x, y)))
                    transitions[(x, y)] = (cur_x, cur_y)
                    num_path[x, y] = num_path[cur_x, cur_y]
                elif matrix[x][y] + costs[cur_x][cur_y] == costs[x][y]:
                    num_path[x, y] += 1

        certain.add((cur_x, cur_y))
    # retrieve the path
    cur_x, cur_y = x_max - 1, y_max - 1
    on_path = np.zeros_like(matrix)
    on_path[-1][-1] = 1

    edges = np.zeros(2*12*12-12*2)
    #import pdb; pdb.set_trace()
    while (cur_x, cur_y) != (0, 0):
        if not graph is None:
            prev_x, prev_y = transitions[(cur_x, cur_y)]
            try:
                edges[graph[tuple(sorted((prev_x*12+prev_y+1, cur_x*12 +cur_y+1)))] - 1] = 1 
            except:
                import pdb; pdb.set_trace()
        cur_x, cur_y = transitions[(cur_x, cur_y)]
        on_path[cur_x, cur_y] = 1.0

    is_unique = num_path[-1, -1] == 1

    if request_transitions:
        return DijkstraOutput(shortest_path=on_path, is_unique=is_unique, transitions=transitions, edges=edges)
    else:
        return DijkstraOutput(shortest_path=on_path, is_unique=is_unique, transitions=None)


def get_solver(neighbourhood_fn):
    def solver(matrix):
        return dijkstra(matrix, neighbourhood_fn).shortest_path

    return solver


class ShortestPath(torch.autograd.Function):
    def __init__(self, lambda_val, neighbourhood_fn="8-grid"):
        self.lambda_val = lambda_val
        self.neighbourhood_fn = neighbourhood_fn
        self.solver = get_solver(neighbourhood_fn)

    def forward(self, weights):
        self.weights = weights.detach().cpu().numpy()
        self.suggested_tours = np.asarray(maybe_parallelize(self.solver, arg_list=list(self.weights)))
        return torch.from_numpy(self.suggested_tours).float().to(weights.device)

    def backward(self, grad_output):
        assert grad_output.shape == self.suggested_tours.shape
        grad_output_numpy = grad_output.detach().cpu().numpy()
        weights_prime = np.maximum(self.weights + self.lambda_val * grad_output_numpy, 0.0)
        better_paths = np.asarray(maybe_parallelize(self.solver, arg_list=list(weights_prime)))
        gradient = -(self.suggested_tours - better_paths) / self.lambda_val
        return torch.from_numpy(gradient).to(grad_output.device)
