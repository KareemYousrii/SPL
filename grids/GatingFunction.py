import torch
from collections import OrderedDict, defaultdict
class DenseGatingFunction(torch.nn.Module):

    def __init__(self, beta, gate_layers=[128, 256, 256], num_reps=1, gate_dropout=None):
        """
        Build a gating function that is a densly connected feedforward network
        that maps one input embedding to a series of tensors representing the
        einet's parameters
        """

        super(DenseGatingFunction, self).__init__()

        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False
        #torch.manual_seed(1)

        beta.num_reps = num_reps
        layers_shapes = gate_layers

        gate_layers = []
        for i, l in enumerate(layers_shapes[:-1]):
            gate_layers.extend([ (f"linear{i+1}",torch.nn.Linear(l, layers_shapes[i+1], device='cuda:0')),
                                (f"relu{i+1}",torch.nn.ReLU())])
            if gate_dropout:
                gate_layers.append((f"dropout{i+1}",torch.nn.Dropout(p=args.gate_dropout)))

        print(gate_layers)
        self.gate = torch.nn.Sequential(OrderedDict(gate_layers))
        
        num_branches = defaultdict(list)
        for node in beta.positive_iter():
            if node.is_decomposition() and len(node.positive_elements) > 1:
                num_branches[len(node.positive_elements)].append(node)
            #elif node.is_mixing():
            #    num_branches[len(node.elements)].append(node)
            elif node.is_true():
                num_branches[2].append(node)

        beta.num_branches = num_branches

        # Output_shapes: a list of (num_decision_nodes, num_children) tuples
        self.output_shapes = [(len(v), k, num_reps) for k,v in num_branches.items()] + [(num_reps,)] + [(len(beta.elements), num_reps)]
        flattened_shapes = [len(v)*k*num_reps for k,v in num_branches.items()] + [num_reps] + [len(beta.elements)*num_reps]

        self.outputs = []
        for i, o in enumerate(flattened_shapes):
            setattr(self, f"output{i}", torch.nn.Sequential(torch.nn.Linear(layers_shapes[-1], o, device='cuda:0'),
                                             #torch.nn.ReLU()
                                             ))
            self.outputs.append(getattr(self, f"output{i}"))

        self.initialize()

    def forward(self, x):
        x = self.gate(x)
        return [o(x).reshape(-1, *s) for o, s in  zip(self.outputs, self.output_shapes)]

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def initialize(self):
        self.gate.apply(self.init_weights)
        for o in self.outputs:
            self.init_weights(o)
