import sys
sys.path.insert(0, './grids/pypsdd')

import torch
import torch.nn.functional as F
from pypsdd import Vtree, SddManager, PSddManager, SddNode, Inst, io
from pypsdd import UniformSmoothing, Prior

class CircuitMPE:
    def __init__(self, vtree_filename, sdd_filename):

        # Load the Sdd
        self.vtree = Vtree.read(vtree_filename)
        self.manager = SddManager(self.vtree)
        self.alpha = io.sdd_read(sdd_filename, self.manager)

        # Convert to psdd
        self.pmanager = PSddManager(self.vtree)

        # Storing psdd
        self.beta = self.pmanager.copy_and_normalize_sdd(self.alpha, self.vtree)

    def overparameterize(self, S=2):
        self.beta = self.beta.overparameterize(S)

    def rand_params(self):
        self.beta.rand_parameters()

    def set_params(self, thetas, log_space=True):

        self.beta.mixing = thetas[-2].log_softmax(dim=1) if log_space else thetas[-2].softmax(dim=1)
        self.beta.root_mixing = thetas[-1].log_softmax(dim=1) if log_space else thetas[-1].softmax(dim=1)
        self.beta.theta = self.beta.root_mixing

        for theta, grouping in zip(thetas[:-2], self.beta.num_branches.items()):

            # (batch_size x num_sum_nodes x num_children, K)
            # -> (num_sum_nodes x batch_size x num_children, K)
            #assert(theta.size(1) == len(grouping[1]) and theta.size(2) == grouping[0])

            theta = theta.permute(1, 2, 0, 3) #(num_sum_node x num_children x batch_size x K)
            if log_space:
                theta = F.log_softmax(theta, dim=1)
            else:
                theta = theta.softmax(dim=1)

            for param, node in zip(theta, grouping[1]):

                # shape: (batch_size x num_children, K)
                node.theta = param

    def compute_mpe_inst(self, lit_weights, binary_encoding=True):
        mpe_inst = self.beta.get_weighted_mpe(lit_weights)[1]
        if binary_encoding:
            # Sort by variable, but ignoring negatives
            mpe_inst.sort(key=lambda x: abs(x))
            return [int(x > 0) for x in mpe_inst]
        else:
            return mpe_inst

    def get_mpe_inst(self, batch_size):
        mpe_inst = self.beta.get_mpe(batch_size)
        argmax = self.beta.mixing.argmax(dim=-1)
        return mpe_inst[torch.arange(batch_size), :, argmax]

    def weighted_model_count(self, lit_weights):
        return self.beta.weighted_model_count(lit_weights)

    def get_norm_ac(self, litleaves):
        return self.beta.generate_normalized_ac(litleaves)

    def get_tf_ac(self, litleaves, log_space=False):
        return self.beta.generate_tf_ac(litleaves, log_space)

    def get_torch_ac(self, litleaves):
        return self.beta.generate_normalized_torch_ac(litleaves)

    def generate_torch_ac_stable(self, litleaves):
        return self.beta.generate_normalized_torch_ac_stable(litleaves)

    # Mainly used for debugging purposes
    def pr_inst(self, inst):
        return self.beta.pr_inst(inst)
    
    def entropy_kld(self):
        import math

        pmanager = PSddManager(self.vtree)
        gamma = pmanager.copy_and_normalize_sdd(self.alpha, self.vtree)
        prior = UniformSmoothing(1.0)
        prior.initialize_psdd(gamma)

        # log model_count(beta) - ent(beta)
        kld = self.beta.kl_psdd(gamma)
        mc = self.beta.model_count()
        entropy = -kld + math.log(mc)

        return entropy

    def Shannon_entropy(self, litleaves):
        return self.beta.Shannon_entropy(litleaves)

    def Shannon_entropy_stable(self):
        return self.beta.Shannon_entropy_stable()

    def get_models(self):
        return self.beta.models(self.vtree)

    def cross_entropy(self, target, log_space=True):
        ll = self.beta.ll(target, log_space=log_space)
        ll = (ll + self.beta.mixing).logsumexp(dim=-1)
        return -ll

    def ff_cross_entropy(self, target, litleaves, log_space=True):
        Z = self.beta.generate_tf_ac(litleaves, log_space)
        ll = self.beta.ff_ll(litleaves, target)
        ll = ll - Z
        ll = (ll + self.beta.mixing).logsumexp(dim=-1)
        return -ll

    def cross_entropy_psdd(self, target):
        return self.beta.log_likelihood(target)

    def parameterize_ff(self, litleaves):
        self.beta.parameterize_ff(litleaves)

