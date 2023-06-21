import sys
sys.path.insert(0, './hmc-utils/pypsdd')

import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from pypsdd import Vtree, SddManager, PSddManager, SddNode, Inst, io
from pypsdd import UniformSmoothing, Prior

""" This is the main way in which SDDs should be used to compute semantic loss.
Construct an instance from a given SDD and vtree file, and then use the available
functions for computing the most probable explanation, weighted model count, or
constructing a tensorflow circuit for integrating semantic loss into a project.
"""


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
        self.beta.vtree = self.vtree

    def overparameterize(self, S=2):
        self.beta = self.beta.overparameterize(S)
        self.beta.vtree = self.vtree

    def rand_params(self):
        self.beta.rand_parameters()

    def set_params(self, thetas, log_space=True):

        #self.beta.mixing = thetas[-1].log_softmax(dim=1) if log_space else thetas[-1].softmax(dim=1)
        self.beta.mixing = thetas[-2].log_softmax(dim=1) if log_space else thetas[-2].softmax(dim=1)
        self.beta.root_mixing = thetas[-1].log_softmax(dim=1) if log_space else thetas[-1].softmax(dim=1)
        self.beta.theta = self.beta.root_mixing
        # self.beta.num_branches is an ordered dict of num_children: [sum_nodes]
        for theta, grouping in zip(thetas[:-2], self.beta.num_branches.items()):

            # (batch_size x num_sum_nodes x num_children, K)
            # -> (num_sum_nodes x batch_size x num_children, K)
            assert(theta.size(1) == len(grouping[1]) and theta.size(2) == grouping[0])

            theta = theta.transpose(0,1)
            if log_space:
                theta = F.log_softmax(theta, dim=2)
            else:
                theta = theta.softmax(dim=2)

            assert(len(theta) == len(grouping[1]))
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

    def get_sample(self, batch_size):
        mpe_inst = self.beta.sample(batch_size)
        sample = Categorical(probs=self.beta.mixing.exp()).sample()
        return mpe_inst[torch.arange(batch_size), :, sample]

    def get_marginals(self):
        return self.beta.mars()

    def weighted_model_count(self, lit_weights):
        return self.beta.weighted_model_count(lit_weights)

    def get_norm_ac(self, litleaves):
        return self.beta.generate_normalized_ac(litleaves)

    def get_tf_ac(self, litleaves):
        return self.beta.generate_tf_ac(litleaves)

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

    def cross_entropy_psdd(self, target):
        return self.beta.log_likelihood(target)

if __name__ == '__main__':
    import torch
    from torch import log
    torch.set_printoptions(precision=8)
    torch.cuda.set_device(1)

    # Testing overparameterization
    import pysdd.sdd as pysdd
    mgr = pysdd.SddManager(var_count=3)
    alpha=mgr.true()
    vtree = pysdd.Vtree(var_count=3, var_order=list(range(1, 3+1)), vtree_type='balanced')

    alpha.save('abc_true.sdd'.encode())
    vtree.save('abc_true.vtree'.encode())

    cmpe = CircuitMPE('abc_true.vtree', 'abc_true.sdd')
    nodes = cmpe.overparameterize()
    import pdb; pdb.set_trace()
    print(cmpe.beta.weighted_model_count(torch.tensor([[0.,1.], [0.,1.], [0.,1.]])))
    io.psdd_save_as_dot(cmpe.beta, 'abc_true.dot')
    exit()

    # Testing nll
    import pysdd.sdd as pysdd #import Vtree, SddManager, WmcManager, Fnf
    mgr = pysdd.SddManager(var_count=4)
    alpha = (mgr.vars[1]&mgr.vars[2])|(mgr.vars[3]&mgr.vars[4])

    # Saving circuit & vtree to disk
    alpha.save(str.encode('./abcd.sdd'))
    vtree = alpha.vtree()
    vtree.save(str.encode('./abcd.vtree'))
    cmpe = CircuitMPE('./abcd.vtree', './abcd.sdd')

    for node in cmpe.beta.as_positive_list():
        if node.is_decomposition() or node.is_true():
            node.theta = torch.tensor([[0.5, 0.5]])
    nll = cmpe.cross_entropy(torch.tensor([[1, 1, 0, 0]]))
    assert(torch.isclose(nll, torch.tensor(3.4657), atol=1e-05))

    for node in cmpe.beta.as_positive_list():
        if node.is_decomposition() or node.is_true():
            node.theta = torch.tensor([[0.2, 0.8]])
    nll = cmpe.cross_entropy(torch.tensor([[1, 1, 0, 0]]))
    assert(torch.isclose(nll, torch.tensor(6.6609), atol=1e-05))

    for node in cmpe.beta.as_positive_list():
        if node.is_decomposition() or node.is_true():
            node.theta = torch.tensor([[1.0, 1.0]])
    nll = cmpe.cross_entropy(torch.tensor([[1, 1, 0, 0]]))
    assert(torch.isclose(nll, torch.tensor(0.)))
    print("nll all good")
    exit()

    vtree = Vtree.read('abcd_constraint.vtree')
    manager = SddManager(vtree)
    alpha = io.sdd_read('abcd_constraint.sdd', manager)

    # Convert to psdd
    pmanager = PSddManager(vtree)

    # Storing psdd
    beta = pmanager.copy_and_normalize_sdd(alpha, vtree)
    prior = UniformSmoothing(1.0)
    prior.initialize_psdd(beta)

    # An sdd for the formula (a & b) | (c & d)
    c = CircuitMPE('abcd_constraint.vtree', 'abcd_constraint.sdd')

    # literal weights are of the form [[-a, a], [-b, b], [-c, c], [-d, d]]
    lit_weights = torch.tensor([[0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.9, 0.1]], device=torch.cuda.current_device())

    # Weighted model counts of both the normalized and unnormalized circuits
    wmc = c.get_norm_ac(lit_weights)

    print(c.entropy_kld())

    # Test 1
    # An sdd for the formula (a & b) | (c & d)
    c = CircuitMPE('abcd_constraint.vtree', 'abcd_constraint.sdd')

    # literal weights are of the form [[-a, a], [-b, b], [-c, c], [-d, d]]
    lit_weights = torch.tensor([[0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.9, 0.1]], device=torch.cuda.current_device())

    # Weighted model counts of both the normalized and unnormalized circuits
    wmc = c.get_tf_ac(lit_weights)
    wmc_normalized = c.get_torch_ac(lit_weights)

    # assert the wmc of the normalized and unnormalized circuits match
    assert(c.get_tf_ac(lit_weights) == 0.0976)
    assert(c.get_torch_ac(lit_weights) == 0.0976)
    
    # Entropy of the probability distribution
    weights = torch.tensor([0.0224, 0.0096, 0.0056, 0.0324, 0.0036, 0.0216, 0.0024], device=torch.cuda.current_device())
    probs = weights/wmc
    entropy = -sum([p*log(p) for p in probs])

    # Circuit Entropy
    circuit_entropy = c.Shannon_entropy(lit_weights)

    # Assert the circuit's entropy and the entropy of the groundtruth distribution match
    assert(torch.isclose(circuit_entropy, entropy))

    ## Check probabilities of all the models of the formula
    #assert(torch.isclose(c.pr_inst([-1, -2, 3, 4]), torch.tensor(0.2295), atol=1e-04))
    #assert(torch.isclose(c.pr_inst([-1, 2, 3, 4]), torch.tensor(0.0984), atol=1e-04))
    #assert(torch.isclose(c.pr_inst([1, -2, 3, 4]), torch.tensor(0.0574), atol=1e-04))
    #assert(torch.isclose(c.pr_inst([1, 2, -3, -4]), torch.tensor(0.3320), atol=1e-04))
    #assert(torch.isclose(c.pr_inst([1, 2, -3, 4]), torch.tensor(0.0369), atol=1e-04))
    #assert(torch.isclose(c.pr_inst([1, 2, 3, -4]), torch.tensor(0.2213), atol=1e-04))
    #assert(torch.isclose(c.pr_inst([1, 2, 3, 4]), torch.tensor(0.0246), atol=1e-04))


    ## Test 2
    ## An sdd for the formula true
    #c = CircuitMPE('abcd_constraint.vtree', 'true_constraint.sdd')

    ## literal weights are of the form [[-a, a], [-b, b], [-c, c], [-d, d]]
    #lit_weights = torch.tensor([[0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.9, 0.1]], device=torch.cuda.current_device())

    #models = [[0, 0, 0, 0],
    #         [0, 0, 0, 1],
    #         [0, 0, 1, 0],
    #         [0, 0, 1, 1],
    #         [0, 1, 0, 0],
    #         [0, 1, 0, 1],
    #         [0, 1, 1, 0],
    #         [0, 1, 1, 1],
    #         [1, 0, 0, 0],
    #         [1, 0, 0, 1],
    #         [1, 0, 1, 0],
    #         [1, 0, 1, 1],
    #         [1, 1, 0, 0],
    #         [1, 1, 0, 1],
    #         [1, 1, 1, 0],
    #         [1, 1, 1, 1]]

    #probs = []
    #for model in models:
    #    prob = 1
    #    for i, val in enumerate(model):
    #        prob *= lit_weights[i][val]
    #    probs += [prob]

    ## Weighted model counts of both the normalized and unnormalized circuits
    #wmc = c.get_tf_ac(lit_weights)
    #wmc_normalized = c.get_torch_ac(lit_weights)

    #assert(wmc == wmc_normalized == 1)

    ## Brute force entropy
    #entropy = -sum([p*log(p) for p in probs])

    ## Circuit Entropy
    #circuit_entropy = c.Shannon_entropy()
    #assert(circuit_entropy == entropy)

    ## Test 3
    ## An sdd for the formula (P | L) & (-A | P) & (-K | (A | L))
    #c = CircuitMPE('LKPA_constraint.vtree', 'LKPA_constraint.sdd')

    ## literal weights form     [[-L,    L], [-K,    K], [-P,    P], [-A,    A]]
    #lit_weights = torch.tensor([[0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.9, 0.1]], device=torch.cuda.current_device())

    ## Weighted model counts of both the normalized and unnormalized circuits
    #wmc = c.get_tf_ac(lit_weights)
    #wmc_normalized = c.get_torch_ac(lit_weights)

    #print(c.get_tf_ac(lit_weights))
    #print(c.get_torch_ac(lit_weights))
    #assert(c.get_tf_ac(lit_weights) == 0.4216)
    #assert(c.get_torch_ac(lit_weights) == 0.4216)
    #
    ## Entropy of the probability distribution
    #weights = torch.tensor([0.2016, 0.0224, 0.0096, 0.0756, 0.0504, 0.0056, 0.0324, 0.0216, 0.0024], device=torch.cuda.current_device())
    #probs = weights/wmc
    #entropy = -sum([p*log(p) for p in probs])

    ## Circuit Entropy
    #circuit_entropy = c.Shannon_entropy()

    ## Assert the circuit's entropy and the entropy of the groundtruth distribution match
    #assert(torch.isclose(circuit_entropy, entropy))
