import heapq
from .data import InstMap

EPS = 1e-12
import torch
from torch import log

DEVICE = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cuda")#torch.device(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")

from torch import Tensor    
def logsumexp(tensor: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    with torch.no_grad():
        m, _ = torch.max(tensor, dim=dim, keepdim=True)
        m = m.masked_fill_(torch.isneginf(m), 0.)

    z = (tensor - m).exp_().sum(dim=dim, keepdim=True)
    mask = z == 0
    z = z.masked_fill_(mask, 1.).log_().add_(m)
    z = z.masked_fill_(mask, -float('inf'))

    if not keepdim:
        z = z.squeeze(dim=dim)
    return z

#class SafeLogAddExp(torch.autograd.Function):
#    @staticmethod
#    def forward(ctx, x, y):
#        ctx.save_for_backward(x, y)
#        return torch.logaddexp(x,y)
#
#    @staticmethod
#    def backward(ctx, grad_output):
#        import pdb; pdb.set_trace()
#        x, y = ctx.saved_tensors
#        grad_input = grad_output.clone()
#        grad_input[torch.isnan(grad_output)] = 0
#        return grad_input

class SafeLogAddExp(torch.autograd.Function):
    """Implements a torch function that is exactly like logaddexp, 
    but is willing to zero out nans on the backward pass."""
    
    @staticmethod
    def forward(ctx, input, other):            
        with torch.enable_grad():
            output = torch.logaddexp(input, other) # internal copy of output
        ctx.save_for_backward(input, other, output)
        return output.clone()

    @staticmethod
    def backward(ctx, grad_output):
        input, other, output = ctx.saved_tensors
        grad_input, grad_other = torch.autograd.grad(output, (input, other), grad_output, only_inputs=True)
        mask = torch.isinf(input).logical_and(input == other)
        grad_input[mask] = 0
        grad_other[mask] = 0
        return grad_input, grad_other

#logaddexp = SafeLogAddExp.apply

#def logaddexp(x: Tensor, y: Tensor) -> Tensor:
#    with torch.no_grad():
#        m = torch.maximum(x, y)
#        m = m.masked_fill_(torch.isneginf(m), 0.)
#
#    z = (x - m).exp_() + (y - m).exp_()
#    mask = z == 0
#    z = z.masked_fill_(mask, 1.).log_().add_(m)
#    z = z.masked_fill_(mask, -float('inf'))
#
#    return z

def logaddexp(x: Tensor, y: Tensor) -> Tensor:
    with torch.no_grad():
        m = torch.maximum(x, y)
        m = m.masked_fill_(torch.isneginf(m), 0.)

    z = (x - m).exp_() + (y - m).exp_()
    mask = z == 0
    z = z.masked_fill_(mask, 1.).log_().add_(m)
    z = z.masked_fill_(mask, -float('inf'))

    return z

class SddNode:
    """Sentential Decision Diagram (SDD)"""

    # typedef's
    FALSE,TRUE,LITERAL,DECOMPOSITION, MIXING = 0,1,2,3,4

########################################
# CONSTRUCTOR + BASIC FUNCTIONS
########################################

    def __init__(self,node_type,alpha,vtree,manager):
        """Constructor

        node_type is FALSE, TRUE, LITERAL or DECOMPOSITION
        alpha is a literal if node_type is LITERAL
        alpha is a list of elements if node_type is DECOMPOSITION
        vtree is the vtree that the node is normalized for
        manager is the SDD manager"""

        self.node_type = node_type
        self.vtree = vtree
        if self.is_false() or self.is_true():
            self.literal,self.elements = None,None
        elif self.is_literal():
            self.literal,self.elements = alpha,None
        else:# self.is_decomposition() or self.is_mixing()
            self.literal,self.elements = None,alpha
        if manager is None: self.id = None
        else: self.id = manager.new_id()
        self.data = None  # data field
        self._bit = False  # internal bit field
        self._array = None # internal array field
        # this is needed by normalized SDDs later
        self.is_false_sdd = node_type == SddNode.FALSE

    def __repr__(self,use_index=False):
        if use_index: index = lambda n: n.index
        else:         index = lambda n: n.id
        if self.is_false():
            st = 'F %d' % index(self)
        elif self.is_true():
            st = 'T %d' % index(self)
        elif self.is_literal():
            st = 'L %d %d %d' % (index(self),self.vtree.id,self.literal)
        elif self.is_mixing():
            st = 'M ' + ' '.join([str(elem) for elem in self.elements])
        else: # self.is_decomposition()
            els = self.elements
            st_el = " ".join( '%d %d' % (index(p),index(s)) for p,s in els )
            st = 'D %d %d %d %s' % (index(self),self.vtree.id,len(els),st_el)
        return st

    @classmethod
    def _dummy_true(cls):
        """used for enumeration"""
        return cls(SddNode.TRUE,None,None,None)

    def is_false(self):
        """Returns true if node is FALSE, and false otherwise"""
        return self.node_type is SddNode.FALSE

    def is_true(self):
        """Returns true if node is TRUE, and false otherwise"""
        return self.node_type is SddNode.TRUE

    def is_literal(self):
        """Returns true if node is LITERAL, and false otherwise"""
        return self.node_type is SddNode.LITERAL

    def is_mixing(self):
        """Returns true if node is MIXING, and false otherwise"""
        return self.node_type is SddNode.MIXING

    def is_decomposition(self):
        """Returns true if node is DECOMPOSITION, and false otherwise"""
        return self.node_type is SddNode.DECOMPOSITION

    def count(self):
        """Returns the number of decision nodes in the SDD"""
        return sum( 1 for n in self if n.is_decomposition() )

    def size(self):
        """Returns the aggregate size of decision nodes in the SDD"""
        return sum( len(n.elements) for n in self if n.is_decomposition() )

    def _node_count(self):
        """Returns the number of decision and terminal nodes in the SDD"""
        return sum( 1 for n in self )

########################################
# TRAVERSAL
########################################

    def __iter__(self,first_call=True,clear_data=False):
        """post-order (children before parents) generator"""
        if self._bit: return
        self._bit = True

        if self.is_mixing():
            for e in elements:
                for node in e.__iter__(first_call=False): yield node

        if self.is_decomposition():
            for p,s in self.elements:
                for node in p.__iter__(first_call=False): yield node
                for node in s.__iter__(first_call=False): yield node
        yield self

        if first_call:
            self.clear_bits(clear_data=clear_data)

    def post_order(self,clear_data=False):
        """post-order generator"""
        return self.__iter__(first_call=True,clear_data=clear_data)

    def pre_order(self,first_call=True,clear_data=False):
        """pre-order generator"""
        if self._bit: return
        self._bit = True

        yield self

        if self.is_mixing():
            for e in elements:
                for node in e.pre_order(first_call=False): yield node

        if self.is_decomposition():
            for p,s in self.elements:
                for node in p.pre_order(first_call=False): yield node
                for node in s.pre_order(first_call=False): yield node

        if first_call:
            self.clear_bits(clear_data=clear_data)

    def clear_bits(self,clear_data=False):
        """Recursively clears bits.  For use when recursively navigating an
        SDD by marking bits (not for use with SddNode.as_list).

        Set clear_data to True to also clear data."""
        if self._bit is False: return
        self._bit = False
        if clear_data: self.data = None

        if self.is_mixing():
            for e in self.elements:
                e.clear_bits(clear_data=clear_data)

        if self.is_decomposition():
            for p,s in self.elements:
                p.clear_bits(clear_data=clear_data)
                s.clear_bits(clear_data=clear_data)

    def as_list(self,reverse=False,clear_data=True):
        """iterating over an SDD's nodes, as a list

        This is faster than recursive traversal of an SDD, as done in
        __iter__() and pre_order().  If clear_data=True, then the data
        fields of each node is reset to None.  If reverse=True, then
        the list is traversed backwards (parents before children).

        Examples
        --------

        >>> for i,node in enumerate(self.as_list()):
        ...     node.data = i

        """
        self._linearize()
        if reverse:
            for node in reversed(self._array):
                yield node
        else:
            for node in self._array:
                yield node
        if clear_data:
            for node in self._array:
                node.data = None


    def _linearize(self):
        """linearize SDD"""
        if self._array is not None: return # already linearized
        self._array = list(self)

    def _is_bits_and_data_clear(self):
        """sanity check, for testing"""
        for node in self.as_list(clear_data=False):
            if node._bit is not False or \
               node.data is not None:
                return False
        return True

########################################
# QUERIES
########################################

    def model_count(self,vtree):
        """Compute model count of SDD relative to vtree.

        It is much faster to linearize the SDD compared to computing
        the model count recursively, if model counting may times."""
        if self.is_false():   return 0
        if self.is_true():    return 1 << vtree.var_count
        if self.is_literal(): return 1 << (vtree.var_count-1)
        if not self.vtree.is_sub_vtree_of(vtree):
            msg = "node vtree is not sub-vtree of given vtree"
            raise AssertionError(msg)

        # self is decomposition node
        for node in self.as_list(clear_data=True):
            if node.is_false():     count = 0
            elif node.is_true():    count = 1
            elif node.is_literal(): count = 1
            else: # node.is_decomposition()
                count = 0
                left_vc  = node.vtree.left.var_count
                right_vc = node.vtree.right.var_count
                for prime,sub in node.elements:
                    if sub.is_false(): continue
                    prime_vc = 0 if prime.is_true() else prime.vtree.var_count
                    sub_vc = 0 if sub.is_true() else sub.vtree.var_count
                    left_gap = left_vc - prime_vc
                    right_gap = right_vc - sub_vc
                    prime_mc = prime.data << left_gap
                    sub_mc = sub.data << right_gap
                    count += prime_mc*sub_mc
            node.data = count
        gap = ( vtree.var_count - node.vtree.var_count )
        return count << gap

    def is_model_marker_torch(self,inst,clear_bits=True,clear_data=True):
        """Returns None if inst is not a model, otherwise it returns the
        value/element that is satisfied.

        inst should be of type Inst or InstMap.

        Performs recursive test, which can be faster than linear
        traversal as in SddNode.as_list.

        If clear_data is set to false, then each node.data will point
        to the value/element that satisfies the SDD (if one exists).

        clear_bits is for internal use."""
        if self._bit: return self.data
        self._bit = True

        if  self.is_false(): is_model = None
        elif self.is_true():
            if hasattr(self.vtree,'var'): # normalized SDD
                is_model = inst[self.vtree.var - 1]
            else: # trimmed SDD
                is_model = True #AC: not normalized, self.data = ?
        elif self.is_literal(): 
            #if inst.is_compatible(self.literal):
            if (node.literal > 0 and inst[node.literal-1]) or (node.literal < 0 and inst[-node.literal-1]):
                is_model = inst[self.vtree.var - 1]
            else: is_model = None
        else: # self.is_decomposition()
            is_model = None
            for p,s in self.elements:
                if s.is_false_sdd: continue # for normalized SDDs
                if p.is_model_marker(inst,clear_bits=False) is None: continue
                if s.is_model_marker(inst,clear_bits=False) is None: continue
                is_model = (p,s)
                break
        self.data = is_model

        if clear_bits:
            self.clear_bits(clear_data=clear_data)
        return is_model

    def is_model_marker(self,inst,clear_bits=True,clear_data=True):
        """Returns None if inst is not a model, otherwise it returns the
        value/element that is satisfied.

        inst should be of type Inst or InstMap.

        Performs recursive test, which can be faster than linear
        traversal as in SddNode.as_list.

        If clear_data is set to false, then each node.data will point
        to the value/element that satisfies the SDD (if one exists).

        clear_bits is for internal use."""
        if self._bit: return self.data
        self._bit = True

        if  self.is_false(): is_model = None
        elif self.is_true():
            if hasattr(self.vtree,'var'): # normalized SDD
                is_model = inst[self.vtree.var]
            else: # trimmed SDD
                is_model = True #AC: not normalized, self.data = ?
        elif self.is_literal(): 
            if inst.is_compatible(self.literal):
                is_model = inst[self.vtree.var]
            else: is_model = None
        else: # self.is_decomposition()
            is_model = None
            for p,s in self.elements:
                if s.is_false_sdd: continue # for normalized SDDs
                if p.is_model_marker(inst,clear_bits=False) is None: continue
                if s.is_model_marker(inst,clear_bits=False) is None: continue
                is_model = (p,s)
                break
        self.data = is_model

        if clear_bits:
            self.clear_bits(clear_data=clear_data)
        return is_model


    def is_model(self,inst):
        """Returns True if inst (of type Inst or InstMap) is a model
        of the SDD, and False otherwise."""
        return self.is_model_marker(inst) is not None

    def models(self,vtree,lexical=False):
        """A generator for the models of an SDD.
        
        If lexical is True, then models will be given in lexical
        (sorted) order.  This is typically slower than when
        lexical=False."""

        iterator = self._models_lexical(vtree) if lexical \
                   else self._models_recursive(vtree)
        return iterator

    def _models_recursive(self,vtree):
        """Recursive model enumeration"""

        if self.is_false():
            return
        elif vtree.is_leaf():
            if self.is_true():
                yield InstMap.from_literal(-vtree.var)
                yield InstMap.from_literal(vtree.var)
            elif self.is_literal():
                yield InstMap.from_literal(self.literal)
        elif self.vtree == vtree:
            for prime,sub in self.elements:
                if sub.is_false_sdd: continue # for normalized SDDs
                for left_model in prime._models_recursive(vtree.left):
                    for right_model in sub._models_recursive(vtree.right):
                        yield left_model.concat(right_model)
        else: # there is a gap
            true_node = SddNode._dummy_true()
            if self.is_true():
                prime,sub = true_node,true_node
            #elif self.vtree.is_sub_vtree_of(vtree.left):
            elif self.vtree.id < vtree.id:
                prime,sub = self,true_node
            #elif self.vtree.is_sub_vtree_of(vtree.right):
            elif self.vtree.id > vtree.id:
                prime,sub = true_node,self
            else:
                msg = "node vtree is not sub-vtree of given vtree"
                raise AssertionError(msg)
            for left_model in prime._models_recursive(vtree.left):
                for right_model in sub._models_recursive(vtree.right):
                    yield left_model.concat(right_model)

    def _models_lexical(self,vtree):
        """Lexical model enumeration"""
        enum = SddEnumerator(vtree)
        return enum.enumerator(self)

    def minimum_cardinality(self):
        "AC: NEED TO TEST"
        """Computes the minimum cardinality model of an SDD."""
        if self.is_false():   return 0
        if self.is_true():    return 0
        if self.is_literal(): return 1 if self.literal > 0 else 0

        # self is decomposition node
        for node in self.as_list(clear_data=True):
            if node.is_false():     card = 0
            elif node.is_true():    card = 0
            elif node.is_literal(): card = 1 if node.literal > 0 else 0
            else: # node.is_decomposition()
                card = None
                for prime,sub in node.elements:
                    if sub.is_false(): continue
                    element_card = prime.data + sub.data
                    if card is None:
                        card = element_card
                    else:
                        card = min(card,element_card)
            node.data = card
        return card

########################################
# NORMALIZED SDDS
########################################

class NormalizedSddNode(SddNode):
    """Normalized Sentential Decision Diagram (SDD)"""

    def __init__(self,node_type,alpha,vtree,manager):
        """Constructor

        node_type is FALSE, TRUE, LITERAL or DECOMPOSITION
        alpha is a literal if node_type is LITERAL
        alpha is a list of elements if node_type is DECOMPOSITION
        vtree is the vtree that the node is normalized for
        manager is the PSDD manager"""

        SddNode.__init__(self,node_type,alpha,vtree,manager)
        self.negation = None
        #self.is_false_sdd = node_type == SddNode.FALSE
        if node_type is SddNode.DECOMPOSITION:
            self.positive_elements = \
                tuple( (p,s) for p,s in self.elements if not s.is_false_sdd )
        self._positive_array = None

    def _positive_node_count(self):
        """Returns the number of (decision and terminal) nodes in the SDD"""
        count = 0
        for node in self.positive_iter():
            count += 1
        return count

    def positive_iter(self,first_call=True,clear_data=False):
        """post-order (children before parents) generator, skipping false SDD
        nodes"""
        if self._bit: return
        if self.is_false_sdd: return
        self._bit = True

        if self.is_mixing():
            for e in self.elements:
                for node in e.positive_iter(first_call=False): yield node

        if self.is_decomposition():
            for p,s in self.elements:
                if s.is_false_sdd: continue
                for node in p.positive_iter(first_call=False): yield node
                for node in s.positive_iter(first_call=False): yield node

        yield self

        if first_call:
            self.clear_bits(clear_data=clear_data)

    def as_positive_list(self,reverse=False,clear_data=True):
        """iterating over an SDD's nodes, as a list.  See SddNode.as_list"""
        if self._positive_array is None:
            self._positive_array = list(self.positive_iter())
        if reverse:
            for node in reversed(self._positive_array):
                yield node
        else:
            for node in self._positive_array:
                yield node
        if clear_data:
            for node in self._positive_array:
                node.data = None

    def model_count(self,evidence=InstMap(),clear_data=True):
        """Compute model count of a normalized SDD.

        SddNode.model_count does not assume the SDD is normalized."""
        for node in self.as_list(clear_data=clear_data):
            if node.is_false():
                count = 0
            elif node.is_true():
                count = 1 if node.vtree.var in evidence else 2
            elif node.is_literal():
                count = 1 if evidence.is_compatible(node.literal) else 0
            else: # node.is_decomposition()
                count = sum( p.data*s.data for p,s in node.elements )
            node.data = count

        return count

    def get_weighted_mpe(self, lit_weights, clear_data=True):
        """Compute the MPE instation given weights associated with literals.

        Assumes the SDD is normalized.
        """
        for node in self.as_positive_list(clear_data=clear_data):
            if node.is_false():
                # No configuration on false
                data = (0, [])
            elif node.is_true():
                # Need to pick max assignment for variable here
                b_ind = max([0,1], key=lambda x: lit_weights[node.vtree.var-1][x])
                # If it's a 0, then -lit number, else lit number
                data = (lit_weights[node.vtree.var-1][b_ind], [pow(-1,b_ind+1) * node.vtree.var])
            elif node.is_literal():
                if node.literal > 0:
                    data = (lit_weights[node.literal-1][1], [node.literal])
                else:
                    data = (lit_weights[-node.literal-1][0], [node.literal])
            else: # node is_decomposition()
                data = max(((p.data[0] * s.data[0], p.data[1] + s.data[1]) for p,s in node.positive_elements)
                        , key=lambda x: x[0])
            node.data = data

        # Need to put the literals in ascending order,
        # sorting by the absolute value of the literal
        indices = data.abs().argsort(dim=-1)
        return data.gather(1, indices)

########################################
# Start Determinstic and SD PCs
########################################
    
    def overparameterize(self, S=2, manager=None, first_call=True, clear_data=False):
        """This function overparameterizes a DSD PC by replicating
        sum nodes, and taking cross-products of sum nodes over disjoint
        sets of variables
        """
        import copy
        import itertools
        import random
        if self._bit: return self.nodes
        if self.is_false_sdd: return []
        self._bit = True

        if self.is_literal():
            nodes = [self]

        elif self.is_true():
            nodes = [self] + [copy.copy(self) for _ in range(1, S)]
            assert(len(nodes) == S)

        else:#self.is_decomposition():
            #TODO: canonicalize node and update unique
            elements = []
            for i, (p,s) in enumerate(self.elements):
                if s.is_false_sdd:
                    continue
                left = p.overparameterize(S, first_call=False)
                right = s.overparameterize(S, first_call=False)
                elements.extend(list(itertools.product(left, right)))
            self.elements = tuple(elements)
            nodes = [self] + [copy.copy(self) for _ in range(1, S)]
            assert(len(nodes) == S)

        self.nodes = nodes

        if first_call:
            nodes = NormalizedSddNode(SddNode.MIXING, nodes, None, manager)
            nodes._bit = True
            nodes.clear_bits(clear_data=clear_data)

        return nodes

    def rand_parameters(self):
        """Initialize all parameters to zero.
        
        ka: Hasn't been tested
        """
        for node in self.positive_iter():
            if node.is_literal(): continue
            node.theta = torch.rand(966, len(node.positive_elements), device=DEVICE, requires_grad=True)
            node.theta = torch.nn.functional.softmax(node.theta, dim=-1) #dict( (el,0.0) for el in root.positive_elements )
            #node.theta_sum = 0.0


    @torch.no_grad()
    def get_mpe(self, batch_size, clear_data=True):
        """Compute the MPE instation given weights associated with literals.

        Assumes the SDD is normalized.
        """
        tint = torch.int16
        for node in self.as_positive_list(clear_data=clear_data):

            if not node.data is None:
                data = node.data
                continue

            if node.is_false():
                # No configuration on false
                data = torch.tensor([], device=DEVICE, dtype=tint)

            elif node.is_true():
                data = torch.where(node.theta.argmax(dim=0) > 0, torch.tensor(node.vtree.var, device=DEVICE, dtype=tint), torch.tensor(-node.vtree.var, device=DEVICE, dtype=tint))
                data = data.unsqueeze(dim=-2)
                #assert(len(data.shape) == 3 and data.shape[0] == len(node.theta) and data.shape[1] == 1)

            elif node.is_literal():
                data = torch.tensor([node.literal], device=DEVICE, dtype=tint).unsqueeze(0).unsqueeze(-1).expand(batch_size, 1, self.num_reps)
                #data = torch.full((batch_size, 1, self.num_reps), node.literal, device=DEVICE, dtype=torch.int)

            elif node.is_mixing():

                node.theta = node.theta.transpose(0,1) #Shape: (num_children x batch_size)
                #assert(node.theta.shape[1] == batch_size and node.theta.shape[0] == len(node.elements))

                data = torch.tensor([], device=DEVICE, dtype=tint)
                for element in node.elements:
                    data = torch.cat((data, element.data.unsqueeze(0)), dim=0)

                max_branch = node.theta.argmax(dim=0, keepdim=True)
                max_branch = max_branch.unsqueeze(dim=-2).expand((1, *data.shape[1:]))

                data = torch.gather(data, 0, max_branch.type(torch.int64)).squeeze(dim=0)
                #assert(len(data.shape) == 3)

            else: # node is_decomposition()

                if len(node.positive_elements) == 1:
                    max_branch = torch.tensor([0], device=DEVICE, dtype=tint).unsqueeze(-1).unsqueeze(-1).expand(1, batch_size, self.num_reps)
                    #max_branch = torch.zeros((1, batch_size, self.num_reps), device=DEVICE, dtype=torch.int16)

                else:
                    max_branch = node.theta.argmax(dim=0, keepdim=True)

                data = torch.stack([torch.cat((p.data, s.data), dim=-2) for p, s in node.positive_elements])
                #data = torch.tensor([], device=DEVICE, dtype=tint)
                #for p, s in node.positive_elements:
                #    a = torch.cat((p.data, s.data), dim=-2).unsqueeze(dim=0)
                #    data = torch.cat((data, a), dim=0)

                max_branch = max_branch.unsqueeze(dim=-2).expand((1, *data.shape[1:]))
                data = torch.gather(data, 0, max_branch.type(torch.int64)).squeeze(dim=0).type(tint)
                #assert(len(data.shape) == 3)

            node.data = data
            node.theta = None
            max_branch = None

        # Need to put the literals in ascending order,
        # sorting by the absolute value of the literal
        indices = data.abs().argsort(dim=-2)
        return data.gather(1, indices)

    def ff_ll(self, litleaves, target, clear_data=True):
        """ Generates a torch arithmetic circuit according to the weighted
        model counting procedure for this SDD. We populate both the weighted
        model counts (in each node's data field) as well as the parameters
        (thetas) for decision nodes and true leaf nodes.

        Assumes the SDD is normalized.
        """
        K = self.mixing.size(1)
        for node in self.as_positive_list(clear_data=clear_data):

            # Use cache
            if node.data is not None:
                data = node.data
                continue

            if node.is_false():
                data = torch.tensor(-300., device=DEVICE)

            elif node.is_true():
                # We associate parameters for true literals, normalized
                # according to the corresponding vtree
                data = litleaves[node.vtree.var - 1][target[:, node.vtree.var - 1] > 0]

            elif node.is_literal():
                data = (target[:, abs(node.literal)-1] == (node.literal>0))#.log()#.float()
                data = torch.where(data.unsqueeze(-1).expand(-1, K), litleaves[abs(node.literal) - 1][node.literal>1], torch.tensor(-300., device=DEVICE))

            elif node.is_decomposition():
                if len(node.positive_elements) == 1:
                    p, s = node.positive_elements[0]
                    data = p.data + s.data
                    
                else:
                    primes, subs = zip(*node.positive_elements)
                    primes = torch.stack([p.data for p in primes])
                    subs = torch.stack([s.data for s in subs])
                    data = (primes + subs).logsumexp(dim=0)

            node.data = data
        return data

    def parameterize_ff(self, litleaves, clear_data=True):
        """ Generates a torch arithmetic circuit according to the weighted
        model counting procedure for this SDD. We populate both the weighted
        model counts (in each node's data field) as well as the parameters
        (thetas) for decision nodes and true leaf nodes.

        Assumes the SDD is normalized.
        """
        bsz = litleaves[0][0].size(0)
        for node in self.as_positive_list(clear_data=clear_data):

            # Use cache
            if node.data is not None:
                data = node.data
                continue

            if node.is_false():
                data = torch.full(bsz, -300, device=DEVICE)#torch.tensor(-300., device=DEVICE)

            elif node.is_true():
                # We associate parameters for true literals, normalized
                # according to the corresponding vtree
                node.theta = torch.stack([litleaves[node.vtree.var - 1][0], litleaves[node.vtree.var - 1][1]])
                data = torch.zeros((bsz, self.num_reps), device=DEVICE)

            elif node.is_literal():
                if node.literal > 0:
                    data = litleaves[node.literal-1][1]
                else:
                    data = litleaves[-node.literal-1][0]

            elif node.is_decomposition():

                # Calculate, and normalize the distribution at every
                # decision node
                #primes = []
                #subs = []
                #for (p, s) in node.positive_elements:
                #    primes += [p.data]
                #    subs += [s.data]
                #primes = torch.stack(primes)
                #subs = torch.stack(subs)
                #node.theta = primes+subs

                if len(node.positive_elements) == 1:
                    p, s = node.positive_elements[0]
                    data = p.data + s.data
                    node.theta = torch.full_like(data, 0.) 
                    
                else:
                    primes, subs = zip(*node.positive_elements)
                    primes = torch.stack([p.data for p in primes])
                    subs = torch.stack([s.data for s in subs])
                    node.theta = primes + subs
                    data = node.theta.logsumexp(dim=0)

                    # Normalize
                    node.theta = node.theta - data


                #node.theta = torch.stack([p.data + s.data\
                #        for p, s in node.positive_elements])
                # Calculate the partition function
                #Z = torch.logsumexp(node.data, dim=0)

                # Normalize
                #node.theta = node.data - Z
                #assert(torch.allclose(node.theta.logsumexp(dim=0), torch.tensor(0.), atol=1e-05))

            node.data = data

    def ll_bak(self, target, log_space=True, clear_data=True):
        """ Calculate the probabilty of particular
        model of the underlying circuit. Assumes that
        the circuit is normalized.
        """
        for node in self.as_positive_list(clear_data=clear_data):

                if node.data is not None:
                    data = node.data
                    continue

                if node.is_false():
                    #data = torch.tensor(0.0) if not log_space else torch.tensor(-float('inf'), device=DEVICE)
                    data = torch.tensor(-300., device=DEVICE)

                elif node.is_true():
                    #data = torch.where((target[:, node.vtree.var - 1] > 0).unsqueeze(-1).expand(-1, 1), node.theta[:, 1], node.theta[:, 0])
                    data = torch.where((target[:, node.vtree.var - 1] > 0), node.theta[:, 1], node.theta[:, 0])

                elif node.is_literal():
                    if node.literal > 0:
                        data = (target[:, node.literal - 1] ==  1)#.log()#.float()
                    else:
                        data = (target[:, -node.literal - 1] ==  0)#.log()#.float()
                    data = torch.where(data == 1, torch.tensor(0., device=DEVICE), torch.tensor(-300., device=DEVICE))

                    #if log_space:
                    #    data = data.log()
                    #data = data.unsqueeze(-1).expand(-1, 1)

                #elif node.is_mixing():

                #    #node.theta = node.theta.transpose(0,1) #Shape: (num_children x batch_size)

                #    data = torch.tensor([], device=DEVICE)
                #    for element in node.elements:
                #        data = torch.cat((data, element.data.unsqueeze(0)), dim=0)
                #    #data = torch.stack((e.data for e in node.elements))
                #    #import pdb; pdb.set_trace()
                #    data = logsumexp(node.theta + data, dim=0)

                else: # node.is_decomposition


                        #data = None
                        ##data = torch.tensor(-float('inf'), device=DEVICE)
                        #for i, (p, s) in enumerate(node.positive_elements):
                        #    if data is None:
                        #        data = node.theta[i] + p.data + s.data
                        #    else:
                        #        data = logaddexp(node.theta[i] + p.data + s.data, data)

                        if len(node.positive_elements)  == 1:
                            p, s = node.positive_elements[0]
                            data = node.theta[0] + p.data + s.data
                        else:
                            #for i, (p,s) in enumerate(node.positive_elements):
                            #    if s.is_false():
                            #        import pdb; pdb.set_trace()
                            #primes = torch.zeros_like(node.theta)
                            #subs = torch.zeros_like(node.theta)
                            #for i, (p, s) in enumerate(node.positive_elements):
                            #    primes[i], subs[i] = p.data, s.data
                            #data = (primes + subs + node.theta).logsumexp(dim=0)

                            #data = None
                            #for i, (p, s) in enumerate(node.positive_elements):
                            #    if data is None:
                            #        data = node.theta[i] + p.data_ + s.data_
                            #    else:
                            #        data = torch.logaddexp(node.theta[i] + p.data_ + s.data_, data)

                            #primes = torch.tensor([], device=DEVICE)
                            #subs = torch.tensor([], device=DEVICE)
                            #for p, s in node.positive_elements:
                            #    primes = torch.cat((p.data_.unsqueeze(0), primes))
                            #    subs = torch.cat((s.data_.unsqueeze(0), subs))
                            #data = (primes + subs + node.theta).logsumexp(dim=0)

                            primes, subs = zip(*node.positive_elements)
                            primes = torch.stack(tuple(p.data for p in primes))
                            subs = torch.stack(tuple(s.data for s in subs))
                            data = (primes + subs + node.theta).logsumexp(dim=0)

                            #data = torch.stack(tuple(torch.stack((p.data_, s.data_)) for p, s in node.positive_elements))
                            #primes, subs = data.unbind(axis=1)
                            #data = (primes + subs + node.theta).logsumexp(dim=0)

                            #import pdb; pdb.set_trace()
                            #data = torch.stack([node.theta[i] + p.data + s.data \
                            #        for i, (p, s) in enumerate(node.positive_elements)]).logsumexp(dim=0)
                        #for i, (p, s) in enumerate(node.positive_elements):
                        #    if data is None:
                        #        data = node.theta[i] + p.data + s.data
                        #    else:
                        #        data = torch.logaddexp(node.theta[i] + p.data + s.data, data)

                node.data = data

        if log_space:
            return data

        data = torch.log(data)
        return data

    def ll(self, target, log_space=True, clear_data=True):
        """ Calculate the probabilty of particular
        model of the underlying circuit. Assumes that
        the circuit is normalized.
        """
        K = self.mixing.size(1)
        for node in self.as_positive_list(clear_data=clear_data):

            if node.data is not None:
                data = node.data
                continue

            if node.is_false():
                data = torch.tensor(-300., device=DEVICE)

            elif node.is_true():
                data = torch.where((target[:, node.vtree.var - 1] > 0), node.theta[:, 1], node.theta[:, 0])

            elif node.is_literal():
                if node.literal > 0:
                    data = (target[:, node.literal - 1] ==  1)#.log()#.float()
                else:
                    data = (target[:, -node.literal - 1] ==  0)#.log()#.float()
                #data = ((data.float() - 1) * 300).unsqueeze(-1).expand(-1, 1)
                #data = data * torch.tensor(0., device=DEVICE) + (~data) * torch.tensor(-300., device=DEVICE)
                data = torch.where(data.unsqueeze(-1).expand(-1, K), torch.tensor(0., device=DEVICE), torch.tensor(-300., device=DEVICE))

            elif node.is_mixing():
                data = torch.stack([element.data for element in node.elements])
                data = (node.theta + data).logsumexp(dim=0)

            else: # node.is_decomposition

                #node.theta = node.theta.squeeze(-1).transpose(0,1) #Shape: (num_children x batch_size)
                if len(node.positive_elements)  == 1:
                    p, s = node.positive_elements[0]
                    #data = node.theta[0] + p.data + s.data
                    data = p.data + s.data
                else:
                    primes, subs = zip(*node.positive_elements)
                    primes = torch.stack([p.data for p in primes])
                    subs = torch.stack([s.data for s in subs])
                    data = (primes + subs + node.theta).logsumexp(dim=0)

            node.data = data

        if log_space:
            return data

        data = torch.log(data)
        return data

########################################
# End Determinstic and SD PCs
########################################

    def weighted_model_count(self, lit_weights, clear_data=True):
        """ Compute weighted model count given literal weights

        Assumes the SDD is normalized.
        """
        for node in self.as_list(clear_data=clear_data):
            if node.is_false():
                data = 0
            elif node.is_true():
                data = 1
            elif node.is_literal():
                if node.literal > 0:
                    data = lit_weights[node.literal-1][1]
                else:
                    data = lit_weights[-node.literal-1][0]
            else: # node is_decomposition
                data = sum(p.data * s.data for p,s in node.elements)
            node.data = data
        return data

    def generate_tf_ac(self, litleaves, log_space=False, clear_data=True):
        """ Generates a tensorflow arithmetic circuit according to the weighted model counting procedure for this SDD.

        Assumes the SDD is normalized.
        """
        # Going to need pytorch for this, but not for the rest of the project, so import here
        import torch
        for node in self.as_positive_list(clear_data=clear_data):

            # Cache
            if node.data is not None:
                data = node.data
                continue

            if node.is_false():
                data = torch.tensor(0.0 if not log_space else -300., device=DEVICE)

            elif node.is_true():
                data = torch.full((litleaves[0][0].size(0), self.num_reps), 1.0 if not log_space else 0., device=DEVICE)

            elif node.is_literal():
                if node.literal > 0:
                    data = litleaves[node.literal-1][1]
                else:
                    data = litleaves[-node.literal-1][0]

            else: # node.is_decomposition
                primes, subs = zip(*node.positive_elements)
                primes = torch.stack([p.data for p in primes])
                subs = torch.stack([s.data for s in subs])

                if not log_space:
                    data = (primes*subs).sum(dim=0)
                else:
                    data = (primes+subs).logsumexp(dim=0)
                #data = sum([p.data * s.data for p,s in node.elements])
            node.data = data
        return data

    def generate_normalized_ac(self, litleaves, clear_data=False):
        """ Generates a torch arithmetic circuit according to the weighted
        model counting procedure for this SDD. We populate both the weighted
        model counts (in each node's data field) as well as the parameters
        (thetas) for decision nodes and true leaf nodes.

        Assumes the SDD is normalized.
        """
        for node in self.as_list(clear_data=clear_data):

            if node.is_false():
                data = 0.0

            elif node.is_true():

                # We associate parameters for true literals, normalized
                # according to the corresponding vtree
                node.theta = [litleaves[node.vtree.var - 1][0], litleaves[node.vtree.var - 1][1]]

                data = 1.0

            elif node.is_literal():

                # We associate parameters for true literals, normalized
                # according to the corresponding vtree
                node.theta = [node.literal < 0, node.literal > 0]

                if node.literal > 0:
                    data = litleaves[node.literal-1][1]

                else:
                    data = litleaves[-node.literal-1][0]

            else: # node.is_decomposition

                # Calculate, and normalize the distribution at every
                # decision node
                node.theta = [p.data * s.data for p, s in node.elements]

                # Calculate the partition function
                Z = sum(node.theta)
                if Z == 0: Z = 1

                # Normalize
                node.theta = [p/Z for p in node.theta]
                node.theta = dict(zip(node.elements,node.theta))

                # Weighted Model Count
                data = sum([p.data * s.data for p,s in node.elements])

            node.data = data
            node.theta_sum = 1.0

        return data

    def generate_normalized_torch_ac_stable(self, litleaves, clear_data=False):
        """ Generates a torch arithmetic circuit according to the weighted
        model counting procedure for this SDD. We populate both the weighted
        model counts (in each node's data field) as well as the parameters
        (thetas) for decision nodes and true leaf nodes.

        Assumes the SDD is normalized.
        """
        import torch

        for node in self.as_positive_list(clear_data=clear_data):

            if node.is_true():

                # We associate parameters for true literals, normalized
                # according to the corresponding vtree
                node.theta = torch.stack([litleaves[node.vtree.var - 1][0], litleaves[node.vtree.var - 1][1]])

            elif node.is_decomposition():

                # Calculate, and normalize the distribution at every
                # decision node
                node.theta = torch.stack([p.data.clamp(min=1e-16).log() + s.data.clamp(min=1e-16).log()\
                        for p, s in node.positive_elements])

                # Calculate the partition function
                Z = torch.logsumexp(node.theta, dim=0)

                # Normalize
                node.theta = torch.exp(node.theta - Z)

                try:
                    # Sanity check: assert no nan parameters
                    assert(torch.isnan(node.theta).sum() == 0)
                except:
                    import pdb; pdb.set_trace()



    def wmc(alpha, lit_weights, log_space=True):
        d = {}
        for node in as_list(alpha):
            if node.is_false():
                data = torch.tensor(0.0 if not log_space else -float('inf'), device=torch.cuda.current_device())
            elif node.is_true():
                data = torch.tensor(1.0 if not log_space else 0.0, device=torch.cuda.current_device())
            elif node.is_literal():
                if node.literal > 0:
                    data = lit_weights[node.literal-1][1]
                else:
                    data = lit_weights[-node.literal-1][0]
            else:
                if log_space:
                    try:
                        data = logsumexp(torch.stack([d[p.id] + d[s.id] for p, s in node.elements()], dim=-1), dim=-1)
                        if (data > 0).any():
                            import pdb; pdb.set_trace()
                    except:
                        import pdb; pdb.set_trace()
                        print("problem")
                else:
                    data = sum([d[p.id]*d[s.id] for p, s in node.elements()])
            d[node.id] = data
        return data


    def Shannon_entropy_stable(self, clear_data=False):
        """
        Follows the algorithm from Shih and Ermon (2020) for calculating
        the entropy of a selective SPN.
        """
        EPS = 1e-12
        import torch
        from torch import log

        for node in self.as_positive_list(clear_data=clear_data):

            if node.is_false():

                # Entropy of an indicator variable
                entropy = torch.tensor(0.0, device=DEVICE)
                                       # torch.cuda.current_device())

            elif node.is_true():

                # Entropy of a Bernoulli random variable
                entropy = node.theta[0] * log(node.theta[0] + EPS)\
                        + node.theta[1] * log(node.theta[1] + EPS)

            elif node.is_literal():

                # Entropy of an indicator variable
                entropy = torch.tensor(0.0, device=DEVICE)
                # torch.cuda.current_device())

            else: # node.is_decomposition

                # Entropy of a sum node is the entropy of the distribution over
                # its children + the weighted entropy of each of its children;
                # each child is a product node, whose entropy is in turn just the
                # sum of entropies of its children
                entropy = torch.tensor(0.0, device=DEVICE)#torch.cuda.current_device())
                for i, child_theta in enumerate(node.theta):
                    assert(len(node.theta) == len(node.positive_elements))

                    # Entropy of the current child
                    p,s = node.positive_elements[i]
                    child_entropy = p.entropy + s.entropy

                    # Add to the parent's entropy the entropy of the current child based on the distribution
                    # + the weighted child's entropy
                    entropy = entropy + (child_theta * log(child_theta + EPS) + child_theta * child_entropy)

                    # Sanity check: assert no nan parameters
                    try:
                        assert(torch.isnan(entropy).sum() == 0)
                    except:
                        import pdb; pdb.set_trace()

            node.entropy = entropy

        return -entropy

    def generate_normalized_torch_ac(self, litleaves, clear_data=False):
        """ Generates a torch arithmetic circuit according to the weighted
        model counting procedure for this SDD. We populate both the weighted
        model counts (in each node's data field) as well as the parameters
        (thetas) for decision nodes and true leaf nodes.

        Assumes the SDD is normalized.
        """
        import torch
        for node in self.as_list(clear_data=clear_data):

            if node.is_false():
                #data = torch.tensor(0.0, device=torch.cuda.current_device())
                data = torch.tensor(0.0)

            elif node.is_true():

                # We associate parameters for true literals, normalized
                # according to the corresponding vtree
                node.theta = torch.stack([litleaves[node.vtree.var - 1][0], litleaves[node.vtree.var - 1][1]])

                data = torch.tensor(1.0)
                #data = torch.tensor(1.0, device=torch.cuda.current_device())

            elif node.is_literal():

                # We associate parameters for true literals, normalized
                # according to the corresponding vtree
                node.theta = torch.Tensor([node.literal < 0, node.literal > 0])

                if node.literal > 0:
                    data = litleaves[node.literal-1][1]

                else:
                    data = litleaves[-node.literal-1][0]

            else: # node.is_decomposition

                # Calculate, and normalize the distribution at every
                # decision node
                node.theta = torch.stack([p.data * s.data for p, s in node.elements])
                #try:
                #    node.theta = [p.data * s.data for p, s in node.elements]
                #    node.theta = torch.cat(node.theta).view(-1, 128)
                #except:
                #    node.theta = torch.stack(node.theta)

                # Calculate the partition function
                Z = torch.sum(node.theta, dim=0)
                Z = Z.masked_fill(Z <= 0, 1).clamp(min=1e-16) # Is there a better way of doing this?

                # Normalize
                node.theta = node.theta / Z

                # Weighted Model Count
                data = sum([p.data * s.data for p,s in node.elements])

            node.data = data

        return data

    def Shannon_entropy(self, litleaves, clear_data=False):
        for node in self.as_list(clear_data=clear_data):

            if node.is_false():
                entropy = torch.tensor(0.0)

            elif node.is_true():
                # Entropy of a Bernoulli random variable
                entropy = litleaves[node.vtree.var - 1][0] * torch.log(litleaves[node.vtree.var - 1][0] + EPS)\
                        + litleaves[node.vtree.var - 1][1] * torch.log(litleaves[node.vtree.var - 1][1] + EPS)

            elif node.is_literal():
                entropy = torch.tensor(0.0)

            else: # node.is_decomposition
                # Entropy of a sum node is the entropy of the distribution over
                # its children + the weighted entropy of each of its children;
                # each child is a product node, whose entropy is in turn just the
                # sum of entropies of its children

                res = []
                for child in node.elements:
                     child_entropy = child[0].entropy + child[1].entropy
                     child_theta = torch.div(child[0].data * child[1].data, (node.data + EPS))
                     res.append(child_theta * torch.log(child_theta + EPS) + child_theta * child_entropy)
                entropy = sum(res)

            node.entropy = entropy

        return -entropy


    def pr_inst(self, inst, clear_data=False):
        """ Calculate the probabilty of particular
        model of the underlying circuit. Assumes that
        the circuit is normalized.
        """
        import torch
        for node in self.as_list(clear_data=clear_data):

            if node.is_false():
                data = torch.tensor(0.0, device=DEVICE)#torch.cuda.current_device())

            elif node.is_true():
                data = node.theta[1] if inst[node.vtree.var - 1] > 0 else node.theta[0] 

            elif node.is_literal():
                if node.literal > 0:
                    data = torch.tensor(inst[node.literal - 1] > 0, device=DEVICE)#torch.cuda.current_device())
                else:
                    data = torch.tensor(inst[-node.literal - 1] <  0, device=DEVICE)#torch.cuda.current_device())

            else: # node.is_decomposition
                data = 0
                for i, child_theta in enumerate(node.theta):
                    p,s = node.elements[i]
                    data += child_theta * p.data * s.data

            node.data = data

        return data


    def hard_sampling(self, clear_data=False):
        """ Calculate the probabilty of particular
        model of the underlying circuit. Assumes that
        the circuit is normalized.
        """

        import torch
        from torch.distributions.bernoulli import Bernoulli
        from torch.distribution.categorical import Categorical

        for node in self.as_list(clear_data=clear_data):

            if node.is_false():
                inst = torch.tensor(0, device=DEVICE)#torch.cuda.current_device())

            elif node.is_true():
                distribution = Bernoulli(node.theta)
                samples = distribution.sample()
                samples[samples == 0] = samples[samples == 0] - 1
                inst = samples * node.vtree.var

            elif node.is_literal():
                inst = node.literal

            else: # node.is_decomposition
                distribution = Categorical(node.theta)
                samples = distribution.sample()
                #p, s = unzip(node.elements[samples])?
                #inst = torch.stack(p, s)?

            node.inst = inst

        return data

    def gumbel_sampling(self, clear_data=False):
        """ Calculate the probabilty of particular
        model of the underlying circuit. Assumes that
        the circuit is normalized.
        """

        import torch
        from torch.distributions.bernoulli import Bernoulli
        from torch.distribution.categorical import Categorical

        for node in self.as_list(clear_data=clear_data):

            if node.is_false():
                inst = torch.tensor(0, device=DEVICE)#torch.cuda.current_device())

            elif node.is_true():
                distribution = Bernoulli(node.theta)
                samples = distribution.sample()
                samples[samples == 0] = samples[samples == 0] - 1
                inst = samples * node.vtree.var

            elif node.is_literal():
                inst = node.literal

            else: # node.is_decomposition
                distribution = Categorical(node.theta)
                samples = distribution.sample()
                #p, s = unzip(node.elements[samples])?
                #inst = torch.stack(p, s)?

            node.inst = inst

        return data


########################################
# MODEL ENUMERATION
########################################

class SddEnumerator:
    """Manager for lexical model enumeration.

    Caching only nodes (caching elements may not help much?)"""

    @staticmethod
    def _element_update(element_enum,inst):
        """This is invoked after inst.concat(other)"""
        pass

    def __init__(self,vtree):
        self.vtree = vtree
        self.true = SddNode._dummy_true()
        self.node_cache = dict( (v,dict()) for v in vtree )
        self.terminal_enumerator = SddTerminalEnumerator

    def lookup_node_enum(self,node,vtree):
        cache = self.node_cache[vtree]
        if node in cache:
            return cache[node].cached_iter()
        else:
            enum = SddNodeEnumerator(node,vtree,self)
            cache[node] = enum
            return enum.cached_iter()

    def enumerator(self,node):
        return SddNodeEnumerator(node,self.vtree,self)

class SddTerminalEnumerator:
    """Enumerator for terminal SDD nodes"""

    def __init__(self,node,vtree):
        self.heap = []

        if node.is_false():
            pass
        elif node.is_literal():
            inst = InstMap.from_literal(node.literal)
            heapq.heappush(self.heap,inst)
        if node.is_true():
            inst = InstMap.from_literal(-vtree.var)
            heapq.heappush(self.heap,inst)
            inst = InstMap.from_literal(vtree.var)
            heapq.heappush(self.heap,inst)

    def __iter__(self):
        return self

    def empty(self):
        return len(self.heap) == 0

    def __next__(self):
        if self.empty(): raise StopIteration()
        return heapq.heappop(self.heap)

    def __cmp__(self,other):
        return cmp(self.heap[0],other.heap[0])

class SddNodeEnumerator:
    """Enumerator for SDD decomposition nodes"""

    def __init__(self,node,vtree,enum_manager):
        self.heap = []
        self.topk = []

        if node.is_false():
            pass
        elif vtree.is_leaf():
            enum = enum_manager.terminal_enumerator(node,vtree)
            if not enum.empty(): heapq.heappush(self.heap,enum)
        elif node.vtree == vtree: # node.is_decomposition
            for prime,sub in node.elements: # initialize
                #if sub.is_false(): continue
                if sub.is_false_sdd: continue
                enum = SddElementEnumerator(prime,sub,node,vtree,enum_manager)
                if not enum.empty(): heapq.heappush(self.heap,enum)
        else: # gap
            true_node = enum_manager.true
            if node.is_true():
                prime,sub = true_node,true_node
            elif node.vtree.is_sub_vtree_of(vtree.left):
                prime,sub = node,true_node
            elif node.vtree.is_sub_vtree_of(vtree.right):
                prime,sub = true_node,node
            else:
                msg = "node vtree is not sub-vtree of given vtree"
                raise AssertionError(msg)
            enum = SddElementEnumerator(prime,sub,node,vtree,enum_manager)
            if not enum.empty(): heapq.heappush(self.heap,enum)
            
    def __iter__(self):
        return self

    def empty(self):
        return len(self.heap) == 0

    def __next__(self):
        while not self.empty():
            enum = heapq.heappop(self.heap)
            model = next(enum)
            self.topk.append(model)
            if not enum.empty(): heapq.heappush(self.heap,enum)
            return model
        raise StopIteration()

    def cached_iter(self):
        k = 0
        while True:
            if k < len(self.topk):
                yield self.topk[k]
                k += 1
            else:
                try:
                    next(self)
                except StopIteration:
                    return

class SddElementEnumerator:
    """Enumerator for SDD elements (prime/sub pairs)"""

    class HeapElement:
        def __init__(self,pinst,siter,element_enum,piter=None):
            self.pinst = pinst
            self.piter = piter
            self.siter = siter
            self.inst = None
            self.element_enum = element_enum
            self.enum_manager = element_enum.enum_manager

            self._try_next()

        def __iter__(self):
            return self

        def empty(self):
            return self.inst is None

        def _try_next(self):
            try:
                sinst = next(self.siter)
                self.inst = self.pinst.concat(sinst)
                self.enum_manager._element_update(self.element_enum,self.inst)
            except StopIteration:
                self.inst = None

        def __next__(self):
            if self.inst is None:
                raise StopIteration()
            else:
                saved_model = self.inst
                self._try_next()
                return saved_model

        def __cmp__(self,other):
            assert self.inst is not None and other.inst is not None
            return cmp(self.inst,other.inst)

    def __init__(self,prime,sub,parent,vtree,enum_manager):
        self.prime = prime
        self.sub = sub
        self.parent = parent
        self.vtree = vtree
        self.enum_manager = enum_manager
        self.heap = []

        piter = enum_manager.lookup_node_enum(prime,vtree.left)
        self._push_next_element_enumerator(piter)

    def __iter__(self):
        return self

    def empty(self):
        return len(self.heap) == 0

    def _push_next_element_enumerator(self,piter):
        try:
            pinst = next(piter)
        except StopIteration:
            pinst = None
        if pinst is not None:
            siter = self.enum_manager.lookup_node_enum(self.sub,self.vtree.right)
            enum = SddElementEnumerator.HeapElement(pinst,siter,self,piter=piter)
            if not enum.empty(): heapq.heappush(self.heap,enum)

    def __next__(self):
        while not self.empty():
            best = heapq.heappop(self.heap)
            inst = next(best)

            if best.piter is not None: # generate next prime model
                piter,best.piter = best.piter,None
                self._push_next_element_enumerator(piter)

            if not best.empty(): heapq.heappush(self.heap,best)
            return inst

        raise StopIteration()

    def __cmp__(self,other):
        assert not self.empty() and not other.empty()
        return cmp(self.heap[0].inst,other.heap[0].inst)

