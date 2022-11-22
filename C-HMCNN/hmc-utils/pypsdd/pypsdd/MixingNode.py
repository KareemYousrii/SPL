class MixingNode:
    def __init__(self, elements, theta=None):
        self.theta = theta
        self.elements = elements
    
    def nll(self):
        import pdb; pdb.set_trace()
        return torch.cat(e.nll() for e in self.elements)

    def mpe(self):
        import pdb; pdb.set_trace()
        return torch.cat(e.mpe() for e in self.elements)
