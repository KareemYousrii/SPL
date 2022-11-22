import sys
import torch
import random
import numpy as np
from torch import nn
import torch.nn.functional as F
from grid_net import Net
from grid_data import GridData

from GatingFunction import DenseGatingFunction
from compute_mpe import CircuitMPE

torch.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)

f = sys.argv[1]
params = sys.argv[1].split('_')
num_reps = int(params[-4])
S = int(params[-3])

# Create CircuitMPE instance
cmpe = CircuitMPE('4-grid-out.vtree.sd', '4-grid-all-pairs-sd.sdd')

if S > 1:
    cmpe.beta.overparameterize(S=S)

# Network and Gating function
model = Net().cuda()
gate = DenseGatingFunction(cmpe.beta, gate_layers=[50] + [128]*2, num_reps=num_reps).cuda()

# Load best model
checkpoint = torch.load(f'{sys.argv[1]}')
model.load_state_dict(checkpoint['nn_state_dict'])
gate.load_state_dict(checkpoint['gf_state_dict'])

# Load data
grid_data = GridData('test.data')
X_valid =  torch.tensor(grid_data.test_data).float().cuda()
y_valid =  torch.LongTensor(grid_data.test_labels).cuda()
y_valid = torch.cat((y_valid, X_valid[:, 24:]), dim=1)

# Set model to eval
model.eval()
gate.eval()

# Forward
output = model(X_valid)
thetas = gate(output)
cmpe.set_params(thetas, log_space=False)

# Get MPE as indicators
mpe = cmpe.get_mpe_inst(X_valid.shape[0])
preds = (mpe > 0).long()

# Percentage that are exactly right
exactly_correct = torch.all(preds == y_valid, dim=1)
percent_exactly_correct = exactly_correct.sum().to(dtype=torch.float)/exactly_correct.size(0)
print("Percentage of validation that are exactly right: %f" % (percent_exactly_correct * 100))

# Percentage of individual labels that are right
individual_correct = (preds == y_valid).sum()
percent_individual_correct = individual_correct.to(dtype=torch.float) / len(preds.flatten())
print("Percentage of individual labels in validation that are right: %f" % (percent_individual_correct * 100))

# Percentage of predictions that satisfy the constraint
wmc = cmpe.weighted_model_count([(1-p, p) for p in preds.unbind(1)])
print("Percentage of predictions that satisfy the constraint %f", 100*sum(wmc)/len(wmc))

## Percentage that are exactly right
#exactly_correct = torch.all(preds == y_valid, dim=1)
#print(exactly_correct.sum())
#percent_exactly_correct = exactly_correct.sum().to(dtype=torch.float)/exactly_correct.size(0)
#print("Percentage of validation that are exactly right: %f" % (percent_exactly_correct * 100))
#
#
## Percentage of individual labels that are right
#individual_correct = (preds == y_valid).sum()
#percent_individual_correct = individual_correct.to(dtype=torch.float) / len(preds.flatten())
#print("Percentage of individual labels in validation that are right: %f" % (percent_individual_correct * 100))
#
## Percentage of predictions that satisfy the constraint
#wmc = [cmpe.weighted_model_count([(1-p, p) for p in np.concatenate((out, inp[24:]))]) for out, inp in zip(np.array(valid_out.cpu().detach() + 0.5, int), X_valid.cpu().detach())]
