import sys
import torch
import random
import numpy as np
from torch import nn
import torch.nn.functional as F
from sushi_net import Net
from sushi_data import SushiData

from GatingFunction import DenseGatingFunction
from compute_mpe import CircuitMPE

torch.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)

# Create CircuitMPE instance
cmpe = CircuitMPE('permutation-4.vtree', 'permutation-4.sdd')

cmpe.beta.overparameterize(S=2)

# Load data
sushi_data = SushiData('sushi.soc')
input_size = sushi_data.train_data.shape[1]
output_size = sushi_data.train_labels.shape[1]
X_valid =  torch.tensor(sushi_data.test_data).float().cuda()
y_valid =  torch.LongTensor(sushi_data.test_labels).cuda()
print(output_size)

# Network and Gating function
model = Net(input_size, output_size).cuda()
gate = DenseGatingFunction(cmpe.beta, gate_layers=[50] + [128]*2, num_reps=2).cuda()

# Load best model
checkpoint = torch.load(f'{sys.argv[1]}')
model.load_state_dict(checkpoint['nn_state_dict'])
gate.load_state_dict(checkpoint['gf_state_dict'])


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
