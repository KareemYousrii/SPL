# python sushi_net.py --entropy_circuit --layers 3 --units 25 --iters 20000 --data test.data --wmc 0.25 --entropy_weight 0.01

import argparse
from torch import nn
import torch.nn.functional as F
import sys
sys.path.insert(0, 'pypsdd')

import numpy as np
from numpy.random import permutation

from sushi_data import SushiData, to_pairwise_comp

from GatingFunction import DenseGatingFunction
from compute_mpe import CircuitMPE

import torch
import random
from torch.distributions.bernoulli import Bernoulli


torch.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)

FLAGS = None

class Net(nn.Module):

    def __init__(self, input_size, output_size, units=25):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, units)
        self.output = nn.Linear(units, 50) 

    def forward(self, x):
        x = self.fc1(x)
        x = F.sigmoid(x)

        x = self.fc2(x)
        x = F.sigmoid(x)

        x = self.fc3(x)
        x = F.sigmoid(x)

        output = self.output(x)
        return output

def main():

    # setting cuda
    device = torch.device("cuda:" + str(FLAGS.device) if torch.cuda.is_available() else "cpu")

    # Import data
    sushi_data = SushiData('sushi.soc')
    input_size = sushi_data.train_data.shape[1]
    output_size = 50#sushi_data.train_labels.shape[1]

    # Create the model
    model = Net(input_size, output_size).cuda()

    # Get supervised part (rest is unsupervised)
    perm = permutation(sushi_data.train_data.shape[0])
    sup_train_inds = perm[:int(sushi_data.train_data.shape[0] * FLAGS.give_labels)]
    unsup_train_inds = perm[int(sushi_data.train_data.shape[0] * FLAGS.give_labels):]

    # Mask out the loss for the unsupervised samples
    ce_weights = torch.zeros([sushi_data.train_data.shape[0], 1]).cuda()
    ce_weights[sup_train_inds, :] = 1

    # Create CircuitMPE instance for predictions
    cmpe = CircuitMPE('permutation-4.vtree', 'permutation-4.sdd')

    if FLAGS.S > 1:
        cmpe.beta.overparameterize(S=FLAGS.S)

    # Gating function
    gate = DenseGatingFunction(cmpe.beta, gate_layers=[50] + [FLAGS.num_units]*FLAGS.num_layers, num_reps=FLAGS.num_reps).to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(list(gate.parameters()) + list(model.parameters()), lr=FLAGS.lr, weight_decay=0.000)
  
    prev_loss = 1e15
    max_coherent = 0

    # Load data
    X = torch.tensor(sushi_data.train_data).float().cuda()
    y = torch.tensor(sushi_data.train_labels).float().cuda()

    for i in range(FLAGS.iters):

        # train
        model.train()
        gate.train()
        optimizer.zero_grad()

        # Forward
        output = model(X)
        thetas = gate(output)
        cmpe.set_params(thetas, log_space=True)
        cross_entropy = cmpe.cross_entropy(y, log_space=True)

        # Calculate loss
        loss = 1.0 * cross_entropy.mean()
        print(f"iter:{i}, loss:{loss}")

        # Backward and step
        loss.backward()
        optimizer.step()

        # Every 1k iterations check accuracy
        if i % 5 == 0 and i != 0:

            print("After %d iterations" % i)

            # Set model to eval
            model.eval()
            gate.eval()

            X_valid =  torch.tensor(sushi_data.valid_data).float().cuda()
            y_valid =  torch.LongTensor(sushi_data.valid_labels).cuda()

            # Parameterize circuit
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

            if max_coherent < percent_exactly_correct * 100:
                max_coherent = percent_exactly_correct * 100
                print("Saving new best model")
                torch.save({
                        'epoch': i,
                        'nn_state_dict': model.state_dict(),
                        'gf_state_dict': gate.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, f'saved_models/best_model_sushi_{FLAGS.lr}_{FLAGS.num_reps}_{FLAGS.S}_{FLAGS.num_layers}_{FLAGS.num_units}.pt')
                print(f'saved_models/best_model_sushi_{FLAGS.lr}_{FLAGS.num_reps}_{FLAGS.S}_{FLAGS.num_layers}_{FLAGS.num_units}.pt')

            print("max so far: ", max_coherent)

            # Percentage of individual labels that are right
            individual_correct = (preds == y_valid).sum()
            percent_individual_correct = individual_correct.to(dtype=torch.float) / len(preds.flatten()) 
            print("Percentage of individual labels in validation that are right: %f" % (percent_individual_correct * 100))

            # Percentage of predictions that satisfy the constraint
            #wmc = cmpe.weighted_model_count([(1-p, p) for p in preds.unbind(1)])
            #print("Percentage of predictions that satisfy the constraint %f", 100*sum(wmc)/len(wmc))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='test.data',
                      help='Input data file to use')
    parser.add_argument('--units', type=int, default=25,
                      help='Number of units per hidden layer')
    parser.add_argument('--layers', type=int, default=3,
                      help='Number of hidden layers')
    parser.add_argument('--iters', type=int, default=20000,
                      help='Number of minibatch steps to do')
    parser.add_argument('--relu', action='store_true',
                      help='Use relu hidden units instead of sigmoid')
    parser.add_argument('--early_stopping', action='store_true',
                      help='Enable early stopping - quit when validation loss is increasing')
    parser.add_argument('--give_labels', type=float, default=1.0,
                      help='Percentage of training examples to use labels for (1.0 = supervised)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--num_reps', type=int, default=1,
                        help='Number of components in the mixture')
    parser.add_argument('--S', type=int, default=1,
                        help='Factor by which to duplicate the sum nodes')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers in the gating function')
    parser.add_argument('--num_units', type=int, default=128,
                        help='Number of units in each layer of the gating function')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU (default:0)')

    FLAGS, unparsed = parser.parse_known_args()
    main()
