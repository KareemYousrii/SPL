import os
import datetime
import json
from time import perf_counter

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import (
    precision_score, 
    average_precision_score, 
    hamming_loss, 
    jaccard_score
)

# Circuit imports
import sys
sys.path.append(os.path.join(sys.path[0],'hmc-utils'))
sys.path.append(os.path.join(sys.path[0],'hmc-utils', 'pypsdd'))

from GatingFunction import DenseGatingFunction
from compute_mpe import CircuitMPE
from pysdd.sdd import SddManager, Vtree

# misc
from common import *

def log1mexp(x):
        assert(torch.all(x >= 0))
        return torch.where(x < 0.6931471805599453094, torch.log(-torch.expm1(-x)), torch.log1p(-torch.exp(-x)))

class ConstrainedFFNNModel(nn.Module):
    """ C-HMCNN(h) model - during training it returns the not-constrained output that is then passed to MCLoss """
    def __init__(self, input_dim, hidden_dim, output_dim, hyperparams, R):
        super(ConstrainedFFNNModel, self).__init__()
        
        self.nb_layers = hyperparams['num_layers']
        self.R = R
        
        fc = []
        for i in range(self.nb_layers):
            if i == 0:
                fc.append(nn.Linear(input_dim, hidden_dim))
            elif i == self.nb_layers-1:
                fc.append(nn.Linear(hidden_dim, output_dim))
            else:
                fc.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc = nn.ModuleList(fc)
        
        self.drop = nn.Dropout(hyperparams['dropout'])
        
        if hyperparams['non_lin'] == 'tanh':
            self.f = nn.Tanh()
        else:
            self.f = nn.ReLU()
        
    def forward(self, x, sigmoid=False, log_sigmoid=False):
        for i in range(self.nb_layers):
            if i == self.nb_layers-1:
                if sigmoid:
                    x = nn.Sigmoid()(self.fc[i](x))
                elif log_sigmoid:
                    x = nn.LogSigmoid()(self.fc[i](x))
                else:
                    x = self.fc[i](x)
            else:
                x = self.f(self.fc[i](x))
                x = self.drop(x)

        
        if self.R is None:
            return x
        
        if self.training:
            constrained_out = x
        else:
            constrained_out = get_constr_out(x, self.R)
        return constrained_out

def main():

    args = parse_args()

    # Set device
    torch.cuda.set_device(int(args.device))

    # Load train, val and test set
    dataset_name, ontology = args.dataset.split("_")[:2]
    hidden_dim = hidden_dims[ontology][dataset_name]
    num_epochs = args.n_epochs

    # Set the hyperparameters 
    hyperparams = {
        'num_layers': 3,
        'dropout': 0.7,
        'non_lin': 'relu',
    }

    # Set seed
    seed_all_rngs(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    
    # Load the datasets
    dataset_name = args.dataset
    data = dataset_name.split('_')[0]
    ontology = dataset_name.split('_')[1]
    from sklearn import preprocessing

    if ('others' in args.dataset):
        train, test = initialize_other_dataset(dataset_name, datasets)
        train.to_eval, test.to_eval = torch.tensor(train.to_eval, dtype=torch.bool),  torch.tensor(test.to_eval, dtype=torch.bool)
    else:
        train, val, test = initialize_dataset(dataset_name, datasets)
        train.to_eval, val.to_eval, test.to_eval = torch.tensor(train.to_eval, dtype=torch.bool), torch.tensor(val.to_eval, dtype=torch.bool), torch.tensor(test.to_eval, dtype=torch.bool)

    different_from_0 = torch.tensor(np.array((test.Y.sum(0)!=0), dtype = bool), dtype=torch.bool)

    # Rescale data and impute missing data
    if ('others' in args.dataset):
        scaler = preprocessing.StandardScaler().fit((train.X.astype(float)))
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit((train.X.astype(float)))
    else:
        scaler = preprocessing.StandardScaler().fit(np.concatenate((train.X, val.X)))
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit(np.concatenate((train.X, val.X)))
        val.X, val.Y = torch.tensor(scaler.transform(imp_mean.transform(val.X))).to(device), torch.tensor(val.Y).to(device)
    train.X, train.Y = torch.tensor(scaler.transform(imp_mean.transform(train.X))).to(device), torch.tensor(train.Y).to(device)
    test.X, test.Y = torch.tensor(scaler.transform(imp_mean.transform(test.X))).to(device), torch.tensor(test.Y).to(device)

    #Create loaders
    train_dataset = [(x, y) for (x, y) in zip(train.X, train.Y)]
    if ('others' not in args.dataset):
        val_dataset = [(x, y) for (x, y) in zip(val.X, val.Y)]
        for (x, y) in zip(val.X, val.Y):
            train_dataset.append((x,y))
    test_dataset = [(x, y) for (x, y) in zip(test.X, test.Y)]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False)

    # We do not evaluate the performance of the model on the 'roots' node (https://dtai.cs.kuleuven.be/clus/hmcdatasets/)
    if 'GO' in dataset_name:
        num_to_skip = 4
    else:
        num_to_skip = 1

    # Prepare circuit: TODO needs cleaning
    if not args.no_constraints:

        if not os.path.isfile('constraints/' + dataset_name + '.sdd') or not os.path.isfile('constraints/' + dataset_name + '.vtree'):
            # Compute matrix of ancestors R
            # Given n classes, R is an (n x n) matrix where R_ij = 1 if class i is ancestor of class j
            R = np.zeros(train.A.shape)
            np.fill_diagonal(R, 1)
            g = nx.DiGraph(train.A)
            for i in range(len(train.A)):
                descendants = list(nx.descendants(g, i))
                if descendants:
                    R[i, descendants] = 1
            R = torch.tensor(R)

            #Transpose to get the ancestors for each node 
            #R = R.transpose(1, 0)
            R = R.unsqueeze(0).to(device)

            # Uncomment below to compile the constraint
            R.squeeze_()
            mgr = SddManager(
                var_count=R.size(0),
                auto_gc_and_minimize=True)

            alpha = mgr.true()
            alpha.ref()
            for i in range(R.size(0)):

               beta = mgr.true()
               beta.ref()
               for j in range(R.size(0)):

                   if R[i][j] and i != j:
                       old_beta = beta
                       beta = beta & mgr.vars[j+1]
                       beta.ref()
                       old_beta.deref()

               old_beta = beta
               beta = -mgr.vars[i+1] | beta
               beta.ref()
               old_beta.deref()

               old_alpha = alpha
               alpha = alpha & beta
               alpha.ref()
               old_alpha.deref()

            # Saving circuit & vtree to disk
            alpha.save(str.encode('constraints/' + dataset_name + '.sdd'))
            alpha.vtree().save(str.encode('constraints/' + dataset_name + '.vtree'))

        # Create circuit object
        cmpe = CircuitMPE('constraints/' + dataset_name + '.vtree', 'constraints/' + dataset_name + '.sdd')

        if args.S > 0:
            cmpe.overparameterize(S=args.S)
            print("Done overparameterizing")

        # Create gating function
        gate = DenseGatingFunction(cmpe.beta, gate_layers=[128] + [256]*args.gates, num_reps=args.num_reps).to(device)
        R = None


    else:
        # Use fully-factorized sdd
        mgr = SddManager(var_count=train.A.shape[0], auto_gc_and_minimize=True)
        alpha = mgr.true()
        vtree = Vtree(var_count = train.A.shape[0], var_order=list(range(1, train.A.shape[0] + 1)))
        alpha.save(str.encode('ancestry.sdd'))
        vtree.save(str.encode('ancestry.vtree'))
        cmpe = CircuitMPE('ancestry.vtree', 'ancestry.sdd')
        cmpe.overparameterize()

        # Gating function
        gate = DenseGatingFunction(cmpe.beta, gate_layers=[462]).to(device)
        R = None

    # We do not evaluate the performance of the model on the 'roots' node (https://dtai.cs.kuleuven.be/clus/hmcdatasets/)
    if 'GO' in dataset_name: 
        num_to_skip = 4
    else:
        num_to_skip = 1 

    # Output path
    if args.exp_id:
        out_path = os.path.join(args.output, args.exp_id)
    else:
        date_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = os.path.join(args.output,  '{}_{}_{}_{}_{}'.format(args.dataset, date_string, args.batch_size, args.gates, args.lr))
    os.makedirs(out_path, exist_ok=True)

    # Tensorboard
    writer = SummaryWriter(log_dir=os.path.join(out_path, "runs"))

    # Dump experiment parameters
    args_out_path = os.path.join(out_path, 'args.json')
    json_args = json.dumps(vars(args))

    print("Starting with arguments:\n%s\n\tdumped at %s", json_args, args_out_path)
    with open(args_out_path, 'w') as f:
        f.write(json_args)

    # Create the model
    # Load train, val and test set
    dataset_name, ontology = args.dataset.split("_")[:2]
    hidden_dim = hidden_dims[ontology][dataset_name]
    num_epochs = args.n_epochs

    model = ConstrainedFFNNModel(input_dims[dataset_name], hidden_dim, 128, hyperparams, R)
    model.to(device)
    print("Model on gpu", next(model.parameters()).is_cuda)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(gate.parameters()), lr=args.lr, weight_decay=args.wd)
    criterion = nn.BCELoss(reduction="none")

    def evaluate(model):
        test_val_t = perf_counter()
        for i, (x,y) in enumerate(test_loader):

            model.eval()
                    
            x = x.to(device)
            y = y.to(device)

            constrained_output = model(x.float(), sigmoid=True)
            predicted = constrained_output.data > 0.5

            # Total number of labels
            total = y.size(0) * y.size(1)

            # Total correct predictions
            correct = (predicted == y.byte()).sum()
            num_correct = (predicted == y.byte()).all(dim=-1).sum()

            # Move output and label back to cpu to be processed by sklearn
            predicted = predicted.to('cpu')
            cpu_constrained_output = constrained_output.to('cpu')
            y = y.to('cpu')

            if i == 0:
                test_correct = num_correct
                predicted_test = predicted
                constr_test = cpu_constrained_output
                y_test = y
            else:
                test_correct += num_correct
                predicted_test = torch.cat((predicted_test, predicted), dim=0)
                constr_test = torch.cat((constr_test, cpu_constrained_output), dim=0)
                y_test = torch.cat((y_test, y), dim =0)

        test_val_e = perf_counter()
        avg_score = average_precision_score(y_test[:,test.to_eval], constr_test.data[:,test.to_eval], average='micro')
        jss = jaccard_score(y_test[:,test.to_eval], predicted_test[:,test.to_eval], average='micro')
        print(f"Number of correct: {test_correct}")
        print(f"avg_score: {avg_score}")
        print(f"test micro AP {jss}\t{(test_val_e-test_val_t):.4f}")

    def evaluate_circuit(model, gate, cmpe, epoch, data_loader, data_split, prefix):

        test_val_t = perf_counter()

        for i, (x,y) in enumerate(data_loader):

            model.eval()
            gate.eval()
                    
            x = x.to(device)
            y = y.to(device)

            # Parameterize circuit using nn
            emb = model(x.float())
            thetas = gate(emb)

            # negative log likelihood and map
            cmpe.set_params(thetas)
            nll = cmpe.cross_entropy(y, log_space=True).mean()

            cmpe.set_params(thetas)
            pred_y = (cmpe.get_mpe_inst(x.shape[0]) > 0).long()

            pred_y = pred_y.to('cpu')
            y = y.to('cpu')

            num_correct = (pred_y == y.byte()).all(dim=-1).sum()

            if i == 0:
                test_correct = num_correct
                predicted_test = pred_y
                y_test = y
            else:
                test_correct += num_correct
                predicted_test = torch.cat((predicted_test, pred_y), dim=0)
                y_test = torch.cat((y_test, y), dim=0)

        dt = perf_counter() - test_val_t
        y_test = y_test[:,data_split.to_eval]
        predicted_test = predicted_test[:,data_split.to_eval]
        
        accuracy = test_correct / len(y_test)
        nll = nll.detach().to("cpu").numpy() / (i+1)
        jaccard = jaccard_score(y_test, predicted_test, average='micro')
        hamming = hamming_loss(y_test, predicted_test)

        print(f"Evaluation metrics on {prefix} \t {dt:.4f}")
        print(f"Num. correct: {test_correct}")
        print(f"Accuracy: {accuracy}")
        print(f"Hamming Loss: {hamming}")
        print(f"Jaccard Score: {jaccard}")
        print(f"nll: {nll}")


        return {
            f"{prefix}/accuracy": (accuracy, epoch, dt),
            f"{prefix}/hamming": (hamming, epoch, dt),
            f"{prefix}/jaccard": (jaccard, epoch, dt),
            f"{prefix}/nll": (nll, epoch, dt),
        }


    for epoch in range(num_epochs):

        if epoch % 5 == 0 and epoch != 0:

            print(f"EVAL@{epoch}")
            perf = {
                **evaluate_circuit(
                    model,
                    gate, 
                    cmpe,
                    epoch=epoch,
                    data_loader=test_loader,
                    data_split=test,
                    prefix="param_sdd/test",
                ),
            }

            for perf_name, (score, epoch, dt) in perf.items():
                writer.add_scalar(perf_name, score, global_step=epoch, walltime=dt)

            writer.flush()

        train_t = perf_counter()

        model.train()
        gate.train()

        tot_loss = 0
        for i, (x, labels) in enumerate(train_loader):

            x = x.to(device)
            labels = labels.to(device)
        
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            if args.no_constraints:

                # Use fully-factorized distribution via circuit
                output = model(x.float(), sigmoid=False)
                thetas = gate(output)
                cmpe.set_params(thetas)
                loss = cmpe.cross_entropy(labels, log_space=True).mean()

            else:
                y = labels
                output = model(x.float(), sigmoid=False)
                thetas = gate(output)
                cmpe.set_params(thetas)
                loss = cmpe.cross_entropy(labels, log_space=True).mean()

            tot_loss += loss
            loss.backward()
            optimizer.step()

        train_e = perf_counter()
        print(f"{epoch+1}/{num_epochs} train loss: {tot_loss/(i+1)}\t {(train_e-train_t):.4f}")

if __name__ == "__main__":
    main()
