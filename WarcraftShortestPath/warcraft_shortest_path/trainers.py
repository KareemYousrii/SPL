import random

import time
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from comb_modules.losses import HammingLoss
from comb_modules.dijkstra import ShortestPath
from logger import Logger
from models import get_model
from utils import AverageMeter, optimizer_from_string, customdefaultdict
from decorators import to_tensor, to_numpy
from . import metrics
from .metrics import compute_metrics
import numpy as np
from collections import defaultdict
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from .visualization import draw_paths_on_image

def get_trainer(trainer_name):
    trainers = {"Baseline": BaselineTrainer,"SL": SLTrainer, "Circuit": CircuitTrainer, "FFCircuit": FFCircuitTrainer, "DijkstraOnFull": DijkstraOnFull}
    return trainers[trainer_name]

def get_neighbors(i,j):
    ret = []
    d = [(-1,0), (-1,-1), (0, -1), (1, -1), (1, 0)]
    for x,y in d:
        ii = i+x
        jj = j+y
        if ii >= 0 and jj >= 0 and ii < ROWS and jj < COLS:
            ret.append((ii,jj))
    return ret

# Circuit paths
import os
import sys
sys.path.append(os.path.join(sys.path[0], '..' ,'grids'))
sys.path.append(os.path.join(sys.path[0], '..' ,'grids', 'pypsdd'))

# Circuit imports
from GatingFunction import DenseGatingFunction
from compute_mpe import CircuitMPE

cmpe = CircuitMPE(f'data/warcraft_shortest_path/12x12/constraint_trimmed.vtree', f'data/warcraft_shortest_path/12x12/constraint_trimmed.sdd')
gate = None

e2i = torch.load('e2i.pt')

@torch.jit.script
def log1mexp(x):
        #assert(torch.all(x >= 0))
        return torch.where(x < 0.6931471805599453094, torch.log(-torch.expm1(-x)), torch.log1p(-torch.exp(-x))).clamp_(max=-1e-5)

# Circuit functions
def parameterize_tiles(output):
    for i in range(2):
        thetas = gates[i](output)
        cmpes[i].set_params(thetas)

def mpe_tiles(bsz):
    up = cmpes[0].get_mpe_inst(bsz).view(-1, 8, 12) 
    down = cmpes[1].get_mpe_inst(bsz).view(-1, 4, 12) 
    grid = torch.cat((up, down), dim=1)
    return (grid > 0).long()

def nll_tiles(target):
    up = cmpes[0].cross_entropy(target[:, :8, :].flatten(start_dim=1)).unsqueeze(0)
    down = cmpes[1].cross_entropy(target[:, 8:12, :].flatten(start_dim=1)).unsqueeze(0)
    return torch.cat((up, down), dim=0).sum(dim=0)


def parameterize(output):
    thetas = gate(output)
    cmpe.set_params(thetas)

def mpe(bsz):
    mpe = cmpe.get_mpe_inst(bsz)
    return (mpe > 0).long()

def nll(target):
    return cmpe.cross_entropy(target.flatten(start_dim=1))

class ShortestPathAbstractTrainer(ABC):
    def __init__(
        self,
        *,
        train_iterator,
        test_iterator,
        metadata,
        use_cuda,
        batch_size,
        optimizer_name,
        optimizer_params,
        model_params,
        fast_mode,
        neighbourhood_fn,
        preload_batch,
        lr_milestone_1,
        lr_milestone_2,
        use_lr_scheduling,
        num_layers,
        num_reps,
        num_units,
        S,
        sl_weight
    ):

        self.fast_mode = fast_mode
        self.use_cuda = use_cuda
        self.optimizer_params = optimizer_params
        self.batch_size = batch_size
        self.test_iterator = test_iterator
        self.train_iterator = train_iterator
        self.metadata = metadata
        self.grid_dim = 12#int(np.sqrt(self.metadata["output_features"]))
        self.neighbourhood_fn = neighbourhood_fn
        self.preload_batch = preload_batch
        self.num_units = num_units
        self.num_layers = num_layers
        self.num_reps = num_reps
        cmpe.beta.num_reps = num_reps
        self.S = S
        self.sl_weight = sl_weight

        self.best_eval = -1

        self.model = None
        self.build_model(**model_params)

        if self.S > 0:
            cmpe.beta.overparameterize(S=self.S)

        global gate
        gate = DenseGatingFunction(cmpe.beta, gate_layers=[self.num_units] + [self.num_units]*self.num_layers, num_reps=self.num_reps).cuda()

        if self.use_cuda:
            self.model.to("cuda")

        self.optimizer = optimizer_from_string(optimizer_name)(list(self.model.parameters()) + list(gate.parameters()), **optimizer_params)
        #self.optimizer = optimizer_from_string(optimizer_name)(list(self.model.parameters()), **optimizer_params)

        self.use_lr_scheduling = use_lr_scheduling
        if use_lr_scheduling:
            self.scheduler = MultiStepLR(self.optimizer, milestones=[lr_milestone_1, lr_milestone_2], gamma=0.1)

        self.epochs = 0
        self.train_logger = Logger(scope="training", default_output="tensorboard")
        self.val_logger = Logger(scope="validation", default_output="tensorboard")

    def train_epoch(self):

        self.epochs += 1
        
        # Training Metrics
        batch_time = AverageMeter("Batch time")
        avg_loss = AverageMeter("Loss")

        # Set model and gating functions to train
        self.model.train()
        #gate.train()

        start = time.time()

        # Training data iterator
        iterator = self.train_iterator.get_epoch_iterator(batch_size=self.batch_size,
                number_of_epochs=1, device='cuda' if self.use_cuda else 'cpu', preload=self.preload_batch)

        # Start epoch
        for i, data in enumerate(iterator):

            #if i == 5:
            #    break

            start_time = time.time()
            input, true_path, true_weights = data["images"], data["labels"],  data["true_weights"]

            loss  = self.forward_pass(input, true_path, train=True, i=i)

            # update batch metrics
            avg_loss.update(loss.item(), input.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - start_time)
            print(batch_time)

        meters = [batch_time, avg_loss]
        meter_str = "\t".join([str(meter) for meter in meters])
        print(f"Epoch: {self.epochs}\t{meter_str}")

        if self.use_lr_scheduling:
            self.scheduler.step()

        #self.train_logger.log(avg_loss.avg, "loss")

        return {
            "train_loss": avg_loss.avg,
        }

    @torch.no_grad()
    def evaluate(self, print_paths=False):
        avg_metrics = defaultdict(AverageMeter)

        self.optimizer.zero_grad(set_to_none=True)
        self.model.eval()
        #gate.eval()

        # Test iterator
        #iterator = self.test_iterator.get_epoch_iterator(batch_size=128,
        #        number_of_epochs=1, shuffle=False, device='cuda' if self.use_cuda else 'cpu', preload=self.preload_batch)

        iterator = self.test_iterator.get_epoch_iterator(batch_size=64,
                number_of_epochs=1, shuffle=False, device='cuda', preload=self.preload_batch)

        a = None
        for i, data in enumerate(iterator):
            input, true_path, true_weights = (
                data["images"],#.contiguous(),
                data["labels"],#.contiguous(),
                data["true_weights"],#.contiguous(),
            )

            #if self.use_cuda:
            #    input = input.cuda()
            #    true_path = true_path.cuda()
            #    true_weights = true_weights.cuda()

            start_time = time.time()
            print("before")
            accuracy, last_suggestion = self.forward_pass(input, true_path, train=False, i=i)
            print("after")
            print(time.time() - start_time)
            suggested_path = last_suggestion["suggested_path"]
            data.update(last_suggestion)

            if a is None:
                a = suggested_path.cpu().detach().numpy()
            else:
                a = np.concatenate((a, suggested_path.cpu().detach().numpy()), axis=0)

            evaluated_metrics = metrics.compute_metrics(true_paths=true_path,
            suggested_paths=suggested_path, true_vertex_costs=true_weights, e2i=e2i)
            avg_metrics["accuracy"].update(accuracy, input.size(0))
            for key, value in evaluated_metrics.items():
                avg_metrics[key].update(value, input.size(0))

            if self.fast_mode:
                break

        for key, avg_metric in avg_metrics.items():
            self.val_logger.log(avg_metric.avg, key=key)
        avg_metrics_values = dict([(key, avg_metric.avg) for key, avg_metric in avg_metrics.items()])

        if avg_metrics_values['below_0.0001_percent_acc'] > self.best_eval:
            self.best_eval = avg_metrics_values['below_0.0001_percent_acc']
            print("Saving new best model")
            torch.save({
                        'epoch': self.epochs,
                        'nn_state_dict': self.model.state_dict(),
                        #'gf_state_dict': gate.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        }, f'checkpoints/{self.num_units}_{self.S}_{self.num_reps}.pt')

        return avg_metrics_values

    @abstractmethod
    def build_model(self, **kwargs):
        pass

    @abstractmethod
    def forward_pass(self, input, true_shortest_paths, train, i):
        pass

    def log(self, data, train, k=None, num=None):
        logger = self.train_logger if train else self.val_logger
        if not train:
            image = self.metadata['denormalize'](data["images"][k]).squeeze().astype(np.uint8)
            suggested_path = data["suggested_path"][k].squeeze()
            labels = data["labels"][k].squeeze()

            suggested_path_im = torch.ones((3, *suggested_path.shape))*255*suggested_path.cpu()
            labels_im = torch.ones((3, *labels.shape))*255*labels.cpu()
            image_with_path = draw_paths_on_image(image=image, true_path=labels, suggested_path=suggested_path, scaling_factor=10)

            logger.log(labels_im.data.numpy().astype(np.uint8), key=f"shortest_path_{num}", data_type="image")
            logger.log(suggested_path_im.data.numpy().astype(np.uint8), key=f"suggested_path_{num}", data_type="image")
            logger.log(image_with_path, key=f"full_input_with_path{num}", data_type="image")



class BaselineTrainer(ShortestPathAbstractTrainer):
    def build_model(self, model_name, arch_params):
        self.model = get_model(
            model_name, out_features=264, in_channels=self.metadata["num_channels"], arch_params=arch_params
        )

    def forward_pass(self, input, label, train, i):

        # Get embedding
        output = self.model(input)
        output = torch.sigmoid(output)
        flat_target = label.view(label.size()[0], -1)

        if train:

            # Cross-Entropy
            criterion = torch.nn.BCELoss(reduction='none')
            loss = criterion(output, flat_target).mean()
            return loss

        else:

            bsz = label.size()[0]
            flat_target = label.view(label.size()[0], -1)

            # Get point-wise accuracy
            accuracy = (output.round() * flat_target).sum() / flat_target.sum()

            # Get suggested_path
            suggested_path = output.view(label.shape).round()
            valid_paths = cmpe.get_tf_ac([[1-p, p] for p in suggested_path.unbind(axis=-1)]).bool()
            last_suggestion = {"vertex_costs": None, "suggested_path": suggested_path}

            return accuracy, last_suggestion, valid_paths

    @torch.no_grad()
    def evaluate(self, print_paths=False):
        avg_metrics = defaultdict(AverageMeter)
        self.model.eval()
        gate.eval()

        # Test iterator
        iterator = self.test_iterator.get_epoch_iterator(batch_size=128,
                number_of_epochs=1, shuffle=False, device='cuda', preload=self.preload_batch)

        a = None
        for i, data in enumerate(iterator):
            input, true_path, true_weights = (
                data["images"],#.contiguous(),
                data["labels"],#.contiguous(),
                data["true_weights"]#.contiguous(),
            )

            #if self.use_cuda:
            #    input = input.cuda()
            #    true_path = true_path.cuda()
            #    true_weights = true_weights.cuda()


            start = time.time()
            accuracy, last_suggestion, valid_paths = self.forward_pass(input, true_path, train=False, i=i)
            print("valid_paths:", valid_paths.sum().item())
            suggested_path = last_suggestion["suggested_path"]

            if a is None:
                a = suggested_path.cpu().detach().numpy()
            else:
                a = np.concatenate((a, suggested_path.cpu().detach().numpy()), axis=0)

            evaluated_metrics = metrics.compute_metrics(true_paths=true_path,
            suggested_paths=suggested_path, true_vertex_costs=true_weights, e2i=e2i, valid_paths=valid_paths)
            avg_metrics["accuracy"].update(accuracy.item(), input.size(0))
            avg_metrics["valid paths"].update((valid_paths.sum()/len(valid_paths)).item(), input.size(0))
            for key, value in evaluated_metrics.items():
                avg_metrics[key].update(value, input.size(0))

            if self.fast_mode:
                break

        for key, avg_metric in avg_metrics.items():
            self.val_logger.log(avg_metric.avg, key=key)
        avg_metrics_values = dict([(key, avg_metric.avg) for key, avg_metric in avg_metrics.items()])

        if avg_metrics_values['below_0.0001_percent_acc'] > self.best_eval:

            self.best_eval = avg_metrics_values['below_0.0001_percent_acc']
            print("Saving new best model")
            torch.save({
                        'epoch': self.epochs,
                        'nn_state_dict': self.model.state_dict(),
                        'gf_state_dict': gate.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        }, f'checkpoints/sl_{self.sl_weight}.pt')

        return avg_metrics_values

class SLTrainer(BaselineTrainer):
    def build_model(self, model_name, arch_params):
        self.model = get_model(
            model_name, out_features=264, in_channels=self.metadata["num_channels"], arch_params=arch_params
        )

    def forward_pass(self, input, label, train, i):

        # Get embedding
        output = self.model(input)
        flat_target = label.view(label.size()[0], -1)

        if train:
            logprobs = F.logsigmoid(output).clamp(max=-1e-7)

            # Cross-Entropy
            criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
            loss = criterion(output, flat_target).mean()

            assert(self.sl_weight != 0)
            semantic_loss = -cmpe.get_tf_ac([[log1mexp(-p), p] for p in logprobs.unbind(axis=-1)], log_space=True) 
            loss += (self.sl_weight * semantic_loss.mean())

            print(f"Loss at iter {i}: {loss}")
            return loss

        else:
            bsz = label.size()[0]
            probs = torch.sigmoid(output)

            # Get point-wise accuracy
            accuracy = (probs.round() * flat_target).sum() / flat_target.sum()

            # Get suggested_path
            suggested_path = probs.view(label.shape).round()
            valid_paths = cmpe.get_tf_ac([[1-p, p] for p in suggested_path.unbind(axis=-1)]).bool()
            last_suggestion = {"vertex_costs": None, "suggested_path": suggested_path}

            return accuracy, last_suggestion, valid_paths

class FFCircuitTrainer(ShortestPathAbstractTrainer):
    def build_model(self, model_name, arch_params):
        self.model = get_model(
            model_name, out_features=(264*self.num_reps + self.num_reps), in_channels=self.metadata["num_channels"], arch_params=arch_params
        )

    def forward_pass(self, input, label, train, i):

        # Get embedding
        output = self.model(input).view(input.size(0), 264+1, self.num_reps)
        output, cmpe.beta.mixing = output.split((264, 1), dim=1)
        logprobs = F.logsigmoid(output).clamp(max=-1e-7)
        cmpe.beta.mixing = cmpe.beta.mixing.squeeze(1).log_softmax(dim=1) 

        # Paremeterize the circuit
        litweights = [[log1mexp(-p), p] for p in logprobs.unbind(axis=1)]
        #cmpe.parameterize_ff(litweights)

        if train:

            # Get the negative log likelihood
            #loss = nll(label).mean() #TODO: Is the reduction correct?
            loss = cmpe.ff_cross_entropy(label, litweights).mean()
            print(f"Loss at iter {i}: {loss.item()}")
            return loss

        else:

            bsz = label.size()[0]
            flat_target = label.view(label.size()[0], -1)

            # parameterize circuit
            cmpe.parameterize_ff(litweights)

            # Get grid mpe
            suggested_path = mpe(bsz)
            
            # Get point-wise accuracy
            accuracy = (suggested_path.view(*flat_target.shape) * flat_target).sum().item() / flat_target.sum().item()
            
            last_suggestion = {"vertex_costs": None, "suggested_path": suggested_path}

            return accuracy, last_suggestion

class CircuitTrainer(ShortestPathAbstractTrainer):
    def build_model(self, model_name, arch_params):
        self.model = get_model(
            model_name, out_features=self.num_units, in_channels=self.metadata["num_channels"], arch_params=arch_params
        )

    def forward_pass(self, input, label, train, i):

        # Get embedding
        output = self.model(input)

        # Paremeterize each of the tiles
        parameterize(output)
        #cmpe.parameterize_ff([[log1mexp(-p), p] for p in output.unbind(axis=-1)])

        if train:

            # Get the negative log likelihood
            loss = nll(label).mean() #TODO: Is the reduction correct?
            print(f"Loss at iter {i}: {loss.item()}")
            return loss

        else:

            bsz = label.size()[0]
            flat_target = label.view(label.size()[0], -1)

            print("Before mpe")
            # Get grid mpe
            suggested_path = mpe(bsz)
            print("After mpe")
            
            # Get point-wise accuracy
            accuracy = (suggested_path.view(*flat_target.shape) * flat_target).sum().item() / flat_target.sum().item()
            
            last_suggestion = {"vertex_costs": None, "suggested_path": suggested_path}

            return accuracy, last_suggestion

class DijkstraOnFull(ShortestPathAbstractTrainer):
    def __init__(self, *, l1_regconst, lambda_val, **kwargs):
        super().__init__(**kwargs)
        self.l1_regconst = l1_regconst
        self.lambda_val = lambda_val
        self.solver = ShortestPath(lambda_val=lambda_val, neighbourhood_fn=self.neighbourhood_fn)
        self.loss_fn = HammingLoss()

    def build_model(self, model_name, arch_params):
        self.model = get_model(
            model_name, out_features=self.metadata["output_features"], in_channels=self.metadata["num_channels"], arch_params=arch_params
        )


    def forward_pass(self, input, true_shortest_paths, train, i):
        output = self.model(input)

        # make grid weights positive
        output = torch.abs(output)
        weights = output.reshape(-1, output.shape[-1], output.shape[-1])

        if i == 0 and not train:
            print(output[0])
        assert len(weights.shape) == 3, f"{str(weights.shape)}"
        shortest_paths = self.solver(weights)

        loss = self.loss_fn(shortest_paths, true_shortest_paths)

        logger = self.train_logger if train else self.val_logger

        last_suggestion = {
            "suggested_weights": weights,
            "suggested_path": shortest_paths
        }

        accuracy = (torch.abs(shortest_paths - true_shortest_paths) < 0.5).to(torch.float32).mean()
        extra_loss = self.l1_regconst * torch.mean(output)
        loss += extra_loss

        return loss, accuracy, last_suggestion
