import sys
import os
import argparse
import random
import torch
import numpy as np
import networkx as nx

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


sys.path.append(os.path.join(sys.path[0], "."))
os.environ["DATA_FOLDER"] = "."
from cutils.parser import *
from cutils import datasets


input_dims = {
    "diatoms": 371,
    "enron": 1001,
    "imclef07a": 80,
    "imclef07d": 80,
    "cellcycle": 77,
    "derisi": 63,
    "eisen": 79,
    "expr": 561,
    "gasch1": 173,
    "gasch2": 52,
    "seq": 529,
    "spo": 86,
}

output_dims_FUN = {
    "cellcycle": 499,
    "derisi": 499,
    "eisen": 461,
    "expr": 499,
    "gasch1": 499,
    "gasch2": 499,
    "seq": 499,
    "spo": 499,
}

output_dims_GO = {
    "cellcycle": 4122,
    "derisi": 4116,
    "eisen": 3570,
    "expr": 4128,
    "gasch1": 4122,
    "gasch2": 4128,
    "seq": 4130,
    "spo": 4116,
}

output_dims_others = {
    "diatoms": 398,
    "enron": 56,
    "imclef07a": 96,
    "imclef07d": 46,
    "reuters": 102,
}

output_dims = {
    "FUN": output_dims_FUN,
    "GO": output_dims_GO,
    "others": output_dims_others,
}

hidden_dims_FUN = {
    "cellcycle": 500,
    "derisi": 500,
    "eisen": 500,
    "expr": 1250,
    "gasch1": 1000,
    "gasch2": 500,
    "seq": 2000,
    "spo": 250,
}

hidden_dims_GO = {
    "cellcycle": 1000,
    "derisi": 500,
    "eisen": 500,
    "expr": 4000,
    "gasch1": 500,
    "gasch2": 500,
    "seq": 9000,
    "spo": 500,
}

hidden_dims_others = {
    "diatoms": 2000,
    "enron": 1000,
    "imclef07a": 1000,
    "imclef07d": 1000,
}

hidden_dims = {
    "FUN": hidden_dims_FUN,
    "GO": hidden_dims_GO,
    "others": hidden_dims_others,
}


def seed_all_rngs(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_data_and_loaders(dataset_name, batch_size, device):

    if "others" in dataset_name:
        train, test = initialize_other_dataset(dataset_name, datasets)
        val = None
    else:
        train, val, test = initialize_dataset(dataset_name, datasets)

    # XXX einet dies unless we use validation here in, e.g., eisen
    preproc_X = (
        train.X if val is None else np.concatenate((train.X, val.X))
    ).astype(float)
    scaler = StandardScaler().fit(preproc_X)
    imputer = SimpleImputer(
        missing_values=np.nan,
        strategy='mean'
    ).fit(preproc_X)

    def process(dataset, shuffle=False):
        if dataset is None:
            return None
        assert np.all(np.isfinite(dataset.X))
        assert np.all(np.unique(dataset.Y.ravel()) == np.array([0, 1]))
        dataset.to_eval = torch.tensor(dataset.to_eval, dtype=torch.bool)
        dataset.X = torch.tensor(
            scaler.transform(imputer.transform(dataset.X))
        ).to(device)
        dataset.Y = torch.tensor(dataset.Y).to(device)
        loader = torch.utils.data.DataLoader(
            dataset=[(x, y) for (x, y) in zip(dataset.X, dataset.Y)],
            batch_size=batch_size,
            shuffle=shuffle
        )
        return loader

    train_loader = process(train, shuffle=True)
    valid_loader = process(val, shuffle=False)
    test_loader = process(test, shuffle=False)

    return train, train_loader, val, valid_loader, test, test_loader


def compute_ancestor_matrix(A, device, transpose=True, no_constraints=False):
    """Compute matrix of ancestors R.

    Given n classes, R is an (n x n) matrix where R_ij = 1 if class i is
    ancestor of class j.
    """
    if no_constraints:
        return None

    R = np.zeros(A.shape)
    np.fill_diagonal(R, 1)
    g = nx.DiGraph(A)
    for i in range(len(A)):
        descendants = list(nx.descendants(g, i))
        if descendants:
            R[i, descendants] = 1
    R = torch.tensor(R)
    if transpose:
        R = R.transpose(1, 0)
    R = R.unsqueeze(0).to(device)

    return R


def get_constr_out(x, R):
    """ Given the output of the neural network x returns the output of MCM given the hierarchy constraint expressed in the matrix R """

    if R is None:
        return x

    c_out = x.double()
    c_out = c_out.unsqueeze(1)
    c_out = c_out.expand(len(x), R.shape[1], R.shape[1])
    R_batch = R.expand(len(x), R.shape[1], R.shape[1])
    final_out, _ = torch.max(R_batch * c_out.double(), dim=2)
    return final_out


def parse_args():

    fmt_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=fmt_class)

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        required=True,
        help='dataset name, must end with: "_GO", "_FUN", or "_others"',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="GPU"
    )
    parser.add_argument(
        "--emb-size",
        type=int,
        default=128,
        help="Embedding layer size"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=1e-5,
        help="Weight decay"
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=200,
        help="Num epochs"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="exp",
        help="Output path to exp result"
    )
    parser.add_argument(
        "--exp-id",
        type=str,
        default=None,
        help="Dataset output suffix"
    )
    parser.add_argument(
        "--no-constraints",
        action="store_true"
    )
    parser.add_argument(
        "--gates", 
        type=int, 
        default=1,
        help='Number of hidden layers in gating function (default: 1)'
    )
    parser.add_argument(
        "--S", 
        type=int, 
        default=0,
        help='PSDD scaling factor (default: 0)'
    )
    parser.add_argument(
        "--num_reps", 
        type=int, 
        default=1,
        help='Number of PSDDs in the ensemble'
    )

    args = parser.parse_args()

    assert "_" in args.dataset
    assert (
        "FUN" in args.dataset
        or "GO" in args.dataset
        or "others" in args.dataset
    )

    return args
