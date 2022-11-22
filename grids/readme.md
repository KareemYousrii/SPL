This sub-repo contains code to train a conditional Determinstic and Structured-Decomposable probabilistic circuits on the 4x4 grid experiment from the
semantic loss paper.

To run it, use

    python -W ignore grid_net.py --iters 20000 --data test.data

where `--iters` can be set to any number of iterations

Most of the changes pertaining to the work in this repo can be found in `cmpe.py`, `pypsdd/pypsdd/sdd.py` as well as `GatingFunction.py`

(One?) Recipe for implementing Semprola:
- Create a PSDD who's logical base is true
- Create an SDD encoding the required logical constraint
- Overparameterize the PSDD
- Multiply the PSDD with the SDD [Missing]

TODOs
- Vectorize MPE computation -- other computations are vectorized.
- Overparameterize the PSDD, making it non-deterministic [Done -- needs more testing]
