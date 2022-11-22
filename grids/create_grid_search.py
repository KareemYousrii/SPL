import itertools
lrs = [1e-3, 5e-3]
num_units = [128]
num_layers = [2]
num_reps = [1, 2, 4, 8]
S = [1, 2, 4, 8]

with open(f'do_search_{lrs[0]}.sh', 'w') as f:
    for (lr, units, layers, reps, s) in itertools.product(lrs, num_units, num_layers, num_reps, S):
        if s == 1 and reps == 1:
            continue
        f.write(f'CUDA_VISIBLE_DEVICES=2 python -W ignore grid_net.py --iters 40000 --data test.data --lr {lr} --num_units {units} --num_layers {layers} --num_reps {reps} --S {s} &> logs/grid_{lr}_{units}_{layers}_{reps}_{s}\n')
