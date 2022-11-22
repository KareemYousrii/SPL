CUDA_VISIBLE_DEVICES=2 python -W ignore grid_net.py --iters 20000 --data test.data --lr 0.001 --num_units 128 --num_layers 2 --num_reps 1 --S 2 &> logs/grid_0.001_128_2_1_2
