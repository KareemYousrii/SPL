CUDA_VISIBLE_DEVICES=2 python -W ignore sushi_net.py --iters 20000 --data test.data --lr 0.001 --num_units 128 --num_layers 2 --num_reps 2 --S 2 &> logs/sushi_0.001_128_2_2_2
