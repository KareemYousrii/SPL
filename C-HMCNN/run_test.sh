#!/bin/bash
for seed in 0 1 2 3 4 5 6 7 8 9
do
    CUDA_VISIBLE_DEVICES=2 python -u test.py --dataset cellcycle_FUN --seed $seed --num_reps 4 --S 4 --gates 2 --lr 1e-3 --batch-size 128 > final/cellcycle_FUN.4.2.4.$seed.txt &
    CUDA_VISIBLE_DEVICES=2 python -u test.py --dataset derisi_FUN --seed $seed --num_reps 4 --S 2 --gates 2 --lr 1e-3 --batch-size 128 > final/derisi_FUN.4.2.2.$seed.txt &
    CUDA_VISIBLE_DEVICES=2 python -u test.py --dataset eisen_FUN --seed $seed --num_reps 4 --S 2 --gates 4 --lr 1e-3 --batch-size 128 > final/eisen_FUN.4.4.2.$seed.txt &
    CUDA_VISIBLE_DEVICES=2 python -u test.py --dataset expr_FUN --seed $seed --gates 4 --lr 1e-3 --batch-size 128 > final/expr_FUN.4.$seed.txt &
    CUDA_VISIBLE_DEVICES=2 python -u test.py --dataset gasch1_FUN --seed $seed --num_reps 4 --S 4 --gates 4 --lr 1e-3 --batch-size 128 > final/gasch1_FUN.4.4.4.$seed.txt &
    CUDA_VISIBLE_DEVICES=2 python -u test.py --dataset gasch2_FUN --seed $seed --num_reps 2 --S 4 --gates 4 --lr 1e-3 --batch-size 128 > final/gasch2_FUN.2.4.4.$seed.txt &
    wait
    CUDA_VISIBLE_DEVICES=2 python -u test.py --dataset seq_FUN --seed $seed --num_reps 8 --S 2 --gates 2 --lr 1e-3 --batch-size 128 > final/seq_FUN.8.2.2.$seed.txt &
    CUDA_VISIBLE_DEVICES=2 python -u test.py --dataset spo_FUN --seed $seed --num_reps 4 --S 4 --gates 4 --lr 1e-3 --batch-size 128 > final/spo_FUN.4.4.4.$seed.txt &
    CUDA_VISIBLE_DEVICES=2 python -u test.py --dataset diatoms_others --seed $seed --gates 2 --lr 1e-3 --batch-size 128 > final/diatoms_others.2.$seed.txt &
    CUDA_VISIBLE_DEVICES=2 python -u test.py --dataset enron_others --seed $seed --num_reps 4 --S 2 --gates 4 --lr 1e-3 --batch-size 128 > final/enron_others.4.4.2.$seed.txt &
    CUDA_VISIBLE_DEVICES=2 python -u test.py --dataset imclef07a_others --seed $seed --num_reps 4 --S 4 --gates 2 --lr 1e-3 --batch-size 128 > final/imclef07a_others.4.2.4.$seed.txt &
    CUDA_VISIBLE_DEVICES=2 python -u test.py --dataset imclef07d_others --seed $seed --num_reps 4 --S 4 --gates 2 --lr 1e-3 --batch-size 128 > final/imclef07d_others.4.2.4.$seed.txt &
    wait
done
