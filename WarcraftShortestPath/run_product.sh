#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python -u  main.py settings/warcraft_shortest_path/overparam/512_2_4.json > logs/overparamff_8.log &
CUDA_VISIBLE_DEVICES=1 python -u  main.py settings/warcraft_shortest_path/overparam/overparamff_4.json > logs/overparamff_4.log &
CUDA_VISIBLE_DEVICES=2 python -u  main.py settings/warcraft_shortest_path/overparam/overparamff_2.json > logs/overparamff_2.log &
CUDA_VISIBLE_DEVICES=2 python -u  main.py settings/warcraft_shortest_path/overparam/overparamff_1.json > logs/overparamff_1.log &
