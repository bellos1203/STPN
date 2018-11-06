#!/usr/bin/env bash

# Usage: CUDA_VISIBLE_DEVICES=[gpu_ID] python3 stpn_main.py --ckpt [ckpt_num] --mode train --training_num [max_iteration_for_training]

CUDA_VISIBLE_DEVICES=1 nohup python3 stpn_main.py --ckpt 1 --mode train --training_num 24000 &

