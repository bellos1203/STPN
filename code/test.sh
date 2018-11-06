#!/usr/bin/env bash

# Usage: CUDA_VISIBLE_DEVICES=[gpu_ID] python3 stpn_main.py --ckpt [ckpt_num] --mode test --test_iter [number_of_iteration_tou_want_to_check]

CUDA_VISIBLE_DEVICES=1 python3 stpn_main.py --ckpt 1 --mode test --test_iter 23900

