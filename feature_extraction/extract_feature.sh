#!/usr/bin/env bash

# Usage : CUDA_VISIBLE_DEVICES=[gpu_ID] python3 extract_feature.py --data [train/test] --stream [rgb/flow]

# CUDA_VISIBLE_DEVICES=1 nohup python3 extract_feature.py --data train --stream rgb &
# CUDA_VISIBLE_DEVICES=2 nohup python3 extract_feature.py --data train --stream flow &
# CUDA_VISIBLE_DEVICES=1 nohup python3 extract_feature.py --data test --stream rgb &
CUDA_VISIBLE_DEVICES=2 nohup python3 extract_feature.py --data test --stream flow &

