#!/bin/bash
python train.py --pretrained twitter.27B.50d --epochs 50 --lr 0.01 --batch-size 64  --val-batch-size 64 --kernel-height 3,4,5 --out-channel 100 --dropout 0.5 --gpu_ids 6
