#!/bin/bash
python train.py --pretrained glove.6B.300d --epochs 150 --lr 0.01 --batch-size 64  --val-batch-size 64 --kernel-height 3,4,5 --out-channel 100 --dropout 0.5