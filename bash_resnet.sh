#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python train_model.py;
CUDA_VISIBLE_DEVICES=0 python resnet_pm.py --epm_flag True --nf 15 --reg_w 2 --base 3.0;
python pruning_resnet.py ;
CUDA_VISIBLE_DEVICES=0 python train_model.py --train_base False;