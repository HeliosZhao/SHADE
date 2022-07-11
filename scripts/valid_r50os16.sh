#!/usr/bin/env bash
echo "Running inference on" ${1}

     python -m torch.distributed.launch --nproc_per_node=1 valid.py \
        --val_dataset cityscapes bdd100k mapillary \
        --arch network.deepv3.DeepR50V3PlusD \
        --date 2207 \
        --bs_mult_val 12 \
        --exp r50os16_val \
        --snapshot ${1}