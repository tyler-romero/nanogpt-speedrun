#!/bin/bash

NUM_GPUS=1

# Single GPU optimization for now
CUDA_VISIBLE_DEVICES="0" uv run torchrun \
    --standalone \
    --nproc_per_node=${NUM_GPUS} \
    src/tuner.py