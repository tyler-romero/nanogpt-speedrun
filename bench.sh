#!/bin/bash

NUM_GPUS=2

uv run torchrun \
    --standalone \
    --nproc_per_node=${NUM_GPUS} \
    src/bench.py
