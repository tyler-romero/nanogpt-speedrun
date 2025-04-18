#!/bin/bash

NUM_GPUS=2
NOTES="${1:-}"

uv run torchrun \
    --standalone \
    --nproc_per_node=${NUM_GPUS} \
    src/train_gpt2.py \
    --notes "${NOTES}"