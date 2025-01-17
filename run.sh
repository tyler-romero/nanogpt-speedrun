# Yields 3.2798 val loss in 6.44B tokens.
# For comparison, the llm.c trainer gets 3.2847 in 10B tokens, which is GPT-2 level quality.
# Speed run is to attain <= 3.28 val loss
# The gain in efficiency over this baseline is due to the following changes:
# 1. Increased learning rate
# 2. Halved per-device batch size from 64 to 32 (~but same training speed)
# 3. Improved learning rate schedule (linear up from 0, then linear down to 0.1 * max)
# 4. Removed all affine scale and bias parameters from the architecture, and switched to
#    RMSNorm (actually this just simplifies the code but doesn't speed up training)

NUM_GPUS=2

CUDA_VISIBLE_DEVICES=0,1 uv run torchrun \
    --standalone \
    --nproc_per_node=${NUM_GPUS} \
    src/train_gpt2.py \
        --input_bin "src/data/fineweb10B/fineweb_train_*.bin" \
        --input_val_bin "src/data/fineweb10B/fineweb_val_*.bin" \
        --output_dir pylog124M \
        --model d12 \
        --batch_size 32 \
        --sequence_length 1024 \
        --total_batch_size 262144 \
        --grad_accum_steps 4 \
        --val_loss_every 128 \
        --num_iterations 24576 \
        --weight_decay 0.1 \
        --learning_rate 0.0015 \
        --warmup_iters 256