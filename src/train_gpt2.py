import os
import sys

with open(sys.argv[0]) as f:
    code = f.read()  # read the code of this file ASAP, for logging

import glob
import subprocess
import time
import uuid
from dataclasses import dataclass

import numpy as np
import torch
import torch._inductor.config as config
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb

###
# NanoGPT speedrun training script
# Heavily inspired by / code borrowed from NanoGPT and https://github.com/KellerJordan/modded-nanogpt
###

# speedrun is to <= this val_loss. A val loss of <3.278 is good evidence that >95% of runs attain below 3.28
SPEEDRUN_TARGET = 3.28

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dim (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.attn_scale = 1 / (2 * config.n_layer) ** 0.5

    def forward(self, x):
        x = x + self.attn_scale * self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x


# -----------------------------------------------------------------------------
# The main GPT-2 model


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying

    def forward(self, idx, targets=None, return_logits=True):
        # forward the GPT model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        for block in self.transformer.h:
            x = block(x)
        x = F.rms_norm(x, (x.size(-1),))

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float()  # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            logits = logits.float()  # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas)
        return optimizer


# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader


def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, 'rb') as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if header[0] != 20240520:
        print('ERROR: magic number mismatch in the data .bin file!')
        print('---> HINT: Are you passing in a correct file with --input_bin?')
        print('---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README')
        print('---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try')
        exit(1)
    assert header[1] == 1, 'unsupported version'
    ntok = header[2]  # number of tokens (claimed)
    return ntok  # for now just return the number of tokens


def _load_data_shard(filename):
    with open(filename, 'rb') as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, 'magic number mismatch in the data .bin file'
        assert header[1] == 1, 'unsupported version'
        ntok = header[2]  # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, 'number of tokens read does not match header?'
    return tokens


class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f'did not find any files that match the pattern {filename_pattern}'

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print0(f'DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files')

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self):  # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()


# -----------------------------------------------------------------------------
# int main


def print0(s, console=False):
    if master_process:
        with open(logfile, 'a') as f:
            if console:
                print(s)
            print(s, file=f)


if __name__ == '__main__':

    @dataclass
    class Hyperparameters:
        input_bin: str = 'src/data/fineweb10B/fineweb_train_*.bin'
        input_val_bin: str = 'src/data/fineweb10B/fineweb_val_*.bin'
        batch_size = 512  # global batch size, in sequences
        device_batch_size: int = 32  # batch size, in sequences, per device
        sequence_length: int = 1024  # sequence length, in tokens
        num_iterations: int = 9664  # number of iterations to run
        learning_rate: float = 0.0018
        warmup_iters: int = 256
        warmdown_iters: int = 2048
        weight_decay: float = 0.1
        val_loss_every: int = 128
        val_tokens: int = (
            10485760  # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
        )
        disable_wandb: bool = False

    args = Hyperparameters()

    # set up DDP (distributed data parallel). torchrun sets this env variable
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), 'for now i think we need CUDA for DDP'
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = 0  # each process gets the exact same seed

    # args error checking and convenience variables
    B, T = args.device_batch_size, args.sequence_length
    assert 1 <= T <= 1024
    # calculate the number of steps to take in the val loop.
    assert args.val_tokens % (B * T * ddp_world_size) == 0
    val_steps = args.val_tokens // (B * T * ddp_world_size)
    # calculate the steps of gradient accumulation required to attain the desired global batch size.
    assert args.batch_size % (B * ddp_world_size) == 0
    grad_accum_steps = args.batch_size // (B * ddp_world_size)
    tokens_per_fwdbwd = B * T * ddp_world_size * grad_accum_steps  # 524288

    # begin logging
    if master_process:
        run_id = uuid.uuid4()
        os.makedirs('logs', exist_ok=True)
        logfile = f'logs/{run_id}.txt'
        print0(logfile, console=True)
        # initialize wandb
        if not args.disable_wandb:
            wandb.init(project='nanogpt-speedrun', name=str(run_id), config=args)

    print0(f'Running pytorch {torch.version.__version__}')
    print(f'using device: {device}')

    print0(
        f'{B=} {T=} {ddp_world_size=} {grad_accum_steps=} {tokens_per_fwdbwd=} {args.batch_size=}',
        console=True,
    )

    # begin by printing this file (the Python code)
    print0('=' * 100)
    print0(code)
    print0('=' * 100)
    # log information about the hardware/software environment this is running on
    print0(f'Running Python {sys.version}')
    print0(f'Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}')
    print0(subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout)
    print0('=' * 100)

    # init the model from scratch
    # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested by @Grad62304977.
    model_config = GPTConfig(vocab_size=50304, n_layer=12, n_head=12, n_embd=768)
    model = GPT(model_config)
    model = model.train().cuda()
    if hasattr(config, 'coordinate_descent_tuning'):
        config.coordinate_descent_tuning = True  # suggested by @Chillee
    print0('compiling the model...')
    model = torch.compile(model)
    ddp_model = DDP(model, device_ids=[ddp_local_rank])
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

    # load tokens
    train_loader = DistributedDataLoader(args.input_bin, B * grad_accum_steps, T, ddp_rank, ddp_world_size)
    val_loader = None
    if args.input_val_bin:
        val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
    x, y = train_loader.next_batch()

    # init the optimizer
    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
    )

    # learning rate decay scheduler (linear warmup and warmdown)
    def get_lr(it):
        assert it <= args.num_iterations
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return args.learning_rate * (it + 1) / args.warmup_iters
        # 2) constant lr for a while
        elif it < args.num_iterations - args.warmdown_iters:
            return args.learning_rate
        # 3) linear warmdown
        else:
            decay_ratio = (args.num_iterations - it) / args.warmdown_iters
            return args.learning_rate * decay_ratio

    tokens_seen = 0
    training_time_ms = 0
    # start the clock
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    # begin training
    train_loader.reset()
    for step in range(args.num_iterations + 1):
        last_step = step == args.num_iterations
        # This effectively ignores timing first 10 steps, which are slower for weird reasons.
        # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
        # steps with dummy data first, and then re-initialize the model and reset the loader.
        if step == 10:
            training_time_ms = 0
            t0 = time.perf_counter()
        timed_steps = float('nan') if step <= 11 else (step - 10) + 1  # <= 11 to avoid bug in val

        # once in a while evaluate the validation dataset
        if (args.val_loss_every > 0 and (step % args.val_loss_every == 0 or last_step)) and (val_loader is not None):
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            # run validation batches
            model.eval()
            val_loader.reset()
            val_loss = 0.0
            for _ in range(val_steps):
                x_val, y_val = val_loader.next_batch()
                with ctx:  # of course, we'd like to use no_grad() here too, but that creates a torch.compile error for some reason
                    _, loss = model(x_val, y_val, return_logits=False)
                    val_loss += loss.detach()
                    del loss
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            val_loss /= val_steps
            # log val loss
            print0(
                f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / (timed_steps - 1):.2f}ms',
                console=True,
            )
            if master_process and not args.disable_wandb:
                wandb.log(
                    {
                        'val_loss': val_loss,
                        'train_time': training_time_ms,
                        'step': step,
                        'tokens_seen': tokens_seen,
                        'step_avg': training_time_ms / (timed_steps - 1),
                    }
                )
            # start the clock again
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            # if we hit the speedrun target, we're done
            if val_loss <= SPEEDRUN_TARGET:
                break

        # bit confusing: we want to make sure to eval on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        if last_step:
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        x, y = train_loader.next_batch()
        with ctx:
            # forward/backward with grad accumulation
            for i, (micro_x, micro_y) in enumerate(
                zip(x.chunk(grad_accum_steps, dim=0), y.chunk(grad_accum_steps, dim=0))
            ):
                _, loss = ddp_model(micro_x, micro_y, return_logits=False)
                train_loss = loss.detach()
                loss.backward()
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # step the optimizer
        optimizer.step()
        # null the gradients
        optimizer.zero_grad(set_to_none=True)

        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        # dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
        if master_process:
            tokens_seen += tokens_per_fwdbwd
            approx_time = training_time_ms + 1000 * (time.perf_counter() - t0)
            tokens_per_second = tokens_seen / (approx_time / 1000) if approx_time > 0 else 0
            print0(
                f'step:{step + 1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time / timed_steps:.2f}ms tokens_seen:{tokens_seen:.2e} tokens/sec:{tokens_per_second:.2e}',
                console=True,
            )
            if not args.disable_wandb:
                wandb.log(
                    {
                        'train_loss': train_loss.item(),
                        'train_time': approx_time,
                        'step': step + 1,
                        'step_avg': approx_time / timed_steps,
                        'tokens_seen': tokens_seen,
                        'lr': lr,
                        'tokens_per_second': tokens_per_second,
                    }
                )

    # -------------------------------------------------------------------------
    print0(f'peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB')
    # clean up nice
    dist.destroy_process_group()
    if master_process and not args.disable_wandb:
        wandb.finish()
