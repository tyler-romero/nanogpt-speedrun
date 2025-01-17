# nanogpt-speedrun
Reproducing GPT-2 (124M) as fast as possible on an RTX 4090.

karparthy:
> The 124M model is the smallest model in the GPT-2 series released by OpenAI in 2019, and is actually quite accessible today, even for the GPU poor...You can train the model with a single GPU too, it would just take proportionally longer (e.g. ~4-24 hours depending on the GPU).

This repo is heavily influced by https://github.com/KellerJordan/modded-nanogpt. The initial baseline here was taken directly from the initial commit of that repo, with minor modifications.

See also: https://github.com/karpathy/llm.c/discussions/481 and https://github.com/tysam-code/hlb-gpt


## setup

```bash
uv sync --all-extras
uv run python src/data/cached_fineweb10B.py
./run.sh
```
