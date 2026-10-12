## Setup

```bash
uv sync --all-groups
```

## Getting started

Trains a character-level Mamba on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories),
logging train loss and validation loss. Training streams through the data in order,
each token once, and every evaluation covers the same `val_tokens` prefix of the val split.

```shell
python main.py --workdir=artifacts/
```

Any field in `configs/default.py` can be overridden, e.g. a quick run on 10% of the data

```shell
python main.py --workdir=artifacts/ \
    --config.train_split="train[:10%]" \
    --config.sequence_length=256
```

Train on GPT-2-tokenized FineWeb, the [modded-nanogpt speedrun](https://github.com/KellerJordan/modded-nanogpt)
data. Each train shard is 100M tokens (200MB), downloaded once into the Hugging Face cache.
For speedrun-comparable val losses evaluate on its fixed 10,485,760 val tokens

```shell
python main.py --workdir=artifacts/ \
    --config.dataset=fineweb \
    --config.fineweb_train_shards=8 \
    --config.val_tokens=10485760
```

Log training logs to wandb

```shell
python main.py --workdir=artifacts/ \
    --config.use_wandb \
    --config.wandb_project=nanollm \
    --config.wandb_entity=sauravmaheshkar
```

## References

* https://github.com/vvvm23/mamba-jax
* https://github.com/johnma2006/mamba-minimal
* https://github.com/radarFudan/mamba-minimal-jax
* https://github.com/jax-ml/jax/discussions/18907
* https://github.com/srush/annotated-mamba/issues/1
* https://github.com/jenriver/bonsai/blob/main/bonsai/models/gemma3/model.py
