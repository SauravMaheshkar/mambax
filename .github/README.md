## Setup

```bash
uv sync --all-groups
```

## Getting started

```shell
python main.py --workdir=artifacts/
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
