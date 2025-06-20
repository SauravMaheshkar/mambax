from model import Mamba


def test_mamba_from_pretrained():
    model = Mamba.from_pretrained(repo_id="state-spaces/mamba-130m")
    assert model is not None
