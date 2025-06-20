from transformers import AutoTokenizer

from model import Mamba


def test_mamba_from_pretrained():
    model = Mamba.from_pretrained(repo_id="state-spaces/mamba-130m")
    assert model is not None


def test_mamba_generate():
    model = Mamba.from_pretrained(repo_id="state-spaces/mamba-130m")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    prompt = "Mamba is the"
    generated = model.generate(tokenizer, prompt)
    assert generated is not None
