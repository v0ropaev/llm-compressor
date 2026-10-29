import pytest
import requests

from llmcompressor.transformers.utils import helpers
from llmcompressor.transformers.utils.helpers import infer_recipe_from_model_path

MODEL_ID = "llm-compressor-tests/model-not-in-cache"


@pytest.mark.unit
@pytest.mark.parametrize(
    "error",
    [
        requests.ConnectionError("connection refused"),
        requests.Timeout("read timed out"),
    ],
    ids=["connection_error", "timeout"],
)
def test_infer_recipe_hub_unreachable(monkeypatch, error):
    # the Hub cannot be reached, but HF_HUB_OFFLINE is not set
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    calls = []

    def fake_head(url, **kwargs):
        calls.append(kwargs)
        raise error

    def fake_download(*args, **kwargs):
        raise AssertionError("hf_hub_download should not be called")

    monkeypatch.setattr(helpers.requests, "head", fake_head)
    monkeypatch.setattr(helpers, "hf_hub_download", fake_download)

    assert infer_recipe_from_model_path(MODEL_ID) is None
    assert len(calls) == 1
    assert calls[0].get("timeout") is not None
