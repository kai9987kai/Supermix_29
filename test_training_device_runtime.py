import os
import sys

import torch


sys.path.append(os.path.join(os.getcwd(), "source"))

from qwen_supermix_pipeline import _resolve_device, _resolve_torch_dtype  # noqa: E402


class _DummyDmlDevice:
    type = "privateuseone"

    def __str__(self) -> str:
        return "privateuseone:0"


def test_resolve_device_auto_falls_back_to_cpu_when_no_accelerator():
    device, info = _resolve_device("auto", device_preference="cuda,npu,xpu,dml,mps,cpu")

    assert str(device) == "cpu"
    assert info["resolved"] == "cpu"
    assert info["requested"] == "auto"


def test_resolve_torch_dtype_auto_uses_float32_on_cpu():
    assert _resolve_torch_dtype("auto", torch.device("cpu"), resolved_backend="cpu") == torch.float32


def test_resolve_torch_dtype_auto_uses_float16_on_dml():
    assert _resolve_torch_dtype("auto", _DummyDmlDevice(), resolved_backend="dml") == torch.float16
