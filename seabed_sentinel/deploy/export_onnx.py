"""ONNX export + numerical parity check against PyTorch."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch


class _Wrapper(torch.nn.Module):
    """Exports sigmoid heat-map probabilities so the runtime needs no post-processing beyond decode."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, x):
        out = self.m(x)
        return torch.sigmoid(out["hm"]), out["wh"], out["reg"]


def export_onnx(model: torch.nn.Module, path: str, tile: int = 256, opset: int = 17) -> str:
    model.eval()
    w = _Wrapper(model).eval()
    x = torch.zeros(1, 1, tile, tile)
    torch.onnx.export(
        w, x, path, opset_version=opset, input_names=["image"], output_names=["hm_prob", "wh", "reg"],
        dynamic_axes={"image": {0: "batch", 2: "rows", 3: "cols"}, "hm_prob": {0: "batch", 2: "r", 3: "c"},
                      "wh": {0: "batch", 2: "r", 3: "c"}, "reg": {0: "batch", 2: "r", 3: "c"}},
        dynamo=False,
    )
    return path


def verify_onnx(model: torch.nn.Module, path: str, tile: int = 256, seed: int = 0) -> Dict[str, float]:
    import onnxruntime as ort
    rng = np.random.default_rng(seed)
    x = rng.random((2, 1, tile, tile), dtype=np.float32)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    hm_o, wh_o, reg_o = sess.run(None, {"image": x})
    with torch.no_grad():
        out = model.eval()(torch.from_numpy(x))
    hm_t = torch.sigmoid(out["hm"]).numpy()
    return {"max_abs_diff_hm": float(np.abs(hm_o - hm_t).max()),
            "max_abs_diff_wh": float(np.abs(wh_o - out["wh"].numpy()).max()),
            "max_abs_diff_reg": float(np.abs(reg_o - out["reg"].numpy()).max()),
            "onnx_size_mb": float(__import__("os").path.getsize(path) / 1e6)}
