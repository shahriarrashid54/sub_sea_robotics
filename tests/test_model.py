import torch

from seabed_sentinel.models import DetectionLoss, SonarCenterNet, decode_detections
from seabed_sentinel.models.ssl import MaskedAutoencoder, random_patch_mask


def test_forward_shapes():
    m = SonarCenterNet(num_classes=6, width=0.5)
    x = torch.rand(2, 1, 128, 160)
    out = m(x)
    assert out["hm"].shape == (2, 6, 32, 40)
    assert out["wh"].shape == (2, 2, 32, 40)
    assert out["seg"].shape == (2, 3, 32, 40)


def test_loss_decreases_when_overfitting_one_batch():
    torch.manual_seed(0)
    m = SonarCenterNet(num_classes=6, width=0.5)
    crit = DetectionLoss()
    x = torch.rand(2, 1, 64, 64)
    hm = torch.zeros(2, 6, 16, 16); hm[:, 2, 8, 8] = 1
    batch = {"hm": hm, "wh": torch.zeros(2, 2, 16, 16), "reg": torch.zeros(2, 2, 16, 16),
             "reg_mask": torch.zeros(2, 1, 16, 16), "seg": torch.zeros(2, 16, 16, dtype=torch.long)}
    batch["wh"][:, :, 8, 8] = 4; batch["reg_mask"][:, 0, 8, 8] = 1
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    first = None
    for _ in range(25):
        l = crit(m(x), batch)["total"]
        opt.zero_grad(); l.backward(); opt.step()
        first = first or float(l)
    assert float(l) < 0.5 * first


def test_decode_respects_threshold_and_topk():
    out = {"hm": torch.full((1, 6, 16, 16), -8.0), "wh": torch.ones(1, 2, 16, 16), "reg": torch.zeros(1, 2, 16, 16)}
    out["hm"][0, 3, 5, 7] = 5.0
    out["hm"][0, 0, 2, 2] = 5.0
    det = decode_detections(out, stride=4, topk=1, score_thresh=0.5)[0]
    assert len(det["boxes"]) == 1
    det = decode_detections(out, stride=4, topk=10, score_thresh=0.5)[0]
    assert len(det["boxes"]) == 2


def test_mc_dropout_returns_uncertainty_map():
    m = SonarCenterNet(num_classes=6, width=0.5, dropout=0.3).eval()
    x = torch.rand(1, 1, 64, 64)
    out, unc = m.forward_mc(x, n=4)
    assert out["hm_prob"].shape == (1, 6, 16, 16)
    assert unc.shape == (1, 16, 16)
    assert torch.isfinite(unc).all()
    assert not m.training


def test_mae_reconstruction_shape():
    mae = MaskedAutoencoder(width=0.5, fpn_ch=32)
    x = torch.rand(2, 1, 64, 64)
    mask = random_patch_mask(x.shape, 16, 0.6, x.device)
    assert mask.shape == x.shape and set(mask.unique().tolist()) <= {0.0, 1.0}
    assert mae(x * (1 - mask)).shape == x.shape
