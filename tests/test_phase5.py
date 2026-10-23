"""Phase 5: models, batch feeder, checkpoints (synthetic data; shapes/behaviour, not accuracy)."""
import numpy as np
import pytest
import torch

from src.data.feature_extraction import AGG_FEATURE_NAMES, aggregate_features
from src.data.normalization import Normalizer
from src.models.model_factory import (MODEL_NAMES, build_model, count_parameters, load_checkpoint, load_model_config,
                                       model_summary, save_checkpoint)
from src.training.data import SplitData
from src.utils.config import META_CHANNELS

W, B = 32, 128


def fake_batch(n=6, seed=0, k_real=10):
    g = torch.Generator().manual_seed(seed)
    mask = torch.zeros(n, W)
    mask[:, :k_real] = 1
    meta = torch.randn(n, W, len(META_CHANNELS), generator=g) * mask.unsqueeze(-1)
    pay = torch.randint(1, 257, (n, W, B), generator=g) * mask.long().unsqueeze(-1)
    return {"mask": mask, "meta": meta, "payload": pay, "agg": torch.randn(n, len(AGG_FEATURE_NAMES), generator=g)}


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_forward_shape_and_param_budget(name):
    m = build_model(name, load_model_config("configs/model.yaml", name)).eval()
    out = m(fake_batch())
    assert out.shape == (6,) and torch.isfinite(out).all()
    n = count_parameters(m)["total"]
    assert n < {"mlp": 5_000, "cnn_meta": 30_000, "cnn_payload": 60_000, "hybrid_resnet": 500_000, "packet_bert": 500_000}[name], n   # compact on purpose
    assert "total parameters" in model_summary(m)


@pytest.mark.parametrize("name", ["cnn_meta", "cnn_payload"])
def test_padding_content_cannot_change_logits(name):
    m = build_model(name, load_model_config("configs/model.yaml", name)).eval()
    b = fake_batch(k_real=7)
    ref = m(b)
    g = torch.Generator().manual_seed(5)
    junk = dict(b)
    pad = (b["mask"] == 0)
    junk["meta"] = torch.where(pad.unsqueeze(-1), torch.randn_like(b["meta"]) * 50, b["meta"])
    junk["payload"] = torch.where(pad.unsqueeze(-1), torch.randint(1, 257, b["payload"].shape, generator=g), b["payload"])
    assert torch.allclose(m(junk), ref, atol=1e-5)


@pytest.mark.parametrize("name", ["cnn_meta", "cnn_payload"])
def test_fully_padded_window_does_not_crash(name):
    m = build_model(name, load_model_config("configs/model.yaml", name)).eval()
    b = fake_batch(k_real=0)
    assert torch.isfinite(m(b)).all()


def test_checkpoint_roundtrip(tmp_path):
    cfg = load_model_config("configs/model.yaml", "cnn_meta")
    m = build_model("cnn_meta", cfg).eval()
    save_checkpoint(tmp_path / "c.pt", m, "cnn_meta", cfg, {"window_size": W}, {"x": 1}, {"seed": 0}, 3, {"val_pr_auc": .9})
    m2, ck = load_checkpoint(tmp_path / "c.pt")
    b = fake_batch()
    assert torch.allclose(m(b), m2(b)) and ck["epoch"] == 3 and ck["label_mapping"]["1"] == "malicious_associated"
    assert ck["normalization"] == {"x": 1}


def test_seeded_init_is_reproducible():
    cfg = load_model_config("configs/model.yaml", "cnn_meta")
    torch.manual_seed(0); a = build_model("cnn_meta", cfg)
    torch.manual_seed(0); b = build_model("cnn_meta", cfg)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


# ---------------------------------------------------------------------------------- batch feeder
def test_splitdata_batches_are_normalised_masked_and_prefix_safe(consolidated_dataset):
    norm = Normalizer.load(consolidated_dataset / "normalization.json")
    d = SplitData(str(consolidated_dataset), "val", norm)
    rows = np.array([3, 0, 5])
    b = d.batch(rows, {"meta", "agg", "payload"})
    raw_meta, raw_mask = np.asarray(d.meta[rows]), np.asarray(d.mask_raw[rows])
    assert np.allclose(b["meta"].numpy(), norm.transform_meta(raw_meta, raw_mask), atol=1e-6)      # order preserved
    assert (b["meta"].numpy()[raw_mask == 0] == 0).all()
    assert np.allclose(b["agg"].numpy(), norm.transform_aggregate(aggregate_features(raw_meta, raw_mask)), atol=1e-5)
    pay, pvl = np.asarray(d.payload[rows]), np.asarray(d.pvl[rows])
    tok = b["payload"].numpy()
    real = (np.arange(B)[None, None, :] < pvl[:, :, None]) & (raw_mask[:, :, None] > 0)
    assert (tok[~real] == 0).all() and (tok[real] == pay[real].astype(np.int64) + 1).all()     # byte 0x00 -> token 1, padding -> 0
    k = 4
    bp = d.batch(rows, {"meta", "agg"}, prefix_k=k)
    assert (bp["mask"].numpy()[:, k:] == 0).all() and (bp["meta"].numpy()[:, k:] == 0).all()
    m4 = raw_mask.copy(); m4[:, k:] = 0
    me4 = raw_meta.copy(); me4[:, k:] = 0
    assert np.allclose(bp["agg"].numpy(), norm.transform_aggregate(aggregate_features(me4, m4)), atol=1e-5)


def test_balanced_sampler_only_resamples_inside_the_loop(consolidated_dataset):
    norm = Normalizer.load(consolidated_dataset / "normalization.json")
    d = SplitData(str(consolidated_dataset), "train", norm)
    n_before = len(d)
    ys = np.concatenate([b["y"].numpy() for b in d.iter_train_batches(64, {"meta"}, np.random.default_rng(0), "balanced")])
    assert len(d) == n_before and 0.35 < ys.mean() < 0.65          # roughly balanced draws; stored split untouched
    with pytest.raises(ValueError):
        list(d.iter_train_batches(64, {"meta"}, np.random.default_rng(0), "bogus"))
