"""Phase 6: metrics, thresholds, training loop, evaluation (synthetic data; plumbing + correctness)."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from src.training import data as tdata
from src.training import evaluate as ev
from src.training.train import train
from src.utils import metrics as M


# ---------------------------------------------------------------------------------- metrics
def test_binary_metrics_hand_computed():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    s = np.array([.9, .8, .2, .7, .1, .1, .0, .3])
    m = M.binary_metrics(y, s, 0.5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 1, 4, 1)
    assert m["confusion_matrix"] == [[4, 1], [1, 2]]
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["fpr"] == pytest.approx(1 / 5) and m["fnr"] == pytest.approx(1 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["per_class"]["benign"]["recall"] == pytest.approx(4 / 5)
    assert m["per_class"]["benign"]["precision"] == pytest.approx(4 / 5)
    assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 4 / 5) / 2)


def test_degenerate_inputs_give_nan_not_crashes():
    assert math.isnan(M.safe_roc_auc(np.zeros(5), np.arange(5)))
    assert math.isnan(M.safe_pr_auc(np.ones(5), np.arange(5)))
    m = M.binary_metrics(np.zeros(4), np.arange(4), 10.0)
    assert math.isnan(m["recall"]) and m["fpr"] == 0.0


@pytest.mark.parametrize("target", [0.0, 0.01, 0.05, 0.2])
def test_threshold_for_fpr_respects_target_even_with_ties(target):
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(500), np.ones(100)].astype(int)
    s = np.r_[np.round(rng.normal(0, 1, 500), 1), np.round(rng.normal(1.5, 1, 100), 1)]    # many ties
    thr = M.threshold_for_fpr(y, s, target)
    assert M.binary_metrics(y, s, thr)["fpr"] <= target + 1e-12
    # and it is not needlessly strict: one step lower would exceed the target (or hit the minimum)
    lower = np.sort(np.unique(s[y == 0]))
    lower = lower[lower < thr]
    if len(lower) and target > 0:
        assert M.binary_metrics(y, s, lower[-1])["fpr"] > target


def test_best_f1_threshold_beats_default():
    rng = np.random.default_rng(1)
    y = (rng.random(400) < 0.3).astype(int)
    s = rng.normal(0, 1, 400) + 2.5 * y - 2
    t = M.best_f1_threshold(y, s)
    assert M.binary_metrics(y, s, t)["f1"] >= M.binary_metrics(y, s, 0.0)["f1"]


def test_calibration_and_temperature_recovery():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2, 20000)
    y = (rng.random(20000) < M.sigmoid(z)).astype(int)
    assert M.calibration(y, M.sigmoid(z))["ece"] < 0.02                 # calibrated -> small ECE
    over = z * 3.0                                                      # over-confident logits
    T = M.fit_temperature(y, over)
    assert 2.5 < T < 3.6
    assert M.calibration(y, M.sigmoid(over / T))["ece"] < M.calibration(y, M.sigmoid(over))["ece"]


def test_capture_level_and_grouped_rates():
    idx = pd.DataFrame({"capture_id": list("aaabbbcc"), "malware_family": ["f", "f", "f", "g", "g", "g", "", ""],
                        "subset": ["", "", "", "", "", "", "2015", "2015"]})
    y = np.array([1, 1, 1, 1, 1, 1, 0, 0])
    s = np.array([5, 5, 5, 5, -5, -5, -5, 5.0])
    cl = M.capture_level(idx, y, s, 0.0, [0.5])
    r = cl["rules"]["alert_rate>=0.5"]
    assert r["capture_tpr"] == pytest.approx(0.5) and r["capture_fpr"] == pytest.approx(1.0)   # c: 1/2 flagged >= 0.5
    g = M.grouped_rates(idx.iloc[:6], y[:6], s[:6], 0.0, "malware_family")
    assert g["f"]["flag_rate"] == pytest.approx(1.0) and g["g"]["flag_rate"] == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------------- training / evaluation
FAST = {"epochs": 2, "batch_size": 64, "max_steps_per_epoch": 0, "patience": 5, "prefix_lengths": [4, 16], "seed": 0}


@pytest.mark.parametrize("model", ["mlp", "cnn_meta", "cnn_payload"])
def test_train_and_evaluate_each_model(consolidated_dataset, tmp_path, model):
    out = train(str(consolidated_dataset), model, tmp_path / model, FAST)
    r = tmp_path / model
    for f in ("best.pt", "config.json", "history.csv", "metrics.json", "model_summary.txt", "predictions_val.csv",
              "predictions_test.csv"):
        assert (r / f).exists(), f
    m = json.loads((r / "metrics.json").read_text())
    assert set(m["splits"]) == {"val", "test"}
    t = m["splits"]["test"]
    assert 0.0 <= t["roc_auc"] <= 1.0 and 0.0 <= t["pr_auc"] <= 1.0
    for op in ("fpr<=0.01", "fpr<=0.05", "max_f1", "logit>=0"):
        assert op in t["at"] and len(t["at"][op]["confusion_matrix"]) == 2
    assert "per_family_recall" in t["breakdown_at_fpr<=0.01"] and "capture_level" in t
    hist = pd.read_csv(r / "history.csv")
    assert len(hist) == 2 and hist["train_loss"].notna().all()
    assert m["run"]["parameters"]["total"] > 0 and out["best_epoch"] in (1, 2)


def test_thresholds_come_from_val_and_test_is_untouched_by_training(consolidated_dataset, tmp_path, monkeypatch):
    seen = []
    orig = tdata.SplitData.__init__

    def spy(self, dataset_dir, split, *a, **k):
        seen.append(split)
        orig(self, dataset_dir, split, *a, **k)

    monkeypatch.setattr(tdata.SplitData, "__init__", spy)
    train(str(consolidated_dataset), "mlp", tmp_path / "r", FAST, evaluate=False)
    assert sorted(set(seen)) == ["train", "val"]               # test split is not even opened during training
    monkeypatch.undo()
    res = ev.evaluate_run(tmp_path / "r")
    pv = pd.read_csv(tmp_path / "r" / "predictions_val.csv")
    thr = M.threshold_for_fpr(pv["y"].to_numpy(), pv["logit"].to_numpy(), 0.01)
    assert res["thresholds_from_val"]["fpr<=0.01"]["logit"] == pytest.approx(thr)
    assert res["splits"]["test"]["at"]["fpr<=0.01"]["threshold"] == pytest.approx(thr)     # same value applied to test


def test_training_is_deterministic_for_a_fixed_seed(consolidated_dataset, tmp_path):
    train(str(consolidated_dataset), "cnn_meta", tmp_path / "a", FAST)
    train(str(consolidated_dataset), "cnn_meta", tmp_path / "b", FAST)
    a = pd.read_csv(tmp_path / "a" / "predictions_test.csv")
    b = pd.read_csv(tmp_path / "b" / "predictions_test.csv")
    assert np.allclose(a["logit"], b["logit"], atol=1e-6)


def test_early_prediction_rows_and_delay(consolidated_dataset, tmp_path):
    train(str(consolidated_dataset), "cnn_meta", tmp_path / "r", {**FAST, "prefix_lengths": [4, 8, 16]})
    rows = json.loads((tmp_path / "r" / "metrics.json").read_text())["early_prediction"]["rows"]
    t = [r for r in rows if r["split"] == "test"]
    assert [r["k_packets"] for r in t] == [4, 8, 16, 32]
    assert all(r["mean_packets_needed"] <= r["k_packets"] for r in t)
    d = [r["median_delay_s"] for r in t]
    assert d == sorted(d)                                       # waiting for more packets never takes less time
    assert t[-1]["frac_windows_already_complete"] == 1.0


def test_external_dataset_uses_checkpoint_normaliser_and_val_thresholds(consolidated_dataset, tmp_path):
    train(str(consolidated_dataset), "mlp", tmp_path / "r", FAST)
    res = ev.evaluate_run(tmp_path / "r", eval_dataset=str(consolidated_dataset), eval_splits=["test"],
                          out_name="metrics_ext.json")
    assert "ext_test" in res["splits"]
    a, b = res["splits"]["test"]["at"]["max_f1"], res["splits"]["ext_test"]["at"]["max_f1"]
    assert a["threshold"] == b["threshold"] and a["confusion_matrix"] == b["confusion_matrix"]


def test_train_refuses_when_a_split_lacks_a_class(consolidated_dataset, tmp_path, monkeypatch):
    import shutil
    bad = tmp_path / "bad"
    shutil.copytree(consolidated_dataset, bad)
    y = np.load(bad / "val_y.npy")
    np.save(bad / "val_y.npy", np.zeros_like(y))
    with pytest.raises(ValueError, match="both classes"):
        train(str(bad), "mlp", tmp_path / "r", FAST)


def test_sampler_and_focal_loss_run(consolidated_dataset, tmp_path):
    train(str(consolidated_dataset), "mlp", tmp_path / "r", {**FAST, "sampler": "balanced", "loss": "focal"})
    assert (tmp_path / "r" / "metrics.json").exists()
