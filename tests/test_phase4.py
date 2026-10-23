import json

import numpy as np
import pandas as pd
import pytest

from src.data import consolidate, label_audit, splits
from src.data.dataset_loader import load_consolidated, load_manifest
from src.data.feature_extraction import aggregate_features
from src.data.normalization import AGG_TRANSFORMS, Normalizer
from src.data.splits import SPLITS, build_groups, check_leakage, make_split
from src.utils.config import META_CHANNELS

CH = {n: i for i, n in enumerate(META_CHANNELS)}


@pytest.fixture(scope="module")
def man(andmal_like_manifest):
    return load_manifest(andmal_like_manifest["manifest"])


# ---------------------------------------------------------------------------------- grouping
def test_groups_merge_captures_of_same_app_and_family(man):
    g_app = build_groups(man, link_families=False)
    # each benign package has two captures -> one group; every malware capture is its own group
    ben = man[man["label"] == "benign"]
    assert g_app[ben.index].nunique() == 20
    assert man.loc[ben.index].groupby(g_app[ben.index])["capture_id"].nunique().eq(2).all()
    mal = man[man["label"] == "malicious_associated"]
    assert g_app[mal.index].nunique() == 32
    g_fam = build_groups(man, link_families=True)
    assert g_fam[mal.index].nunique() == 8                      # 8 families x 4 captures merged by family
    assert g_fam[ben.index].nunique() == 20


# ---------------------------------------------------------------------------------- splits
@pytest.mark.parametrize("strategy", ["group_random", "family_holdout"])
def test_split_has_no_leakage_and_covers_everything(man, strategy):
    sp = make_split(man, strategy, seed=1, val_frac=0.2, test_frac=0.2)
    assert len(sp) == len(man) and sp["example_id"].tolist() == man["example_id"].tolist()
    assert check_leakage(man, sp, strategy) == []
    d = man.merge(sp, on="example_id")
    for s in SPLITS:
        x = d[d["split"] == s]
        assert x["label"].nunique() == 2, (s, x["label"].unique())     # both classes everywhere
    # fractions (examples) roughly as requested, per class
    for lab in ("benign", "malicious_associated"):
        e = d[d["label"] == lab]
        assert 0.05 < (e["split"] == "test").mean() < 0.4


def test_family_holdout_keeps_families_whole_and_honours_forced_families(man):
    sp = make_split(man, "family_holdout", seed=0, val_frac=0.2, test_frac=0.2,
                    test_families=["Fam00", "Fam01"], val_families=["Fam02"])
    d = man.merge(sp, on="example_id")
    fam_split = d[d["malware_family"] != ""].groupby("malware_family")["split"].agg(lambda s: set(s))
    assert all(len(v) == 1 for v in fam_split)                           # families never split
    assert fam_split["Fam00"] == {"test"} and fam_split["Fam01"] == {"test"} and fam_split["Fam02"] == {"val"}
    assert check_leakage(man, sp, "family_holdout") == []


def test_group_random_can_share_families_across_splits_but_not_captures(man):
    sp = make_split(man, "group_random", seed=0, val_frac=0.2, test_frac=0.2)
    d = man.merge(sp, on="example_id")
    assert check_leakage(man, sp, "group_random") == []
    fam_sets = d[d["malware_family"] != ""].groupby("malware_family")["split"].nunique()
    assert (fam_sets > 1).any()   # expected: random capture split mixes families -> optimistic evaluation


def test_split_is_deterministic_and_seed_dependent(man):
    a = make_split(man, "group_random", 7, 0.15, 0.15)
    b = make_split(man, "group_random", 7, 0.15, 0.15)
    c = make_split(man, "group_random", 8, 0.15, 0.15)
    assert a.equals(b) and not a["split"].equals(c["split"])


def test_filters_unknown_local_infra_and_short_windows(man):
    m = man.copy()
    m.loc[m.index[:10], ["label", "binary_label"]] = ["unknown", -1]
    sp = make_split(m, "group_random", 0, 0.15, 0.15, exclude_local_infra=True, min_window_packets=4)
    d = m.merge(sp, on="example_id")
    assert (d.loc[d["split"] == "excluded", "exclude_reason"] != "").all()
    reasons = set(d["exclude_reason"])
    assert {"unknown_label", "local_infra"} <= reasons
    kept = d[d["split"] != "excluded"]
    assert (kept["n_packets"] >= 4).all() and (kept["binary_label"] >= 0).all()
    assert check_leakage(m, sp, "group_random") == []


def test_leakage_checker_detects_violations(man):
    sp = make_split(man, "group_random", 0, 0.15, 0.15)
    bad = sp.copy()
    d = man.merge(bad, on="example_id")
    cap = d.loc[d["split"] == "train", "capture_id"].iloc[0]
    rows = d.index[(d["capture_id"] == cap)]
    bad.loc[rows[:1], "split"] = "test"           # move ONE window of a train capture into test
    errs = check_leakage(man, bad, "group_random")
    assert any(e.startswith("capture_id") for e in errs) and any(e.startswith("flow_id") or e.startswith("group_id") for e in errs)


def test_input_validation(man):
    with pytest.raises(ValueError):
        make_split(man, "group_random", 0, 0.6, 0.6)
    with pytest.raises(ValueError):
        make_split(man, "group_random", 0, 0.15, 0.15, test_families=["Fam00"])
    with pytest.raises(ValueError):
        make_split(man, "family_holdout", 0, 0.15, 0.15, test_families=["Nope"])
    with pytest.raises(ValueError):
        make_split(man, "family_holdout", 0, 0.15, 0.15, test_families=["Fam00"], val_families=["Fam00"])


def test_splits_cli_writes_reports_and_returns_zero(andmal_like_manifest, tmp_path):
    rc = splits.main(["--manifest", str(andmal_like_manifest["manifest"]), "--name", "t", "--strategy", "family_holdout",
                      "--test-families", "Fam03", "--out-dir", str(tmp_path), "--val-frac", "0.2", "--test-frac", "0.2"])
    assert rc == 0
    meta = json.loads((tmp_path / "t.json").read_text())
    assert meta["leakage_errors"] == [] and "Fam03" in meta["summary"]["per_split"]["test"]["families"]
    assert "all checks passed" in (tmp_path / "t.md").read_text()


# ---------------------------------------------------------------------------------- normalisation
def test_normalizer_formulas_padding_and_train_only(man, andmal_like_manifest, tmp_path):
    from src.data.dataset_loader import load_arrays
    arr = load_arrays(man, andmal_like_manifest["processed"])
    meta, mask = arr["meta"], arr["mask"]
    nrm = Normalizer.fit([(meta, mask)], man["example_id"], arr["spec"])
    ch = nrm.spec["meta"]["channels"]
    real = mask.astype(bool)
    pl = np.log1p(meta[..., CH["payload_len"]][real].astype(np.float64))
    assert np.isclose(ch["payload_len"]["mean"], pl.mean()) and np.isclose(ch["payload_len"]["std"], pl.std())
    iat = meta[..., CH["log_iat"]][real].astype(np.float64)
    assert np.isclose(ch["log_iat"]["mean"], iat.mean())
    y = nrm.transform_meta(meta, mask)
    assert y.dtype == np.float32 and np.abs(y[~real]).max() == 0                       # padding stays exactly 0
    assert np.allclose(y[..., CH["direction"]][real], meta[..., CH["direction"]][real])  # untouched channel
    z = y[..., CH["payload_len"]][real]
    assert abs(z.mean()) < 1e-3 and abs(z.std() - 1) < 1e-3
    # manual formula check on one packet
    i, j = np.argwhere(real)[5]
    expect = (np.log1p(meta[i, j, CH["payload_len"]]) - ch["payload_len"]["mean"]) / ch["payload_len"]["std"]
    assert np.isclose(y[i, j, CH["payload_len"]], expect, atol=1e-5)
    # aggregate features
    agg = nrm.transform_aggregate(aggregate_features(meta, mask))
    assert agg.shape == (len(man), len(AGG_TRANSFORMS)) and np.isfinite(agg).all()
    # save/load round trip is exact
    nrm.save(tmp_path / "n.json")
    n2 = Normalizer.load(tmp_path / "n.json")
    assert np.array_equal(n2.transform_meta(meta[:50], mask[:50]), y[:50])
    assert nrm.spec["fit"]["n_examples"] == len(man)


def test_normalizer_fit_statistics_differ_when_data_differs(man, andmal_like_manifest):
    """Guards against accidentally fitting on everything: stats must follow the data given to fit()."""
    from src.data.dataset_loader import load_arrays
    arr = load_arrays(man, andmal_like_manifest["processed"])
    half = len(man) // 2
    a = Normalizer.fit([(arr["meta"][:half], arr["mask"][:half])], man["example_id"][:half], arr["spec"])
    b = Normalizer.fit([(arr["meta"][half:], arr["mask"][half:])], man["example_id"][half:], arr["spec"])
    assert a.spec["meta"]["channels"]["payload_len"]["mean"] != b.spec["meta"]["channels"]["payload_len"]["mean"]
    assert a.spec["fit"]["example_ids_sha1"] != b.spec["fit"]["example_ids_sha1"]


# ---------------------------------------------------------------------------------- consolidation
@pytest.mark.parametrize("payload", [True, False])
def test_consolidate_end_to_end(andmal_like_manifest, tmp_path, payload):
    sp_dir = tmp_path / "splits"
    assert splits.main(["--manifest", str(andmal_like_manifest["manifest"]), "--name", "s", "--out-dir", str(sp_dir),
                        "--val-frac", "0.2", "--test-frac", "0.2"]) == 0
    out = tmp_path / "ds"
    args = ["--manifest", str(andmal_like_manifest["manifest"]), "--processed-root", str(andmal_like_manifest["processed"]),
            "--splits", str(sp_dir / "s.csv"), "--out", str(out)] + ([] if payload else ["--no-payload"])
    assert consolidate.main(args) == 0
    card = json.loads((out / "dataset_card.json").read_text())
    assert card["include_payload"] is payload
    man = load_manifest(andmal_like_manifest["manifest"])
    tr, va, te = (load_consolidated(out, s) for s in SPLITS)
    assert len(tr["y"]) + len(va["y"]) + len(te["y"]) == len(man)
    assert ("payload" in tr) is payload
    assert set(tr["index"]["example_id"]).isdisjoint(te["index"]["example_id"])
    assert set(tr["index"]["capture_id"]).isdisjoint(te["index"]["capture_id"])
    assert (tr["y"] == tr["index"]["binary_label"].to_numpy()).all()
    # the saved normaliser was fitted on the train split only
    nrm = Normalizer.load(out / "normalization.json")
    assert nrm.spec["fit"]["n_examples"] == len(tr["y"]) and nrm.spec["fit"]["split"] == "train"
    import hashlib
    ids = hashlib.sha1("\n".join(sorted(tr["index"]["example_id"])).encode()).hexdigest()
    assert nrm.spec["fit"]["example_ids_sha1"] == ids
    assert consolidate.verify(out, andmal_like_manifest["processed"], man, n_per_split=10) == []


def test_consolidate_rejects_incomplete_split_file(andmal_like_manifest, tmp_path):
    man = load_manifest(andmal_like_manifest["manifest"])
    sp = make_split(man, "group_random", 0, 0.15, 0.15).iloc[:-5]
    p = tmp_path / "bad.csv"
    sp.to_csv(p, index=False)
    rc = consolidate.main(["--manifest", str(andmal_like_manifest["manifest"]), "--processed-root",
                           str(andmal_like_manifest["processed"]), "--splits", str(p), "--out", str(tmp_path / "o")])
    assert rc == 2
