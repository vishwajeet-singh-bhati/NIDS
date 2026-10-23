import pytest

from src.data.labeling import LabelMapper

BASE = {
    "categories": ["benign", "malicious_associated", "confirmed_attack", "unknown"],
    "binary_target": {"benign": 0, "malicious_associated": 1, "confirmed_attack": 1, "unknown": -1},
    "default_rule": {"mapped_label": "unknown", "label_source": "no_rule_matched", "reliability": "none"},
}


def cfg(rules):
    return {**BASE, "rules": rules}


RULES = [
    {"name": "ben", "enabled": True, "path_regex": r"(^|/)benign/(?P<app>[^/]+)/", "original_label": "benign",
     "mapped_label": "benign", "label_source": "dir", "reliability": "weak"},
    {"name": "mal", "enabled": True, "path_regex": r"(^|/)(?P<category>adware)/(?P<family>[^/]+)/",
     "original_label": "from_directory", "mapped_label": "malicious_associated", "label_source": "dir",
     "reliability": "weak"},
    {"name": "off", "enabled": False, "path_regex": r".*", "mapped_label": "confirmed_attack"},
]


def test_rules_groups_and_original_label_preserved():
    m = LabelMapper(cfg(RULES))
    r = m.label_capture("benign/appX/c.pcap")
    assert (r.mapped_label, r.binary_label, r.app_id, r.original_label, r.rule_name) == ("benign", 0, "appX", "benign", "ben")
    r = m.label_capture("adware/dowgin/c.pcap")
    assert r.mapped_label == "malicious_associated" and r.binary_label == 1
    assert r.malware_family == "dowgin" and r.category == "adware" and r.original_label == "dowgin"


def test_unmatched_is_unknown_and_excluded_never_guessed():
    r = LabelMapper(cfg(RULES)).label_capture("mystery/x/c.pcap")
    assert r.mapped_label == "unknown" and r.binary_label == -1 and r.rule_name == "default"


def test_disabled_rules_are_ignored_and_no_rules_warns(caplog):
    m = LabelMapper(cfg([RULES[2]]))
    assert m.label_capture("anything/x.pcap").mapped_label == "unknown"
    assert any("EVERY example will be" in rec.message for rec in caplog.records)


def test_nothing_emits_confirmed_attack_unless_configured():
    m = LabelMapper(cfg(RULES))
    assert all(m.label_capture(p).mapped_label != "confirmed_attack"
               for p in ("benign/a/x.pcap", "adware/f/x.pcap", "zzz/x.pcap"))


def test_override_csv_wins_and_validates(tmp_path):
    csvp = tmp_path / "o.csv"
    csvp.write_text("capture_path,mapped_label,reliability,malware_family\nadware/dowgin/c.pcap,unknown,manual_review,fam1\n")
    m = LabelMapper({**cfg(RULES), "override_csv": str(csvp)})
    r = m.label_capture("adware/dowgin/c.pcap")
    assert r.mapped_label == "unknown" and r.rule_name == "override_csv" and r.malware_family == "fam1"
    csvp.write_text("capture_path,mapped_label\nadware/dowgin/c.pcap,evil\n")
    with pytest.raises(ValueError):
        LabelMapper({**cfg(RULES), "override_csv": str(csvp)}).label_capture("adware/dowgin/c.pcap")


def test_bad_config_rejected():
    with pytest.raises(ValueError):
        LabelMapper(cfg([{"name": "x", "enabled": True, "path_regex": ".", "mapped_label": "bogus"}]))
    bad = dict(BASE, binary_target={"benign": 0})
    with pytest.raises(ValueError):
        LabelMapper({**bad, "rules": []})


# ---------------------------------------------------------------- real CIC-AndMal2017 rules
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def andmal_mapper():
    return LabelMapper.from_file(ROOT / "configs" / "labels.yaml")


def test_andmal2017_rules_on_observed_paths():
    paths = [p for p in (ROOT / "tests/fixtures/andmal2017_paths_sample.txt").read_text().split("\n") if p]
    assert len(paths) > 400
    m = andmal_mapper()
    n_ben = n_mal = 0
    for rel in paths:
        r = m.label_capture(rel)
        top = rel.split("/")[0]
        if top == "Benign":
            n_ben += 1
            assert (r.mapped_label, r.binary_label) == ("benign", 0), rel
            assert r.rule_name == "cic_andmal2017_benign_directory", rel   # no path needed the fallback
            assert r.app_id.startswith(("com.", "air.", "net.", "org.", "de.", "me.", "uk.", "tv.")) or r.app_id, rel
            assert r.subset == "2015" and r.capture_date.startswith("2017-"), rel
            assert not r.app_id.startswith("OK-"), rel
        else:
            n_mal += 1
            assert (r.mapped_label, r.binary_label) == ("malicious_associated", 1), rel
            assert r.rule_name == "cic_andmal2017_malware_directory", rel
            assert r.category == top and r.malware_family == rel.split("/")[1] and r.original_label == r.malware_family
            assert len(r.app_id) == 32 and r.capture_date.startswith("2017-"), rel
    assert n_ben > 100 and n_mal > 100


def test_andmal2017_specific_examples():
    m = andmal_mapper()
    r = m.label_capture("Benign/2015/08_04_2017-be-2015-OK-3-com.isr.stl.pcap")
    assert (r.app_id, r.subset, r.capture_date) == ("com.isr.stl", "2015", "2017-08-04")
    r = m.label_capture("Benign/2015/08_04_2017-be-2015-com.google.android.apps.finance.pcap")
    assert r.app_id == "com.google.android.apps.finance"
    r = m.label_capture("Scareware/AndroidSpy.277/06_22_2017-sc-android.spy.277-drweb-2c5f158e2be5b0a67fe7378d6cff0d2d.pcap")
    assert (r.malware_family, r.category, r.capture_date) == ("AndroidSpy.277", "Scareware", "2017-06-22")
    # unseen naming variants fall back instead of becoming 'unknown'; unknown dirs stay unknown
    r = m.label_capture("Benign/2016/some_odd_name.pcap")
    assert r.mapped_label == "benign" and r.app_id == "" and r.rule_name.endswith("fallback")
    r = m.label_capture("Ransomware/Koler/odd.pcap")
    assert r.mapped_label == "malicious_associated" and r.malware_family == "Koler"
    assert m.label_capture("Other/x/y.pcap").mapped_label == "unknown"
