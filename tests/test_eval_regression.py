"""Re-score the paper predictions with eval/eval_miou.py and compare with the paper numbers.

The prediction files are not shipped with the repository. Point these variables at
test_results_rank0.json files to run the test (skipped otherwise):
    TITOK_PRED_UNAV100   predictions of TiTok on data/test/unav100.json
    TITOK_PRED_CHARADES  predictions of TiTok on data/test/charades_sta.json
"""
import json
import os
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

# (env var, test JSON, sample mIoU %, CountF1) - Tables 2/3 of the paper
CASES = [
    ("TITOK_PRED_UNAV100", "data/test/unav100.json", 65.6926, 0.578037),
    ("TITOK_PRED_CHARADES", "data/test/charades_sta.json", 56.3413, None),
]


@pytest.mark.parametrize("env_var,test_json,miou,count_f1", CASES)
def test_paper_numbers(env_var, test_json, miou, count_f1, tmp_path):
    pred = os.environ.get(env_var)
    if not pred:
        pytest.skip(f"{env_var} not set")
    subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "eval", "eval_miou.py"), pred,
         "--test_json", os.path.join(REPO_ROOT, test_json), "--out_dir", str(tmp_path), "--quiet"],
        check=True, capture_output=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    summary = json.load(open(tmp_path / "sample_miou_summary.json", encoding="utf-8"))
    assert summary["mIoU_%"] == pytest.approx(miou, abs=1e-4)
    if count_f1 is not None:
        assert summary["count_metrics"]["CountF1"] == pytest.approx(count_f1, abs=1e-6)
