"""GDPO reward channels (gdpo/rewards.py).

tests/fixtures/reward_cases.json holds 300 (completion, GT) cases with the outputs of the
reward implementation used for the paper; the consolidated module must reproduce them exactly.
"""
import json
import os

import pytest

import rewards

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "reward_cases.json")
CASES = json.load(open(FIXTURE, encoding="utf-8"))
CHANNELS = {name: (fn, needs_gt) for name, fn, needs_gt in rewards.REWARD_CHANNELS}


def _score(name, completion, gt):
    fn, needs_gt = CHANNELS[name]
    return fn(completion, gt) if needs_gt else fn(completion)


def test_channel_order():
    assert [c[0] for c in rewards.REWARD_CHANNELS] == ["format", "count", "global", "local", "precision"]


@pytest.mark.parametrize("gt_type", [tuple, list])
def test_matches_paper_implementation(gt_type):
    for case in CASES:
        gt = [gt_type(g) for g in case["gt"]]
        for name, expected in case["expected"].items():
            assert _score(name, case["completion"], gt) == expected, (name, case)


def test_perfect_prediction():
    gt = [(5.0, 10.0), (20.0, 26.0)]
    completion = ("From <t0><t0><t5><tdot><t0> to <t0><t1><t0><tdot><t0>. "
                  "From <t0><t2><t0><tdot><t0> to <t0><t2><t6><tdot><t0>.")
    scores = {name: _score(name, completion, gt) for name in CHANNELS}
    assert scores["format"] == 1.0
    assert scores["count"] == 1.0
    assert scores["local"] == pytest.approx(1.0)
    assert scores["global"] == pytest.approx(1.0, abs=1e-5)
    assert scores["precision"] == pytest.approx(1.0)


def test_single_prediction_on_multi_segment_query():
    gt = [(5.0, 10.0), (20.0, 26.0)]
    completion = "From <t0><t0><t5><tdot><t0> to <t0><t2><t6><tdot><t0>."
    assert _score("count", completion, gt) == 0.0
    assert _score("precision", completion, gt) < 1.0


def test_unparseable_output():
    gt = [(5.0, 10.0)]
    for name in CHANNELS:
        assert _score(name, "the dog barks at 5 seconds", gt) == 0.0
