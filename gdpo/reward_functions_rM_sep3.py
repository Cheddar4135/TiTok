#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reward_functions_rM_sep3.py
  TiTok reward used in the paper (GDPO stage).

  Five channels, all in [0, 1] and computed on the whole completion:
    format    - strict 'From <t\\d>{3}<tdot><t\\d> to ... .' list (count neutral)
    count     - segment-count matching
    global    - MUSEG F1 (combines precision and recall)
    local     - NGIoU best match / max(n)
    precision - temporal precision = overlap(pred n gt) / pred_len

  Why a continuous precision channel:
    A count-based false-positive score with an IoU>=0.3 threshold only catches extra
    segments, not over-long ones: a prediction covering the GT keeps IoU above 0.3
    (e.g. an 8 s GT predicted as 13.4 s gives IoU 0.60 and no penalty).
    It also gives no gradient for single segments because of the threshold.
    precision = fraction of predicted time that overlaps the GT.
      - Over-long predictions are penalized continuously: pred_len up -> precision down.
      - Extra segments are penalized continuously: non-overlapping time only grows the denominator.
      - 1.0 means all predicted time lies inside the GT.
      - Recall (covering the GT) is handled by global/local.

  W&B logs each channel as rewards/<channel> (load_reward_module sets fn.__name__).
  reward_weights follow the order [format, count, global, local, precision].
  reward_functions*.py are imported, not modified.
"""
from typing import List, Tuple

# Unchanged channels and helpers are reused from reward_functions_rM_sep
from reward_functions_rM_sep import (
    format_reward,     # strict, count neutral
    global_reward,     # MUSEG F1 (precision + recall)
    local_reward,      # NGIoU best-match / max(n)
    _parse,            # segment parser
)
from reward_functions import _merge_intervals   # union of overlapping intervals


def count_matching_reward(completion: str, gt_intervals, **kwargs) -> float:
    """Segment-count matching.
      - multi-segment GT (>= 2) and a single prediction -> 0.0 (penalize single collapse)
      - otherwise -> 1 - |n_pred - n_gt| / max(n_pred, n_gt, 1)
    """
    if not gt_intervals:
        return 0.0
    n_g = len(gt_intervals)
    if n_g == 0:
        return 0.0
    n_p = len(_parse(completion))
    # A single prediction for a multi-segment query always scores 0 (anti-collapse)
    if n_g >= 2 and n_p == 1:
        return 0.0
    return max(0.0, 1.0 - abs(n_p - n_g) / max(n_p, n_g, 1))


def precision_reward(completion: str, gt_intervals, **kwargs) -> float:
    """Temporal precision = overlap(pred n gt) / pred_len, in [0, 1].

    Continuously penalizes predicted time that falls outside the GT.
    Overlapping predictions are merged (no double counting); GT is merged too.
      - over-long predictions -> larger pred_len -> lower precision
      - extra segments -> pred_len grows without overlap -> lower precision
      - 1.0 = all predicted time on the GT. Recall is handled by global/local.
    """
    if not gt_intervals:
        return 0.0
    preds = _parse(completion)
    if not preds:
        return 0.0
    pred_u = _merge_intervals(list(preds))
    gt_u = _merge_intervals([tuple(g) for g in gt_intervals])
    pred_len = sum(e - s for s, e in pred_u)
    if pred_len <= 0:
        return 0.0
    overlap = 0.0
    for gs, ge in gt_u:
        for ps, pe in pred_u:
            overlap += max(0.0, min(pe, ge) - max(ps, gs))
    return min(1.0, overlap / pred_len)


# Channel spec read by the trainer: (name, fn, needs_gt). Weight order [format, count, global, local, precision].
iou_reward = global_reward   # legacy 'iou' slot of load_reward_module
REWARD_CHANNELS = [
    ("format",    format_reward,          False),
    ("count",     count_matching_reward,  True),
    ("global",    global_reward,          True),
    ("local",     local_reward,           True),
    ("precision", precision_reward,       True),
]

__all__ = [
    "format_reward", "count_matching_reward", "global_reward", "local_reward",
    "precision_reward", "iou_reward", "REWARD_CHANNELS",
]
