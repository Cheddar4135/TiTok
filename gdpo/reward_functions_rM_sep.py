#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reward_functions_rM_sep.py
  Separated-channel reward: instead of merging into one MUSEG r_M, the signal is
  split into four channels so that GDPO normalizes each channel within its group.

    Channels: {format, len, global, local}
      format - strict 'From X to Y.' structure (count neutral: any number of well-formed segments = 1.0)
      len    - 1.0 if #pred segments == #gt segments else 0.0 (binary; graded via env)
      global - MUSEG global matching = F1(precision, recall), includes a precision term
      local  - per-segment NGIoU best-match average / max(n_pred, n_gt)

  Motivation (multi-segment collapse):
    1) The old r_G (NGIoU, no precision) cheaply rewarded one large segment covering all
       GT segments, which attracted the policy to single-segment collapse. global uses the
       original MUSEG F1 so the precision term penalizes that strategy.
    2) r_G and r_L are kept separate instead of averaged into r_M; GDPO standardizes each
       channel within the group, so the local signal is no longer drowned out.
    3) The len channel adds an explicit segment-count signal.

  Limitation of binary len:
       In a group that already collapsed to one segment every rollout gets len=0, so
       std=0 and the advantage is 0 (no recovery gradient). Use LEN_REWARD_MODE=graded
       to keep a gradient near collapse.

  The trainer picks up REWARD_CHANNELS through load_reward_module.

  reward_functions.py is imported, not modified.
"""
import os
import re
from typing import List, Tuple

# Reused from reward_functions.py:
#   decode_vtg_time : VTG time tokens -> seconds
#   _merge_intervals: union of overlapping intervals
#   _SEG_CAPTURE_RE : segment parser (1-4 integer digits + <tdot> + 1 decimal digit)
#   ngiou_pair      : NGIoU of one (gt, pred) pair (used by local)
from reward_functions import (
    decode_vtg_time,
    _merge_intervals,
    _SEG_CAPTURE_RE,
    ngiou_pair,
)

# len channel mode: binary (default) / graded (keeps a gradient near collapse).
#   Read once at import time from the environment.
LEN_REWARD_MODE = os.environ.get("LEN_REWARD_MODE", "binary").lower()


# ============================================================
# Parsing: IoU/count channels use the standard capture (format strictness is the format channel's job)
# ============================================================

def _parse(completion: str) -> List[Tuple[float, float]]:
    """Extract (start, end) segments with start < end from a completion."""
    if not isinstance(completion, str):
        return []
    text = completion.replace("<|im_end|>", "").strip()
    out: List[Tuple[float, float]] = []
    for s_str, e_str in _SEG_CAPTURE_RE.findall(text):
        s = decode_vtg_time(s_str)
        e = decode_vtg_time(e_str)
        if s is not None and e is not None and e > s:
            out.append((s, e))
    return out


# ============================================================
# 1) format - strict, count neutral
# ============================================================
# 1.0 only if the whole response is one or more
# 'From <t\d>{3}<tdot><t\d> to <t\d>{3}<tdot><t\d>.' segments, else 0.0.
#   - Any number of well-formed segments scores 1.0 (no penalty on multiple segments).
#   - Any extra text scores 0.0.
_STRICT_SEG = r"[Ff]rom\s+(?:<t\d>){3}<tdot><t\d>\s+to\s+(?:<t\d>){3}<tdot><t\d>\s*\.\s*"
_STRICT_FULL = re.compile(r"^\s*(?:" + _STRICT_SEG + r")+$")


def format_reward(completion: str, **kwargs) -> float:
    if not isinstance(completion, str):
        return 0.0
    text = completion.replace("<|im_end|>", "").strip()
    if not text:
        return 0.0
    return 1.0 if _STRICT_FULL.match(text) else 0.0


# ============================================================
# 2) len - number of predicted vs GT segments
# ============================================================

def len_matching_reward(completion: str, gt_intervals, **kwargs) -> float:
    """Segment-count reward.
      binary (default): 1.0 if n_pred == n_gt else 0.0
      graded (env)    : 1 - |n_pred - n_gt| / max(n_pred, n_gt, 1)
    """
    if not gt_intervals:
        return 0.0
    n_g = len(gt_intervals)
    n_p = len(_parse(completion))
    if n_g == 0:
        return 0.0
    if LEN_REWARD_MODE == "graded":
        return max(0.0, 1.0 - abs(n_p - n_g) / max(n_p, n_g, 1))
    return 1.0 if n_p == n_g else 0.0


# ============================================================
# 3) global - MUSEG global matching (F1 with a precision term)
# ============================================================
# Same as MUSEG reward_global_matching: merge the predicted union, then
#   P = overlap / Σpred,  R = overlap / Σgt,  F1 = 2PR/(P+R)
# the precision term (sum of pred lengths in the denominator) penalizes over-prediction.

def global_reward(completion: str, gt_intervals, **kwargs) -> float:
    if not gt_intervals:
        return 0.0
    preds = _parse(completion)
    if not preds:
        return 0.0
    pred_u = _merge_intervals(preds)              # MUSEG: merge overlapping predictions
    gt = list(gt_intervals)
    all_sel = sum(e - s for s, e in pred_u)
    all_gt = sum(e - s for s, e in gt)
    overlap = 0.0
    for gs, ge in gt:
        for ps, pe in pred_u:
            overlap += max(0.0, min(pe, ge) - max(ps, gs))
    precision = overlap / (all_sel + 1e-6)
    recall = overlap / (all_gt + 1e-6)
    return 2 * precision * recall / (precision + recall + 1e-6)


# ============================================================
# 4) local - per-segment NGIoU best match
# ============================================================
# Sorted sequential pairing breaks when one segment is off, so instead:
# greedy best match (each pred to the unused gt with the highest NGIoU), then
#   average over max(n_pred, n_gt), penalizing both over- and under-prediction.
#   The pairwise NGIoU formula is the original ngiou_pair.

def local_reward(completion: str, gt_intervals, **kwargs) -> float:
    if not gt_intervals:
        return 0.0
    preds = _parse(completion)
    gt = list(gt_intervals)
    if not preds:
        return 0.0
    n = max(len(preds), len(gt))
    if n == 0:
        return 0.0
    used = set()
    total = 0.0
    for p in preds:
        best, best_j = 0.0, -1
        for j, g in enumerate(gt):
            if j in used:
                continue
            score = ngiou_pair(g, p)
            if score > best:
                best, best_j = score, j
        if best_j >= 0:
            used.add(best_j)
            total += best
    return total / n


# ============================================================
# Channel spec read by the trainer: (name, fn, needs_gt)
# ============================================================
# reward_weights follow the order [format, len, global, local].
iou_reward = global_reward   # legacy 'iou' slot of load_reward_module
REWARD_CHANNELS = [
    ("format", format_reward,        False),
    ("len",    len_matching_reward,  True),
    ("global", global_reward,        True),
    ("local",  local_reward,         True),
]

__all__ = [
    "format_reward", "len_matching_reward", "global_reward", "local_reward",
    "iou_reward", "REWARD_CHANNELS",
]
