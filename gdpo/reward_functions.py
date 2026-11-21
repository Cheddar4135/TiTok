#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reward_functions.py
  Base reward utilities for GDPO (multi-segment temporal grounding).

Expected output format:
  "From <t0><t0><t0><t5><tdot><t2> to <t0><t0><t1><t3><tdot><t9>. From ..."

Rewards:
  1. format_reward - strict "From X to Y." format score in [0, 1]
  2. iou_reward    - MUSEG segment-matching reward r_M (see r_M below)
"""

import os
import re
from typing import List, Tuple

# Weight of r_G in r_M = w*r_G + (1-w)*r_L. Override with env RM_RG_WEIGHT.
# Default 0.5 gives the MUSEG definition r_M = (r_G + r_L) / 2.
RM_RG_WEIGHT = float(os.environ.get("RM_RG_WEIGHT", "0.5"))


# One "From X to Y[.]" segment
# X, Y: (<t\d>){1,4}<tdot><t\d>  (1-4 integer digits + . + 1 decimal digit)
_TIME_TOKEN_RE = r"(?:<t\d>){1,4}<tdot><t\d>"
_SEG_CAPTURE_RE = re.compile(
    r"[Ff]rom\s+(" + _TIME_TOKEN_RE + r")\s+to\s+(" + _TIME_TOKEN_RE + r")"
)
_SEG_MATCH_RE = re.compile(
    r"[Ff]rom\s+" + _TIME_TOKEN_RE + r"\s+to\s+" + _TIME_TOKEN_RE + r"\.?"
)


def decode_vtg_time(token_str: str, max_time: float = 999.9) -> float | None:
    """Decode VTG time tokens, e.g. "<t0><t0><t3><t9><tdot><t0>" -> 39.0.

    max_time: upper bound in seconds (the XXX.Y format covers 0.0~999.9).
    """
    if "<tdot>" not in token_str:
        return None
    parts = token_str.split("<tdot>")
    int_part = re.findall(r"<t(\d)>", parts[0])
    dec_part = re.findall(r"<t(\d)>", parts[1]) if len(parts) > 1 else []
    if not int_part:
        return None
    integer_part = int("".join(int_part))
    decimal_part = int(dec_part[0]) if dec_part else 0
    t = integer_part + decimal_part / 10.0
    return min(t, max_time)


def temporal_iou(ps: float, pe: float, gs: float, ge: float) -> float:
    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = (pe - ps) + (ge - gs) - inter
    return inter / union if union > 0 else 0.0


# Strict format reward (scored per time value)
# Loose matcher: captures any time-token sequence in "from X to Y"
_SEG_LOOSE_RE = re.compile(
    r"[Ff]rom\s+((?:<t\d>|<tdot>)+)\s+to\s+((?:<t\d>|<tdot>)+)"
)


def _score_time_value(s: str) -> float:
    """Strict score in [0, 1] for one time value (X or Y).

    0.5 - contains <tdot>
    0.5 - canonical token layout (<t\\d>{3}<tdot><t\\d>{1})
    """
    score = 0.0
    has_tdot = "<tdot>" in s
    if has_tdot:
        score += 0.5
        parts = s.split("<tdot>")
        if len(parts) == 2:
            int_digits = re.findall(r"<t\d>", parts[0])
            dec_digits = re.findall(r"<t\d>", parts[1])
            if len(int_digits) == 3 and len(dec_digits) == 1:
                score += 0.5
    return score


def format_reward(completion: str, **kwargs) -> float:
    """'From X to Y.' segment-level strict format reward.

    Each segment's X and Y are scored independently and averaged.
    Per time value:
      0.5 - contains <tdot>
      0.5 - canonical token layout (<t\\d>{3}<tdot><t\\d>{1})
    """
    if not isinstance(completion, str):
        return 0.0
    text = completion.replace("<|im_end|>", "").strip()
    if not text:
        return 0.0

    matches = _SEG_LOOSE_RE.findall(text)
    if not matches:
        return 0.0

    scores = []
    for x_str, y_str in matches:
        scores.append(_score_time_value(x_str))
        scores.append(_score_time_value(y_str))

    return sum(scores) / len(scores)


# Interval helpers (and an unused set-IoU recall variant)

def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Merge overlapping or touching intervals and return their union."""
    if not intervals:
        return []
    sorted_iv = sorted(intervals)
    merged = [sorted_iv[0]]
    for s, e in sorted_iv[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged


def _intersection_length(a: List[Tuple[float, float]],
                          b: List[Tuple[float, float]]) -> float:
    """Total intersection length between two merged interval lists."""
    total = 0.0
    for as_, ae in a:
        for bs, be in b:
            total += max(0.0, min(ae, be) - max(as_, bs))
    return total


def iou_reward_setrecall(
    completion: str,
    gt_intervals: List[Tuple[float, float]],
    **kwargs,
) -> float:
    """[unused] Set-IoU recall: coverage of union(GT) by union(pred).

        recall = |union(preds) ∩ union(GTs)| / |union(GTs)|

    No over-prediction penalty (recall only).
    """
    if not gt_intervals:
        return 0.0
    if not isinstance(completion, str):
        return 0.0

    text = completion.replace("<|im_end|>", "").strip()

    preds = []
    for start_str, end_str in _SEG_CAPTURE_RE.findall(text):
        s = decode_vtg_time(start_str)
        e = decode_vtg_time(end_str)
        if s is not None and e is not None and e > s:
            preds.append((s, e))

    if not preds:
        return 0.0

    pred_union = _merge_intervals(preds)
    gt_union = _merge_intervals(list(gt_intervals))
    gt_total = sum(e - s for s, e in gt_union)
    if gt_total <= 0:
        return 0.0

    inter = _intersection_length(pred_union, gt_union)
    return inter / gt_total


# ============================================================
# MUSEG r_M - segment matching reward (used)
#   Paper: "MUSEG: Reinforcing Video Temporal Understanding via
#           Timestamp-Aware Multi-Segment Grounding" (arXiv:2505.20715)
#   r_M = (r_G + r_L) / 2
#     r_G — global NGIoU on union(GT) vs union(pred)
#     r_L — sorted sequential pairing NGIoU (G_i ↔ P_i), mismatch=0
# ============================================================

def _enclosing_interval_length(intervals: List[Tuple[float, float]]) -> float:
    """Length |C| of the smallest interval enclosing all intervals."""
    if not intervals:
        return 0.0
    lo = min(s for s, _ in intervals)
    hi = max(e for _, e in intervals)
    return max(0.0, hi - lo)


def _union_total_length(intervals: List[Tuple[float, float]]) -> float:
    """Total length of the union of an interval list."""
    return sum(e - s for s, e in _merge_intervals(intervals))


def ngiou_segments(pred: List[Tuple[float, float]],
                    gt: List[Tuple[float, float]]) -> float:
    """NGIoU between two interval sets (each union treated as one region).

        NGIoU = ½ · (1 + IoU − |C \\ (G ∪ P)| / |C|)

    IoU = |G n P| / |G u P|; C is the smallest interval enclosing G and P.
    Range [0, 1]; 0 if either set is empty.
    """
    if not pred or not gt:
        return 0.0

    pred_u = _merge_intervals(pred)
    gt_u = _merge_intervals(gt)

    inter_len = _intersection_length(pred_u, gt_u)
    union_len = _union_total_length(list(pred_u) + list(gt_u))
    iou_term = inter_len / union_len if union_len > 0 else 0.0

    c_len = _enclosing_interval_length(list(pred_u) + list(gt_u))
    outside_term = (c_len - union_len) / c_len if c_len > 0 else 0.0

    return 0.5 * (1 + iou_term - outside_term)


def ngiou_pair(g: Tuple[float, float], p: Tuple[float, float]) -> float:
    """NGIoU of a single (g, p) interval pair."""
    return ngiou_segments([p], [g])


def r_M(
    completion: str,
    gt_intervals: List[Tuple[float, float]],
    **kwargs,
) -> float:
    """MUSEG Segment Matching reward (multi-segment temporal grounding).

        r_M = (r_G + r_L) / 2

    r_G (global): NGIoU of union(GT) vs union(pred)
    r_L (local):  sort by start, pair G_i with P_i sequentially and average NGIoU.
                  Unmatched segments score 0.
                  Denominator max(|G|, |P|) penalizes over- and under-prediction.

    Sequential matching after sorting (not Hungarian), as defined in the MUSEG paper.
    """
    if not gt_intervals:
        return 0.0
    if not isinstance(completion, str):
        return 0.0

    text = completion.replace("<|im_end|>", "").strip()
    preds = []
    for start_str, end_str in _SEG_CAPTURE_RE.findall(text):
        s = decode_vtg_time(start_str)
        e = decode_vtg_time(end_str)
        if s is not None and e is not None and e > s:
            preds.append((s, e))

    if not preds:
        return 0.0

    r_G = ngiou_segments(preds, list(gt_intervals))

    pred_sorted = sorted(preds, key=lambda x: x[0])
    gt_sorted = sorted(list(gt_intervals), key=lambda x: x[0])
    n = max(len(pred_sorted), len(gt_sorted))
    if n == 0:
        return 0.0
    local_scores = []
    for i in range(n):
        if i < len(pred_sorted) and i < len(gt_sorted):
            local_scores.append(ngiou_pair(gt_sorted[i], pred_sorted[i]))
        else:
            local_scores.append(0.0)
    r_L = sum(local_scores) / n

    return RM_RG_WEIGHT * r_G + (1.0 - RM_RG_WEIGHT) * r_L


# Name used by the trainer / other modules; implemented by MUSEG r_M.
iou_reward = r_M
