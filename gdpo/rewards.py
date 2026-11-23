#!/usr/bin/env python3
"""
rewards.py - TiTok reward for the GDPO stage (multi-segment temporal grounding).

Expected completion format (video-SALMONN 2+ time tokens, XXX.Y seconds):
    "From <t0><t0><t5><tdot><t2> to <t0><t1><t3><tdot><t9>. From ..."

Five channels, all in [0, 1] and computed on the whole completion. GDPO normalizes
each channel within its rollout group before combining them:
    format    - strict 'From <t\\d>{3}<tdot><t\\d> to ... .' list (count neutral)
    count     - segment-count matching (0 for a single prediction on a multi-segment query)
    global    - MUSEG global matching: F1 of merged prediction vs GT time
    local     - per-segment NGIoU, greedy best match, averaged over max(n_pred, n_gt)
    precision - temporal precision: fraction of predicted time that overlaps the GT

global and local follow MUSEG ("MUSEG: Reinforcing Video Temporal Understanding via
Timestamp-Aware Multi-Segment Grounding", arXiv:2505.20715). The continuous precision
channel penalizes over-long and extra segments, which an IoU-threshold false-positive
count does not.

The trainer reads REWARD_CHANNELS = [(name, fn, needs_gt), ...] (see load_reward_module
in gdpo_trainer.py); reward_weights in the config follow the same order.
"""
import re
from typing import List, Tuple

# ============================================================
# Parsing
# ============================================================
_TIME_TOKEN_RE = r"(?:<t\d>){1,4}<tdot><t\d>"
_SEG_CAPTURE_RE = re.compile(
    r"[Ff]rom\s+(" + _TIME_TOKEN_RE + r")\s+to\s+(" + _TIME_TOKEN_RE + r")"
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
# Interval helpers
# ============================================================

def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Merge overlapping or touching intervals and return their union."""
    if not intervals:
        return []
    sorted_iv = sorted(tuple(iv) for iv in intervals)  # accept [s, e] lists as well as tuples
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


# ============================================================
# Reward channels
# ============================================================

# format: 1.0 only if the whole response is one or more strict
# 'From <t\d>{3}<tdot><t\d> to <t\d>{3}<tdot><t\d>.' segments, else 0.0.
_STRICT_SEG = r"[Ff]rom\s+(?:<t\d>){3}<tdot><t\d>\s+to\s+(?:<t\d>){3}<tdot><t\d>\s*\.\s*"
_STRICT_FULL = re.compile(r"^\s*(?:" + _STRICT_SEG + r")+$")


def format_reward(completion: str, **kwargs) -> float:
    if not isinstance(completion, str):
        return 0.0
    text = completion.replace("<|im_end|>", "").strip()
    if not text:
        return 0.0
    return 1.0 if _STRICT_FULL.match(text) else 0.0


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


# global: MUSEG reward_global_matching. P = overlap / sum(pred), R = overlap / sum(gt),
# F1 = 2PR/(P+R); the precision term penalizes over-prediction.
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


# local: greedy best match (each prediction to the unused GT with the highest NGIoU),
# averaged over max(n_pred, n_gt) to penalize over- and under-prediction.
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


# Channel spec read by the trainer: (name, fn, needs_gt).
REWARD_CHANNELS = [
    ("format",    format_reward,          False),
    ("count",     count_matching_reward,  True),
    ("global",    global_reward,          True),
    ("local",     local_reward,           True),
    ("precision", precision_reward,       True),
]
