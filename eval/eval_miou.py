#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_miou.py - multi-segment temporal grounding evaluator (3 methods).

Input:  inference results (test_results_rank0.json; a single file or a folder searched recursively)
Output: three summary JSONs next to each result file (same structure, different IoU method)
  - pairwise_miou_summary.json : MUSEG style (best-match pred IoU per GT segment)
  - union_miou_summary.json    : union style (GT segment vs union of overlapping preds)
  - sample_miou_summary.json   : R-AVST all_iou style (merged GT vs merged pred per sample; paper mIoU)

Aggregation is per sample (compute per sample, then average over samples).

Prediction parsing:
  - default    : video-SALMONN time tokens  "From <t..> to <t..>. ..."
  - --natural  : lenient (time tokens + HH:MM:SS / M:SS / X.X-Y.Y / from X to Y seconds /
                 second{X} to second{Y} / first N seconds and other natural-language outputs)
  - CoT outputs (with <answer>...</answer>) are parsed inside <answer> only (auto-detected).

GT source, in order:
  1) the file given by --test_json
  2) gt_segments embedded in the result entries
  3) <test_dir>/<leaf>.json (or <leaf>/_full.json) named after the result's parent folder
  Matching: (basename(id/video), gt_label) key; positional fallback only when allowed.

Usage:
  python3 eval_miou.py <OUT_DIR or test_results_rank0.json> --test_json <GT.json>
  python3 eval_miou.py <OUT_DIR> --test_json <GT.json> --natural
"""
import argparse
import datetime as _dt
import json
import os
import re

THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]
METHODS = ["pairwise", "union", "sample"]
OUT_NAMES = {
    "pairwise": "pairwise_miou_summary.json",
    "union": "union_miou_summary.json",
    "sample": "sample_miou_summary.json",
}
MAX_TIME_DEFAULT = 9999.9


# ============================== prediction parsing ==============================
def extract_answer_scope(text):
    """CoT: return the content of <answer>...</answer> if present, otherwise the whole text."""
    spans = re.findall(r"<answer>(.*?)</answer>", text or "", re.DOTALL | re.IGNORECASE)
    if spans:
        return " ".join(spans)
    return text or ""


def _fix(s, e, max_time):
    """Invalid intervals (e<=s) get a conservative minimum length of 0.1 s."""
    if e <= s:
        e = min(s + 0.1, max_time)
    return [min(s, max_time), min(e, max_time)]


# ---- time-token intervals ----
# Matches "From <t..> to <t..>" and separator variants such as "<t..> - <t..>" (dash)
# (prompts without an answer-format instruction may yield 'in <t..> - <t..> seconds').
_TOKTIME = r"(?:<t\d>)+(?:<tdot>(?:<t\d>)+)?"
_TOK_SEG = re.compile(rf"({_TOKTIME})\s*(?:to|-|–|—|~)\s*({_TOKTIME})", re.IGNORECASE)


def _decode_tok(token_str, max_time):
    if "<tdot>" in token_str:
        a, _, b = token_str.partition("<tdot>")
        ip = re.findall(r"<t(\d)>", a)
        dp = re.findall(r"<t(\d)>", b)
    else:
        ip = re.findall(r"<t(\d)>", token_str)
        dp = []
    if not ip:
        return None
    return min(int("".join(ip)) + (int(dp[0]) / 10.0 if dp else 0.0), max_time)


def parse_tokens(text, max_time):
    out = []
    for sa, sb in _TOK_SEG.findall(text):
        s, e = _decode_tok(sa, max_time), _decode_tok(sb, max_time)
        if s is None or e is None:
            continue
        out.append(_fix(s, e, max_time))
    return out


# ---- lenient natural-language parsing ----
_HMS = r"(?:\d+:)?\d{1,2}:\d{2}"


def _hms_to_sec(tok):
    s = 0
    for p in (int(x) for x in tok.split(":")):
        s = s * 60 + p
    return float(s)


def parse_natural(text, max_time):
    """Natural-language time outputs -> [[s,e]]. A consumed mask prevents double matches.

    Priority (specific -> general): time tokens -> HH:MM:SS -> M:SS -> second{} -> from X to Y(sec) ->
    X-Y seconds → first N seconds.
    """
    segs = list(parse_tokens(text, max_time))  # recover time tokens even inside prose
    consumed = [False] * len(text)

    def grab(pattern, conv):
        for m in re.finditer(pattern, text, re.IGNORECASE):
            a, b = m.span()
            if any(consumed[a:b]):
                continue
            r = conv(m)
            if r is not None:
                segs.append(r)
                for i in range(a, b):
                    consumed[i] = True

    # HH:MM:SS - HH:MM:SS  (3 fields first)
    grab(r"(\d{1,2}:\d{2}:\d{2})\s*(?:to|-|–|—|~)\s*(\d{1,2}:\d{2}:\d{2})",
         lambda m: _fix(_hms_to_sec(m.group(1)), _hms_to_sec(m.group(2)), max_time))
    # M:SS (to|-|...) M:SS
    grab(rf"({_HMS})\s*(?:to|-|–|—|~)\s*({_HMS})",
         lambda m: _fix(_hms_to_sec(m.group(1)), _hms_to_sec(m.group(2)), max_time))
    # second{X} to/- second{Y}
    grab(r"second\{?\s*([\d.]+)\s*\}?\s*(?:to|-)\s*second\{?\s*([\d.]+)\s*\}?",
         lambda m: _fix(float(m.group(1)), float(m.group(2)), max_time))
    # from X to Y (seconds)
    grab(r"from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*(?:seconds|secs|sec)?\b",
         lambda m: _fix(float(m.group(1)), float(m.group(2)), max_time))
    # X-Y seconds
    grab(r"\b(\d+(?:\.\d+)?)\s*(?:to|-|–|—)\s*(\d+(?:\.\d+)?)\s*(?:seconds|secs|sec)\b",
         lambda m: _fix(float(m.group(1)), float(m.group(2)), max_time))
    # X.X-Y.Y  (decimal-second ranges, also inside prose without a unit)
    grab(r"(\d+(?:\.\d+)?)\s*(?:to|-|–|—)\s*(\d+(?:\.\d+)?)",
         lambda m: _fix(float(m.group(1)), float(m.group(2)), max_time))
    # first N seconds -> [0, N]
    grab(r"first\s+(\d+(?:\.\d+)?)\s*(?:seconds|secs|sec)\b",
         lambda m: _fix(0.0, float(m.group(1)), max_time))

    segs.sort()
    return segs


def parse_pred(raw, natural, max_time):
    scope = extract_answer_scope(raw)
    return parse_natural(scope, max_time) if natural else parse_tokens(scope, max_time)


# ============================== interval operations ==============================
def tiou(a, b):
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    inter = max(0.0, e - s)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def merge_intervals(intervals):
    if not intervals:
        return []
    s = sorted([list(x) for x in intervals])
    out = [list(s[0])]
    for a, b in s[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def total_len(intervals):
    return sum(max(0.0, b - a) for a, b in intervals)


def intersect_lists(A, B):
    out, i, j = [], 0, 0
    while i < len(A) and j < len(B):
        s = max(A[i][0], B[j][0])
        e = min(A[i][1], B[j][1])
        if e > s:
            out.append([s, e])
        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1
    return out


def union_iou(seg, others):
    """IoU of one segment vs the union of the overlapping others."""
    ov = [o for o in others if min(o[1], seg[1]) > max(o[0], seg[0])]
    if not ov:
        return 0.0
    U = merge_intervals(ov)
    G = [list(seg)]
    inter = total_len(intersect_lists(G, U))
    uni = total_len(merge_intervals(G + U))
    return inter / uni if uni > 0 else 0.0


def best_iou(seg, others):
    return max((tiou(seg, o) for o in others), default=0.0)


def sample_iou_parts(gt, pred):
    """all_iou (R-AVST): merged GT vs merged pred -> (iou, recall_t, precision_t)."""
    G = merge_intervals(gt)
    P = merge_intervals(pred)
    inter = total_len(intersect_lists(G, P))
    gt_len, pred_len = total_len(G), total_len(P)
    uni = total_len(merge_intervals(G + P))
    iou = inter / uni if uni > 0 else 0.0
    rec = inter / gt_len if gt_len > 0 else 0.0
    prc = inter / pred_len if pred_len > 0 else 0.0
    return iou, rec, prc


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _clip01(x):
    return max(0.0, min(1.0, x))


# ============================== per-method aggregation ==============================
def _gt_ious(gt, pred, method):
    if method == "pairwise":
        return [best_iou(g, pred) for g in gt]
    return [union_iou(g, pred) for g in gt]  # union


def _pred_ious(gt, pred, method):
    if method == "pairwise":
        return [best_iou(p, gt) for p in pred]
    return [union_iou(p, gt) for p in pred]  # union


def _frac_ge(arr, th):
    return _mean([1.0 if x >= th else 0.0 for x in arr]) if arr else 0.0


def compute_method_block(samples, method):
    """mIoU / Recall / Precision / F1 for one method.

    Aggregation units:
      - pairwise (Best_IoU, per GT segment) : averaged over all GT segments
      - union    (Union_IoU, per GT segment): same
      - sample   (All_IoU, per sample)      : one all_iou per sample, averaged over samples
    Recall = fraction of units with method-IoU >= θ (same as R@1).
    Precision = fraction of prediction-side units with IoU >= θ; F1 = 2PR/(P+R).
    """
    if method == "sample":
        # All_IoU is a symmetric set IoU (inter/union), one value per sample: P = R = F1 = fraction of samples with all_iou >= θ.
        ious = [sample_iou_parts(gt, pred)[0] for gt, pred in samples if gt]
        miou = 100.0 * _mean(ious)
        rec = {str(t): round(100.0 * _frac_ge(ious, t), 4) for t in THRESHOLDS}
        prc = dict(rec)
        f1 = dict(rec)
    else:
        gt_iou_all, pred_iou_all = [], []   # pooled GT / pred segments
        for gt, pred in samples:
            if not gt:
                continue
            gt_iou_all.extend(_gt_ious(gt, pred, method))   # IoU per GT segment
            pred_iou_all.extend(_pred_ious(gt, pred, method))  # IoU per predicted segment
        miou = 100.0 * _mean(gt_iou_all)
        rec, prc, f1 = {}, {}, {}
        for t in THRESHOLDS:
            r = _frac_ge(gt_iou_all, t)
            p = _frac_ge(pred_iou_all, t)
            rec[str(t)] = round(100.0 * r, 4)
            prc[str(t)] = round(100.0 * p, 4)
            f1[str(t)] = round(100.0 * (2 * r * p / (r + p) if (r + p) > 0 else 0.0), 4)

    return {
        "method": method,
        "mIoU_%": round(miou, 4),
        "Recall_avg_%": round(_mean(list(rec.values())), 4),
        "Precision_avg_%": round(_mean(list(prc.values())), 4),
        "F1_avg_%": round(_mean(list(f1.values())), 4),
        "Recall": rec,
        "Precision": prc,
        "F1": f1,
    }


def compute_count_metrics(samples):
    """CountF1 and its components (CR*, FMR), using only per-sample counts (N_gt, N_pred).

    Subsets (by GT count):
      - multi  S  = {i : N_gt >= 2}
      - single S1 = {i : N_gt == 1}
      - samples with N_gt == 0 are excluded from both.

    1) CR_multi = mean_{i∈S} min(N_gt,N_pred)/max(N_gt,N_pred)
       chance_floor b = mean_{i∈S} 1/N_gt   (expected score of "always predict one")
       CR* = (CR_multi - b)/(1 - b), clipped to [0,1] (0 if worse than chance).
    2) FMR = mean_{i∈S1} 1[N_pred >= 2]      (single segments split; lower is better)
       SingleMiss = mean_{i∈S1} 1[N_pred == 0] (parse failures / no answer)
    3) SingleAcc = mean_{i∈S1} 1[N_pred == 1] = 1 - FMR - SingleMiss
       Parse failures (N_pred=0) do not count as success, so models with many parse
       failures do not get an inflated OSA.
    4) CountF1 = harmonic mean(CR*, SingleAcc) = 2·CR*·SingleAcc/(CR*+SingleAcc),
       0 when both are 0.

    An empty subset gives None for its metric, and CountF1 is then None ('-' in tables).
    """
    multi_ratios, multi_floor, single_npred = [], [], []
    for gt, pred in samples:
        ng, npd = len(gt), len(pred)
        if ng >= 2:
            hi = max(ng, npd)
            multi_ratios.append(min(ng, npd) / hi if hi > 0 else 0.0)
            multi_floor.append(1.0 / ng)
        elif ng == 1:
            single_npred.append(npd)

    out = {"n_multi": len(multi_ratios), "n_single": len(single_npred),
           "CR_multi": None, "chance_floor": None, "CR_star": None,
           "FMR": None, "SingleMiss": None, "SingleAcc": None, "CountF1": None}

    if multi_ratios:
        cr = _mean(multi_ratios)
        b = _mean(multi_floor)
        out["CR_multi"] = round(cr, 6)
        out["chance_floor"] = round(b, 6)
        out["CR_star"] = round(_clip01((cr - b) / (1 - b)) if (1 - b) > 0 else 0.0, 6)
    if single_npred:
        out["FMR"] = round(_mean([1.0 if n >= 2 else 0.0 for n in single_npred]), 6)
        out["SingleMiss"] = round(_mean([1.0 if n == 0 else 0.0 for n in single_npred]), 6)
        out["SingleAcc"] = round(_mean([1.0 if n == 1 else 0.0 for n in single_npred]), 6)

    cs, sa = out["CR_star"], out["SingleAcc"]
    if cs is not None and sa is not None:
        denom = cs + sa
        out["CountF1"] = round(2 * cs * sa / denom, 6) if denom > 0 else 0.0
    return out


def compute_shared_block(samples):
    """Method-independent stats: FP/FN (best-match overlap, per threshold), segment counts, parsing."""
    total_gt = total_pred = 0
    n_fn = {t: 0 for t in THRESHOLDS}
    n_fp = {t: 0 for t in THRESHOLDS}
    gt_counts, pred_counts = [], []
    parse_ok = parse_fail = 0

    for gt, pred in samples:
        gt_counts.append(len(gt))
        pred_counts.append(len(pred))
        parse_ok += 1 if pred else 0
        parse_fail += 0 if pred else 1
        total_gt += len(gt)
        total_pred += len(pred)
        for g in gt:
            bi = best_iou(g, pred)
            for t in THRESHOLDS:
                if bi < t:
                    n_fn[t] += 1
        for p in pred:
            bi = best_iou(p, gt)
            for t in THRESHOLDS:
                if bi < t:
                    n_fp[t] += 1

    n = len(samples)

    def buckets(counts):
        b = {"0": 0, "1": 0, "2": 0, "3": 0, "4+": 0}
        for c in counts:
            b["4+" if c >= 4 else str(c)] += 1
        return {k: round(100.0 * v / max(n, 1), 4) for k, v in b.items()}

    fp_rate = {str(t): round(100.0 * n_fp[t] / max(total_pred, 1), 4) for t in THRESHOLDS}
    fn_rate = {str(t): round(100.0 * n_fn[t] / max(total_gt, 1), 4) for t in THRESHOLDS}

    return {
        "n_samples": n,
        "count_metrics": compute_count_metrics(samples),
        "FP_rate_avg_%": round(_mean(list(fp_rate.values())), 4),
        "FN_rate_avg_%": round(_mean(list(fn_rate.values())), 4),
        "FP_rate_%": fp_rate,
        "FN_rate_%": fn_rate,
        "n_fp_segments": {str(t): n_fp[t] for t in THRESHOLDS},
        "n_fn_segments": {str(t): n_fn[t] for t in THRESHOLDS},
        "gt_segments": {
            "total": total_gt,
            "mean_per_sample": round(_mean(gt_counts), 4),
            "count_buckets_pct": buckets(gt_counts),
        },
        "pred_segments": {
            "total": total_pred,
            "mean_per_sample": round(_mean(pred_counts), 4),
            "count_buckets_pct": buckets(pred_counts),
        },
        "parse_ok": parse_ok,
        "parse_fail": parse_fail,
    }


def build_summaries(samples, meta):
    """Return the three per-method summary dicts ({method: summary})."""
    shared = compute_shared_block(samples)
    out = {}
    for method in METHODS:
        block = compute_method_block(samples, method)
        summary = {
            "label": meta.get("label", ""),
            "testset": meta.get("testset", ""),
            "method": method,
            "n_samples": shared["n_samples"],
        }
        summary.update({k: v for k, v in block.items() if k != "method"})
        summary["thresholds"] = THRESHOLDS
        summary.update({k: v for k, v in shared.items() if k != "n_samples"})
        summary.update({k: v for k, v in meta.items()
                        if k not in ("label", "testset")})
        out[method] = summary
    return out


# ============================== GT source resolution ==============================
def _basename(p):
    return os.path.basename(str(p)) if p else ""


def _coerce_segments(val):
    if not val:
        return []
    if isinstance(val, str):
        out = []
        for m in re.finditer(r"\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)", val):
            out.append([float(m.group(1)), float(m.group(2))])
        return out
    out = []
    for s in val:
        try:
            out.append([float(s[0]), float(s[1])])
        except (TypeError, ValueError, IndexError):
            continue
    return out


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _default_test_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    # <repo>/eval -> <repo>/data/test
    return os.path.normpath(os.path.join(here, "..", "data", "test"))


def resolve_gt_json_path(results_file, test_dir):
    leaf = os.path.basename(os.path.dirname(os.path.abspath(results_file)))
    flat = os.path.join(test_dir, f"{leaf}.json")
    if os.path.isfile(flat):
        return flat
    nested = os.path.join(test_dir, leaf, "_full.json")
    return nested if os.path.isfile(nested) else None


def _gt_from_ref(results, max_time):
    """Parse GT from each entry's 'ref' (video-SALMONN time tokens)."""
    gt_list = [parse_tokens(r.get("ref", "") or "", max_time) for r in results]
    return gt_list, sum(1 for g in gt_list if g), sum(1 for g in gt_list if not g)


def _gt_from_test_json(results, gt_path, source_tag, allow_positional=False):
    """Match gt_segments from test_json.

    Order: (video/id, gt_label) key -> video alone (when unambiguous) ->
    positional (only with allow_positional). Implicit positional matching compares
    against the wrong GT when the result order differs, so it is off by default.
    """
    gt_data = _load_json(gt_path)
    key_map, ambiguous = {}, set()          # (video, gt_label) -> segs
    vid_map, vid_amb = {}, set()            # video -> segs (video-only fallback)
    for item in gt_data:
        vid = _basename(item.get("video") or item.get("id") or item.get("audio"))
        segs = _coerce_segments(item.get("gt_segments"))
        key = (vid, item.get("gt_label", ""))
        if key in key_map and key_map[key] != segs:
            ambiguous.add(key)
        key_map[key] = segs
        if vid in vid_map and vid_map[vid] != segs:
            vid_amb.add(vid)               # same video with different segs -> ambiguous
        vid_map[vid] = segs

    gt_list, matched, unmatched, by_pos = [], 0, 0, 0
    for i, r in enumerate(results):
        vid = _basename(r.get("id") or r.get("video") or r.get("audio"))
        gl = r.get("gt_label", "")
        key = (vid, gl)
        if gl and key in key_map and key not in ambiguous:
            gt_list.append(key_map[key]); matched += 1
        elif vid and vid in vid_map and vid not in vid_amb:
            gt_list.append(vid_map[vid]); matched += 1
        elif allow_positional and len(gt_data) == len(results):
            gt_list.append(_coerce_segments(gt_data[i].get("gt_segments")))
            matched += 1; by_pos += 1
        else:
            gt_list.append([]); unmatched += 1
    if by_pos:
        print(f"[WARN] {os.path.basename(gt_path)}: {by_pos} samples matched by position "
              f"(key/video matching failed). GT is misaligned if the order differs.")
    return gt_list, source_tag, matched, unmatched


def build_gt_for_results(results, results_file, args):
    """GT source priority:
    --gt_ref (forced) > embedded (gt_segments) > ref (auto) > --test_json / leaf-name test_json.

    ref/embedded come first because they are each sample's own GT, so order and
    matching cannot go wrong. test_json relies on (video, gt_label) matching and is
    fragile for results without gt_label.
    """
    n = len(results)
    if args.gt_ref:
        gl, m, u = _gt_from_ref(results, args.max_time)
        return gl, "ref(forced)", m, u
    if all(r.get("gt_segments") is not None for r in results):
        gl = [_coerce_segments(r.get("gt_segments")) for r in results]
        return gl, "embedded", sum(1 for g in gl if g), sum(1 for g in gl if not g)
    if all(r.get("ref") for r in results):
        gl, m, u = _gt_from_ref(results, args.max_time)
        # Use ref only when it parses as video-SALMONN time-token GT. Natural-language refs
        # (HH:MM:SS etc.) give 0 segments with parse_tokens -> fall back to test_json.
        if m >= 0.5 * n:
            return gl, "ref(auto)", m, u
    gt_path = args.test_json or resolve_gt_json_path(results_file, args.test_dir)
    if not gt_path or not os.path.isfile(gt_path):
        raise FileNotFoundError(
            f"GT not found (results={results_file}): no embedded gt_segments/ref and no "
            f"leaf-name match. Pass --test_json or --gt_ref. (test_dir={args.test_dir})"
        )
    gl, src, m, u = _gt_from_test_json(results, gt_path, gt_path, args.allow_positional)
    if u > 0:
        print(f"[WARN] test_json matching failed for {u}/{n} samples (excluded with gt=[]). "
              f"If these results carry ref/gt_segments, those are more accurate.")
    return gl, src, m, u


# ============================== evaluate one result file ==============================
def write_and_report(results_file, summaries, args):
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(results_file))
    os.makedirs(out_dir, exist_ok=True)
    for method in METHODS:
        path = os.path.join(out_dir, OUT_NAMES[method])
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(summaries[method], f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)

    if not args.quiet:
        s0 = summaries["pairwise"]
        SEP = "=" * 60
        print(f"\n{SEP}")
        print(f"  {results_file}")
        print(f"  GT: {s0.get('gt_source')} (matched={s0.get('gt_matched')}, "
              f"unmatched={s0.get('gt_unmatched')})  natural={s0.get('natural')}")
        print(f"  Samples: {s0['n_samples']}  GT segs: {s0['gt_segments']['total']}  "
              f"Pred segs: {s0['pred_segments']['total']}  "
              f"parse_ok/fail: {s0['parse_ok']}/{s0['parse_fail']}")
        print(f"  {'method':9s} | mIoU%  | R@.1/.3/.5/.7  | F1_avg | FP/FN_avg%")
        for m in METHODS:
            s = summaries[m]
            r = s["Recall"]
            print(f"  {m:9s} | {s['mIoU_%']:5.2f} | "
                  f"{r['0.1']:.1f}/{r['0.3']:.1f}/{r['0.5']:.1f}/{r['0.7']:.1f} | "
                  f"{s['F1_avg_%']:5.2f} | {s['FP_rate_avg_%']:.1f}/{s['FN_rate_avg_%']:.1f}")
        print(f"  [SAVED] {', '.join(OUT_NAMES[m] for m in METHODS)}  -> {out_dir}")
        print(SEP)

    if args.progress_log:
        snap = {"timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
                "results_file": results_file,
                **{m: summaries[m] for m in METHODS}}
        with open(args.progress_log, "a") as f:
            f.write(json.dumps(snap, ensure_ascii=False) + "\n")


def _scale_percent(segs, dur, max_time):
    """Predictions given as a percentage (0~100) of the video duration -> seconds."""
    if not dur:
        return []
    return [_fix(s / 100.0 * dur, e / 100.0 * dur, max_time) for s, e in segs]


def evaluate_one(results_file, args):
    results = _load_json(results_file)
    gt_list, gt_source, matched, unmatched = build_gt_for_results(results, results_file, args)
    samples = []
    for r, gt in zip(results, gt_list):
        pred = parse_pred(r.get("pred", ""), args.natural, args.max_time)
        if args.pred_percent:
            pred = _scale_percent(pred, r.get(args.duration_key), args.max_time)
        samples.append((gt, pred))
    testset = args.testset or os.path.basename(os.path.dirname(os.path.abspath(results_file)))
    meta = {"label": args.label, "testset": testset,
            "gt_source": gt_source, "gt_matched": matched,
            "gt_unmatched": unmatched, "natural": bool(args.natural)}
    summaries = build_summaries(samples, meta)
    write_and_report(results_file, summaries, args)
    return summaries


def find_results_files(root):
    if os.path.isfile(root):
        return [root]
    found = []
    for dirpath, _dirs, files in os.walk(root):
        if "test_results_rank0.json" in files:
            found.append(os.path.join(dirpath, "test_results_rank0.json"))
    return sorted(found)


def main():
    ap = argparse.ArgumentParser(
        description="Multi-segment temporal grounding mIoU (pairwise/union/sample)")
    ap.add_argument("results", help="test_results_rank0.json or a folder to search recursively")
    ap.add_argument("--test_json", default=None, help="GT json (default: embedded GT / leaf-name match)")
    ap.add_argument("--test_dir", default=None, help="folder for leaf-name GT lookup (default <repo>/data/test)")
    ap.add_argument("--natural", action="store_true",
                    help="lenient parsing (HH:MM:SS/M:SS/seconds/second{} ...). Default: time tokens only")
    ap.add_argument("--label", default="", help="model label stored in the summary")
    ap.add_argument("--testset", default="", help="test-set tag stored in the summary")
    ap.add_argument("--gt_ref", action="store_true",
                    help="parse GT from each entry's 'ref' (time tokens), forced. Default: auto-detect")
    ap.add_argument("--pred_percent", action="store_true",
                    help="predictions are %% of the video duration; convert to seconds")
    ap.add_argument("--allow_positional", action="store_true",
                    help="allow positional GT matching when test_json matching fails (risky; default off)")
    ap.add_argument("--duration_key", default="duration",
                    help="duration field used by --pred_percent")
    ap.add_argument("--max_time", type=float, default=MAX_TIME_DEFAULT)
    ap.add_argument("--out_dir", default=None, help="only for a single result file; default: next to the result")
    ap.add_argument("--progress_log", default=None, help="append JSONL snapshots here")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.test_dir is None:
        args.test_dir = _default_test_dir()

    files = find_results_files(args.results)
    if not files:
        raise SystemExit(f"[error] no test_results_rank0.json found: {args.results}")
    # In folder mode, always write next to each result (avoids mixing outputs)
    if len(files) > 1:
        args.out_dir = None

    if not args.quiet:
        print(f"[INFO] {len(files)} result files  (natural={args.natural})")

    for rf in files:
        try:
            evaluate_one(rf, args)
        except Exception as e:  # noqa: BLE001
            print(f"[SKIP] {rf}: {e}")


if __name__ == "__main__":
    main()
