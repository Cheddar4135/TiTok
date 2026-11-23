#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
maketable.py - collect every evaluation summary below a folder and write a single
tab-separated table to <folder>/table.txt (overwritten on every run).

One row per result folder (a folder containing pairwise_miou_summary.json); n_samples<500 is skipped.
Rows are sorted by sample_mIoU, descending.

Columns (default, tab-separated, 12):
  ID  ckpt  sample_mIoU  F1@0.1  F1@0.3  F1@0.5  F1@0.7  CountF1  USA  OSA
  n_samples  testset

  - sample_mIoU : mIoU_% from sample_miou_summary.json (per-sample IoU of merged GT vs merged pred).
                  The F1 columns use the pairwise method, so the column name keeps 'sample'
                  to make the mix explicit; this is the mIoU reported in the paper.
  - F1@θ    : from pairwise_miou_summary.json (best-match per segment)
  - CountF1 : harmonic mean of USA and OSA; 0 if either is 0, so a model must both
              recover repeated events and avoid splitting single ones.
  - USA     : = CR*, chance-corrected count ratio on multi-segment queries (N_gt>=2):
              (CR_multi-b)/(1-b), clipped to [0,1]. 0 = chance, 1 = perfect.
  - OSA     : = SingleAcc, fraction of single-segment queries (N_gt==1) answered with exactly one segment.
      Note: parse failures (N_pred=0) count as failures, so SingleAcc != 1-FMR.
      Note: values are read from the stored summaries; summaries written before this
        definition used N_pred<=1. Re-score results that contain parse failures.
      Note: older summaries without a count_metrics block are read from top-level CR_star/SingleAcc;
        without SingleAcc the column shows '-' (1-FMR would count N_pred=0 as success).

  --full : 21 columns (adds F1@0.9, CR/FMR, SCR, R@θ, mean gt/pred counts).

Usage:  python3 eval/maketable.py outputs/gdpo
"""
import argparse
import json
import os
import re

THS = ["0.1", "0.3", "0.5", "0.7", "0.9"]
MAIN_THS = ["0.1", "0.3", "0.5", "0.7"]          # thresholds in the default table

HEADER = (["ID", "ckpt", "sample_mIoU"]
          + [f"F1@{t}" for t in MAIN_THS]
          + ["CountF1", "USA", "OSA", "n_samples", "testset"])

HEADER_FULL = (["ID", "ckpt", "sample_mIoU"]
               + [f"F1@{t}" for t in THS]
               + ["CountF1", "USA", "OSA", "SCR"]
               + [f"R@{t}" for t in THS]
               + ["gt(mean)", "pred(mean)", "n_samples", "testset"])


def _num(x):
    try:
        return f"{float(x):.2f}"
    except (TypeError, ValueError):
        return "-"


def _isnum(x):
    try:
        float(x); return True
    except (TypeError, ValueError):
        return False


def _series_cols(d, keys=THS):
    """Per-threshold values as a list of columns."""
    if not isinstance(d, dict):
        return ["-" for _ in keys]
    return [_num(d.get(k)) for k in keys]


def _scr(gmean, pmean):
    """Segment Count Ratio = min/max, only when both are positive."""
    if not (_isnum(gmean) and _isnum(pmean)):
        return "-"
    g, p = float(gmean), float(pmean)
    hi = max(g, p)
    if hi <= 0:
        return "-"
    return f"{min(g, p) / hi:.2f}"


def _count_metrics(p):
    """Extract USA/OSA/CountF1 from both the current (count_metrics block) and the old (flat) schema.

    USA = CR_star, OSA = SingleAcc; '-' when SingleAcc is missing
    (1-FMR counts N_pred=0 as success, so it is not used as a substitute).
    """
    cm = p.get("count_metrics")
    if not isinstance(cm, dict):
        cm = p                                   # old schema: CR_star/FMR/CountF1 at the top level
    osa = cm.get("SingleAcc")
    return _num(cm.get("CountF1")), _num(cm.get("CR_star")), _num(osa)


def parse_path(dirpath):
    """Result folder -> (ID, ckpt, testset). Works regardless of the root
    (a leaf folder can be passed directly). The component before checkpoint-* is the ID."""
    parts = [p for p in os.path.abspath(dirpath).split(os.sep) if p]
    testset = parts[-1] if parts else "-"
    ckpt, ck_idx = "-", None
    for i, p in enumerate(parts):
        m = re.fullmatch(r"checkpoint-(\w+)", p)
        if m:
            ckpt, ck_idx = m.group(1), i
            break
    if ck_idx is not None and ck_idx > 0:
        ID = parts[ck_idx - 1]                      # run name right before checkpoint-*
    else:
        ID = "-"
        for stage in ("gdpo", "base", "sft", "merged"):   # no checkpoint component
            if stage in parts and parts.index(stage) + 1 < len(parts):
                ID = parts[parts.index(stage) + 1]
                break
    return ID, ckpt, testset


def _load(path):
    try:
        return json.load(open(path))
    except (json.JSONDecodeError, OSError):
        return {}


def main():
    ap = argparse.ArgumentParser(description="Collect evaluation summaries below a folder into table.txt")
    ap.add_argument("path", help="root folder to scan (e.g. outputs/gdpo); table.txt is written here")
    ap.add_argument("--out", default="table.txt", help="output file name (default table.txt)")
    ap.add_argument("--full", action="store_true",
                    help="print all columns (adds F1@0.9, SCR, R@θ, mean gt/pred counts)")
    args = ap.parse_args()
    header = HEADER_FULL if args.full else HEADER
    root = os.path.abspath(args.path)
    if not os.path.isdir(root):
        raise SystemExit(f"[error] not a directory: {root}")

    rows = []
    for dirpath, _d, files in os.walk(root):
        if "pairwise_miou_summary.json" not in files:
            continue
        p = _load(os.path.join(dirpath, "pairwise_miou_summary.json"))
        s = _load(os.path.join(dirpath, "sample_miou_summary.json"))
        if (p.get("n_samples") or 0) < 500:
            continue
        ID, ckpt, testset = parse_path(dirpath)
        sample = _num(s.get("mIoU_%"))

        gmean = _num(p.get("gt_segments", {}).get("mean_per_sample"))
        pmean = _num(p.get("pred_segments", {}).get("mean_per_sample"))
        scr = _scr(gmean, pmean)

        countf1, usa, osa = _count_metrics(p)
        n = str(p.get("n_samples", "-"))

        if args.full:
            rows.append([ID, ckpt, sample]
                        + _series_cols(p.get("F1"))
                        + [countf1, usa, osa, scr]
                        + _series_cols(p.get("Recall"))
                        + [gmean, pmean, n, testset])
        else:
            rows.append([ID, ckpt, sample]
                        + _series_cols(p.get("F1"), MAIN_THS)
                        + [countf1, usa, osa, n, testset])

    # sort by sample_mIoU (column 2), descending
    rows.sort(key=lambda r: (float(r[2]) if _isnum(r[2]) else -1.0), reverse=True)

    lines = ["\t".join(header)] + ["\t".join(r) for r in rows]
    out_path = os.path.join(root, args.out)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, out_path)
    print(f"[SAVED] {out_path}  ({len(rows)} rows, {len(header)} columns)")


if __name__ == "__main__":
    main()
