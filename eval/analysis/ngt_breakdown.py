#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ngt_breakdown.py — 기존 test_results_rank0.json 을 N_gt(1/2/3/4+) 별로 재집계.

  - mIoU : eval_miou.compute_method_block 그대로 (union / pairwise / sample)
  - Count: eval_miou.compute_count_metrics 그대로
      USA = CR_star  (N_gt>=2 버킷에서만 정의, b 는 버킷 내 mean 1/N_gt)
      OSA = SingleAcc (N_gt==1 버킷에서만 정의)
      CountF1 은 USA·OSA 를 모두 갖는 ALL 행에서만 정의
  - GT 매칭/파싱도 eval_miou 와 동일 → ALL 행은 기존 *_miou_summary.json 과 일치.

사용
  python3 ngt_breakdown.py NAME=<results.json|dir> [NAME=...] [--csv out.csv] [--json out.json]
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import eval_miou as em  # noqa: E402

BUCKETS = ["1", "2", "3", "4+", "ALL"]


def _bucket(n):
    return "4+" if n >= 4 else str(n)


def load_samples(results_file, natural=False):
    args = argparse.Namespace(gt_ref=False, test_json=None, test_dir=em._default_test_dir(),
                              allow_positional=False, max_time=em.MAX_TIME_DEFAULT)
    results = em._load_json(results_file)
    gt_list, src, matched, unmatched = em.build_gt_for_results(results, results_file, args)
    samples = [(gt, em.parse_pred(r.get("pred", ""), natural, em.MAX_TIME_DEFAULT))
               for r, gt in zip(results, gt_list)]
    return samples, src, unmatched


def breakdown(samples):
    groups = {k: [] for k in BUCKETS}
    for gt, pred in samples:
        if not gt:
            continue
        groups[_bucket(len(gt))].append((gt, pred))
        groups["ALL"].append((gt, pred))
    rows = {}
    for k, ss in groups.items():
        if not ss:
            rows[k] = {"n": 0}
            continue
        cm = em.compute_count_metrics(ss)
        rows[k] = {
            "n": len(ss),
            "mIoU_union": em.compute_method_block(ss, "union")["mIoU_%"],
            "mIoU_pairwise": em.compute_method_block(ss, "pairwise")["mIoU_%"],
            "mIoU_sample": em.compute_method_block(ss, "sample")["mIoU_%"],
            "mean_N_pred": round(sum(len(p) for _, p in ss) / len(ss), 3),
            "exact_count_%": round(100.0 * sum(len(g) == len(p) for g, p in ss) / len(ss), 2),
            "USA": cm["CR_star"], "OSA": cm["SingleAcc"], "CountF1": cm["CountF1"],
            "CR": cm["CR_multi"], "b": cm["chance_floor"],
        }
    return rows


def _f(x, pct=False):
    if x is None:
        return "-"
    return f"{100 * x:.2f}" if pct else (f"{x:.2f}" if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+", metavar="NAME=PATH")
    ap.add_argument("--natural", action="store_true")
    ap.add_argument("--csv")
    ap.add_argument("--json")
    a = ap.parse_args()

    out = {}
    for spec in a.preds:
        name, path = spec.split("=", 1)
        if os.path.isdir(path):
            path = os.path.join(path, "test_results_rank0.json")
        samples, src, unmatched = load_samples(path, a.natural)
        out[name] = {"path": path, "gt_source": src, "gt_unmatched": unmatched,
                     "rows": breakdown(samples)}

    cols = ["N_gt", "n", "mIoU(U)", "mIoU(P)", "mIoU(S)", "N_pred", "exact%",
            "USA", "OSA", "CountF1", "CR", "b"]
    for name, r in out.items():
        print(f"\n[{name}]  gt={os.path.basename(str(r['gt_source']))} "
              f"unmatched={r['gt_unmatched']}")
        print("  ".join(c.rjust(8) for c in cols))
        for k in BUCKETS:
            x = r["rows"][k]
            if not x["n"]:
                print(k.rjust(8) + "       0"); continue
            cells = [k, str(x["n"]), _f(x["mIoU_union"]), _f(x["mIoU_pairwise"]),
                     _f(x["mIoU_sample"]), _f(x["mean_N_pred"]), _f(x["exact_count_%"]),
                     _f(x["USA"], 1), _f(x["OSA"], 1), _f(x["CountF1"], 1),
                     _f(x["CR"], 1), _f(x["b"], 1)]
            print("  ".join(c.rjust(8) for c in cells))

    if a.csv:
        keys = ["n", "mIoU_union", "mIoU_pairwise", "mIoU_sample", "mean_N_pred",
                "exact_count_%", "USA", "OSA", "CountF1", "CR", "b"]
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "N_gt"] + keys)
            for name, r in out.items():
                for k in BUCKETS:
                    w.writerow([name, k] + [r["rows"][k].get(c) for c in keys])
        print(f"\n[SAVED] {a.csv}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"[SAVED] {a.json}")


if __name__ == "__main__":
    main()
