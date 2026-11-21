#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
error_analysis_ck2000.py — ckpt2000 / UnAV-100 test 저장 예측으로 오류 유형·세분 지표·paired bootstrap.

파싱/GT/기존 지표는 count_metrics_ck2000.py · eval_miou.py 함수를 import 해서 그대로 쓴다
(load_samples, miou_f1, count_f1, count_errors, compute_method_block, sample_iou_parts, ...).
GT 없는 샘플 제외, 파싱 실패 = 빈 예측(N_pred=0) 도 기존과 동일.

 1. 오류 유형 (겹침 = 교집합 길이 > 0, 원본 segment 그대로)
    merge   : GT>=2개와 겹치는 pred segment 가 있는 샘플 비율 / 그런 pred 수 ÷ 전체 pred 수
    split   : pred>=2개와 겹치는 GT segment 가 있는 샘플 비율
    miss    : 어떤 pred 와도 안 겹치는 GT 수 ÷ 전체 GT 수
    spurious: 어떤 GT 와도 안 겹치는 pred 수 ÷ 전체 pred 수
    (N_gt>=2 전부, N_gt==1 은 split·spurious)
 2. GT 간격 3분위 (N_gt>=2): start 정렬 후 min_i (s_{i+1} - e_i) (1e-6 반올림). 경계 q1,q2 =
    np.quantile(.,[1/3,2/3]); short <= q1 < mid <= q2 < long.
    그룹은 TiTok(ref 토큰) GT 로 한 번 정하고 9개 설정에 같은 샘플 인덱스로 적용
    (w/o special tokens 의 natural GT 는 최대 0.05s 차이; 지표 계산은 각 설정 자신의 GT).
 3. N_gt = 2, 3, 4, >=5 세분
 4. union temporal Prec/Rec = eval_miou.sample_iou_parts (빈 예측 → 0), 길이비 = |∪pred|/|∪GT|
 5. pairwise P_k / R_k (compute_method_block "pairwise")
 6. joint success = 1[N_pred==N_gt and sample IoU >= t]
 7. paired bootstrap (샘플 복원추출, 9개 설정 공통 인덱스, corpus 지표 매 반복 재계산)

단위: mIoU·F1·P·R·Prec·Rec·joint·collapse·merge 등 비율 = %, USA·OSA·CountF1 = 0~1 (기존 표 관례).
출력: <OUT>/ea{1..7}_*.csv, .tex
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from count_metrics_ck2000 import (  # noqa: E402
    B_EXPECTED, F1_KEYS, G, RUNS, count_errors, count_f1, fmt, fmt_signed, load_samples, miou_f1)
from eval_miou import best_iou, compute_method_block, merge_intervals, sample_iou_parts, total_len  # noqa: E402

KS = ["0.3", "0.5", "0.7"]
TS = [0.5, 0.7]


# ============================== 샘플별 통계 ==============================
def _ov(a, b):
    return min(a[1], b[1]) - max(a[0], b[0]) > 0


def seg_errors(gt, pred):
    gt_k = [sum(1 for p in pred if _ov(g, p)) for g in gt]   # GT 별 겹치는 pred 수
    pr_k = [sum(1 for g in gt if _ov(g, p)) for p in pred]   # pred 별 겹치는 GT 수
    return {"merge_segs": sum(1 for k in pr_k if k >= 2), "split_segs": sum(1 for k in gt_k if k >= 2),
            "miss_segs": sum(1 for k in gt_k if k == 0), "spur_segs": sum(1 for k in pr_k if k == 0)}


def min_gap(gt):
    s = sorted(gt)
    return round(min(s[i + 1][0] - s[i][1] for i in range(len(s) - 1)), 6)  # 0.1s 토큰의 float 잡음 제거


def per_sample(gt, pred):
    iou, rec, prc = sample_iou_parts(gt, pred)
    ng, npd = len(gt), len(pred)
    d = {"ngt": ng, "npred": npd, "iou": iou, "urec": rec, "uprec": prc,
         "len_ratio": total_len(merge_intervals(pred)) / total_len(merge_intervals(gt)),
         **seg_errors(gt, pred)}
    gi = [best_iou(g, pred) for g in gt]
    pi = [best_iou(p, gt) for p in pred]
    for k in KS:
        d[f"gt_hit{k}"] = sum(1 for x in gi if x >= float(k))
        d[f"pred_hit{k}"] = sum(1 for x in pi if x >= float(k))
    for t in TS:
        d[f"joint{t}"] = 1.0 if (npd == ng and iou >= t) else 0.0
    return d


def _m(xs):
    return sum(xs) / len(xs) if xs else None


def _ratio(a, b):
    return a / b if b else None


def _pct(x):
    return None if x is None else 100.0 * x


# ============================== bootstrap 지표 (가중합) ==============================
BOOT_METRICS = ["mIoU", "F1@0.3", "F1@0.5", "F1@0.7", "USA", "OSA_le1", "CountF1_le1",
                "collapse", "Prec_union", "Rec_union", "joint@0.5", "joint@0.7", "merge_sample"]


def vectors(ps):
    a = {k: np.array([p[k] for p in ps], dtype=float) for k in ps[0]}
    multi = (a["ngt"] >= 2).astype(float)
    single = (a["ngt"] == 1).astype(float)
    v = {"one": np.ones(len(ps)), "iou": a["iou"], "ngt": a["ngt"], "npred": a["npred"],
         "multi": multi, "single": single,
         "cr": multi * np.minimum(a["ngt"], a["npred"]) / np.maximum(a["ngt"], a["npred"]),
         "invg": multi / a["ngt"], "le1": single * (a["npred"] <= 1),
         "coll": multi * (a["npred"] == 1), "uprec": a["uprec"], "urec": a["urec"],
         "mergeS": multi * (a["merge_segs"] > 0)}
    for k in KS:
        v[f"gh{k}"], v[f"ph{k}"] = a[f"gt_hit{k}"], a[f"pred_hit{k}"]
    for t in TS:
        v[f"j{t}"] = a[f"joint{t}"]
    return v


def boot_metrics(W, v):
    """W: (B, n) 재표본 가중치(복원추출 횟수). 반환 {metric: (B,)}; W=ones(1,n) 이면 점추정."""
    S = {k: W @ x for k, x in v.items()}
    out = {"mIoU": 100 * S["iou"] / S["one"]}
    for k in KS:
        r, p = S[f"gh{k}"] / S["ngt"], S[f"ph{k}"] / S["npred"]
        out[f"F1@{k}"] = 100 * np.where(r + p > 0, 2 * r * p / np.maximum(r + p, 1e-300), 0.0)
    cr, b = S["cr"] / S["multi"], S["invg"] / S["multi"]
    usa = np.maximum(0.0, (cr - b) / (1 - b))
    osa = S["le1"] / S["single"]
    out.update({"USA": usa, "OSA_le1": osa,
                "CountF1_le1": np.where(usa + osa > 0, 2 * usa * osa / np.maximum(usa + osa, 1e-300), 0.0),
                "collapse": 100 * S["coll"] / S["multi"],
                "Prec_union": 100 * S["uprec"] / S["one"], "Rec_union": 100 * S["urec"] / S["one"],
                "joint@0.5": 100 * S["j0.5"] / S["one"], "joint@0.7": 100 * S["j0.7"] / S["one"],
                "merge_sample": 100 * S["mergeS"] / S["multi"]})
    return out


# ============================== 출력 도우미 ==============================
def write_csv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_tex(path, comments, colspec, header, body):
    L = [f"% {c}" for c in comments] + [r"\begin{tabular}{" + colspec + "}", r"\toprule"]
    L += header + [r"\midrule"] + body + [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def row_tex(cells):
    return " & ".join(cells) + r" \\"


def grouped_body(rows, key_col, label_fn, cells_fn, names):
    """설정별 multirow 블록."""
    body = []
    for i, nm in enumerate(names):
        rs = [r for r in rows if r["setting"] == nm]
        for j, r in enumerate(rs):
            first = (r"\multirow{%d}{*}{%s}" % (len(rs), r["_tex"])) if j == 0 else ""
            body.append(row_tex([first, label_fn(r[key_col])] + cells_fn(r)))
        if i < len(names) - 1:
            body.append(r"\midrule")
    return body


# ============================== main ==============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{G}/error_analysis_ck2000")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=500)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    data, checks, missing = [], [], []
    for name, tex, d, ts, nat, ref_miou, ref_f1, ref_cf1 in RUNS:
        samples, gt_src = load_samples(d, ts, nat)
        if samples is None:
            missing.append(name)
            print(f"[MISSING] {name}: {d}")
            continue
        samples = [(gt, pr) for gt, pr in samples if gt]
        pairs = [(len(gt), len(pr)) for gt, pr in samples]
        mf, cf = miou_f1(samples), count_f1(pairs)
        items = [("mIoU", mf["mIoU"], ref_miou, 1)]
        items += [(k, mf[k], r, 1) for k, r in zip(F1_KEYS, ref_f1)]
        items += [("CountF1_le1", cf["CountF1_le1"], ref_cf1, 2)]
        checks.append((name, [(k, v, r, round(v, nd) == r) for k, v, r, nd in items], cf["b"]))
        ps = [per_sample(gt, pr) for gt, pr in samples]
        data.append({"name": name, "tex": tex, "samples": samples, "pairs": pairs, "ps": ps,
                     "mf": mf, "cf": cf})
    if missing:
        sys.exit(f"누락 설정 {missing}: 9개 설정이 모두 있어야 함")
    names = [x["name"] for x in data]
    n = len(data[0]["samples"])

    # ---------------- 검증 ----------------
    print("=== 검증 1: 기존 표 대조 (count_metrics_ck2000 과 동일) ===")
    b_ok, mism = True, []
    for name, items, b in checks:
        b_ok &= round(b, 3) == B_EXPECTED
        print(f"{name:20s} b={b:.6f}  " + "  ".join(
            f"{k} {v:.4f}/{r}{'' if ok else ' MISMATCH'}" for k, v, r, ok in items))
        mism += [(name, k, v, r) for k, v, r, ok in items if not ok]
    print(f"b == {B_EXPECTED} (3자리): {'OK' if b_ok else 'MISMATCH'}")
    print("불일치: " + ("없음" if not mism else
          "; ".join(f"{nm} {k} {v:.4f} vs {r}" for nm, k, v, r in mism)))

    print("\n=== 검증 2: bootstrap 가중합 공식(W=1) vs 기존 함수 ===")
    ok2 = True
    for x in data:
        pt = {k: float(v[0]) for k, v in boot_metrics(np.ones((1, n)), vectors(x["ps"])).items()}
        ref = {"mIoU": x["mf"]["mIoU"], **{k: x["mf"][k] for k in F1_KEYS},
               "USA": x["cf"]["USA"], "OSA_le1": x["cf"]["OSA_le1"], "CountF1_le1": x["cf"]["CountF1_le1"]}
        # compute_method_block 는 4자리 반올림 → 허용오차 1e-4 (%), count 지표는 1e-12
        bad = [(k, pt[k], r) for k, r in ref.items()
               if abs(pt[k] - r) > (1e-12 if k in ("USA", "OSA_le1", "CountF1_le1") else 5e-5 + 1e-9)]
        ok2 &= not bad
        print(f"{x['name']:20s} " + ("OK" if not bad else f"MISMATCH {bad}"))
        x["pt"] = pt
    print(f"전체: {'OK' if ok2 else 'MISMATCH'}")

    print("\n=== 검증 3: 샘플 정렬 (N_gt 동일, GT 좌표 차 최대) ===")
    ref_gt = [gt for gt, _ in data[0]["samples"]]
    for x in data[1:]:
        ngt_same = all(len(a) == len(g) for a, (g, _) in zip(ref_gt, x["samples"]))
        dmax = max(abs(u - w) for a, (g, _) in zip(ref_gt, x["samples"])
                   for sa, sg in zip(a, g) for u, w in zip(sa, sg))
        print(f"{x['name']:20s} n={len(x['samples'])} N_gt 동일={ngt_same} max|ΔGT|={dmax:.5f}s")
        assert len(x["samples"]) == n and ngt_same

    # ================= 1. 오류 유형 =================
    r1 = []
    for x in data:
        ps = x["ps"]
        M = [p for p in ps if p["ngt"] >= 2]
        S1 = [p for p in ps if p["ngt"] == 1]
        npM, ngM, npS = sum(p["npred"] for p in M), sum(p["ngt"] for p in M), sum(p["npred"] for p in S1)
        r1.append({
            "setting": x["name"], "_tex": x["tex"],
            "n_multi": len(M), "n_pred_multi": npM, "n_gt_seg_multi": ngM,
            "merge_sample": _pct(_m([1.0 if p["merge_segs"] else 0.0 for p in M])),
            "merge_segs": sum(p["merge_segs"] for p in M),
            "merge_seg_ratio": _pct(_ratio(sum(p["merge_segs"] for p in M), npM)),
            "split_sample": _pct(_m([1.0 if p["split_segs"] else 0.0 for p in M])),
            "miss_segs": sum(p["miss_segs"] for p in M),
            "miss_ratio": _pct(_ratio(sum(p["miss_segs"] for p in M), ngM)),
            "spur_segs": sum(p["spur_segs"] for p in M),
            "spur_ratio": _pct(_ratio(sum(p["spur_segs"] for p in M), npM)),
            "n_single": len(S1), "n_pred_single": npS,
            "split_sample_single": _pct(_m([1.0 if p["split_segs"] else 0.0 for p in S1])),
            "spur_segs_single": sum(p["spur_segs"] for p in S1),
            "spur_ratio_single": _pct(_ratio(sum(p["spur_segs"] for p in S1), npS)),
        })
    c1 = ["setting", "n_multi", "n_pred_multi", "n_gt_seg_multi", "merge_sample", "merge_segs",
          "merge_seg_ratio", "split_sample", "miss_segs", "miss_ratio", "spur_segs", "spur_ratio",
          "n_single", "n_pred_single", "split_sample_single", "spur_segs_single", "spur_ratio_single"]
    write_csv(os.path.join(args.out, "ea1_error_types.csv"), r1, c1)
    write_tex(os.path.join(args.out, "ea1_error_types.tex"),
              ["ckpt2000, UnAV-100 test. Overlap = intersection length > 0. All values in %.",
               "Merge (sample) = P(some pred overlaps >=2 GT); Merge (seg) = such preds / all preds.",
               "Split = P(some GT overlaps >=2 preds); Miss = GT overlapping no pred / all GT;",
               "Spurious = preds overlapping no GT / all preds."],
              "l" + "c" * 8,
              [r" & \multicolumn{6}{c}{$N_{gt}\geq 2$} & \multicolumn{2}{c}{$N_{gt}=1$} \\",
               r"\cmidrule(lr){2-7}\cmidrule(lr){8-9}",
               r"Setting & Merge (sample) & Merge (seg) & Split (sample) & Miss & Spurious & \#Pred"
               r" & Split (sample) & Spurious \\"],
              [row_tex([r["_tex"], fmt(r["merge_sample"], 1), fmt(r["merge_seg_ratio"], 1),
                        fmt(r["split_sample"], 1), fmt(r["miss_ratio"], 1), fmt(r["spur_ratio"], 1),
                        str(r["n_pred_multi"]), fmt(r["split_sample_single"], 1),
                        fmt(r["spur_ratio_single"], 1)]) for r in r1])

    # ================= 2. GT 간격 3분위 =================
    multi_idx = [i for i, g in enumerate(ref_gt) if len(g) >= 2]
    gaps = {i: min_gap(ref_gt[i]) for i in multi_idx}
    q1, q2 = (float(q) for q in np.quantile([gaps[i] for i in multi_idx], [1 / 3, 2 / 3]))
    grp = {i: ("short" if gaps[i] <= q1 else "mid" if gaps[i] <= q2 else "long") for i in multi_idx}
    GROUPS = ["short", "mid", "long"]
    gstat = {gname: [gaps[i] for i in multi_idx if grp[i] == gname] for gname in GROUPS}
    print(f"\n[GT 간격] q1={q1:.4f}s q2={q2:.4f}s  " + "  ".join(
        f"{gname}: n={len(v)} [{min(v):.2f},{max(v):.2f}]" for gname, v in gstat.items())
        + f"  (min gap<=0 샘플 {sum(1 for i in multi_idx if gaps[i] <= 0)})")
    r2 = []
    for x in data:
        for gname in GROUPS:
            idx = [i for i in multi_idx if grp[i] == gname]
            sub = [x["samples"][i] for i in idx]
            ps = [x["ps"][i] for i in idx]
            pw = compute_method_block(sub, "pairwise")
            r2.append({"setting": x["name"], "_tex": x["tex"], "group": gname,
                       "q1_s": q1, "q2_s": q2, "gap_min_s": min(gstat[gname]), "gap_max_s": max(gstat[gname]),
                       "n": len(sub), "mIoU": compute_method_block(sub, "sample")["mIoU_%"],
                       "F1@0.5": pw["F1"]["0.5"], "F1@0.7": pw["F1"]["0.7"],
                       "exact": _pct(_m([1.0 if p["npred"] == p["ngt"] else 0.0 for p in ps])),
                       "collapse": _pct(_m([1.0 if p["npred"] == 1 else 0.0 for p in ps])),
                       "merge_sample": _pct(_m([1.0 if p["merge_segs"] else 0.0 for p in ps]))})
    c2 = ["setting", "group", "q1_s", "q2_s", "gap_min_s", "gap_max_s", "n", "mIoU", "F1@0.5", "F1@0.7",
          "exact", "collapse", "merge_sample"]
    write_csv(os.path.join(args.out, "ea2_gt_gap.csv"), r2, c2)
    glab = {g: f"{g} (n={len(gstat[g])})" for g in GROUPS}
    write_tex(os.path.join(args.out, "ea2_gt_gap.tex"),
              ["ckpt2000, UnAV-100 test, N_gt>=2. Per-sample min gap between adjacent GT segments (s).",
               f"Tertile boundaries: q1={q1:.2f}s, q2={q2:.2f}s; short<=q1<mid<=q2<long. "
               + ", ".join(f"{g}: n={len(gstat[g])}, gap in [{min(gstat[g]):.2f},{max(gstat[g]):.2f}]"
                           for g in GROUPS) + ".",
               "Groups from TiTok (token) GT, same samples for all settings. Exact/Collapse/Merge in %."],
              "llcccccc",
              [r"Setting & Gap & mIoU & F1@0.5 & F1@0.7 & Exact acc. & Collapse & Merge (sample) \\"],
              grouped_body(r2, "group", lambda g: glab[g],
                           lambda r: [fmt(r["mIoU"], 1), fmt(r["F1@0.5"], 1), fmt(r["F1@0.7"], 1),
                                      fmt(r["exact"], 1), fmt(r["collapse"], 1), fmt(r["merge_sample"], 1)],
                           names))

    # ================= 3. N_gt 세분 =================
    NB = [("2", lambda g: g == 2), ("3", lambda g: g == 3), ("4", lambda g: g == 4), (">=5", lambda g: g >= 5)]
    r3 = []
    for x in data:
        for bk, f in NB:
            idx = [i for i, (g, _) in enumerate(x["pairs"]) if f(g)]
            sub = [x["samples"][i] for i in idx]
            pr = [x["pairs"][i] for i in idx]
            pw = compute_method_block(sub, "pairwise")
            e = count_errors(pr)
            r3.append({"setting": x["name"], "_tex": x["tex"], "N_gt": bk, "n": len(sub),
                       "mIoU": compute_method_block(sub, "sample")["mIoU_%"],
                       "F1@0.5": pw["F1"]["0.5"], "F1@0.7": pw["F1"]["0.7"],
                       "MAE": e["MAE"], "signed": e["signed"], "exact": _pct(e["exact"]),
                       "collapse": _pct(_m([1.0 if p == 1 else 0.0 for _, p in pr]))})
    c3 = ["setting", "N_gt", "n", "mIoU", "F1@0.5", "F1@0.7", "MAE", "signed", "exact", "collapse"]
    write_csv(os.path.join(args.out, "ea3_ngt_fine.csv"), r3, c3)
    write_tex(os.path.join(args.out, "ea3_ngt_fine.tex"),
              ["ckpt2000, UnAV-100 test, N_gt = 2 / 3 / 4 / >=5. mIoU = sample, F1 = pairwise.",
               "Exact acc. and Collapse (P(N_pred=1)) in %."],
              "llcccccccc",
              [r"Setting & $N_{gt}$ & \#Samples & mIoU & F1@0.5 & F1@0.7 & Count MAE & Signed err."
               r" & Exact acc. & Collapse \\"],
              grouped_body(r3, "N_gt", lambda b: r"$\geq$5" if b == ">=5" else b,
                           lambda r: [str(r["n"]), fmt(r["mIoU"], 1), fmt(r["F1@0.5"], 1), fmt(r["F1@0.7"], 1),
                                      fmt(r["MAE"], 2), fmt_signed(r["signed"]), fmt(r["exact"], 1),
                                      fmt(r["collapse"], 1)], names))

    # ================= 4. union Prec/Rec =================
    SUBS = [("all", lambda g: True), ("1", lambda g: g == 1), (">=2", lambda g: g >= 2)]
    r4 = []
    for x in data:
        for sk, f in SUBS:
            ps = [p for p in x["ps"] if f(p["ngt"])]
            lr = [p["len_ratio"] for p in ps]
            r4.append({"setting": x["name"], "_tex": x["tex"], "subset": sk, "n": len(ps),
                       "n_empty_pred": sum(1 for p in ps if p["npred"] == 0),
                       "Prec": _pct(_m([p["uprec"] for p in ps])), "Rec": _pct(_m([p["urec"] for p in ps])),
                       "len_ratio_median": float(np.median(lr)), "len_ratio_mean": _m(lr)})
    c4 = ["setting", "subset", "n", "n_empty_pred", "Prec", "Rec", "len_ratio_median", "len_ratio_mean"]
    write_csv(os.path.join(args.out, "ea4_union_prec_rec.csv"), r4, c4)
    slab = {"all": "All", "1": r"$N_{gt}=1$", ">=2": r"$N_{gt}\geq 2$"}
    write_tex(os.path.join(args.out, "ea4_union_prec_rec.tex"),
              ["ckpt2000, UnAV-100 test. Union-level temporal Prec = |U_pred ∩ U_gt| / |U_pred|,",
               "Rec = |U_pred ∩ U_gt| / |U_gt|, per-sample mean, in %. Empty prediction -> Prec = Rec = 0.",
               "Len. ratio = |U_pred| / |U_gt| (empty prediction -> 0)."],
              "llcccccc",
              [r"Setting & Subset & \#Samples & \#Empty pred & Prec & Rec & Len. ratio (median)"
               r" & Len. ratio (mean) \\"],
              grouped_body(r4, "subset", lambda s: slab[s],
                           lambda r: [str(r["n"]), str(r["n_empty_pred"]), fmt(r["Prec"], 1), fmt(r["Rec"], 1),
                                      fmt(r["len_ratio_median"], 2), fmt(r["len_ratio_mean"], 2)], names))

    # ================= 5. pairwise P_k / R_k =================
    r5 = []
    for x in data:
        for sk, f in SUBS:
            sub = [s for s, (g, _) in zip(x["samples"], x["pairs"]) if f(g)]
            pw = compute_method_block(sub, "pairwise")
            row = {"setting": x["name"], "_tex": x["tex"], "subset": sk, "n": len(sub)}
            for k in KS:
                row[f"P@{k}"], row[f"R@{k}"], row[f"F1@{k}"] = pw["Precision"][k], pw["Recall"][k], pw["F1"][k]
            r5.append(row)
    c5 = ["setting", "subset", "n"] + [f"{m}@{k}" for k in KS for m in ("P", "R", "F1")]
    write_csv(os.path.join(args.out, "ea5_pairwise_pr.csv"), r5, c5)
    write_tex(os.path.join(args.out, "ea5_pairwise_pr.tex"),
              ["ckpt2000, UnAV-100 test. Pairwise (best-IoU) segment-level P_k / R_k / F1_k from eval_miou, in %."],
              "llc" + "c" * 9,
              [r" & & & \multicolumn{3}{c}{$k=0.3$} & \multicolumn{3}{c}{$k=0.5$} & \multicolumn{3}{c}{$k=0.7$} \\",
               r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}",
               r"Setting & Subset & \#Samples & P & R & F1 & P & R & F1 & P & R & F1 \\"],
              grouped_body(r5, "subset", lambda s: slab[s],
                           lambda r: [str(r["n"])] + [fmt(r[f"{m}@{k}"], 1) for k in KS for m in ("P", "R", "F1")],
                           names))

    # ================= 6. joint success =================
    JS = [("all", lambda g: True), ("1", lambda g: g == 1), ("2", lambda g: g == 2), (">=3", lambda g: g >= 3)]
    r6 = []
    for x in data:
        for sk, f in JS:
            ps = [p for p in x["ps"] if f(p["ngt"])]
            r6.append({"setting": x["name"], "_tex": x["tex"], "subset": sk, "n": len(ps),
                       **{f"joint@{t}": _pct(_m([p[f"joint{t}"] for p in ps])) for t in TS}})
    write_csv(os.path.join(args.out, "ea6_joint_success.csv"), r6,
              ["setting", "subset", "n"] + [f"joint@{t}" for t in TS])
    jl = {"all": "All", "1": "1", "2": "2", ">=3": r"$\geq$3"}
    write_tex(os.path.join(args.out, "ea6_joint_success.tex"),
              ["ckpt2000, UnAV-100 test. Joint success = P(N_pred = N_gt and sample IoU >= t), in %."],
              "l" + "c" * 8,
              [r" & \multicolumn{4}{c}{$t=0.5$} & \multicolumn{4}{c}{$t=0.7$} \\",
               r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
               r"Setting & " + " & ".join([jl[s] if s == "all" else f"$N_{{gt}}={jl[s]}$" if s != ">=3"
                                         else r"$N_{gt}\geq 3$" for s, _ in JS] * 2) + r" \\"],
              [row_tex([x["tex"]] + [fmt(next(r for r in r6 if r["setting"] == x["name"] and r["subset"] == s)
                                         [f"joint@{t}"], 1) for t in TS for s, _ in JS]) for x in data])

    # ================= 7. paired bootstrap =================
    rng = np.random.default_rng(args.seed)
    vecs = [vectors(x["ps"]) for x in data]
    acc = [{m: [] for m in BOOT_METRICS} for _ in data]
    done = 0
    while done < args.n_boot:
        c = min(args.chunk, args.n_boot - done)
        idx = rng.integers(0, n, size=(c, n))
        W = np.bincount((idx + n * np.arange(c)[:, None]).ravel(), minlength=c * n).reshape(c, n).astype(float)
        for si, v in enumerate(vecs):
            bm = boot_metrics(W, v)
            for m in BOOT_METRICS:
                acc[si][m].append(bm[m])
        done += c
    boot = [{m: np.concatenate(a[m]) for m in BOOT_METRICS} for a in acc]
    pts = [{k: float(v[0]) for k, v in boot_metrics(np.ones((1, n)), vv).items()} for vv in vecs]
    r7 = []
    for si, x in enumerate(data):
        for m in BOOT_METRICS:
            dlt = boot[si][m] - boot[0][m]
            lo, hi = (float(q) for q in np.percentile(dlt, [2.5, 97.5]))
            r7.append({"setting": x["name"], "_tex": x["tex"], "metric": m,
                       "value": pts[si][m], "value_TiTok": pts[0][m], "diff_point": pts[si][m] - pts[0][m],
                       "diff_boot_mean": float(dlt.mean()), "ci_lo": lo, "ci_hi": hi,
                       "zero_in_ci": bool(lo <= 0 <= hi)})
    write_csv(os.path.join(args.out, "ea7_bootstrap.csv"), r7,
              ["setting", "metric", "value", "value_TiTok", "diff_point", "diff_boot_mean", "ci_lo", "ci_hi",
               "zero_in_ci"])
    mlab = {"mIoU": "mIoU", "F1@0.3": "F1@0.3", "F1@0.5": "F1@0.5", "F1@0.7": "F1@0.7", "USA": "USA",
            "OSA_le1": r"OSA$_{\leq 1}$", "CountF1_le1": r"CountF1$_{\leq 1}$", "collapse": "Collapse",
            "Prec_union": "Prec (union)", "Rec_union": "Rec (union)", "joint@0.5": "Joint@0.5",
            "joint@0.7": "Joint@0.7", "merge_sample": "Merge (sample)"}
    nd = {m: (3 if m in ("USA", "OSA_le1", "CountF1_le1") else 2) for m in BOOT_METRICS}

    def cells7(r):
        k = nd[r["metric"]]
        return [fmt(r["value"], k), fmt_signed(r["diff_boot_mean"], k),
                f"[{r['ci_lo']:+.{k}f}, {r['ci_hi']:+.{k}f}]", "yes" if r["zero_in_ci"] else "no"]
    write_tex(os.path.join(args.out, "ea7_bootstrap.tex"),
              [f"ckpt2000, UnAV-100 test (n={n}). Paired bootstrap, {args.n_boot} resamples of samples with",
               f"replacement, seed={args.seed}, identical indices for all settings; corpus metrics recomputed",
               "per resample. Delta = setting - TiTok, mean over resamples; 95% percentile CI.",
               "USA / OSA / CountF1 on 0-1 scale; others in % (points)."],
              "llcccc",
              [r"Setting & Metric & Value & $\Delta$ vs TiTok & 95\% CI & 0 in CI \\"],
              grouped_body(r7, "metric", lambda m: mlab[m], cells7, names))

    print(f"\n[bootstrap] n_boot={args.n_boot} seed={args.seed} n={n}")
    print(f"[SAVED] {args.out}")


if __name__ == "__main__":
    main()
