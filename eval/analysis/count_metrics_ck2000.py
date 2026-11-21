#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
count_metrics_ck2000.py — ckpt2000 / UnAV-100 test 저장 예측으로 count 지표 재계산 (추론 없음).

파싱/GT 매칭/mIoU·F1 집계는 eval_miou.py 함수를 그대로 쓴다 (eval.sh 와 동일 설정):
  - pred : parse_pred(r["pred"], natural, MAX_TIME=9999.9)  (natural 설정만 --natural)
  - GT   : build_gt_for_results (ref(auto) > test_json=<data>/<TESTSET>/_full.json)
  - mIoU : compute_method_block(..., "sample")   (표의 mIoU)
  - F1@θ : compute_method_block(..., "pairwise") (표의 F1)
파싱 실패 응답은 기존과 동일하게 빈 예측(N_pred=0)으로 채점한다.

count 지표 (샘플별 N_gt, N_pred):
  S_multi={N_gt>=2}, S_single={N_gt==1}
  CR  = mean_{S_multi} min/max,  b = mean_{S_multi} 1/N_gt
  USA = max(0,(CR-b)/(1-b))
  OSA_le1 = mean_{S_single} 1[N_pred<=1]  (기존 eval_miou.py 정의)
  OSA_eq1 = mean_{S_single} 1[N_pred==1]  (Overleaf Eq. 11 정의)
  CountF1_le1 / CountF1_eq1 = HM(USA, OSA_*) (둘 다 0 이면 0)
  Count MAE / signed error / exact-count acc : 전체, single, multi

count 분포 지표 (count_dist_ck2000.csv / .tex):
  Collapse = P(N_pred==1 | N_gt>=2), N_gt==2, N_gt>=3 별도
  D = N_pred - N_gt 분포 (N_gt>=2): <=-3, -2, -1, 0, +1, >=+2
  N_pred 분포 (N_gt==1): 0, 1, 2, >=3

출력: <OUT>/count_summary_ck2000.csv, count_by_ngt_ck2000.csv, count_dist_ck2000.csv, *.tex
"""
import argparse
import csv
import json
import os
import sys
from types import SimpleNamespace

EV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EV)
from eval_miou import build_gt_for_results, compute_method_block, parse_pred  # noqa: E402

MAX_TIME = 9999.9  # eval.sh 기본값
WS = "/workspace"
G = f"{WS}/outputs/gdpo"
P = "sft_7b_unav_v8_rl_rMsep3_uncha_batch4_noscaling"

# (표 이름, LaTeX 이름, 결과 폴더, TESTSET, natural, 기존 mIoU, 기존 F1@0.3/0.5/0.7, 기존 CountF1)
# w/o RL mIoU 는 기존 표의 반올림 오류(56.8)를 56.7 로 정정한 값.
RUNS = [
    ("TiTok", "TiTok",
     f"{G}/{P}/checkpoint-2000/fps5_tti/unav100_titok", "unav100_titok", False, 65.7, (63.8, 50.3, 38.6), 0.58),
    ("w/o special tokens", "w/o special tokens",
     f"{G}/sft_7b_unav_v8_chronus_rl_rMsep3_uncha_batch4_natural/checkpoint-2000/fps5_natural/unav100_titok_natural",
     "unav100_titok_natural", True, 63.9, (60.6, 47.3, 35.5), 0.51),
    ("w/o audio", "w/o audio",
     f"{G}/{P}_noaudio/checkpoint-2000/fps5_tti/unav100_titok_noaudio", "unav100_titok_noaudio", False, 57.8, (53.6, 40.4, 29.4), 0.52),
    ("w/o RL (SFT only)", "w/o RL (SFT only)",
     f"{WS}/outputs/base/salmonn2p_7b_unav_v8/fps5_tti/unav100_titok", "unav100_titok", False, 56.7, (26.9, 19.1, 14.4), 0.16),
    ("with GRPO", "with GRPO",
     f"{G}/{P}_GRPO/checkpoint-2000/fps5_tti/unav100_titok", "unav100_titok", False, 64.4, (64.2, 50.1, 36.3), 0.62),
    ("w/o TTI", "w/o TTI",
     f"{G}/{P}_ttioff/checkpoint-2000/fps5_off/unav100_titok", "unav100_titok", False, 61.2, (59.8, 45.6, 32.7), 0.53),
    ("w/o r_global", r"w/o $r_{\text{global}}$",
     f"{G}/{P}_noglobal/checkpoint-2000/fps5_tti/unav100_titok", "unav100_titok", False, 61.4, (60.7, 46.4, 33.7), 0.57),
    ("w/o r_prec", r"w/o $r_{\text{prec}}$",
     f"{G}/{P}_noprecision/checkpoint-2000/fps5_tti/unav100_titok", "unav100_titok", False, 65.5, (62.9, 49.6, 36.7), 0.59),
    ("w/o r_count", r"w/o $r_{\text{count}}$",
     f"{G}/{P}_nocount/checkpoint-2000/fps5_tti/unav100_titok", "unav100_titok", False, 63.9, (64.0, 50.3, 36.9), 0.57),
]
B_EXPECTED = 0.397
BUCKETS = ["1", "2", ">=3"]


def bucket(n_gt):
    return "1" if n_gt == 1 else ("2" if n_gt == 2 else ">=3")


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def load_samples(run_dir, testset, natural):
    rf = os.path.join(run_dir, "test_results_rank0.json")
    if not os.path.isfile(rf):
        return None, None
    with open(rf) as f:
        results = json.load(f)
    args = SimpleNamespace(gt_ref=False, max_time=MAX_TIME, allow_positional=False,
                           test_json=f"{WS}/data/{testset}/_full.json", test_dir=None)
    gt_list, gt_src, _m, _u = build_gt_for_results(results, rf, args)
    samples = [(gt, parse_pred(r.get("pred", ""), natural, MAX_TIME))
               for r, gt in zip(results, gt_list)]
    return samples, gt_src


def count_errors(pairs):
    """pairs [(n_gt, n_pred)] -> MAE, signed, exact acc."""
    if not pairs:
        return {"n": 0, "MAE": None, "signed": None, "exact": None}
    return {"n": len(pairs),
            "MAE": _mean([abs(p - g) for g, p in pairs]),
            "signed": _mean([p - g for g, p in pairs]),
            "exact": _mean([1.0 if p == g else 0.0 for g, p in pairs])}


def _hm(a, b):
    return 2 * a * b / (a + b) if (a + b) > 0 else 0.0


def osa_pair(pairs):
    """S_single 에서 OSA_le1 (N_pred<=1, 기존 eval_miou) / OSA_eq1 (N_pred==1, Eq. 11)."""
    single = [p for g, p in pairs if g == 1]
    return {"OSA_le1": _mean([1.0 if p <= 1 else 0.0 for p in single]),
            "OSA_eq1": _mean([1.0 if p == 1 else 0.0 for p in single]),
            "n_single_pred0": sum(1 for p in single if p == 0)}


def count_f1(pairs):
    multi = [(g, p) for g, p in pairs if g >= 2]
    cr = _mean([min(g, p) / max(g, p) for g, p in multi])
    b = _mean([1.0 / g for g, _ in multi])
    usa = max(0.0, (cr - b) / (1.0 - b))
    o = osa_pair(pairs)
    return {"CR": cr, "b": b, "USA": usa, **o,
            "CountF1_le1": _hm(usa, o["OSA_le1"]), "CountF1_eq1": _hm(usa, o["OSA_eq1"])}


DIFF_BINS = [("d<=-3", lambda d: d <= -3), ("d=-2", lambda d: d == -2), ("d=-1", lambda d: d == -1),
             ("d=0", lambda d: d == 0), ("d=+1", lambda d: d == 1), ("d>=+2", lambda d: d >= 2)]
SINGLE_BINS = [("pred=0", lambda p: p == 0), ("pred=1", lambda p: p == 1),
               ("pred=2", lambda p: p == 2), ("pred>=3", lambda p: p >= 3)]


def count_dist(pairs):
    """Collapse rate, (N_pred-N_gt) 분포 (N_gt>=2), N_pred 분포 (N_gt==1). 비율은 0~1."""
    multi = [(g, p) for g, p in pairs if g >= 2]
    m2 = [(g, p) for g, p in multi if g == 2]
    m3 = [(g, p) for g, p in multi if g >= 3]
    single = [p for g, p in pairs if g == 1]
    out = {"n_multi": len(multi), "n_gt2": len(m2), "n_gt3p": len(m3), "n_single": len(single)}
    for k, sub in [("collapse_multi", multi), ("collapse_gt2", m2), ("collapse_gt3p", m3)]:
        out[k] = _mean([1.0 if p == 1 else 0.0 for _, p in sub])
    for k, f in DIFF_BINS:
        out[k] = _mean([1.0 if f(p - g) else 0.0 for g, p in multi])
    for k, f in SINGLE_BINS:
        out[k] = _mean([1.0 if f(p) else 0.0 for p in single])
    return out


def miou_f1(samples):
    s = compute_method_block(samples, "sample")
    p = compute_method_block(samples, "pairwise")
    return {"mIoU": s["mIoU_%"], "F1@0.3": p["F1"]["0.3"],
            "F1@0.5": p["F1"]["0.5"], "F1@0.7": p["F1"]["0.7"]}


def fmt(x, nd):
    return "--" if x is None else f"{x:.{nd}f}"


def fmt_signed(x, nd=2):
    return "--" if x is None else f"{x:+.{nd}f}"


F1_KEYS = ["F1@0.3", "F1@0.5", "F1@0.7"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{G}/count_analysis_ck2000")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    summary, by_ngt, dist, checks, missing = [], [], [], [], []
    for name, tex, d, ts, nat, ref_miou, ref_f1, ref_cf1 in RUNS:
        samples, gt_src = load_samples(d, ts, nat)
        if samples is None:
            missing.append((name, d))
            print(f"[MISSING] {name}: {d}")
            continue
        n_unmatched = sum(1 for gt, _ in samples if not gt)
        samples = [(gt, pr) for gt, pr in samples if gt]  # eval_miou 와 동일하게 GT 없는 샘플 제외
        pairs = [(len(gt), len(pr)) for gt, pr in samples]
        n_parse_fail = sum(1 for _, p in pairs if p == 0)

        mf = miou_f1(samples)
        cf = count_f1(pairs)
        dist.append({"setting": name, **count_dist(pairs), "_tex": tex})
        e_all = count_errors(pairs)
        e_single = count_errors([x for x in pairs if x[0] == 1])
        e_multi = count_errors([x for x in pairs if x[0] >= 2])
        summary.append({
            "setting": name, "n": len(pairs), "n_gt_unmatched": n_unmatched,
            "n_parse_fail": n_parse_fail, "gt_source": gt_src, "natural_parse": nat,
            **mf, **cf,
            "MAE_all": e_all["MAE"], "MAE_single": e_single["MAE"], "MAE_multi": e_multi["MAE"],
            "signed_all": e_all["signed"], "signed_single": e_single["signed"],
            "signed_multi": e_multi["signed"],
            "exact_all": e_all["exact"], "exact_single": e_single["exact"],
            "exact_multi": e_multi["exact"],
            "n_single": e_single["n"], "n_multi": e_multi["n"], "_tex": tex,
        })
        for bk in BUCKETS:
            sub = [(gt, pr) for gt, pr in samples if bucket(len(gt)) == bk]
            sp = [(len(gt), len(pr)) for gt, pr in sub]
            e = count_errors(sp)
            o = osa_pair(sp) if bk == "1" else {"OSA_le1": None, "OSA_eq1": None}
            by_ngt.append({"setting": name, "N_gt": bk, "n": len(sub), **miou_f1(sub),
                           "MAE": e["MAE"], "signed": e["signed"], "exact": e["exact"],
                           "OSA_le1": o["OSA_le1"], "OSA_eq1": o["OSA_eq1"], "_tex": tex})

        # 검증: 표 반올림 수준 일치 (mIoU/F1 1자리, CountF1_le1 2자리)
        items = [("mIoU", mf["mIoU"], ref_miou, 1)]
        items += [(k, mf[k], r, 1) for k, r in zip(F1_KEYS, ref_f1)]
        items += [("CountF1_le1", cf["CountF1_le1"], ref_cf1, 2)]
        checks.append((name, [(k, v, r, round(v, nd) == r) for k, v, r, nd in items], cf["b"]))

    # ---------------- 검증 리포트 ----------------
    print("\n=== 검증 ===")
    b_ok, mism = True, []
    for name, items, b in checks:
        b_ok &= round(b, 3) == B_EXPECTED
        print(f"{name:20s} b={b:.6f}  " + "  ".join(
            f"{k} {v:.4f}/{r}{'' if ok else ' MISMATCH'}" for k, v, r, ok in items))
        mism += [(name, k, v, r) for k, v, r, ok in items if not ok]
    print(f"b == {B_EXPECTED} (3자리): {'OK' if b_ok else 'MISMATCH'}")
    print("불일치: " + ("없음" if not mism else
          "; ".join(f"{n} {k} {v:.4f} vs {r} (d={v - r:+.4f})" for n, k, v, r in mism)))
    if missing:
        print(f"누락 설정: {[n for n, _ in missing]}")

    # ---------------- CSV ----------------
    s_cols = ["setting", "n", "n_single", "n_multi", "n_parse_fail", "n_single_pred0",
              "n_gt_unmatched", "natural_parse", "gt_source",
              "mIoU", "F1@0.3", "F1@0.5", "F1@0.7",
              "CR", "b", "USA", "OSA_le1", "OSA_eq1", "CountF1_le1", "CountF1_eq1",
              "MAE_all", "MAE_single", "MAE_multi",
              "signed_all", "signed_single", "signed_multi",
              "exact_all", "exact_single", "exact_multi"]
    with open(os.path.join(args.out, "count_summary_ck2000.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=s_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary)
    g_cols = ["setting", "N_gt", "n", "mIoU", "F1@0.3", "F1@0.5", "F1@0.7",
              "MAE", "signed", "exact", "OSA_le1", "OSA_eq1"]
    with open(os.path.join(args.out, "count_by_ngt_ck2000.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=g_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(by_ngt)

    # ---------------- LaTeX ----------------
    L = [r"% ckpt2000, UnAV-100 test (n=3455). mIoU = sample mIoU, F1 = pairwise.",
         r"% OSA_le1 = P(N_pred<=1 | N_gt=1) (eval_miou.py), OSA_eq1 = P(N_pred==1 | N_gt=1) (Eq. 11).",
         r"% #0 = number of samples with N_gt=1 and N_pred=0.",
         r"% MAE / Signed / Exact: All / Single (N_gt=1) / Multi (N_gt>=2).",
         r"% w/o RL (SFT only) = SFT start model (base/salmonn2p_7b_unav_v8), no RL step.",
         r"\begin{tabular}{l" + "c" * 19 + "}", r"\toprule",
         r" & & \multicolumn{3}{c}{F1} & & \multicolumn{2}{c}{OSA} & \multicolumn{2}{c}{CountF1} & "
         r" & \multicolumn{3}{c}{Count MAE} & \multicolumn{3}{c}{Signed count err.}"
         r" & \multicolumn{3}{c}{Exact-count acc.} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){7-8}\cmidrule(lr){9-10}"
         r"\cmidrule(lr){12-14}\cmidrule(lr){15-17}\cmidrule(lr){18-20}",
         r"Setting & mIoU & @0.3 & @0.5 & @0.7 & USA & $\leq$1 & $=$1 & $\leq$1 & $=$1 & \#0"
         r" & All & Single & Multi & All & Single & Multi & All & Single & Multi \\", r"\midrule"]
    for r in summary:
        L.append(" & ".join([r["_tex"], fmt(r["mIoU"], 1)] + [fmt(r[k], 1) for k in F1_KEYS]
                            + [fmt(r["USA"], 2), fmt(r["OSA_le1"], 2), fmt(r["OSA_eq1"], 2),
                               fmt(r["CountF1_le1"], 2), fmt(r["CountF1_eq1"], 2),
                               str(r["n_single_pred0"]),
                               fmt(r["MAE_all"], 2), fmt(r["MAE_single"], 2), fmt(r["MAE_multi"], 2),
                               fmt_signed(r["signed_all"]), fmt_signed(r["signed_single"]),
                               fmt_signed(r["signed_multi"]),
                               fmt(r["exact_all"], 2), fmt(r["exact_single"], 2),
                               fmt(r["exact_multi"], 2)]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(args.out, "count_summary_ck2000.tex"), "w") as f:
        f.write("\n".join(L) + "\n")

    L = [r"% ckpt2000, UnAV-100 test, per-N_gt breakdown. mIoU = sample mIoU, F1 = pairwise.",
         r"% OSA_le1 = P(N_pred<=1), OSA_eq1 = P(N_pred==1); defined for N_gt=1 only.",
         r"\begin{tabular}{llcccccccccc}", r"\toprule",
         r"Setting & $N_{gt}$ & \#Samples & mIoU & F1@0.3 & F1@0.5 & F1@0.7 & Count MAE"
         r" & Signed err. & Exact acc. & OSA$_{\leq 1}$ & OSA$_{=1}$ \\", r"\midrule"]
    names = [r["setting"] for r in summary]
    for i, nm in enumerate(names):
        rows = [r for r in by_ngt if r["setting"] == nm]
        for j, r in enumerate(rows):
            lab = {"1": "1", "2": "2", ">=3": r"$\geq$3"}[r["N_gt"]]
            first = r"\multirow{3}{*}{" + r["_tex"] + "}" if j == 0 else ""
            L.append(" & ".join([first, lab, str(r["n"]), fmt(r["mIoU"], 1), fmt(r["F1@0.3"], 1),
                                 fmt(r["F1@0.5"], 1), fmt(r["F1@0.7"], 1), fmt(r["MAE"], 2),
                                 fmt_signed(r["signed"]), fmt(r["exact"], 2),
                                 fmt(r["OSA_le1"], 2), fmt(r["OSA_eq1"], 2)]) + r" \\")
        if i < len(names) - 1:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(args.out, "count_by_ngt_ck2000.tex"), "w") as f:
        f.write("\n".join(L) + "\n")
    # ---------------- count 분포 ----------------
    d_rates = ["collapse_multi", "collapse_gt2", "collapse_gt3p"] + \
        [k for k, _ in DIFF_BINS] + [k for k, _ in SINGLE_BINS]
    for r in dist:  # 분포 합 = 1 확인
        for grp in (DIFF_BINS, SINGLE_BINS):
            tot = sum(r[k] for k, _ in grp)
            assert abs(tot - 1.0) < 1e-9, (r["setting"], tot)
    with open(os.path.join(args.out, "count_dist_ck2000.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["setting", "n_multi", "n_gt2", "n_gt3p", "n_single"] + d_rates,
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(dist)

    def pct(x):
        return "--" if x is None else f"{100 * x:.1f}"

    nrow = dist[0] if dist else {}
    L = [r"% ckpt2000, UnAV-100 test. All values in %.",
         r"% Collapse = P(N_pred=1 | N_gt) for N_gt>=2 / N_gt=2 / N_gt>=3"
         f" (n={nrow.get('n_multi')}/{nrow.get('n_gt2')}/{nrow.get('n_gt3p')}).",
         r"% Delta = N_pred - N_gt over N_gt>=2. N_pred distribution over N_gt=1"
         f" (n={nrow.get('n_single')}).",
         r"\begin{tabular}{l" + "c" * len(d_rates) + "}", r"\toprule",
         r" & \multicolumn{3}{c}{Collapse rate} & \multicolumn{6}{c}{$N_{pred}-N_{gt}$ ($N_{gt}\geq 2$)}"
         r" & \multicolumn{4}{c}{$N_{pred}$ ($N_{gt}=1$)} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-10}\cmidrule(lr){11-14}",
         r"Setting & $\geq$2 & $=$2 & $\geq$3 & $\leq$$-$3 & $-$2 & $-$1 & 0 & $+$1 & $\geq$$+$2"
         r" & 0 & 1 & 2 & $\geq$3 \\", r"\midrule"]
    for r in dist:
        L.append(" & ".join([r["_tex"]] + [pct(r[k]) for k in d_rates]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(args.out, "count_dist_ck2000.tex"), "w") as f:
        f.write("\n".join(L) + "\n")

    print(f"\n[SAVED] {args.out}")


if __name__ == "__main__":
    main()
