#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_crab.py — Crab+ 추론 결과 채점 래퍼 (avicuna/chronus 패턴 복제).

Crab+ (crab_plus_git/eval_unav/infer_crab_unav.py) 는 레이블 무관 프롬프트
("Please describe the events and time range that occurred in the video.") 로
영상의 모든 이벤트를 `<event>label, (s e)(s e)</event>` 형태로 낸다. 시간은 Crab+ 학습 라벨
(AVUIE_2/unav) 규약대로 영상 길이를 0~10 으로 정규화한 값.
그래서 채점 전에 래퍼에서:
  1) results 의 chunk_*.jsonl / chunk_*.json 전부 concat (id 별 마지막 기록)
  2) raw_pred 에서 gt_label 과 같은 레이블(대소문자·공백 무시)의 <event> (또는 <answer>) 블록만 골라 (s e) 구간 추출
     — 해당 레이블 블록이 없으면 pred="" (파싱 실패로 집계, 제외하지 않음)
     — <event> 밖 서술문("From 6 seconds to 10 seconds ...")은 레이블 귀속이 불명확해 사용하지 않음
  3) 0~10 → 초: s * duration / 10 (avicuna 의 0~100 % 환산과 같은 방식)
  4) pred = "from S to E seconds. ..." (초) 로 기록 → eval_miou.py --natural (Table 2 와 동일 채점기)
  5) maketable.py → <eval_dir>/table.txt

사용:
  python3 eval_crab.py --results_dir /workspace/outputs/base/Crab_Plus/unav100_multiseg/results
"""
import argparse, glob, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_RESULTS = "/workspace/outputs/base/Crab_Plus/unav100_multiseg/results"
# <event>label, (s e)..</event> 와 <answer>label, (s e)..</answer> 를 동일하게 취급 (B안 레이블 질의에서
# 모델이 AVQA 학습 태그인 <answer> 로 감싸 답하는 경우가 있음). 여는/닫는 태그는 같은 종류끼리만 짝지음.
_EVENT = re.compile(r"<(event|answer)>\s*([^<]*?)\s*,\s*([^<]*?)</\1>", re.DOTALL | re.IGNORECASE)
_SEG = re.compile(r"\(\s*(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*\)")


def _norm(s):
    return re.sub(r"\s+", " ", s or "").strip().lower()


def merge_results(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "chunk_*.json*")))
    if not files:
        raise SystemExit(f"[eval_crab] chunk_*.json(l) 없음: {results_dir}")
    rows = {}
    for f in files:
        it = [json.loads(l) for l in open(f) if l.strip()] if f.endswith(".jsonl") else json.load(open(f))
        for r in it:
            rows[r["id"]] = r  # id 별 마지막 기록 (재시도 반영)
    print(f"[eval_crab] merged {len(files)} files → {len(rows)} samples")
    return list(rows.values())


def extract(raw, label, duration, single_name_fallback=False, info=None):
    """raw 에서 label 의 <event>/<answer> 구간만 → 초 단위 [[s,e]] (0~10 → ×duration/10).

    single_name_fallback (B안 레이블 질의 전용): 정확 일치 블록이 없을 때, 블록들의 레이블 이름이
    한 종류뿐이면(블록 1개, 또는 같은 이름 블록 여러 개) 이름과 무관하게 그 블록들을 답으로 사용
    ("man speak" vs "man speaking" 같은 표기 차이 구제). 서로 다른 이름이 2종 이상이면
    어느 것이 질의에 대한 답인지 알 수 없으므로 파싱 실패([]) 로 둔다.
    """
    blocks = [(_norm(name), body) for _tag, name, body in _EVENT.findall(raw or "")]
    use = [body for name, body in blocks if name == _norm(label)]
    if not use and single_name_fallback and len({name for name, _ in blocks}) == 1:
        use = [body for _, body in blocks]
        if info is not None:
            info["fallback"] = blocks[0][0]
    segs = []
    for body in use:
        for a, b in _SEG.findall(body):
            segs.append([float(a) * duration / 10.0, float(b) * duration / 10.0])
    return segs


def main():
    ap = argparse.ArgumentParser(description="Crab+ 결과 채점 → table.txt")
    ap.add_argument("--results_dir", default=DEF_RESULTS)
    ap.add_argument("--eval_dir", default=None, help="기본 = results_dir 형제 'eval'")
    ap.add_argument("--label", default="Crab_Plus")
    ap.add_argument("--testset", default="unav100_multiseg")
    ap.add_argument("--single_name_fallback", action="store_true",
                    help="B안(레이블 질의) 전용: 정확 일치 실패 시 블록 이름이 한 종류뿐이면 그 블록을 답으로 사용")
    args = ap.parse_args()

    eval_dir = args.eval_dir or os.path.join(os.path.dirname(args.results_dir.rstrip("/")), "eval")
    os.makedirs(eval_dir, exist_ok=True)

    conv, n_err, n_noevent, n_nolabel, n_fb = [], 0, 0, 0, 0
    for r in merge_results(args.results_dir):
        raw = r.get("raw_pred", "")
        n_err += bool(r.get("error"))
        info = {}
        segs = extract(raw, r["gt_label"], r["duration"], args.single_name_fallback, info)
        n_fb += "fallback" in info
        if not _EVENT.search(raw or ""):
            n_noevent += 1
        elif not segs:
            n_nolabel += 1
        conv.append({"id": r["id"], "gt_label": r["gt_label"], "gt_segments": r["gt_segments"],
                     "duration": r["duration"],
                     "pred": ". ".join(f"from {s:.2f} to {e:.2f} seconds" for s, e in segs),
                     "raw_pred": raw, **({"fallback_label": info["fallback"]} if info else {})})
    print(f"[eval_crab] {len(conv)} samples: error={n_err}, <event>/<answer> 블록 없음={n_noevent}, "
          f"gt_label 블록 없음={n_nolabel} (둘 다 pred='' 로 포함, 제외 안 함)"
          + (f", 단일이름 fallback 적용={n_fb}" if args.single_name_fallback else ""))

    rank0 = os.path.join(eval_dir, "test_results_rank0.json")
    with open(rank0, "w") as f:
        json.dump(conv, f, ensure_ascii=False, indent=2)
    print(f"[eval_crab] wrote {rank0}")

    cmd = [sys.executable, os.path.join(HERE, "eval_miou.py"), rank0,
           "--natural", "--label", args.label, "--testset", args.testset]
    print("[eval_crab] $", " ".join(cmd)); subprocess.run(cmd, check=True)

    cmd = [sys.executable, os.path.join(HERE, "maketable.py"), eval_dir]
    print("[eval_crab] $", " ".join(cmd)); subprocess.run(cmd, check=True)

    table = os.path.join(eval_dir, "table.txt")
    print(f"\n[eval_crab] DONE → {table}\n")
    if os.path.exists(table):
        with open(table) as f:
            print(f.read())


if __name__ == "__main__":
    main()
