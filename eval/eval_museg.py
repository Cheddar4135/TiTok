#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_museg.py — MUSEG(UnAV-100 멀티세그) 추론 결과 채점 래퍼.

infer_unav_chunks.py 결과 = [{"id","question","output"}, ...] (chronus 와 동일 스키마).
따라서 eval_chronus.py 와 동일 파이프라인:
  1) results/chunk_*.json 병합
  2) id 로 test_json(unav100_museg.json) 매칭 → gt_segments embed, output→pred
  3) eval/test_results_rank0.json 저장
  4) eval_miou.py --natural  (museg 형식 X.XX-X.XX 소수초, finditer 관대 파싱)
  5) maketable.py → eval/table.txt

사용(인자 없이도 표준 경로로 동작):
  python3 eval_museg.py
"""
import argparse, glob, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = "/home/aix23102/audiolm/workspace"
DEF_RESULTS = f"{WS}/outputs/base/MUSEG/unav100_multiseg/results"
DEF_TESTJSON = f"{WS}/data/test/unav100_museg.json"


def _load(p):
    with open(p) as f:
        return json.load(f)


def merge_results(results_dir):
    if os.path.isfile(results_dir):
        return _load(results_dir)
    chunks = sorted(glob.glob(os.path.join(results_dir, "chunk_*.json")))  # _smoke.json 제외
    if not chunks:
        raise SystemExit(f"[eval_museg] chunk_*.json 없음: {results_dir}")
    out = []
    for c in chunks:
        out.extend(_load(c))
    print(f"[eval_museg] merged {len(chunks)} chunks → {len(out)} samples")
    return out


_THINK_TAG = re.compile(r"</?think>", re.IGNORECASE)


def _dedupe_loop(text):
    """잘린 <think> 는 대개 같은 문장을 반복하다 max_new_tokens 에 걸린 것이라
    같은 구간이 수십 번 언급된다("The seventh clip-clop is from 0.00-23.00 ...").
    반복 언급은 하나의 주장이므로 채점기와 같은 파서로 구간을 뽑아 처음 등장한 것만
    남기고, 그 구간들을 채점기가 그대로 다시 읽을 수 있는 정규형 텍스트로 낸다.
    """
    sys.path.insert(0, HERE)
    import eval_miou as EM
    uniq = []
    for seg in EM.parse_natural(text, EM.MAX_TIME_DEFAULT):
        if seg not in uniq:
            uniq.append(seg)
    return " ".join(f"{s:.2f}-{e:.2f} seconds." for s, e in uniq)


def _pred_text(r, truncated_fallback=True, loop_dedupe=True):
    """output(= 추론 스크립트가 뽑은 <answer> 내용) 을 pred 로.

    <think> 가 반복 루프로 max_new_tokens 에 잘려 </answer> 가 아예 없는 경우 output 이
    빈 문자열이 된다(UnAV-100 3455 중 124건). 이때는 raw 에서 think 태그만 떼어낸 원문을
    pred 로 넘겨 채점기의 자연어 파서가 서술 속 구간을 회수하게 한다.
    <answer></answer> 로 명시적으로 빈 답을 낸 경우는 그대로 빈 pred.
    loop_dedupe: fallback 원문 속 반복 언급 구간을 한 번만 센다 (_dedupe_loop).
    """
    out = r.get("output", "") or ""
    if out.strip() or not truncated_fallback:
        return out, False
    raw = r.get("raw", "") or ""
    if "</answer>" in raw.lower():
        return out, False
    text = _THINK_TAG.sub("", raw).strip()
    return (_dedupe_loop(text) if loop_dedupe else text), True


def transform(results, test_json, truncated_fallback=True, loop_dedupe=True):
    gt = {x["id"]: x for x in _load(test_json)}
    conv, miss, n_fb = [], 0, 0
    for r in results:
        g = gt.get(r["id"])
        if g is None:
            miss += 1
            continue
        pred, fb = _pred_text(r, truncated_fallback, loop_dedupe)
        n_fb += fb
        conv.append({
            "id": r["id"],
            "gt_label": g.get("gt_label", ""),
            "gt_segments": g.get("gt_segments", []),
            "pred": pred,   # output → pred (잘린 샘플은 raw fallback)
        })
    if miss:
        print(f"[eval_museg] [WARN] {miss} 샘플 id 매칭 실패 → 제외")
    print(f"[eval_museg] transformed {len(conv)} samples (output→pred, gt embedded), "
          f"truncated→raw fallback {n_fb}건")
    return conv


def main():
    ap = argparse.ArgumentParser(description="MUSEG 결과 채점 → table.txt")
    ap.add_argument("--results_dir", default=DEF_RESULTS)
    ap.add_argument("--test_json", default=DEF_TESTJSON)
    ap.add_argument("--eval_dir", default=None)
    ap.add_argument("--label", default="MUSEG-7B")
    ap.add_argument("--testset", default="unav100_multiseg")
    ap.add_argument("--no_truncated_fallback", action="store_true",
                    help="구 동작: </answer> 없이 잘린 샘플도 빈 pred 로 둔다")
    ap.add_argument("--no_loop_dedupe", action="store_true",
                    help="fallback 원문의 반복 구간을 중복 제거하지 않는다 (2026-10-04 이전 동작)")
    args = ap.parse_args()

    base = args.results_dir if os.path.isdir(args.results_dir) else os.path.dirname(args.results_dir)
    eval_dir = args.eval_dir or os.path.join(os.path.dirname(base), "eval")
    os.makedirs(eval_dir, exist_ok=True)

    results = merge_results(args.results_dir)
    conv = transform(results, args.test_json, truncated_fallback=not args.no_truncated_fallback,
                     loop_dedupe=not args.no_loop_dedupe)
    rank0 = os.path.join(eval_dir, "test_results_rank0.json")
    with open(rank0, "w") as f:
        json.dump(conv, f, ensure_ascii=False, indent=2)
    print(f"[eval_museg] wrote {rank0}")

    cmd = [sys.executable, os.path.join(HERE, "eval_miou.py"), rank0,
           "--natural", "--label", args.label, "--testset", args.testset]
    print("[eval_museg] $", " ".join(cmd))
    subprocess.run(cmd, check=True)

    cmd = [sys.executable, os.path.join(HERE, "maketable.py"), eval_dir]
    print("[eval_museg] $", " ".join(cmd))
    subprocess.run(cmd, check=True)

    table = os.path.join(eval_dir, "table.txt")
    print(f"\n[eval_museg] DONE → {table}\n")
    if os.path.exists(table):
        with open(table) as f:
            print(f.read())


if __name__ == "__main__":
    main()
