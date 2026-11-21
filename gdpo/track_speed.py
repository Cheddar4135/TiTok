#!/usr/bin/env python3
"""
track_speed.py — train.log(tqdm) 만으로 GDPO 런 속도/진행률/ETA 추적. (GPU·서버 무관, log 만 공유되면 됨)

  python3 gdpo/track_speed.py                      # 최근 30분 내 log 갱신된 런 전부
  python3 gdpo/track_speed.py ttioff nocount       # 런 이름 일부로 지정 (부분 일치)
  python3 gdpo/track_speed.py --all                # checkpoints/gdpo 의 모든 런
  python3 gdpo/track_speed.py --active-min 120     # '활성' 기준 분 변경

출력:
  - 진행률, step 평균(val/save step 제외), 최근 100 step 평균, val+save step 소요
  - 전체 평균(s/step, val 포함) 기준 남은 시간/완료 예상 (log 마지막 갱신 시각 기준)
  - 공통 step 구간(1~min(last)) 에서의 런 간 직접 비교
  ※ resume 으로 tqdm 경과시간이 리셋되면 마지막 세션만 사용.
"""
import argparse, glob, os, re, time

GDPO_CKPT = os.environ.get("GDPO_CKPT_DIR", "/workspace/checkpoints/gdpo")
PAT = re.compile(r"(\d+)/(\d+) \[(?:(\d+):)?(\d+):(\d+)<")


def parse(log):
    s = open(log, errors="ignore").read()
    pts, total = [], None
    for m in PAT.finditer(s):
        st, total = int(m.group(1)), int(m.group(2))
        t = int(m.group(3) or 0) * 3600 + int(m.group(4)) * 60 + int(m.group(5))
        if pts and t < pts[-1][1]:   # resume → 경과시간 리셋, 마지막 세션만
            pts = []
        pts.append((st, t))
    d = {}
    for st, t in pts:
        d[st] = t
    return d, total, "torchcodec unavailable" in s


def fmt(ts):
    return time.strftime("%m-%d %H:%M", time.localtime(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="런 이름 일부 (부분 일치)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--active-min", type=float, default=30)
    ap.add_argument("--save-steps", type=int, default=200)
    a = ap.parse_args()

    logs = sorted(glob.glob(f"{GDPO_CKPT}/*/train.log"))
    if a.runs:
        logs = [l for l in logs if any(r in os.path.basename(os.path.dirname(l)) for r in a.runs)]
    elif not a.all:
        logs = [l for l in logs if time.time() - os.path.getmtime(l) < a.active_min * 60
                and "aborted" not in l]
    if not logs:
        print("대상 런 없음"); return

    now, res = time.time(), {}
    for log in logs:
        name = os.path.basename(os.path.dirname(log))
        d, total, decord = parse(log)
        if len(d) < 2:
            print(f"## {name}: step 기록 부족 (아직 로딩 중?)"); continue
        ks = sorted(d); last = ks[-1]
        steps = [(k, d[k] - d[k - 1]) for k in ks if k - 1 in d]
        is_val = lambda k: k > 1 and k % a.save_steps == 1     # save/val 직후 step
        normal = [x for k, x in steps if not is_val(k) and k > 1]  # step0→1(워밍업) 제외
        recent = [x for k, x in steps if k > last - 100 and not is_val(k) and k > 1]
        val = [(k - 1, x) for k, x in steps if is_val(k)]
        avg = (d[last] - d[ks[0]]) / (last - ks[0])
        mt = os.path.getmtime(log)
        remain = (total - last) * avg
        res[name] = d
        stale = "" if now - mt < 600 else f"  ⚠️ {int((now - mt) / 60)}분째 갱신 없음"
        print(f"## {name}")
        print(f"   step {last}/{total} ({last / total * 100:.1f}%)  log갱신 {fmt(mt)}{stale}"
              f"  decoder={'decord' if decord else 'torchcodec'}")
        if normal:
            print(f"   학습 step 평균 {sum(normal) / len(normal):.1f}s"
                  + (f" | 최근100 {sum(recent) / len(recent):.1f}s" if recent else ""))
        if val:
            print("   val+save step: " + ", ".join(f"{k}→{x}s" for k, x in val))
        print(f"   전체 평균 {avg:.1f}s/step → 남은 {remain / 3600:.1f}h, 완료 예상 {fmt(mt + remain)}")

    if len(res) > 1:
        c = min(max(d) for d in res.values())
        if c > 1:
            print(f"\n== 공통 구간 step 1~{c} 비교 (s/step)")
            for name, d in res.items():
                if 1 in d and c in d:
                    print(f"   {(d[c] - d[1]) / (c - 1):6.1f}  {name}")


if __name__ == "__main__":
    main()
