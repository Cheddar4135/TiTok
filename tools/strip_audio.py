#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
strip_audio.py — 학습/val json 의 no-audio 판 생성.

  각 샘플에서 "audio" 키 제거 + use_audio=False. 그 외(순서/프롬프트/GT) 전부 그대로.
    → 샘플 수·순서 동일 = 같은 seed 에서 RandomSampler 데이터 순서 동일.

  ⚠️ 둘 다 필요한 이유 (qwenvl/data/dataset.py _get_item):
     757: use_audio=True 면 video 파일에서 오디오를 추출  → use_audio=False 필요
     771: "audio" 키가 있으면 use_audio 와 무관하게 로드  → "audio" 키 제거 필요

  사용: python tools/strip_audio.py IN.json OUT.json
"""
import json
import sys


def main():
    src, dst = sys.argv[1], sys.argv[2]
    data = json.load(open(src, encoding="utf-8"))
    n_audio = 0
    out = []
    for d in data:
        d = dict(d)
        if d.pop("audio", None) is not None:
            n_audio += 1
        d["use_audio"] = False
        out.append(d)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"{src} → {dst}: {len(out)}개 (audio 키 제거 {n_audio}개, use_audio=False 전체)")


if __name__ == "__main__":
    main()
