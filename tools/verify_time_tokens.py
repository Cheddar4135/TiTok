#!/usr/bin/env python3
"""
Verify the tokenizer of a checkpoint with time tokens added.
- <t0>~<t9>, <tdot> (11 tokens) each encode to a single id
- sec_to_time_token_str() output splits into exactly 6 tokens
- print the id sequences of representative times (0.0, 1.2, 59.9, 9999.9)
"""
import argparse
import sys
from transformers import AutoTokenizer


VTG_TIME_TOKENS = [f"<t{i}>" for i in range(10)] + ["<tdot>"]


def sec_to_time_token_str(sec: float) -> str:
    # Same logic as the helper in dataset.py. Range 0.0~9999.9 s, precision 0.1 s.
    sec = max(0.0, min(9999.9, sec))
    tenths = round(sec * 10)
    i = tenths // 10
    f = tenths % 10
    d1, d2, d3, d4 = (i // 1000) % 10, (i // 100) % 10, (i // 10) % 10, i % 10
    return f"<t{d1}><t{d2}><t{d3}><t{d4}><tdot><t{f}>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="checkpoint with time tokens added")
    args = ap.parse_args()

    print(f"[1/4] Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print(f"  Vocab size: {len(tokenizer)}")

    print(f"\n[2/4] Check that each time token encodes to a single id")
    all_ok = True
    token_ids = {}
    for tok in VTG_TIME_TOKENS:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        status = "OK" if len(ids) == 1 else "FAIL"
        if len(ids) != 1:
            all_ok = False
        token_ids[tok] = ids[0] if len(ids) == 1 else ids
        print(f"  {tok:<8} -> {ids}  [{status}]")

    if not all_ok:
        print("\n[ERROR] Some tokens do not encode to a single id.")
        print("        Run tools/add_time_tokens.py first.")
        sys.exit(1)

    print(f"\n[3/4] Check sec_to_time_token_str output")
    test_cases = [0.0, 0.1, 1.2, 12.3, 59.9, 123.4, 9999.9, 99999.0, -1.0]
    expected_id_count = 6
    for sec in test_cases:
        s = sec_to_time_token_str(sec)
        ids = tokenizer.encode(s, add_special_tokens=False)
        status = "OK" if len(ids) == expected_id_count else "FAIL"
        if len(ids) != expected_id_count:
            all_ok = False
        print(f"  {sec:>8.1f}s -> {s!r:<50} ids={ids} [{status}]")

    print(f"\n[4/4] Decode a representative sequence (1.2 s)")
    s = sec_to_time_token_str(1.2)
    ids = tokenizer.encode(s, add_special_tokens=False)
    decoded = tokenizer.decode(ids)
    print(f"  encoded:  {s}")
    print(f"  ids:      {ids}")
    print(f"  decoded:  {decoded}")
    round_trip_ok = decoded.replace(" ", "") == s.replace(" ", "")
    print(f"  round-trip: {'OK' if round_trip_ok else 'FAIL'}")
    if not round_trip_ok:
        all_ok = False

    print(f"\n{'=' * 60}")
    print("RESULT:", "ALL PASSED" if all_ok else "FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
