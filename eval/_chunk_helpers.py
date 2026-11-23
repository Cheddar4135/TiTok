#!/usr/bin/env python3
"""Helpers for chunked inference in eval/eval.sh.

Subcommands:
  split            : test_json → chunks_dir/chunk_NNNN.json (idempotent)
  build_full       : chunks_dir/chunk_*.json → chunks_dir/_full.json (concat)
  append           : atomically append chunk results to the master file
  resume_offset    : choose the next chunk from .chunk_idx, master and chunks_dir
  truncate_master  : keep only the first N entries of master (align to chunk boundaries)
  summary_oneline  : print one line of metrics from a summary JSON
"""
import argparse
import glob
import json
import os


def _atomic_write_json(path, obj, indent=None):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
    os.replace(tmp, path)


def _load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def cmd_split(args):
    data = _load_json(args.test_json)
    if data is None:
        raise SystemExit(f"[split] cannot read {args.test_json}")
    cs = args.chunk_size
    os.makedirs(args.chunks_dir, exist_ok=True)
    n_chunks = (len(data) + cs - 1) // cs if data else 0
    for i in range(n_chunks):
        path = os.path.join(args.chunks_dir, f"chunk_{i:04d}.json")
        if not os.path.exists(path):
            _atomic_write_json(path, data[i * cs : (i + 1) * cs], indent=2)
    last = len(data) - (n_chunks - 1) * cs if n_chunks > 0 else 0
    print(f"{len(data)} {n_chunks} {cs} {last}")


def cmd_build_full(args):
    out = os.path.join(args.chunks_dir, "_full.json")
    if os.path.exists(out) and not args.force:
        n = len(_load_json(out, []))
        print(f"{n} existing")
        return
    data = []
    for cf in sorted(glob.glob(os.path.join(args.chunks_dir, "chunk_*.json"))):
        data.extend(_load_json(cf, []))
    _atomic_write_json(out, data, indent=2)
    print(f"{len(data)} built")


def cmd_append(args):
    new_items = _load_json(args.chunk_results, [])
    existing = _load_json(args.master, [])
    existing.extend(new_items)
    _atomic_write_json(args.master, existing, indent=2)
    print(f"{len(existing)} {len(new_items)}")


def cmd_resume_offset(args):
    """Print the next chunk index, master length and chunk-aligned expected_n.

    Priority:
      1) the value in .chunk_idx if the file exists
      2) estimate from the master length and the cumulative chunk sizes in chunks_dir
      3) otherwise 0
    """
    n_master = len(_load_json(args.master, []))

    chunk_sizes = []
    if args.chunks_dir and os.path.isdir(args.chunks_dir):
        for cf in sorted(glob.glob(os.path.join(args.chunks_dir, "chunk_*.json"))):
            chunk_sizes.append(len(_load_json(cf, [])))

    next_idx = 0
    if args.chunk_idx_file and os.path.exists(args.chunk_idx_file):
        try:
            with open(args.chunk_idx_file) as f:
                next_idx = int(f.read().strip() or "0")
        except (ValueError, OSError):
            next_idx = 0
    elif chunk_sizes:
        cum = 0
        for ci, sz in enumerate(chunk_sizes):
            if cum + sz <= n_master:
                cum += sz
                next_idx = ci + 1
            else:
                break

    expected_n = sum(chunk_sizes[:next_idx]) if chunk_sizes else 0
    print(f"{next_idx} {n_master} {expected_n}")


def cmd_truncate_master(args):
    data = _load_json(args.master, [])
    if len(data) <= args.keep:
        print(len(data))
        return
    _atomic_write_json(args.master, data[: args.keep], indent=2)
    print(args.keep)


def cmd_summary_oneline(args):
    d = _load_json(args.summary, {})
    if not d:
        print("  (no summary)")
        return
    r = d.get("Recall", {})
    print(
        f"  n={d.get('n_samples', 0)}"
        f"  mIoU={d.get('mIoU_union_%', 0):.2f}%"
        f"  R@.3={r.get('0.3', 0):.2f}%"
        f"  R@.5={r.get('0.5', 0):.2f}%"
        f"  R@.7={r.get('0.7', 0):.2f}%"
        f"  FP={d.get('FP_rate_%', 0):.2f}%"
        f"  FN={d.get('FN_rate_%', 0):.2f}%"
    )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("split")
    p.add_argument("--test_json", required=True)
    p.add_argument("--chunks_dir", required=True)
    p.add_argument("--chunk_size", type=int, required=True)
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("build_full")
    p.add_argument("--chunks_dir", required=True)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_build_full)

    p = sub.add_parser("append")
    p.add_argument("--master", required=True)
    p.add_argument("--chunk_results", required=True)
    p.set_defaults(func=cmd_append)

    p = sub.add_parser("resume_offset")
    p.add_argument("--master", required=True)
    p.add_argument("--chunk_idx_file", default=None)
    p.add_argument("--chunks_dir", default=None)
    p.set_defaults(func=cmd_resume_offset)

    p = sub.add_parser("truncate_master")
    p.add_argument("--master", required=True)
    p.add_argument("--keep", type=int, required=True)
    p.set_defaults(func=cmd_truncate_master)

    p = sub.add_parser("summary_oneline")
    p.add_argument("--summary", required=True)
    p.set_defaults(func=cmd_summary_oneline)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
