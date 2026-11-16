#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M2 基准划分：把 validation 固定切成「开发集 dev1200」与「留出集 holdout746」。

规则（写进 report/M2.md 与 data/splits/split_manifest.json）：
  1. 只切 validation（有标签、且本来就是调参集），train 不参与切分，test_inputs 不碰；
  2. dev100 是课程给定的冒烟集，100 条全部放进 dev1200，保证冒烟集不落在留出集里；
  3. dev1200 的标签配额按 validation 的标签比例分配（四舍五入），
     剩余部分按标签分层随机抽（seed=0），其余全部进 holdout746；
  4. 划分结果与源文件 SHA-256 一起冻结，之后不再改动。

用法（包根目录）：
    python oj/make_split.py
产物：
    data/splits/dev1200.jsonl
    data/splits/holdout746.jsonl
    data/splits/split_manifest.json
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import random
import sys
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(DATA, "splits")
SEED = 0
DEV_N = 1200


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dump(rows: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    val_path = os.path.join(DATA, "validation.jsonl")
    dev100_path = os.path.join(DATA, "dev100.jsonl")
    rows = load(val_path)
    dev100 = load(dev100_path)
    by_id = {r["item_id"]: r for r in rows}
    dev100_ids = {r["item_id"] for r in dev100}
    if not dev100_ids <= set(by_id):
        raise SystemExit("dev100 不是 validation 的子集，停止划分")

    label_of = lambda r: int(r["target"]["label"])
    total = len(rows)
    if not (len(dev100_ids) < DEV_N < total):
        raise SystemExit("dev 大小不合理")

    # 目标配额：与 validation 的标签比例一致
    label_totals = collections.Counter(label_of(r) for r in rows)
    quota: dict[int, int] = {}
    assigned = 0
    labels_sorted = sorted(label_totals)
    for i, lab in enumerate(labels_sorted):
        if i == len(labels_sorted) - 1:
            quota[lab] = DEV_N - assigned
        else:
            quota[lab] = round(DEV_N * label_totals[lab] / total)
            assigned += quota[lab]

    # dev100 优先占用配额
    dev = [by_id[i] for i in sorted(dev100_ids)]
    dev_counts = collections.Counter(label_of(r) for r in dev)
    for lab in labels_sorted:
        if dev_counts[lab] > quota[lab]:
            raise SystemExit(f"dev100 在标签 {lab} 上已超过配额，需调整 DEV_N")

    # 其余配额从 validation \ dev100 分层随机抽
    rng = random.Random(SEED)
    pool = [r for r in rows if r["item_id"] not in dev100_ids]
    for lab in labels_sorted:
        need = quota[lab] - dev_counts[lab]
        cand = sorted([r for r in pool if label_of(r) == lab], key=lambda r: r["item_id"])
        if need > len(cand):
            raise SystemExit(f"标签 {lab} 可抽条数不足")
        dev.extend(rng.sample(cand, need))

    dev_ids = {r["item_id"] for r in dev}
    holdout = [r for r in rows if r["item_id"] not in dev_ids]
    dev.sort(key=lambda r: r["item_id"])
    holdout.sort(key=lambda r: r["item_id"])

    os.makedirs(OUT, exist_ok=True)
    dev_path = os.path.join(OUT, "dev1200.jsonl")
    hold_path = os.path.join(OUT, "holdout746.jsonl")
    dump(dev, dev_path)
    dump(holdout, hold_path)

    def stats(rs: list[dict]) -> dict:
        c = collections.Counter(label_of(r) for r in rs)
        return {"n": len(rs), "label_counts": {str(k): c[k] for k in sorted(c)},
                "label_share": {str(k): round(c[k] / len(rs), 4) for k in sorted(c)}}

    manifest = {
        "schema": "pawsx-zh.benchmark-split/1",
        "dataset": "pawsx_zh",
        "dataset_version": "course-v1-20261028",
        "created": date.today().isoformat(),
        "seed": SEED,
        "rule": {
            "source": "data/validation.jsonl",
            "dev_size": DEV_N,
            "dev100_forced_into_dev": True,
            "stratified_by": "target.label",
            "quota": {str(k): quota[k] for k in sorted(quota)},
            "remainder": "holdout",
        },
        "splits": {
            "dev1200": {"file": "data/splits/dev1200.jsonl", "purpose": "调阈值/早停/提示词与消融选择",
                        **stats(dev), "sha256": sha256(dev_path)},
            "holdout746": {"file": "data/splits/holdout746.jsonl",
                           "purpose": "留出验收；开发期不读标签、不用于任何选择", **stats(holdout),
                           "sha256": sha256(hold_path)},
        },
        "source_sha256": {
            "data/validation.jsonl": sha256(val_path),
            "data/dev100.jsonl": sha256(dev100_path),
            "data/train.jsonl": sha256(os.path.join(DATA, "train.jsonl")),
            "data/test_inputs.jsonl": sha256(os.path.join(DATA, "test_inputs.jsonl")),
            "data/compare50.jsonl": sha256(os.path.join(DATA, "compare50.jsonl")),
        },
        "checks": {
            "dev100_subset_of_dev1200": dev100_ids <= dev_ids,
            "dev_and_holdout_disjoint": not (dev_ids & {r["item_id"] for r in holdout}),
            "no_duplicate_ids": len(dev_ids) + len(holdout) == len(rows) == len(set(dev_ids) | {r["item_id"] for r in holdout}),
            "dev_and_holdout_cover_validation": len(dev) + len(holdout) == len(rows),
        },
    }
    manifest_path = os.path.join(OUT, "split_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(json.dumps({"dev1200": stats(dev), "holdout746": stats(holdout), "checks": manifest["checks"]},
                     ensure_ascii=False, indent=2))
    print(f"-> {dev_path}\n-> {hold_path}\n-> {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
