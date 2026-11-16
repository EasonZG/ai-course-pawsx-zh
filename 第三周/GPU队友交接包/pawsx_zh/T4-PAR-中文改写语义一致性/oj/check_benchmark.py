#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M2 冻结自检：验证划分未变、无重叠、判分规则文件可读。

用法（包根目录）：
    python oj/check_benchmark.py
返回码 0 表示冻结记录与当前文件一致；1 表示有人改过文件或划分被破坏。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    freeze_path = os.path.join(ROOT, "data", "BENCHMARK_FREEZE.json")
    if not os.path.exists(freeze_path):
        print("❌ 找不到 data/BENCHMARK_FREEZE.json，先运行 python oj/freeze_benchmark.py")
        return 1
    freeze = json.load(open(freeze_path, encoding="utf-8"))

    bad = []
    for rel, want in freeze["sha256"].items():
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            bad.append((rel, "missing"))
        elif sha256(path) != want:
            bad.append((rel, "changed"))
    if bad:
        print("❌ 冻结文件与记录不一致：")
        for rel, why in bad:
            print(f"  {why:8s} {rel}")
        return 1

    dev = load_jsonl(os.path.join(ROOT, "data/splits/dev1200.jsonl"))
    hold = load_jsonl(os.path.join(ROOT, "data/splits/holdout746.jsonl"))
    dev100 = load_jsonl(os.path.join(ROOT, "data/dev100.jsonl"))
    val = load_jsonl(os.path.join(ROOT, "data/validation.jsonl"))

    checks = {
        "dev1200=1200": len(dev) == 1200,
        "holdout746=746": len(hold) == 746,
        "dev+holdout=validation": len(dev) + len(hold) == len(val),
        "ids_disjoint": not ({r["item_id"] for r in dev} & {r["item_id"] for r in hold}),
        "dev100_in_dev": {r["item_id"] for r in dev100} <= {r["item_id"] for r in dev},
        "labels_only_0_1": all(r["target"]["label"] in (0, 1) for r in dev + hold),
        "no_dup_ids": len({r["item_id"] for r in dev + hold}) == len(dev) + len(hold),
    }
    rules = json.load(open(os.path.join(ROOT, "data/scoring_rules.json"), encoding="utf-8"))
    checks["rules_task_single_label"] = rules.get("task") == "single_label"
    checks["rules_labels_ok"] = rules.get("labels") == ["different", "paraphrase"]

    for name, ok in checks.items():
        print(("✅ " if ok else "❌ ") + name)
    print(f"\n冻结时间：{freeze['frozen_at']}；数据集：{freeze['dataset_version']}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
