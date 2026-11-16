#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M2 冻结记录：把划分、判分规则和审计/评测脚本的 SHA-256 写进一个文件。

用法（包根目录）：
    python oj/freeze_benchmark.py
产物：
    data/BENCHMARK_FREEZE.json
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

FILES = [
    "data/train.jsonl",
    "data/validation.jsonl",
    "data/dev100.jsonl",
    "data/test_inputs.jsonl",
    "data/compare50.jsonl",
    "data/splits/dev1200.jsonl",
    "data/splits/holdout746.jsonl",
    "data/splits/split_manifest.json",
    "data/scoring_rules.json",
    "oj/analyze_benchmark.py",
    "oj/make_split.py",
    "oj/eval_split.py",
    "oj/run.py",
    "oj/scoring.py",
    "oj/selfcheck.py",
    "oj/test_scoring_rules.py",
    "oj/check_benchmark.py",
]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    missing = [p for p in FILES if not os.path.exists(os.path.join(ROOT, p))]
    if missing:
        print("缺少文件：", missing)
        return 1
    tz = timezone(timedelta(hours=8))
    record = {
        "schema": "pawsx-zh.benchmark-freeze/1",
        "dataset": "pawsx_zh",
        "dataset_version": "course-v1-20261028",
        "module": "M2 数据与基准",
        "task_id": "T4-PAR-⑩ 中文改写语义一致性核查系统",
        "group": "赵国（队长）、钟承轩、李洋",
        "frozen_at": datetime.now(tz).isoformat(timespec="seconds"),
        "seed": 0,
        "split": {
            "dev": "data/splits/dev1200.jsonl（1200 条，含 dev100）",
            "holdout": "data/splits/holdout746.jsonl（746 条，开发期不用）",
            "platform_test": "test 400（平台冻结，本地无标签）",
            "compare": "compare50（平台三轨横比，本地无标签）",
        },
        "statement": [
            "dev1200 与 holdout746 互斥且覆盖 validation 全部 1946 条。",
            "holdout746 不参与阈值、早停、提示词或超参数选择。",
            "test_inputs 与 compare50 的标签不使用、不推断、不进入训练或提示词。",
            "本文件记录后，划分与判分规则不再改动；若要改，必须升版本并重跑全部实验。",
        ],
        "sha256": {p: sha256(os.path.join(ROOT, p)) for p in FILES},
    }
    out = os.path.join(ROOT, "data", "BENCHMARK_FREEZE.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(f"-> {out}")
    for p, h in record["sha256"].items():
        print(f"  {h[:16]}…  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
