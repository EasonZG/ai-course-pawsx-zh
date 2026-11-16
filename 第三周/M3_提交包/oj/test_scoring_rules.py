#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M2 判分规则最小测试：把 data/scoring_rules.json 里的口径逐条钉死。

用法（包根目录）：
    python oj/test_scoring_rules.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scoring  # noqa: E402

META = {"task": "single_label", "labels": ["different", "paraphrase"]}
GOLD = {"a": {"label": 1}, "b": {"label": 0}, "c": {"label": 1}, "d": {"label": 0}}


def run(preds):
    return scoring.score(META, GOLD, preds)


def main() -> int:
    cases = []

    m = run([{"id": "a", "prediction": {"label": 1}},
             {"id": "b", "prediction": {"label": 0}},
             {"id": "c", "prediction": {"label": 1}},
             {"id": "d", "prediction": {"label": 0}}])
    cases.append(("全对 = 1.0", m["accuracy"] == 1.0 and m["valid_output_rate"] == 1.0))

    m = run([{"id": "a", "prediction": {"label": 1}},
             {"id": "b", "prediction": {"label": 0}},
             {"id": "c", "prediction": {"label": 1}}])
    cases.append(("缺行留在分母（3/4）", m["accuracy"] == 0.75 and m["missing"] == 1))

    m = run([{"id": "a", "prediction": {"label": 1}},
             {"id": "a", "prediction": {"label": 1}},
             {"id": "b", "prediction": {"label": 0}},
             {"id": "c", "prediction": {"label": 1}},
             {"id": "d", "prediction": {"label": 0}}])
    cases.append(("重复 ID 整条作废（3/4）", m["accuracy"] == 0.75 and m["duplicate_ids"] == 1))

    m = run([{"id": "a", "prediction": {"label": "paraphrase"}},
             {"id": "b", "prediction": {"label": 0}},
             {"id": "c", "prediction": {"label": 1}},
             {"id": "d", "prediction": {"label": 0}}])
    cases.append(("标签名判非法（3/4）", m["accuracy"] == 0.75 and m["valid_output_count"] == 3))

    m = run([{"id": "a", "prediction": {"label": [1]}},
             {"id": "b", "prediction": {"label": 0}},
             {"id": "c", "prediction": {"label": 1}},
             {"id": "d", "prediction": {"label": 0}}])
    cases.append(("多答案/列表判非法（3/4）", m["accuracy"] == 0.75))

    m = run([{"id": "a", "prediction": {"label": True}},
             {"id": "b", "prediction": {"label": 0}},
             {"id": "c", "prediction": {"label": 1}},
             {"id": "d", "prediction": {"label": 0}}])
    cases.append(("bool 判非法（3/4）", m["accuracy"] == 0.75))

    m = run([{"id": "a", "prediction": {"label": 2}},
             {"id": "b", "prediction": {"label": 0}},
             {"id": "c", "prediction": {"label": 1}},
             {"id": "d", "prediction": {"label": 0}}])
    cases.append(("越界整数判非法（3/4）", m["accuracy"] == 0.75))

    m = run([{"id": "a", "prediction": {"label": "1"}},
             {"id": "b", "prediction": {"label": "0"}},
             {"id": "c", "prediction": {"label": 1}},
             {"id": "d", "prediction": {"label": 0}}])
    cases.append(("数字字符串可接受（4/4）", m["accuracy"] == 1.0))

    m = run([{"id": "a", "prediction": {"label": 1}},
             {"id": "b", "prediction": {"label": 0}},
             {"id": "c", "prediction": {"label": 1}},
             {"id": "d", "prediction": {"label": 0}},
             {"id": "zzz", "prediction": {"label": 0}}])
    cases.append(("未知 ID 忽略（4/4）", m["accuracy"] == 1.0 and m["unknown_records"] == 1))

    ok = True
    for name, passed in cases:
        print(("✅ " if passed else "❌ ") + name)
        ok &= passed
    print(f"\n{sum(p for _, p in cases)}/{len(cases)} 通过")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
