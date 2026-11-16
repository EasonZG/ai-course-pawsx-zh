#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M2 基准判分：对「有标签的留出集」按冻结规则算分。

判分规则（与平台 oj/scoring.py 同口径，细节见 data/scoring_rules.json）：
  - 单标签，输出 {"label": 0|1}，整数下标，0=different、1=paraphrase；
  - 逐条 exact match；标签名/越界/空值/缺行/重复 ID 一律算错，且不缩小分母；
  - 主指标 accuracy；附加 macro-F1、每类 P/R/F1、混淆矩阵；
  - 95% bootstrap 置信区间：对「句对」重采样 1000 次，百分位法，seed=0。

用法（包根目录）：
  python oj/eval_split.py --gold data/splits/holdout746.jsonl --pred runs/trad/results.holdout.jsonl
产物：
  终端表格；加 --out 时另存 metrics.json。
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scoring  # noqa: E402

LABELS = ["different", "paraphrase"]


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def gold_label(row: dict) -> int | None:
    t = row.get("target")
    if isinstance(t, dict):
        t = t.get("label")
    if isinstance(t, bool):
        return None
    if isinstance(t, int):
        return t
    if isinstance(t, str) and t.strip().isdigit():
        return int(t.strip())
    return None


def pred_label(row: dict) -> int | None:
    out = row.get("output", row.get("prediction"))
    if isinstance(out, str):
        out = scoring.parse_output(out)
    if isinstance(out, dict):
        out = out.get("label")
    if isinstance(out, bool):
        return None
    if isinstance(out, int):
        return out
    if isinstance(out, float) and out.is_integer():
        return int(out)
    if isinstance(out, str) and out.strip().isdigit():
        return int(out.strip())
    return None


def f1(tp: int, fp: int, fn: int) -> float:
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else 0.0


def metrics_from_conf(conf: list[list[int]], n: int) -> dict:
    out = {"accuracy": (conf[0][0] + conf[1][1]) / n if n else 0.0}
    per = {}
    for c in (0, 1):
        tp = conf[c][c]
        fp = sum(conf[r][c] for r in (0, 1) if r != c)
        fn = sum(conf[c][j] for j in (0, 1) if j != c)
        per[LABELS[c]] = {
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": f1(tp, fp, fn),
            "support": tp + fn,
        }
    out["macro_f1"] = sum(v["f1"] for v in per.values()) / 2
    out["per_label"] = per
    out["confusion"] = {"rows_gold_cols_pred": conf,
                        "row_order": LABELS, "col_order": LABELS}
    return out


def bootstrap_ci(correct: list[int], n_boot: int = 1000, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    n = len(correct)
    vals = []
    for _ in range(n_boot):
        s = sum(correct[rng.randrange(n)] for _ in range(n))
        vals.append(s / n)
    vals.sort()
    return [vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot) - 1]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="有 target 的 jsonl（dev1200 / holdout746 / dev100）")
    ap.add_argument("--pred", required=True, help="results.jsonl（item_id/output/status/usage）")
    ap.add_argument("--out", default=None, help="可选：metrics.json 输出路径")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--errors", type=int, default=5, help="打印前 N 条错误样例")
    args = ap.parse_args()

    gold_rows = [r for r in load_jsonl(args.gold) if gold_label(r) in (0, 1)]
    pred_rows = load_jsonl(args.pred)
    meta = {"task": "single_label", "labels": LABELS}
    gold = {r["item_id"]: {"label": gold_label(r)} for r in gold_rows}

    # 1) 平台同源判分（含缺行/重复/非法值口径）
    preds = [{"id": r.get("item_id"), "prediction": r.get("output")} for r in pred_rows]
    platform = scoring.score(meta, gold, preds)

    # 2) 逐条对齐，算混淆矩阵与 bootstrap（重复 ID 视为无效）
    cnt = collections.Counter(r.get("item_id") for r in pred_rows)
    dup = {k for k, v in cnt.items() if v > 1}
    by_id = {}
    for r in pred_rows:
        by_id.setdefault(r.get("item_id"), r)

    conf = [[0, 0], [0, 0]]
    correct: list[int] = []
    errors = []
    invalid = 0
    for g in gold_rows:
        iid, y = g["item_id"], gold_label(g)
        p = pred_label(by_id.get(iid, {})) if iid in by_id and iid not in dup else None
        if p not in (0, 1):
            invalid += 1
            p = -1
        else:
            conf[y][p] += 1
        correct.append(int(p == y))
        if p != y and len(errors) < args.errors:
            errors.append({"item_id": iid, "gold": y, "pred": p,
                           "text_pair": g.get("text_pair"),
                           "status": by_id.get(iid, {}).get("status", "missing")})

    n = len(gold_rows)
    m = metrics_from_conf(conf, n)
    m.update({
        "split": os.path.basename(args.gold),
        "n": n,
        "covered": n - platform["missing"],
        "coverage": (n - platform["missing"]) / n if n else 0.0,
        "missing": platform["missing"],
        "duplicate_ids": platform["duplicate_ids"],
        "invalid_or_duplicate": invalid,
        "valid_output_rate": platform["valid_output_rate"],
        "accuracy_ci95": bootstrap_ci(correct, args.boot, args.seed),
        "platform_scoring": {k: platform[k] for k in
                             ("total", "prediction_records", "missing", "duplicate_ids",
                              "unknown_records", "accuracy", "macro_f1", "valid_output_count",
                              "valid_output_rate")},
        "pass_gate_0.6675": m["accuracy"] >= 0.6675,
        "errors": errors,
    })

    print(f"=== {m['split']}  n={n}  覆盖={m['coverage']:.3f}  合法输出率={m['valid_output_rate']:.3f}")
    print(f"accuracy  {m['accuracy']:.4f}   95% CI [{m['accuracy_ci95'][0]:.4f}, {m['accuracy_ci95'][1]:.4f}]")
    print(f"macro-F1  {m['macro_f1']:.4f}")
    print("混淆矩阵（行=gold，列=pred，0=different 1=paraphrase）：")
    for i, name in enumerate(LABELS):
        print(f"  gold {i} {name:10s} -> pred0 {conf[i][0]:4d} / pred1 {conf[i][1]:4d}")
    for name, v in m["per_label"].items():
        print(f"  {name:10s} P={v['precision']:.4f} R={v['recall']:.4f} F1={v['f1']:.4f} n={v['support']}")
    if errors:
        print("错误样例：")
        for e in errors:
            print(f"  {e['item_id']} gold={e['gold']} pred={e['pred']} status={e['status']}")
    if m["missing"] or m["duplicate_ids"] or invalid:
        print(f"⚠ 缺行 {m['missing']} / 重复 {m['duplicate_ids']} / 非法或重复 {invalid}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=2)
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
