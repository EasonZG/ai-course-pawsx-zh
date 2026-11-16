#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M2 数据审计：统计条数、标签分布、长度分布，以及各切片的句子/句对重叠。

用法（包根目录）：
    python oj/analyze_benchmark.py
输出：
    runs/benchmark/data_audit.json   机器可读的审计结果
    runs/benchmark/data_audit.md     可直接粘进报告的表
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "runs", "benchmark")
SPLITS = ["train", "validation", "dev100", "test_inputs", "compare50"]


def load(name: str) -> list[dict]:
    path = os.path.join(DATA, f"{name}.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pair_key(r: dict) -> tuple[str, str]:
    a, b = r.get("text_pair") or ["", ""]
    return a, b


def unordered_key(r: dict) -> tuple[str, str]:
    return tuple(sorted(pair_key(r)))


def sentences(rows: list[dict]) -> set[str]:
    out: set[str] = set()
    for r in rows:
        out.update(r.get("text_pair") or [])
    return out


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def main() -> int:
    data = {n: load(n) for n in SPLITS}
    report: dict = {"dataset": "pawsx_zh", "data_dir": DATA, "splits": {}}

    # ---------- 基础统计
    for name, rows in data.items():
        labels = [r.get("target", {}).get("label") for r in rows]
        known = [x for x in labels if isinstance(x, int)]
        lens = [max(len(r["text_pair"][0]), len(r["text_pair"][1])) for r in rows]
        lens_sorted = sorted(lens)
        dist = collections.Counter(known)
        report["splits"][name] = {
            "file": f"data/{name}.jsonl",
            "sha256": sha256(os.path.join(DATA, f"{name}.jsonl")),
            "n": len(rows),
            "n_with_label": len(known),
            "label_counts": {str(k): v for k, v in sorted(dist.items())},
            "label_share": {str(k): round(v / len(known), 4) for k, v in sorted(dist.items())} if known else {},
            "id_unique": len({r["item_id"] for r in rows}) == len(rows),
            "char_len": {
                "mean": round(statistics.mean(lens), 2),
                "median": statistics.median(lens),
                "p90": lens_sorted[int(0.90 * len(lens))],
                "p95": lens_sorted[int(0.95 * len(lens))],
                "max": max(lens),
                "share_le_64": round(sum(x <= 64 for x in lens) / len(lens), 4),
                "share_le_96": round(sum(x <= 96 for x in lens) / len(lens), 4),
                "share_le_128": round(sum(x <= 128 for x in lens) / len(lens), 4),
            },
        }

    # ---------- 叠层/泄漏检查
    val_ids = {r["item_id"] for r in data["validation"]}
    test_ids = {r["item_id"] for r in data["test_inputs"]}
    dev_ids = {r["item_id"] for r in data["dev100"]}
    cmp_ids = {r["item_id"] for r in data["compare50"]}
    report["id_relations"] = {
        "dev100_subset_of_validation": dev_ids <= val_ids,
        "dev100_overlap_validation": len(dev_ids & val_ids),
        "compare50_subset_of_test_inputs": cmp_ids <= test_ids,
        "compare50_overlap_test_inputs": len(cmp_ids & test_ids),
    }

    pairs = {n: {pair_key(r) for r in rows} for n, rows in data.items()}
    unord = {n: {unordered_key(r) for r in rows} for n, rows in data.items()}
    sents = {n: sentences(rows) for n, rows in data.items()}

    def overlap_block(a: str, b: str) -> dict:
        inter_s = sents[a] & sents[b]
        inter_p = pairs[a] & pairs[b]
        inter_u = unord[a] & unord[b]
        rows_b = data[b]
        both_sents = sum(1 for r in rows_b if r["text_pair"][0] in sents[a] and r["text_pair"][1] in sents[a])
        return {
            "a": a,
            "b": b,
            "shared_sentences": len(inter_s),
            "b_sentences": len(sents[b]),
            "shared_sentence_share_of_b": round(len(inter_s) / max(1, len(sents[b])), 4),
            "shared_ordered_pairs": len(inter_p),
            "shared_ordered_pair_share_of_b": round(len(inter_p) / max(1, len(rows_b)), 4),
            "shared_unordered_pairs": len(inter_u),
            "b_rows_both_sentences_in_a": both_sents,
            "b_rows_both_sentences_share": round(both_sents / max(1, len(rows_b)), 4),
        }

    report["overlap"] = [
        overlap_block("train", "validation"),
        overlap_block("train", "test_inputs"),
        overlap_block("validation", "test_inputs"),
    ]

    # ---------- 重复句对（train 内部）
    train_pairs = collections.Counter(pair_key(r) for r in data["train"])
    report["train_duplicate_ordered_pairs"] = sum(v - 1 for v in train_pairs.values() if v > 1)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "data_audit.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ---------- Markdown 表
    lines = ["# pawsx_zh 数据审计", "", "## 1. 切片与标签分布", "",
             "| 切片 | 条数 | 有标签 | label 0 (different) | label 1 (paraphrase) | 最大句长均值 | ≤64 | ≤128 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in SPLITS:
        s = report["splits"][name]
        lc = s["label_counts"]
        n0, n1 = lc.get("0", 0), lc.get("1", 0)
        lines.append(
            f"| {name} | {s['n']} | {s['n_with_label']} | {n0} | {n1} | "
            f"{s['char_len']['mean']} | {pct(s['char_len']['share_le_64'])} | {pct(s['char_len']['share_le_128'])} |")
    lines += ["", "## 2. 句子/句对重叠", "",
              "| A | B | 共享句子（占 B） | 共享有序句对（占 B） | B 中两句都出现在 A |",
              "| --- | --- | ---: | ---: | ---: |"]
    for o in report["overlap"]:
        lines.append(
            f"| {o['a']} | {o['b']} | {o['shared_sentences']} ({pct(o['shared_sentence_share_of_b'])}) | "
            f"{o['shared_ordered_pairs']} ({pct(o['shared_ordered_pair_share_of_b'])}) | "
            f"{o['b_rows_both_sentences_in_a']} ({pct(o['b_rows_both_sentences_share'])}) |")
    lines += ["", "## 3. 集合关系与重复", "",
              f"- dev100 ⊂ validation：{report['id_relations']['dev100_subset_of_validation']}"
              f"（{report['id_relations']['dev100_overlap_validation']}/100）",
              f"- compare50 ⊂ test_inputs：{report['id_relations']['compare50_subset_of_test_inputs']}"
              f"（{report['id_relations']['compare50_overlap_test_inputs']}/50）",
              f"- train 内重复有序句对：{report['train_duplicate_ordered_pairs']}"]
    with open(os.path.join(OUT, "data_audit.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
