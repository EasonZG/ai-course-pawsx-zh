#!/usr/bin/env python3
"""例题：中文改写语义一致性（T4-PAR ⑩）的传统方法轨 —— **零依赖「词语重合」对照基线**。

这个例题不是解法，是**下界对照**（课程 ⑩ 要求的"以词语重合方法作初始对照"）：

    判据：字二元组 Jaccard + 包含度（两句共有的二字组，占短句多少）
    阈值：在 data/validation.jsonl 上网格搜索，直接选验证集准确率最高的一组

它跑得比谁都快（400 条 < 1 秒），分数也稳定地**不够**：

    平台正式切片（test 400）  0.5575   门禁 0.6675   → 差 11 个百分点
    冻结 bert + 线性头        0.5275   （比词面重合还差）
    微调 bert（20k/2ep）      0.6975   （test 400，教师参考基线）
    微调 bert（全量/3ep）     0.7400   （compare 50，上界参考）

为什么把"更贵"的冻结编码器也列出来：因为它在这道题上**不如词面重合**——
PAWS-X 训练集是机翻、验证/测试是人工翻译，且对手句对用词高度重合，
标签取决于细微语序/指代差别；冻结特征学不到，必须让任务相关的参数参与训练。

所以这个例题想让你先得到三件事（也是报告里必须写出来的）：
  1. 协议通了：一个包、一个入口、三种 method，结果逐行落盘、可被判分；
  2. 下界在哪：词面重合只到 0.56，**调阈值/加词表最多换 1-2 个点**，不是解决方案；
  3. 差距是什么：把 0.56 与门禁 0.6675 之间的 10 个点，用语义表示（微调/检索/LLM）补上。

用法（在提交包根目录）：
    python oj/run.py --method trad --split test --input data/compare50.jsonl --out runs/trad/cmp.jsonl --seed 0
    python oj/selfcheck.py            # dev100 自检，数字与平台同口径

换方法：这是 trad 轨。llm / agent 轨请另写实现但**共用同一个入口**（平台只换 --method 调你）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# 词面重合的两个判据（都在 [0,1]）：Jaccard 看"整体像不像"，包含度看"短句被覆盖多少"
DEFAULT_JAC = 0.86
DEFAULT_CONT = 0.95
TUNE_ROWS = 1200          # 调阈值用的验证集条数（够用且快；全量 1946 条也行）


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def bigrams(text: str) -> set[str]:
    s = "".join((text or "").split())
    if len(s) < 2:
        return {s} if s else set()
    return {s[i:i + 2] for i in range(len(s) - 1)}


def pair_features(item: dict) -> tuple[float, float]:
    a, b = item.get("text_pair") or ["", ""]
    A, B = bigrams(a), bigrams(b)
    inter = len(A & B)
    union = len(A | B) or 1
    jac = inter / union
    cont = inter / (min(len(A), len(B)) or 1)
    return jac, cont


def label_of(row: dict) -> int:
    t = row.get("target")
    if isinstance(t, dict):
        return int(t["label"])
    if isinstance(t, int):
        return int(t)
    return -1


def tune_threshold(root: str, tune_path: str | None = None) -> tuple[float, float, str]:
    """在开发集上网格搜索阈值（M2 冻结协议：只用 dev1200，不碰 holdout/test）。"""
    path = tune_path or os.path.join(root, "data", "splits", "dev1200.jsonl")
    if not os.path.exists(path):
        path = os.path.join(root, "data", "validation.jsonl")
    if not os.path.exists(path):
        return DEFAULT_JAC, DEFAULT_CONT, "没有开发集，用默认阈值"
    rows = [r for r in load_jsonl(path) if label_of(r) >= 0][:TUNE_ROWS]
    if not rows:
        return DEFAULT_JAC, DEFAULT_CONT, "开发集没有 target，用默认阈值"
    feats = [pair_features(r) for r in rows]
    ys = [label_of(r) for r in rows]
    best = (-1.0, DEFAULT_JAC, DEFAULT_CONT)
    for jac_th in [i / 100 for i in range(60, 100)]:            # 0.60 ~ 0.99
        for cont_th in [i / 100 for i in range(80, 101)]:       # 0.80 ~ 1.00
            correct = 0
            for (jac, cont), y in zip(feats, ys):
                pred = 1 if (jac >= jac_th and cont >= cont_th) else 0
                correct += pred == y
            acc = correct / len(ys)
            if acc > best[0]:
                best = (acc, jac_th, cont_th)
    return best[1], best[2], f"在 {len(rows)} 条验证集上调阈值：准确率 {best[0]:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="trad", choices=["trad", "llm", "agent"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tune", default=None, help="调参用的有标签开发集，默认 data/splits/dev1200.jsonl")
    args = ap.parse_args()

    if args.method != "trad":
        raise SystemExit("这个例题只实现了 trad 轨（词面重合对照）；"
                         "llm/agent 轨请参照 examples/starter/oj/run.py 自己补，"
                         "并保持 usage 里 tokens_in / tool_calls 如实记账。")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    jac_th, cont_th, why = tune_threshold(root, args.tune)
    items = load_jsonl(args.input)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    t0 = time.time()
    with open(args.out, "w", encoding="utf-8") as f:
        for it in items:
            t1 = time.time()
            jac, cont = pair_features(it)
            label = 1 if (jac >= jac_th and cont >= cont_th) else 0   # 1 = 改写（同义）
            f.write(json.dumps({
                "item_id": it["item_id"],
                "output": {"label": label},          # 必须是整数下标（labels: 0=different, 1=paraphrase）
                "status": "ok",
                "latency_ms": round((time.time() - t1) * 1000, 3),
                "usage": {"tokens_in": 0, "tokens_out": 0, "cost": 0.0, "tool_calls": 0},
            }, ensure_ascii=False) + "\n")
            f.flush()

    print(f"{why}；预测 {len(items)} 条，用时 {time.time() - t0:.2f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
