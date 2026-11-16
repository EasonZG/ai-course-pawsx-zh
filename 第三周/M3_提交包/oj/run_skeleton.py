#!/usr/bin/env python3
"""统一评测入口骨架 —— 三种方法（trad / llm / agent）都从这一个文件进出。

平台这样调用它（在判题容器的提交包根目录里，cwd = 包根）：
    python oj/run.py --method trad --split dev --input items.jsonl --out results.jsonl --seed 0

你只需要保证三件事（缺一条就会被判题打回）：
  1. 读完 --input 的每一行（JSON，至少含 item_id；具体字段见题目卡 data/metadata.json）；
  2. 往 --out 写 results.jsonl，**每行 5 个键**：
         {"item_id": "...", "output": <按任务形态>, "status": "ok|fail|timeout|error",
          "latency_ms": 12, "usage": {"tokens_in":0,"tokens_out":0,"cost":0.0,"tool_calls":0}}
     - 失败的样也要写一行（status=fail），不要静默丢掉 —— 分母是全部测试题，丢行等于自己扣分；
     - 每写一行就 flush()：平台超时会直接 kill 进程，没落盘的部分会丢。
  3. output 的形状严格按 metadata.json 的 task（写成别的样子会被判无效，不会被"猜"成对的）：
         single_label : {"label": <整数下标>}          # 下标就是 metadata.json 里 labels 的序号
         multilabel   : {"labels": [<整数下标>, ...]}   # 非空、不重复
         multi_aspect : {"aspects": {"<方面名>": <-2|-1|0|1>, ...}}  # 必须覆盖全部方面

三种方法怎么区分：平台用同一个包、同一个入口，只换 --method。
    trad  : 不许调用任何 LLM（usage.tokens_in 必须为 0）
    llm   : 每个测试样例恰好调用一次模型，不许用工具/检索
    agent : 至少一次真实工具调用，能读到返回并继续决策（轨迹要能回放）
"""
from __future__ import annotations

import argparse
import json
import os
import time


# --------------------------------------------------------------------------- 工具
def read_items(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def emit(f, item_id, output, status="ok", latency_ms=0, usage=None):
    """写一行结果并立即落盘。"""
    f.write(json.dumps({
        "item_id": item_id,
        "output": output,
        "status": status,
        "latency_ms": int(latency_ms),
        "usage": usage or {"tokens_in": 0, "tokens_out": 0, "cost": 0.0, "tool_calls": 0},
    }, ensure_ascii=False) + "\n")
    f.flush()


# --------------------------------------------------------------------------- 三轨实现
def predict_trad(item, ctx):
    """传统方法：不得调用 LLM。在这里接你的模型（见 examples/T3-INT-clinc150 的完整例子）。"""
    # TODO: 用 ctx 里的模型/词典/规则产出预测
    raise NotImplementedError("还没实现 trad 轨")


def predict_llm(item, ctx):
    """单 LLM 基线：一次调用，不调用工具、不检索、不自修正。

    key 由平台注入（OJ_LLM_BASE_URL / OJ_LLM_MODEL / OJ_LLM_API_KEY），不要写死在代码里。
    """
    # TODO: 一次 chat completion -> 解析成 output；usage 里如实填 tokens_in/tokens_out/cost
    raise NotImplementedError("还没实现 llm 轨")


def predict_agent(item, ctx):
    """Agent：至少一次工具调用 + 读返回 + 继续决策。

    工具建议（任选 1-3 个，都要记账）：
      - 检索工具：从 data/train.jsonl 里找相似样例（返回 top-k 文本与标签）
      - 校验工具：检查输出是否满足 metadata.json 的标签/方面集合
    每次模型决策与工具调用都要落到 usage / 轨迹文件里（runs/<method>/traces/*.jsonl）。
    """
    # TODO: 循环：模型决策 -> 调工具 -> 读结果 -> 再决策 -> 输出
    raise NotImplementedError("还没实现 agent 轨")


# --------------------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="trad", choices=["trad", "llm", "agent"])
    ap.add_argument("--split", default="dev")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    items = read_items(args.input)
    ctx = {"seed": args.seed, "split": args.split, "root": os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}
    fn = {"trad": predict_trad, "llm": predict_llm, "agent": predict_agent}[args.method]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for item in items:
            t0 = time.time()
            try:
                output, usage = fn(item, ctx)
                emit(f, item["item_id"], output, "ok", (time.time() - t0) * 1000, usage)
            except NotImplementedError:
                # 未实现也要写一行，否则覆盖率 < 1 会被判题直接判失败
                emit(f, item["item_id"], None, "fail", (time.time() - t0) * 1000)
            except Exception:
                emit(f, item["item_id"], None, "error", (time.time() - t0) * 1000)
    print(f"{args.method}: {len(items)} 行 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
