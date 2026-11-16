#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M3 统一入口：--method trad（字符 n-gram 线性模型）/ --method bert（可选离线 BERT）。

平台调用方式保持不变：
    python oj/run_m3.py --method trad --split test --input data/test_inputs.jsonl \
        --out runs/m3/results.test.jsonl --seed 0

约定：
- 每行输出 item_id/output/status/latency_ms/usage 五个键；label 为整数 0/1；
- usage 全为 0，传统方法不调用任何 LLM；
- 逐行 flush；单条失败仍写一行（status=error，label=-1），不丢分母；
- 输入 target 不参与推理；不读取任何训练/开发标签；
- 模型缺失时明确失败，不静默回退重训。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from m3_model import ModelError, TradModel  # noqa: E402

ZERO_USAGE = {"tokens_in": 0, "tokens_out": 0, "cost": 0.0, "tool_calls": 0}
TZ = timezone(timedelta(hours=8))


def utf8_stdio() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass


def validate_item(it: dict) -> tuple[str, str, str]:
    iid = it.get("item_id")
    if not isinstance(iid, str) or not iid.strip():
        raise ValueError("item_id 缺失或不是非空字符串")
    pair = it.get("text_pair")
    if not isinstance(pair, list) or len(pair) < 2:
        raise ValueError("text_pair 必须是至少包含两句的列表")
    a, b = pair[0], pair[1]
    if not isinstance(a, str) or not isinstance(b, str):
        raise ValueError("text_pair 的前两句必须是字符串")
    return iid, a, b


def main() -> int:
    utf8_stdio()
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="trad", choices=["trad", "bert"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-dir", default=None, help="trad 模型目录，默认 models/trad")
    ap.add_argument("--bert-model", default=None, help="bert 模型目录，默认 models/bert")
    args = ap.parse_args()

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sidecar = out_path + ".errors.jsonl"
    if os.path.exists(sidecar):
        os.remove(sidecar)

    model_desc = ""
    predictor = None
    try:
        if args.method == "trad":
            model_dir = args.model_dir or os.path.join(ROOT, "models", "trad")
            predictor = TradModel(model_dir)
            model_desc = predictor.info().get("model_sha256", "")[:16]
        else:
            # 延迟导入：没有 torch/transformers 时给出明确错误，而不是伪造结果
            from run_bert import BertPredictor  # noqa: WPS433
            model_dir = args.bert_model or os.path.join(ROOT, "models", "bert")
            predictor = BertPredictor(model_dir)
            model_desc = getattr(predictor, "model_hash", "")[:16]
    except Exception as e:
        print(f"[M3] 模型加载失败（method={args.method}）：{e}", file=sys.stderr)
        return 2

    try:
        with open(args.input, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
            n = ok = err = 0
            t_all = time.time()
            err_f = None
            for lineno, line in enumerate(fin, 1):
                if not line.strip():
                    continue
                n += 1
                t0 = time.perf_counter()
                it = None
                try:
                    it = json.loads(line)
                    if not isinstance(it, dict):
                        raise ValueError("输入行不是 JSON 对象")
                    iid, a, b = validate_item(it)
                    # 注意：it 中的 target 不参与任何计算；入口从不读取标签
                    if args.method == "trad":
                        label, _score, _stats = predictor.predict_pair(a, b)
                    else:
                        label, _prob = predictor.predict_pair(a, b)
                    row = {
                        "item_id": iid,
                        "output": {"label": int(label)},
                        "status": "ok",
                        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                        "usage": dict(ZERO_USAGE),
                    }
                    ok += 1
                except Exception as e:
                    err += 1
                    row = {
                        "item_id": it.get("item_id") if isinstance(it, dict) and isinstance(it.get("item_id"), str) else "",
                        "output": {"label": -1},
                        "status": "error",
                        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                        "usage": dict(ZERO_USAGE),
                    }
                    if err_f is None:
                        err_f = open(sidecar, "w", encoding="utf-8")
                    err_f.write(json.dumps({
                        "at": datetime.now(TZ).isoformat(timespec="milliseconds"),
                        "line": lineno,
                        "item_id": row["item_id"],
                        "error_type": type(e).__name__,
                        "error": str(e),
                        "traceback": traceback.format_exc(limit=3),
                    }, ensure_ascii=False) + "\n")
                    err_f.flush()
                    print(f"[M3] 第 {lineno} 行失败 item_id={row['item_id']!r}: {e}", file=sys.stderr)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
            if err_f is not None:
                err_f.close()
            elapsed = time.time() - t_all
    except Exception as e:
        print(f"[M3] 输入读取/输出写入失败：{e}", file=sys.stderr)
        return 3

    print(json.dumps({
        "method": args.method,
        "split": args.split,
        "model": model_desc,
        "n": n,
        "ok": ok,
        "error": err,
        "elapsed_s": round(elapsed, 3),
        "items_per_s": round(n / elapsed, 2) if elapsed > 0 else None,
        "out": out_path,
        "errors_sidecar": sidecar if err else None,
        "usage_all_zero": True,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
