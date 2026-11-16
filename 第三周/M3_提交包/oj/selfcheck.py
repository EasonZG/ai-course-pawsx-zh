#!/usr/bin/env python3
"""本地自检：用**与平台同源的判分器**跑 dev 集，给出和平台一致的数字。

用法（在提交包根目录）：
    python oj/selfcheck.py                 # 默认 --method trad，跑 data/dev100.jsonl
    python oj/selfcheck.py --method llm
    python oj/selfcheck.py --dev data/dev100.jsonl --entry oj/run.py

它做四件事：
  1. 按平台的方式调用你的入口（同参数、同 cwd）；
  2. 用同源判分器算指标（缺行/重复 ID/非法标签都留在分母里 —— 和平台一模一样）；
  3. 和题目卡里的验收下界对照（注意：下界标注为"未校准"时只是参考）；
  4. 给出可行动的诊断（覆盖率不足、标签名写成字符串、空输出……）。

注意：dev 集有答案可以自检，**正式成绩用 compare 集（无答案）由平台判**。别把 dev 成绩当最终成绩。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import scoring  # noqa: E402  （与平台 judge/scoring.py 同一份文件）


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="trad")
    ap.add_argument("--dev", default=os.path.join(ROOT, "data", "dev100.jsonl"))
    ap.add_argument("--entry", default=os.path.join(ROOT, "oj", "run.py"))
    ap.add_argument("--meta", default=os.path.join(ROOT, "data", "metadata.json"))
    ap.add_argument("--gate", default=os.path.join(ROOT, "验收.json"))
    args = ap.parse_args()

    if not os.path.exists(args.dev):
        print(f"找不到 dev 集：{args.dev}（题目卡里说明了它叫什么）")
        return 2
    rows = load_jsonl(args.dev)
    meta = scoring.load_meta(args.meta)
    meta.pop("trivial_baseline", None)
    gold = {r["item_id"]: r["target"] for r in rows if "target" in r}
    items = [{k: v for k, v in r.items() if k != "target"} for r in rows]

    tmp = tempfile.mkdtemp()
    items_path = os.path.join(tmp, "items.jsonl")
    out_path = os.path.join(tmp, "results.jsonl")
    with open(items_path, "w", encoding="utf-8") as f:
        for r in items:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cmd = [sys.executable, args.entry, "--method", args.method, "--split", "dev",
           "--input", items_path, "--out", out_path, "--seed", "0"]
    print("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.stdout.strip():
        print(proc.stdout.strip()[:1500])
    if proc.returncode != 0:
        print("入口退出码非 0：", proc.returncode)
        print((proc.stderr or "")[-1500:])
        return 1

    if not os.path.exists(out_path):
        print("❌ 没有产出 results.jsonl（检查 --out 参数是否是最后才写的）")
        return 1
    preds = [{"id": r.get("item_id"), "prediction": r.get("output")} for r in load_jsonl(out_path)]
    m = scoring.score(meta, gold, preds)

    print("\n=== 本地指标（与平台同口径）")
    print(f"  题目数 {m['total']} / 收到的预测 {m['prediction_records']} / 缺失 {m['missing']} / "
          f"重复 {m['duplicate_ids']} / 未知 ID {m['unknown_records']}")
    print(f"  合法输出率 {m['valid_output_rate']:.3f}（非法/缺失都算错，不缩小分母）")
    for k in ("accuracy", "macro_f1", "micro_f1", "exact_match", "aspect_accuracy",
              "oos_precision", "oos_recall", "oos_f1", "in_domain_macro_f1"):
        if k in m and isinstance(m[k], (int, float)):
            print(f"  {k:18s} {m[k]:.4f}")
    if m.get("hint"):
        print(f"  ⚠ {m['hint']}")

    # 与验收下界对照
    if os.path.exists(args.gate):
        g = json.load(open(args.gate, encoding="utf-8"))
        mins = g.get("min") or g.get("provisional_min") or {}
        status = g.get("status", "")
        print(f"\n=== 验收下界对照（{status or '无状态'}）")
        print("  ⚠ 注意：下界是在平台判题用的正式切片上校准的（compare 50 或 test 100/400，见题目卡 §8），这里显示的是 dev100 的分数。")
        print("    两者不是同一批题，只能看方向（差得多就该担心），不能当成'我过没过'。")
        for k, v in mins.items():
            got = m.get(k)
            if isinstance(got, (int, float)):
                flag = "✅" if got >= v else "❌"
                print(f"  {flag} {k}: {got:.4f} vs 下界 {v}")
        if "未校准" in status or "uncalibrated" in status:
            print("  提示：下界尚未校准（S2 教师试做后填正式值），这里只作参考，不是最终门禁。")

    # 诊断
    print("\n=== 诊断")
    if m["missing"]:
        print(f"  ❌ 有 {m['missing']} 题没交结果 → 平台判题会因覆盖率 < 1 直接判失败")
    if m["duplicate_ids"]:
        print(f"  ❌ 有 {m['duplicate_ids']} 个 item_id 重复 → 重复的整条作废（不是最后一条生效）")
    if m["unknown_records"]:
        print(f"  ⚠ 有 {m['unknown_records']} 条预测的 item_id 不在题目里，被忽略")
    if m["valid_output_rate"] < 1:
        print(f"  ❌ {int((1 - m['valid_output_rate']) * m['total'])} 条输出格式非法 → 看题目卡的输出格式")
    if m["valid_output_rate"] == 1 and not m["missing"] and not m["duplicate_ids"]:
        print("  ✅ 格式与覆盖率都没问题，可以提交")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
