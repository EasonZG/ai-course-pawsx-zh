# T4-PAR 中文改写语义一致性 · 开始三步

```bash
# 1) 自检工具就位（不需要装任何依赖）
python oj/selfcheck.py --method trad        # 会调 oj/run.py 跑 dev100 并给出与平台同口径的分数

# 2) 实现你的传统方法（在 oj/run.py 里接上你的模型；三轨结构见 oj/run_skeleton.py）
#    三轨正式比较判的是 compare50；本地可以先在 test_inputs 上跑通流程
python oj/run.py --method trad --split test \
    --input data/compare50.jsonl --out runs/trad/results.compare.jsonl --seed 0
python oj/run.py --method trad --split test \
    --input data/test_inputs.jsonl --out runs/trad/results.test.jsonl --seed 0

# 3) 打包提交
python oj/selfcheck.py                       # 提交前最后一遍
zip -r my-submission.zip oj data manifest.json report
```

- **判哪一刀由 `manifest.json` 的 `splits` 最后一项决定**：M3 验收用 `["dev","test"]`（pawsx/clinc 是 400 条，
  门禁就在这一刀上校准）；M8 三轨横比用 `["dev","compare"]`（50 条，成本友好，但只能检出 12pp 以上差异）。
- 先读 [题目卡.md](题目卡.md)：输入输出形状、指标口径、验收红线都在里面。
- `oj/scoring.py` 与平台判分器是**同一份文件**，所以 selfcheck 的分数就是平台会给的分数（dev 集）。
- 三种方法（trad / llm / agent）共用 `oj/run.py`，只换 `--method`。

## M2 基准（开发/验收分离）

`validation` 已冻结切成两份（`seed=0`，按标签分层）：

| 文件 | 条数 | 用途 |
| --- | ---: | --- |
| `data/splits/dev1200.jsonl` | 1,200 | 调阈值、早停、提示词/超参、消融；含全部 `dev100` |
| `data/splits/holdout746.jsonl` | 746 | 留出验收，开发期不读标签、不参与选择 |
| `data/BENCHMARK_FREEZE.json` | — | 数据/划分/规则/脚本的 SHA-256 冻结记录 |

```bash
python oj/analyze_benchmark.py       # 数据审计 -> runs/benchmark/data_audit.md
python oj/make_split.py              # 生成划分（只在首次/升版本时跑）
python oj/freeze_benchmark.py        # 写冻结记录
python oj/check_benchmark.py         # 冻结自检，9 项全绿
python oj/test_scoring_rules.py      # 判分口径 9 条最小测试
python oj/eval_split.py --gold data/splits/holdout746.jsonl \
  --pred runs/trad/results.holdout.jsonl --out runs/benchmark/trad.holdout.metrics.json
```

判分规则见 `data/scoring_rules.json`：单标签整数 exact match；缺行/重复 ID/标签名/越界/多答案都算错且不缩小分母；主指标 accuracy，附 macro-F1、每类 P/R/F1、混淆矩阵、1,000 次 bootstrap 95% CI。`oj/run.py` 的 `--tune` 默认只读 `dev1200`，不要再拿完整 `validation` 调参。
