# 第三周 GPU 实施流程与审查方式

日期：2026-11-16。项目：中文匹配·⑩ 中文改写语义一致性核查系统，T4-PAR，pawsx_zh。

## 1 当前交接的目标和状态

用户本轮要求：先把工作流程和审查方式写具体，再把项目交给有 GPU 的队友，附上供对方 agent 阅读的 AGENT.md。此文是执行依据，取代此前的本机 CPU 路线。

**当前状态为待 GPU 实现、训练与验收。** 第一周选题和第二周冻结基准已完成；第三周原始课程包中的 `report/M3.md` 仍是模板，`oj/run.py` 仍是词面对照。不得把本文件、历史基线、参考成绩或本机中间产物当作 M3 已完成的证据。

Codex 负责方案和审查。实现由 OpenCode 的 `opencode-go/deepseek-v4.1-flash#max` 执行；若队友环境没有该模型或工具，先如实报告可用能力，不擅自替换指定实现模型。GPU 训练由队友设备执行，不需要原电脑在线，也不需要原电脑路径或凭据。

## 2 对方 agent 首先阅读什么

按以下顺序阅读，路径相对于交接包根目录：

1. `AGENT.md` / `AGENTS.md` 与本文。
2. `第一周/第一周选题报告_中文改写语义一致性核查系统.docx`：主假设、BERT 方案和原分工。
3. `第二周/第二周实验报告_数据与基准.docx` 与 `pawsx_zh/T4-PAR-中文改写语义一致性/report/M2.md`：冻结数据和判分协议，以第二周的新划分覆盖第一周较早的 validation 用法。
4. `第三周/README.md` 与 `第三周/lab-report-template (1).docx`：本周要求、Word 四节模板。
5. 原课程包的 `题目卡.md`、`README.md`、`data/metadata.json`、`data/BENCHMARK_FREEZE.json`、`data/scoring_rules.json`、`manifest.json`、`oj/run.py`、`oj/selfcheck.py`、`oj/eval_split.py`。

读完先产出 `第三周/M3_执行记录.md`，列实际设备、理解的任务、预计文件变动和各阶段状态。不必等待额外确认才能开展用户已经授权的实现。

## 3 数据和推理边界

| 数据 | 条数 | 允许用途 |
| --- | ---: | --- |
| train.jsonl | 44,898 | 唯一任务训练来源，含两类标签 |
| data/splits/dev1200.jsonl | 1,200 | 超参、早停、选 checkpoint、消融分析；包含全部 dev100 |
| data/splits/holdout746.jsonl | 746 | 模型与配置锁定后的一次留出评测 |
| dev100.jsonl | 100 | 格式与入口冒烟；不是独立泛化证据 |
| test_inputs.jsonl | 1,965 | 无标签，仅预测、验格式、覆盖率 |
| compare50.jsonl | 50 | 无标签，为后续三轨同数据横比保存预测 |

禁止用完整 validation 调参，因为它包含 holdout746；禁止下载、检索或补全上游测试标签；禁止把 test/compare 当训练数据。模型正常输出测试预测是允许的，“不推断测试标签”指不借助外部答案或额外模型构造 gold。

传统轨推理不调用 LLM 或远端生成 API。BERT 编码器微调属于允许的小模型训练；OpenCode 辅助写代码与被试模型推理是两件事，报告必须分开说明。输出 `usage.tokens_in/tokens_out/cost/tool_calls` 均为 0。

## 4 目录与版本处理

原始 `pawsx_zh/T4-PAR-中文改写语义一致性/` 作为只读历史基准。复制为 `第三周/M3_提交包/` 后新增实现；不覆盖前两周材料。

M2 冻结清单含 `oj/run.py` 等脚本。为保持原哈希校验，在 M3 副本中保留这些文件原字节，新增 `oj/run_m3.py` 作为统一 M3 入口，`manifest.entry` 指向它。原 `selfcheck.py` 已支持 `--entry`，直接复用，不必改冻结脚本。

M3 manifest 填真实 task_id（⑩）、module=M3、methods=[trad]、splits=[dev,test]、seed=0；group 使用已分配组号，若未知则沿用现值并在执行记录中明确待人工填入，不能编造组号。不得误称未实现 llm/agent 已可用。

若实际平台只接受固定 `oj/run.py` 且不读取 manifest.entry，应先确认平台约束，再在独立的最终平台包使用适配入口，并保留完整 M2 原包和其哈希记录作为历史证据，另生成 M3 代码版本清单。不能改写 M2 冻结清单把变化伪装成原版本。

建议新增的职责文件（由对方 agent 实现；交接包不声称已存在）：

| 文件或目录（M3 包内） | 职责 |
| --- | --- |
| oj/train_bert.py | 加载训练/开发集、训练、保存 checkpoint、开发集选模 |
| oj/run_m3.py | 只加载本地 checkpoint，按课程 CLI 输出 JSONL |
| oj/m3_model.py | 模型加载和句对推理，训练与推理共用 tokenizer 配置 |
| oj/analyze_m3.py | 逐条错误、分桶、配对 bootstrap |
| oj/test_m3.py | 数据隔离、模型更新、输出协议、离线加载等必要测试 |
| configs/m3/*.json | 明确的实验配置，含 seed 和精度模式 |
| models/trad/ | 选定权重、tokenizer、配置，不依赖外部缓存 |
| runs/m3/ | 环境、逐 epoch 日志、选择依据、全部预测、指标、错误分析、自检 |
| report/M3.md | 课程四节实验报告 |
| requirements-m3.lock.txt | 实际成功使用的完整依赖版本 |

## 5 分阶段执行流程

### 阶段 A 环境与基准检查

1. 使用项目独立虚拟环境；记录 OS、Python、GPU 型号/显存、驱动、PyTorch、CUDA、transformers 等版本。CUDA 与 PyTorch 安装按队友设备适配，参考实际官方安装文档，不照搬本机环境。不要把虚拟环境打包。
2. 必须确认 `torch.cuda.is_available()` 为 True，并用实际小批次验证张量和模型均在 GPU。没有可用 GPU 时停止全量训练，输出准确错误和所需修复条件。
3. 在原包和 M3 副本各运行 `python oj/check_benchmark.py` 与 `python oj/test_scoring_rules.py`；预期分别 9 项冻结检查和 9 项判分测试通过。记录 stdout、stderr、退出码。
4. 读取 train/dev 的结构和标签映射（0=different，1=paraphrase），核对 44,898/1,200 条。冻结检查会读取 holdout 以核验划分，这是基准检查，不得将其样本内容或标签提供给训练和模型选择。
5. 从可信模型仓库获取 `google-bert/bert-base-chinese` 的原始预训练权重和 tokenizer，记录来源及实际 revision/文件哈希。使用普通 BERT 预训练权重，不得偷换成已在本任务测试集训练的模型。网络不可用时允许队友预先准备权重，记录来源。

**审查 A：** 环境日志、冻结检查和判分测试齐全；数据、模型来源可追溯；无基准文件改动。失败先修复再训练。

### 阶段 B 完成实现并小批次验证

1. 句对编码为 `[CLS] A [SEP] B [SEP]`；正确提供 attention_mask、token_type_ids，使用 tokenizer 的 pair 编码，`truncation=longest_first`、动态 padding。
2. 实现二分类头，显式设置 label 映射；所有 BERT 参数和分类头均参与训练，不冻结编码器。
3. 从 train 内固定抽 64–128 条，执行 2–5 个优化步骤和一次模型保存/重载，不碰 holdout。验证 loss 有限、至少一个编码器参数和分类头参数确实变化；只看 loss 下降不能证明训练正确。
4. 推理只接收 item_id/task/text_pair，target 即便出现在文件中也被忽略；不得在推理时训练、调阈值或读取开发标签。
5. 测试改变/删除输入 target 后同句对预测完全一致；测试 checkpoint 缺失时明确报错而非静默下载/回退；测试错误输入保留可识别 item_id 的失败行；测试合法 label 只为整数0/1。
6. 对同一已保存权重检查批量/单条推理一致性；CUDA 计时须同步，分别报告模型加载时间与每例推理时间，批量耗时分摊需注明。
7. 统计 train/dev 的 tokenizer 后句对长度和 64/128 截断比例。第二周的“单句字符长度”不能直接当作 BERT 句对 token 长度；长序列必须真实截断而非报告为全部覆盖。

**审查 B：** 必要测试通过，参数更新有证据，GPU 显存和吞吐可接受；调试输出不得混入最终 runs。烟测模型不得作为最终成绩。

### 阶段 C 正式训练与开发集消融

固定下列配置；若实际 OOM，可按规定降低 micro batch 并记录变更，不能不留记录地换方案。

| 参数 | 值 |
| --- | --- |
| backbone | google-bert/bert-base-chinese，约 0.1B，记录实测总参数/可训练参数 |
| train | 全量 44,898 条，每 epoch 训练前 shuffle |
| dev | 仅 dev1200，用于每 epoch 评价和 checkpoint 选择 |
| epochs | 3 |
| lr / optimizer | 2e-5 / AdamW |
| weight_decay / warmup_ratio | 0.01 / 0.1 |
| effective batch | 32；micro batch16×累积2或8×累积4均可 |
| gradient clipping / dropout | 1.0 / 0.1 |
| seed | 0，覆盖 Python/NumPy/Torch/数据加载随机性 |
| precision | CUDA 支持时 fp16 或 bf16，记录实际模式；否则 fp32 |
| prediction | softmax argmax，不以 holdout 校准阈值 |

最小实验矩阵如下，不引入大规模网格搜索：

| 实验 | max_length | loss | 用途 |
| --- | ---: | --- | --- |
| E0 | — | — | 原词面对照，仅 dev1200 和 dev100，可复用 M2 留出预测 |
| E1 主配置 | 128 | 普通交叉熵 | 验证第一周主假设的任务模型 |
| E2 长度消融 | 64 | 普通交叉熵 | 比较截断与速度影响 |
| E3 权重消融 | 128 | 类别加权交叉熵 | 比较类别偏斜影响 |

E3 类别权重仅由 train 计数计算：`w_c = N / (2*n_c)`，归一尺度和实际权重写入配置。其余配置、训练样本、seed保持一致。各实验分别保存真实每 epoch 的 loss、dev accuracy、macro-F1、每类召回、训练时长和峰值显存。

每个实验先按 dev accuracy 最大、macro-F1 次之、epoch 更早再次之选择 checkpoint；再按相同指标选 E1/E2/E3 的最终模型，最后并列按 E1、E2、E3 顺序选择。保存所有候选的开发成绩。资源有限可优先完成 E1，将未执行的消融明确记为未执行，不能空填数字。

**审查 C：** 检查配置和日志能对应到模型；全量训练条数和优化步骤合理；选择依据只含 dev。不得用 holdout 选择 epoch、loss、长度或学习率。训练没有达到理想分数也保留结果，不把性能问题改写成运行失败。

### 阶段 D 锁定与一次留出评测

1. 写 `runs/m3/selection.json`，记录选择时间、候选成绩、最终配置、训练数据/代码/模型哈希。该文件先于 holdout 评测生成。
2. 最终模型对 holdout746 生成一次完整预测，由冻结 `eval_split.py` 判分；保存评测时间和审计记录。重复读取同一预测计算审查指标是允许的，但不得看结果后重新选模或训练替换，再把它叫首次留出结果。
3. 同一最终模型生成 dev1200、dev100、test_inputs、compare50 全部预测。前三类有标签的数据分别报告成绩；test/compare 只报告输出完整性和运行时间。
4. 与历史词面对照做同批配对比较。原包若没有保存 holdout 的词面逐条预测，先核对旧 `run.py` 与 M2 哈希，再仅用 dev1200 重新得到原规则并生成词面 holdout 预测；标注为“重放冻结对照”，不重新选择规则，不据此改模型。只有同批逐条预测可算配对区间，不能拿两个点分数假造配对 bootstrap。

**审查 D：** selection 时间早于 holdout 评测，checkpoint 哈希一致，分母不缩小，所有 split 行数和 ID 集合与输入一致，所有 usage 为 0。保留负结果；本地 0.6675 以上也不等于平台门禁通过。

### 阶段 E 错误分析、报告和交付

1. 输出完整错题清单，至少包括 item_id、两句文本、gold、pred、概率、status、推测原因。报告引用至少三条真实错例，FP/FN均有时各覆盖一个，标注噪声只列“疑似”而不擅自改 gold。
2. 分桶统计 Jaccard（<0.3、0.3–0.6、0.6–0.9、≥0.9）、句对 token 长度（≤64、65–128、>128）、相同句对，展示每桶样本数与正确数；边界值和空桶处理要明确，空桶不报虚构比例。
3. 指标统一：accuracy、macro-F1、每类P/R/F1、混淆矩阵、覆盖率、失败数、95% bootstrap区间（1000次，seed0）。配对提升区间跨0时不声称可靠提升。
4. `report/M3.md` 保留课程报告结构；Word使用第三周原模板并保留“一、实验目标及完成情况；二、核心材料；三、问题分析与改进；四、分工安排”。内容包括真实训练参数范围、命令、产物、指标、三条错例、局限、AI使用及下一步。
5. 原分工是赵国统筹数据/评测、钟承轩负责传统模型、李洋负责格式及后续LLM/Agent。报告区分原定职责和实际完成记录；未经人工复核不得写“组员已交叉验证”。
6. Word先渲染为PDF/页面PNG并逐页检查，修正断表、缺字、溢出、空白页；工具不可用如实说明，不把结构检查当作视觉检查完成。
7. 整理 `README_M3.md`、实际依赖版本、下载模型的方法和离线推理方法；打包完整模型、tokenizer、数据、代码、结果、日志、manifest、Markdown/Word报告。排除虚拟环境、下载缓存、调试checkpoint、凭据、临时文件。
8. 解压至新目录，离线用实际入口跑至少 dev100 的完整100条并判分，核对与打包前相同。不能借用训练设备缓存才能启动。模型过大时明确包大小和传输方式，不能悄悄漏掉权重。

**审查 E：** 报告数字从保存预测重新计算一致，Word四节齐全，交付包可离线重载，未执行项准确披露。平台test400分数仅在真实平台评测后填写；题目卡0.6975/0.74是教师参考值，不是本组成绩。

## 6 命令约定

以下命令在 `第三周/M3_提交包/` 根目录、激活实际训练环境后执行。`train_bert.py`、`run_m3.py`、`test_m3.py`、配置文件由接手 agent 按本文实现，交接阶段尚未验收；不能把此命令清单当作已经运行的日志。

```text
python oj/check_benchmark.py
python oj/test_scoring_rules.py
python oj/test_m3.py
python oj/train_bert.py --config configs/m3/E1.json
python oj/train_bert.py --config configs/m3/E2.json
python oj/train_bert.py --config configs/m3/E3.json
```

训练脚本必须保存选模依据，最终模型导出到 `models/trad/` 后再执行：

```text
python oj/run_m3.py --method trad --split dev --input data/splits/dev1200.jsonl --out runs/m3/results.dev1200.jsonl --seed 0
python oj/run_m3.py --method trad --split holdout --input data/splits/holdout746.jsonl --out runs/m3/results.holdout746.jsonl --seed 0
python oj/eval_split.py --gold data/splits/holdout746.jsonl --pred runs/m3/results.holdout746.jsonl --out runs/m3/metrics.holdout746.json
python oj/selfcheck.py --method trad --entry oj/run_m3.py
python oj/run_m3.py --method trad --split test --input data/test_inputs.jsonl --out runs/m3/results.test.jsonl --seed 0
python oj/run_m3.py --method trad --split compare --input data/compare50.jsonl --out runs/m3/results.compare.jsonl --seed 0
```

补齐 dev1200/dev100 的 `eval_split` 指标和永久结果文件；selfcheck 使用临时目录，不能仅凭控制台记录替代永久逐条输出。Windows 命令行设置 `PYTHONUTF8=1`、`PYTHONIOENCODING=utf-8`，Linux 用等价环境设置。不要把原电脑 `D:` 路径写死进代码。

## 7 最终审查表与回传

| 检查点 | 通过证据 | 不通过时 |
| --- | --- | --- |
| 基准未变 | 原包与M3冻结检查退出码0，9项全通过 | 恢复冻结文件并定位改动 |
| 判分一致 | 9项判分测试通过，复用冻结判分器 | 修复新增逻辑，不改gold |
| 真实训练 | GPU/环境日志、训练参数数目、参数变化与epoch日志、checkpoint | 不写“已训练完成” |
| 数据隔离 | train/dev来源哈希、代码读取路径、selection与holdout时间顺序 | 披露泄漏并停止声称无污染 |
| 预测完整 | 1200/746/100/1965/50条，ID匹配、无重复、五键格式 | 补齐失败状态、重验覆盖，不删分母 |
| 推理独立 | 改target不变、离线加载、缺模型明确失败 | 修复后重新验证 |
| 结果可信 | 保存预测复算指标、bootstrap配置、错例可定位 | 更正报告、保留原日志 |
| 报告合格 | 四节、训练范围、命令、指标、≥3错例、AI说明、分工 | 补齐真实内容，不填猜测 |
| 可携带 | 新目录离线dev100预测一致，manifest入口真实可用 | 补齐模型/tokenizer/依赖说明 |
| 平台门禁 | 平台真实test400报告 | 保持“待平台验收”状态 |

队友最终回传：完整 `M3_提交包` 或其zip、Word报告、执行记录、依赖锁、环境与训练日志、全部逐条预测和指标、selection.json、模型哈希、必要测试日志、离线自检日志；若模型单独传输需同时给文件列表和SHA-256。Codex据此逐项审查，不只接受agent一句“已完成”。

训练后不要擅自上传平台、发送给老师或调用付费资源。交给用户审阅并决定最终提交。
