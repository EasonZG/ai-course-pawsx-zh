# 给 GPU 队友 Agent 的开场提示词

将下面整段粘贴给接手项目的 agent。它假定当前工作目录已经是本交接包根目录；所有路径均相对根目录。

---

你是接手“中文匹配·⑩ 中文改写语义一致性核查系统（T4-PAR，pawsx_zh）”第三周 M3 的 GPU 侧实现 agent。当前交接包状态是**待 GPU 实现、训练与验收**，不是完成版；不要假设包内已有 M3 训练结果或 Word 报告。

## 第一步：读材料和确认状态

1. 完整阅读根目录 `AGENT.md`、`GPU交接_工作流程与审查方式.md`、`交接清单.md`。
2. 阅读 `第一周/第一周选题报告_中文改写语义一致性核查系统.docx`，以及 `第二周/第二周实验报告_数据与基准.docx`、`pawsx_zh/T4-PAR-中文改写语义一致性/report/M2.md`。以第二周的 dev1200/holdout746 冻结划分为准，第一周较早的 validation 用法不得覆盖它。
3. 阅读 `第三周/README.md`、`第三周/lab-report-template (1).docx`，以及原课程包的 `题目卡.md`、`README.md`、`manifest.json`、`data/metadata.json`、`data/BENCHMARK_FREEZE.json`、`data/scoring_rules.json`、`oj/run.py`、`oj/selfcheck.py`、`oj/eval_split.py`。
4. 先写 `第三周/M3_执行记录.md`（可用 `第三周/M3_执行记录_模板.md`），记录实际设备、Python/CUDA/PyTorch/transformers 版本、GPU 型号与显存、任务理解、预计文件变动和阶段状态。不要等额外确认才开始已授权工作。

## 第二步：环境与基准检查（阶段 A）

- 使用项目独立虚拟环境；记录实际版本，不把虚拟环境打包。
- 必须确认 `torch.cuda.is_available()` 为 True，并用小批次验证模型和张量确实在 GPU 上。没有可用 GPU 就停止全量训练，报告缺失条件和修复建议，不得用 CPU 结果冒充 GPU 训练。
- 在原包和复制出的 `第三周/M3_提交包/` 中分别运行 `python oj/check_benchmark.py` 与 `python oj/test_scoring_rules.py`，预期 9 项冻结检查和 9 项判分测试通过；保存 stdout、stderr、退出码。
- 核对 train 44,898 条、dev1200 1,200 条、标签映射 0=different、1=paraphrase。
- 从可信仓库获取 `google-bert/bert-base-chinese` 原始预训练权重与 tokenizer，记录来源、revision 和文件哈希；不得换成已在本题测试集训练过的模型。

## 第三步：实现与小批次验证（阶段 B）

- 从原课程包复制出 `第三周/M3_提交包/`，保留原 `oj/run.py` 等冻结文件原字节；新增 `oj/run_m3.py` 作为统一入口，`manifest.entry` 指向它。原 `selfcheck.py` 已支持 `--entry`，优先复用，不改冻结脚本。
- 句对编码为 `[CLS] A [SEP] B [SEP]`，使用 tokenizer 的 pair 编码，提供 `attention_mask`、`token_type_ids`，`truncation=longest_first`，动态 padding；所有 BERT 参数和分类头都参与训练，不冻结编码器。
- 从 train 固定抽 64–128 条，执行 2–5 个优化步骤和一次保存/重载；验证 loss 有限、至少一个编码器参数和分类头参数确实变化。烟测模型不得作为最终成绩。
- 推理只接收 `item_id/task/text_pair`；target 即使出现也忽略；不得在推理时训练、调阈值或读取开发标签。
- 必要测试：改/删 target 后同句对预测一致；checkpoint 缺失时明确报错；错误输入保留可识别 `item_id` 的失败行；合法 label 只为整数 0/1；批量与单条推理一致；统计 tokenizer 后句对长度和 64/128 截断比例。

## 第四步：正式训练与消融（阶段 C）

按 `GPU交接_工作流程与审查方式.md` 的固定配置执行：全量 train 44,898，dev1200 选模，3 epochs，lr=2e-5，AdamW，weight_decay=0.01，warmup_ratio=0.1，有效 batch 32，gradient clipping 1.0，dropout 0.1，seed=0；按实际显存选择 fp16/bf16/fp32 并记录。至少完成 E1 主配置（max_length=128，普通交叉熵）；E2（max_length=64）和 E3（类别加权交叉熵）资源允许就做，未执行必须明确写“未执行”，不能空填数字。

所有实验保存每 epoch 的 loss、dev accuracy、macro-F1、每类召回、训练时长、峰值显存；checkpoint 选择只看 dev1200，按 accuracy 最高、macro-F1 次之、epoch 更早兜底。不得用 holdout 选 epoch、loss、长度或学习率。

## 第五步：锁定与一次 holdout（阶段 D）

- 先写 `runs/m3/selection.json`，记录选择时间、候选成绩、最终配置、训练数据/代码/模型哈希；该文件必须早于 holdout 评测。
- 最终模型对 `data/splits/holdout746.jsonl` 生成一次完整预测，用冻结 `oj/eval_split.py` 判分；保存评测时间和审计记录。看结果后不得重新选模或替换模型再称“首次 holdout”。
- 同一模型生成 dev1200、dev100、test_inputs、compare50 全部预测；test/compare 只报告完整性、格式和运行时间，不报虚构 accuracy。
- 与历史词面对照做同批配对 bootstrap；只有同批逐条预测才能算配对区间。区间跨 0 时不得声称可靠提升。

## 第六步：错误分析、报告与交付（阶段 E）

- 输出完整错题清单：`item_id`、两句文本、gold、pred、概率、status、推测原因；报告引用至少三条真实错例，FP/FN 尽量各覆盖，疑似标注噪声只写“疑似”。
- 按 Jaccard、token 长度、相同句对分桶，展示每桶样本数与正确数；空桶不报虚构比例。
- 指标统一：accuracy、macro-F1、每类 P/R/F1、混淆矩阵、覆盖率、失败数、1000 次 seed=0 bootstrap 95% 区间。
- `report/M3.md` 保留课程四节；Word 使用 `第三周/lab-report-template (1).docx`，保留“一、实验目标及完成情况；二、核心材料；三、问题分析与改进；四、分工安排”。数字必须从保存预测复算。
- 原分工：赵国统筹数据/评测，钟承轩负责传统模型，李洋负责格式及后续 LLM/Agent。报告分开写“原定职责”和“实际完成记录”；未经人工复核不得写“组员已交叉验证”。
- 打包完整模型/tokenizer、数据、代码、结果、日志、manifest、Markdown/Word 报告；排除虚拟环境、下载缓存、调试 checkpoint、凭据和临时文件。解压到新目录，离线用实际入口跑 dev100 100 条并判分，核对与打包前一致。

## 硬性边界

- 传统轨不调用 LLM 或远端生成 API；推理 `usage.tokens_in/tokens_out/cost/tool_calls` 全为 0。
- 只读 `data/train.jsonl` 训练；只用 `dev1200` 选模；`holdout746` 只评一次；test/compare 标签不下载、不检索、不补全。
- 不修改原包和 M2 冻结文件；不把参考值或本机中间产物写成实测；不伪造 GPU 日志、训练轨迹、指标或报告。
- 组号若未知，沿用 `g00` 并在执行记录写明“待人工确认”，不得编造。
- 所有路径相对本包根目录；不要依赖原电脑绝对路径、本机技能或本机缓存。
- 不擅自上传平台、发送给老师或使用付费云资源。

## 开始前请先回复

请先用一段话回复：当前设备与 GPU 型号/显存、`torch.cuda.is_available()` 结果、Python/PyTorch/CUDA/transformers 版本、指定实现模型与工具是否可用、是否发现包内缺文件。确认后再按阶段 A→E 执行，并在每个阶段结束时更新 `第三周/M3_执行记录.md`。

完成后按 `GPU交接_工作流程与审查方式.md` 第 7 节回传；最终状态只能是“待人工/平台验收”，不能自称作业已完成。
