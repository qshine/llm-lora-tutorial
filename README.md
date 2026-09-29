# LoRA 微调与 vLLM 部署练习：小学科学探索伙伴

写给不做算法、想从零学习 LoRA 微调的开发者，不需要机器学习背景。在一张 RTX 4090 上亲手完成：QLoRA 微调 → 保存与重载 → BF16 合并 → vLLM 部署成 OpenAI 兼容 API → 本地调用与压测，并弄清每一步显存花在哪里。练习任务是用中文回答小学生的科学问题。

## 打开电子书

```bash
./book_review.sh
```

脚本会准备阅读环境，并打开 <http://127.0.0.1:8000>。首次需要 Python 3.11 和网络。端口被占用时用 `./book_review.sh 8001`，按 `Ctrl+C` 停止。阅读无需 GPU 或模型权重。

## 一份数据，直接训练

只维护 [data/dataset.json](data/dataset.json)：`facts` 是 60 张参考事实卡，平级的 `train`、`val`、`test` 分别是 500 段训练、50 段验证、50 段测试对话。事实卡按 `fact_ids` 拼进每题的 system 提示词，作为回答依据，不是学习目标。训练部分按轮展开为 1150 个回答目标。资料来源、分类和审核记录都在同一个文件中。

[数据说明](data/README.md) 用完整的昼夜样例解释每个字段、哪些进入模型、哪些只是审核信息。Notebook 的 D1～D7 注释标出读取、筛选、上下文构造、分词、训练器入口、损失范围和保存位置。

只检查数据格式，不需要 PyTorch：

```bash
python3.11 -m lab.data --input data/dataset.json --allow-draft
```

Notebook 直接读取这个 JSON，在内存中生成训练样本，无需预先运行 `lab.prepare`。训练时自动在 `runs/science-001/dataset.json` 留一份实际输入副本。旧 `data/processed/seed-v1`、`seed-v2` 保留用于复查；只有命令行实验才需要新生成冻结快照，见教程第 4.6 节。

## 阅读入口

- [实操教程](tasks/plan.md)：样例、命令、字段含义与结果解释。
- [学习清单](tasks/todo.md)：区分代码检查与待完成的训练实验。
- [验证记录](tasks/verification.md)：已经运行的检查及结果。
- [电子书维护](book/README.md)：构建、链接检查与发布方法。

## 在 GPU 上逐格执行

运行 `python3 tools/build_gpu_zip.py` 生成 `llm-lora-tutorial-gpu.zip`，上传到 GPU 数据盘并解压。按教程第 3 章准备环境、从魔搭下载完整模型并校验，再打开 [01_gpu_train.ipynb](notebooks/01_gpu_train.ipynb)，选择 **LoRA · Python 3.11** 内核。

Notebook 针对 RTX 4090 / 24 GB，以直接的 Python 代码展示原模型回答、LoRA 配置、1 个 Epoch 训练（约 72 步）、loss 曲线、保存、重载、人工评估与 BF16 合并。每段代码下面解释用途和理由；环境安装与模型下载放在电子书中。最后附 GPU 终端 vLLM 部署和本地调用示例；教程第 10 章还包括 KV Cache 显存估算、直接挂载 adapter、OpenAI SDK 流式调用和 `vllm bench serve` 压测。

默认产物在 `runs/science-001/`：`training-complete.json` 记录训练完成，`adapter/` 是 LoRA 增量，`merged-model/` 是供 vLLM 使用的完整模型。可生成回答不代表质量提升，需查看 `answer-review.json` 并完成评审。

全部对话均为合成草稿，覆盖小学 1–6 年级，按 1–2、3–4、5–6 分段。50 个训练主题各 10 条路径，主题间隔离 val/test；主题内部分知识性回答复用。仍需逐条核对事实、来源和适读性。4B GPU 训练、效果评测和 vLLM 服务待实际执行；不能保证微调优于完整提示词基线。

## 代码地图

| 文件 | 职责 |
|---|---|
| `lab/data.py`、`prepare.py`、`common.py` | 校验源数据、展开问答、构造提示词与训练样本 |
| `lab/audit.py` | 用真实 TRL collator 检查监督目标 |
| `lab/download.py`、`train.py`、`training.py` | 固定模型版本、训练和恢复 |
| `lab/infer.py`、`evaluate.py` | 生成对照回答、盲评与结果统计 |
| `lab/merge.py` | 独立合并导出 |
| `notebooks/01_gpu_train.ipynb`、教程第 4～5 章 | 直接训练代码、产物验证和 API 调用示例 |

`models/`、`runs/`、`data/processed/` 保存本地产物。生成物不纳入源码版本控制。
