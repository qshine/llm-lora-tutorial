<div align="center">

# LoRA 微调与 vLLM 部署实操手册

**小学科学探索伙伴 · 在一张 RTX 4090 上从零走通 QLoRA 微调、合并导出与 vLLM 部署**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3110/)
[![Model: Qwen3-4B](https://img.shields.io/badge/Model-Qwen3--4B--Instruct--2507-8C432E)](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
[![GPU: RTX 4090](https://img.shields.io/badge/GPU-RTX%204090%20·%2024%20GB-76B900?logo=nvidia&logoColor=white)](tasks/plan.md)
[![E-book](https://img.shields.io/badge/电子书-在线阅读-A7492D)](https://qshine.github.io/llm-lora-tutorial/)

**简体中文** · [English](README.en.md) · [日本語](README.ja.md) · [한국어](README.ko.md)

[在线阅读](https://qshine.github.io/llm-lora-tutorial/) · [实操教程](tasks/plan.md) · [GPU Notebook](notebooks/01_gpu_train.ipynb) · [数据说明](data/README.md)

</div>

---

## 简介

写给**不做算法、想从零学习 LoRA 微调的开发者**——后端、前端、客户端、数据开发都可以，不需要机器学习背景。你会在一张 RTX 4090 上亲手完成：

```text
QLoRA 微调 → 保存与重载 → BF16 合并 → vLLM 部署成 OpenAI 兼容 API → 本地调用与压测
```

并弄清每一步显存花在哪里、文件在哪里、怎样确认这一步真的成功了。练习任务是用中文回答小学生的科学问题：对话风格鲜明，微调前后的变化容易看出来。

## 特点

| | |
|---|---|
| 📖 **完整电子书** | 9 章正文 + 5 个附录，从概念、数据、租 GPU 到部署、评估与排错；支持中文、English、日本語、한국어 一键切换 |
| 🧪 **可直接运行的 Notebook** | 直接展示 `LoraConfig`、`SFTTrainer`、`trainer.train()`，每格代码下写明做什么、为什么、会看到什么 |
| 🗂️ **一份数据** | `data/dataset.json` 含 60 张事实卡与 500 / 50 / 50 段训练、验证、测试对话，按主题隔离 |
| 🚀 **训练到部署全链路** | 合并导出、vLLM 服务、挂载 adapter、OpenAI SDK 流式调用、`vllm bench serve` 压测 |
| ✅ **测试守护** | 教程里的命令、附录代码和部署示例都由测试与真实代码逐字比对 |

## 快速开始

### 1. 阅读电子书

在线阅读：<https://qshine.github.io/llm-lora-tutorial/>（页面右上角可切换中文 / English / 日本語 / 한국어）。也可以在本地预览：

```bash
./book_review.sh
```

脚本会准备阅读环境并打开 <http://127.0.0.1:8000>。首次需要 Python 3.11 和网络；端口被占用时用 `./book_review.sh 8001`，按 `Ctrl+C` 停止。阅读无需 GPU 或模型权重。

### 2. 检查数据

只检查数据格式，不需要 PyTorch：

```bash
python3.11 -m lab.data --input data/dataset.json --allow-draft
```

### 3. 在 GPU 上逐格训练

运行 `python3 tools/build_gpu_zip.py` 生成 `llm-lora-tutorial-gpu.zip`，上传到 GPU 数据盘并解压。按教程第 3 章准备环境、从魔搭下载完整模型并校验，再打开 [01_gpu_train.ipynb](notebooks/01_gpu_train.ipynb)，选择 **LoRA · Python 3.11** 内核。英文版 [01_gpu_train.en.ipynb](notebooks/01_gpu_train.en.ipynb) 运行完全相同的代码，只是说明和注释为英文。

Notebook 针对 RTX 4090 / 24 GB：原模型回答 → LoRA 配置 → 1 个 Epoch 训练（约 72 步）→ loss 曲线 → 保存与重载 → 人工评估 → BF16 合并，最后附 vLLM 部署和本地调用示例。默认产物在 `runs/science-001/`：

| 产物 | 含义 |
|---|---|
| `training-complete.json` | 训练完成记录 |
| `adapter/` | LoRA 增量权重 |
| `merged-model/` | 供 vLLM 使用的完整模型 |
| `answer-review.json` | 训练前后回答对照，需人工评审 |

## 数据

只维护 [data/dataset.json](data/dataset.json)：`facts` 是 60 张参考事实卡，平级的 `train`、`val`、`test` 分别是 500 段训练、50 段验证、50 段测试对话。事实卡按 `fact_ids` 拼进每题的 system 提示词，作为回答依据，不是学习目标。训练部分按轮展开为 1150 个回答目标。资料来源、分类和审核记录都在同一个文件中。

[数据说明](data/README.md) 用完整的昼夜样例解释每个字段、哪些进入模型、哪些只是审核信息。Notebook 的 D1～D7 注释标出读取、筛选、上下文构造、分词、训练器入口、损失范围和保存位置。

Notebook 直接读取这个 JSON，在内存中生成训练样本，无需预先运行 `lab.prepare`。训练时自动在 `runs/science-001/dataset.json` 留一份实际输入副本。只有命令行实验才需要新生成冻结快照，见教程第 4.6 节。

> [!NOTE]
> 全部对话均为合成草稿，覆盖小学 1–6 年级，按 1–2、3–4、5–6 分段；50 个训练主题各 10 条路径，主题间隔离 val/test。仍需逐条核对事实、来源和适读性。不能保证微调一定优于完整提示词基线。

## 项目结构

| 路径 | 职责 |
|---|---|
| `tasks/plan.md` | 实操教程（电子书正文） |
| `notebooks/01_gpu_train.ipynb` | GPU 训练主入口，对应教程第 4～5 章 |
| `data/dataset.json` | 唯一数据源 |
| `lab/data.py`、`prepare.py`、`common.py` | 校验源数据、展开问答、构造提示词与训练样本 |
| `lab/audit.py` | 用真实 TRL collator 检查监督目标 |
| `lab/download.py`、`train.py`、`training.py` | 固定模型版本、训练和恢复 |
| `lab/infer.py`、`evaluate.py` | 生成对照回答、盲评与结果统计 |
| `lab/merge.py` | 独立合并导出 |
| `book/`、`mkdocs*.yml` | 多语言电子书构建 |
| `tests/` | 数据、训练、Notebook、教程与电子书测试 |

`models/`、`runs/`、`data/processed/` 保存本地产物，不纳入版本控制。

## 更多文档

- [实操教程](tasks/plan.md)：样例、命令、字段含义与结果解释。
- [学习清单](tasks/todo.md)：区分代码检查与待完成的训练实验。
- [验证记录](tasks/verification.md)：已经运行的检查及结果。
- [电子书维护](book/README.md)：构建、链接检查与发布方法。

## 许可证

本项目以 [MIT License](LICENSE) 发布。基座模型 [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) 不随仓库分发，使用时遵循其自身的许可证。
