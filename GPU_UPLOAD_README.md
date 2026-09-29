# GPU 上传包：单文件数据、逐格训练

适用：单卡 RTX 4090 / 24 GB；镜像 2.7.1-CUDA12.6-Py3.11-Ubuntu22。
上传到数据盘并解压，在解压后的 llm-lora-tutorial 目录打开 GPU 终端。
新旧项目放不同目录，保留旧模型缓存和训练结果。

## 1. 准备独立训练环境

```bash
python3.11 -m venv .venv-train
.venv-train/bin/python -m pip install --upgrade pip
.venv-train/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
.venv-train/bin/python -m pip install -r requirements-train.txt -r requirements-download.txt -r requirements-notebook.txt
.venv-train/bin/python -m pip check
.venv-train/bin/python -m ipykernel install --user --name lora-train --display-name 'LoRA · Python 3.11'
```

连不上 download.pytorch.org（Network is unreachable）时，第三条换成阿里云 PyTorch 镜像，再继续后面的命令：

```bash
.venv-train/bin/python -m pip install 'torch==2.8.0+cu126' -f https://mirrors.aliyun.com/pytorch-wheels/cu126/
```

## 2. 从魔搭下载完整权重并核对

```bash
.venv-train/bin/python -m lab.download --source modelscope --output models/qwen3-4b
.venv-train/bin/python -m lab.download --check-only --output models/qwen3-4b
```

不要加 --tokenizer-only；必须显示完整 12 文件校验通过。
失败时保留 model-lock.json 和 .cache，网络恢复后重跑；文件损坏则移走该文件备份后重下。
备用源：把 --source modelscope 换成 --source hf-mirror，校验方式相同。
本包包含分词器和下载校验清单，不含约 8 GB 的完整权重。

## 3. 只维护一份数据

数据源只有 data/dataset.json：facts 保存 60 张事实卡；平级的 train、val、test 分别保存 500、50、50 段对话。只训练 train，展开为 1150 个回答目标。
覆盖小学三个学段，所有对话仍是合成草稿，需核对科学事实、适读性和观察建议。
每个字段与使用位置见 data/README.md；来源、分组、审核记录都保留。
Notebook 直接读取这一个 JSON，在内存中构造上下文、Token 和监督掩码。
不需要 lab.prepare，也不需要 data/processed/。进阶 CLI 实验需要时才按电子书生成快照。

```bash
.venv-train/bin/python -m lab.data --input data/dataset.json --allow-draft
```

## 4. 开始训练

在 JupyterLab 打开 notebooks/01_gpu_train.ipynb，选择 LoRA · Python 3.11 内核。
按顺序逐格执行。每个代码格下方解释用途；D1～D7 注释标明数据在哪里被使用。
默认 RUN_DIR 为 runs/science-001，新实验换成 science-002，不覆盖旧结果。
修改数据后需重新读取、分词和训练；旧内存变量不会自动改变。

默认跑满 1 个 Epoch（约 72 个优化器步）。训练时另开终端运行 nvidia-smi -l 2 观察显存。

- dataset.json：本次真正读入的数据副本；日常仍只编辑 data/dataset.json。
- loss-curve.png：训练/验证 loss 曲线。
- training-complete.json：实际训练步数、参数变化及保存完成记录。
- adapter/：训练好的 LoRA 增量，须搭配固定版本基座。
- answer-review.json：训练前后回答及待填写人工评审。
- merged-model/：合并后的完整 BF16 模型，可供 vLLM 部署。

Notebook 最后写明 GPU 终端 vLLM 部署、本地 SSH 隧道和 API 测试例子；挂载 adapter、流式调用和压测见电子书第 5 章。
vLLM 运行在独立 .venv-serve；先核对宿主机驱动支持 CUDA 12.9，关闭训练内核释放显存。
数据盘 50 GB 请关注剩余空间，完整训练与部署建议扩容到 100 GB。
保存 Notebook 并备份 adapter、版本锁、数据副本、评审及必要检查点，再按需停止实例。
本地 CPU 检查不等于实际 4090 训练通过；训练前后回答不同，不代表质量一定提升，需要人工对比。
