# 本次实际验证记录

## 2026-10-11：加入 GPU 租用推荐（智川云邀请链接）

| 检查 | 实际结果 |
|---|---|
| 内容 | 首页新增“租 GPU：作者在用的智川云”，第 3.1 节新增“在哪里租”（价格、新人券、选型、数据盘与镜像源经验），第 0.4 节准备表的账号与预算两行改为人民币示例；均附邀请链接与邀请码 `zt4f01m0`，并注明是作者邀请链接及代金券领取规则 |
| 价格来源 | 用户提供的 2026-10-11 页面截图：RTX 4090D 24 GB 按量 ¥1.25/时，包日 ¥29、包周 ¥200、包月 ¥800；邀请规则取自用户提供的邀请页截图。gpuhome.cc 为前端渲染页面，内置浏览器打开为空白，未能直接核对 |
| 回归测试 / 电子书 | **106 passed**；`mkdocs build --strict` 通过，52 个页面、27,728 处链接与资源检查通过 |
| 价格截图 | 用户提供的租用页截图转为 `images/gpuhome-rtx4090d-price-2026-10-11.png`，嵌入首页与第 3.1 节价格表下方 |

## 2026-10-11：首页加入作者介绍，读者定位改为非算法开发者

| 检查 | 实际结果 |
|---|---|
| 内容 | 首页新增“关于作者”（简介、X、微信公众号、小红书及两个二维码）；首页、教程开头、README、站点描述、CLAUDE.md 的读者定位改为“不做算法、想从零学习 LoRA 微调的开发者”；第 1 章标题与“后端类比”改为通用的开发类比 |
| 图片 | `images/author-wechat-qrcode.jpg`；小红书二维码从名片截图中裁出为 `images/author-xiaohongshu-qrcode.jpg`；`book/build.py` 改为同时发布 `images/` 下的 PNG 和 JPG |
| 回归测试 / 电子书 | **106 passed**；`mkdocs build --strict` 通过，52 个页面、27,716 处链接与资源检查通过；本地预览目测桌面与手机宽度排版正常 |

## 2026-10-11：Notebook 第 13 节改为 curl 调用

| 检查 | 实际结果 |
|---|---|
| 内容 | Notebook 第 13 节去掉 Python 示例，改为 5 组 curl：健康检查、最小请求、按训练格式（system = 角色 + 学段 + 事实卡）、多轮、流式；教程第 5.5 节新增同样的“先用 curl 直接调用”，原 Python 与 SDK 示例保留 |
| 请求体 | 两个 JSON 文件由 `build_messages` 生成；用 Qwen 分词器算得单轮请求 `prompt_tokens` 为 395；在本机用 bash 实际执行 heredoc，生成的文件都是合法 JSON |
| 测试 | `test_tutorial.py` 新增 curl 测试：模型名与 `--served-model-name` 一致、`temperature=0`、含流式示例，且文件中的 messages 与 `build_messages` 输出完全相同；Python 调用示例改从 `plan.md` 读取并执行。**106 passed** |
| 电子书 / 打包 | `mkdocs build --strict` 通过，52 个页面、27,708 处链接与资源检查通过；GPU 上传包重新生成 |
| 尚未核对 | 真实 vLLM 服务上的 curl 响应 |

## 2026-10-11：RTX 4090 上首次完整 1 Epoch 训练（用户在 GPU 主机实测）

| 检查 | 实际结果 |
|---|---|
| 环境 | RTX 4090 24 GB（驱动 580.119.02，CUDA 上限 13.0）；镜像 2.7.1-CUDA12.6-Py3.11-Ubuntu22；官方 PyTorch 源不可达，改用国内镜像安装 torch |
| 加载 | NF4 基座加载后 `nvidia-smi` 显示 4353 MiB |
| 训练 | `runs/science-002`，72/72 步，耗时 26:19；训练平均 loss 2.233；逐步训练 loss 首/末 5.813 → 1.55 |
| 验证 loss | 第 12/24/36/48/60/72 步：2.506 / 2.186 / 2.034 / 1.976 / 1.961 / 1.960；Mean Token Accuracy 0.462 → 0.573 |
| 失败教训 | 此前两次未重启内核就重复执行训练格，在已训练模型上继续训练：训练 loss 降到约 0.5，验证 loss 回升到 2.3（过拟合）；`runs/science-001` 已删除 |
| 文档 | 曲线截图存为 `images/loss-curve-rtx4090-2026-10-11.png`，加入第 4.5 节并附逐段解读；`book/build.py` 改为发布 `images/` 下全部 PNG |
| 尚未核对 | 第 8 步完成记录（峰值显存、adapter 文件）、关闭 LoRA 的基座验证 loss、第 10 步回答对比、合并与 vLLM |

## 2026-10-10：PyTorch 官方源不可达时的阿里云镜像备用

| 检查 | 实际结果 |
|---|---|
| 触发 | GPU 主机执行 `pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126` 报 `Network is unreachable`（用户实际输出） |
| 镜像核对 | 在开发机查询 `mirrors.aliyun.com/pytorch-wheels`：cu126 目录有 `torch-2.8.0+cu126-cp311-manylinux_2_28_x86_64`；cu129 目录有 vLLM 0.21.0 所需的 `torch 2.11.0`、`torchaudio 2.11.0`、`torchvision 0.26.0` 的 cp311 Linux 包（依赖版本取自 PyPI 上 vLLM 0.21.0 元数据）。**尚未在 GPU 主机上实际用镜像安装** |
| 文档 | 第 3.3 节、第 5.4 节、Notebook 部署格、`GPU_UPLOAD_README.md` 加入 `-f https://mirrors.aliyun.com/pytorch-wheels/...` 备用命令；GPU 上传包重新生成（31 个文件，3.89 MiB） |
| 回归测试 / 电子书 | **105 passed**；`mkdocs build --strict` 通过，52 个页面、27,600 处链接与资源检查通过 |

## 2026-10-10：第 3.3 节说明上传包内容

| 检查 | 实际结果 |
|---|---|
| 内容 | 第 3.3 节新增打包命令和“上传包里有什么”：按 `unzip -l` 实际列出的 31 个文件写目录树与用途，另列包里没有的内容及来源；`requirements-train.txt` 注释中的旧章节号改为 3.1 |
| 打包 | `tools/build_gpu_zip.py` 重新生成：31 个文件，3.89 MiB |
| 回归测试 / 电子书 | **105 passed**；`mkdocs build --strict` 通过，52 个页面、27,594 处链接与资源检查通过 |

## 2026-10-10：精简第 2 章后半部分

| 检查 | 实际结果 |
|---|---|
| 内容 | 第 2.4 节补充“多轮对话是否训练多次”和多轮历史累积；原 2.6～2.11 改为 2.6 七步回顾、2.7 改数据与换数据、2.8 可选本地检查；重复的观察题说明删除（第 4.5 节已有）；审核要求移到第 6.1 节，命令行快照说明移到第 4.6 节，few-shot 选取说明移到第 6.4 节；相关章节号引用同步 |
| 回归测试 | **105 passed**（教程中示例命令少了两条重复的 `lab.prepare`/`lab.audit`，每个 CLI 仍至少有一条示例） |
| 电子书 | `mkdocs build --strict` 通过；52 个页面、27,561 处本地链接与资源检查通过 |

## 2026-10-10：补充“为什么需要事实卡”

| 检查 | 实际结果 |
|---|---|
| 内容 | 第 2.2 节新增“为什么需要事实卡：三个常见疑问”（基座已懂常识为何还要卡；卡是输入不是学习目标、训练后不能依赖它记住；为何训练时也必须放），`data/README.md` 同步；第 0.1 节、第 2.5 节加指向；第 2.1 节表格下补充“为什么还要一份 val” |
| 回归测试 | **107 passed** |
| 电子书 | `mkdocs build --strict` 通过；52 个页面、27,594 处本地链接与资源检查通过；本地预览目测第 2 章引用块与对比表排版正常 |

## 2026-10-10：dataset.json 改为 train/val/test 平级（schema 4）

| 检查 | 实际结果 |
|---|---|
| 数据转换 | 顶层改为 `facts`、`train`、`val`、`test` 四个平级数组，对话内删除 `split` 字段，内容不变；`lab.data --allow-draft --formal` 输出 `{'train': 500, 'val': 50, 'test': 50}` |
| 读取逻辑 | `lab.data.tag_splits` 按所在数组在内存中补 `split`；旧版 `episodes` 布局和对话自带 `split` 都会报错；`validate`、`prepare`、`infer` 不变 |
| 命令行快照 | 用固定 Qwen 分词器执行 `lab.prepare --allow-draft --formal`：9 个文件，train 1150 条、val 115 条 SFT 样本，`val.episodes.jsonl` 中 50 段均为 `split: val` |
| 文档 | 第 2.2 节新增“事实卡有什么用、在哪里用到”（校验、提示词拼装、Notebook、快照、A/B/C 推理、盲评、部署调用）；`data/README.md`、README、GPU 上传说明同步 |
| 回归测试 | **107 passed**（新增平级布局、旧布局拒绝、Notebook D1 读取三项测试） |
| 电子书 | `mkdocs build --strict` 通过；52 个页面、27,571 处本地链接与资源检查通过；GPU 上传包重新生成（31 个文件，3.88 MiB） |

## 2026-10-10：章节重排，取消“进阶与参考”折叠

| 检查 | 实际结果 |
|---|---|
| 结构 | 原 0～13 章合并为 0～8 章 + 附录：0 目标与准备（新写全流程、两台机器配合、准备清单）、1 概念、2 训练数据（原 3～5 章）、3 GPU、4 训练、5 部署、6 效果评估（原 8～9 章）、7 成本排错验收、8 开发者参考（原 11、13 章）；侧栏为一层平铺目录，正文内折叠块改为普通小节 |
| 新增说明 | 第 2～6 章开头加“本章速览”（做什么/需要准备/做完得到/耗时）；第 2.5 节加入用固定 Qwen 分词器实际渲染的 `day-night-01:0`（408 Token，监督 22 Token）；第 4.1 节解释一个 step 内部流程；第 5 章开头画出一次 API 请求的路径 |
| 引用同步 | `plan.md`、Notebook、README、GPU_UPLOAD_README、`todo.md`、`book/README.md` 中的章节号按新编号更新；GPU 上传包重新生成（31 个文件，3.88 MiB） |
| 回归测试 | **103 passed** |
| 电子书 | `mkdocs build --strict` 通过；52 个页面、28,089 处本地链接与资源检查通过；本地预览目测第 0、2、4、5 章的图示与速览框排版正常 |

## 2026-10-04：面向部署学习的改版

| 检查 | 实际结果 |
|---|---|
| 训练量 | Notebook 由样本数算出步数：1150 条 train ÷ 累积 16 → **72 步（1 Epoch）**；新增 115 条 val 的 eval loss（每 12 步）和 `loss-curve.png` |
| 数据清理 | 662 条回答去掉拼接留下的半角空格；`increase_depth_when_requested`、`ask_for_evidence_without_blame`、`explain_then_invite_exploration` 各 60 条改写；`lab.data --allow-draft --formal` 通过 |
| 真实分词 | 固定 Qwen tokenizer 重新生成：1150 条 train、115 条 val；最长 550 Token，平均 459 Token，平均监督 36 Token |
| CLI 草稿门槛 | `lab.train --allow-draft` 可跑 epoch 对照，manifest 记录 `draft`；冻结候选和测试集推理仍要求审核后的数据 |
| 部署章节 | 第 10 章新增 KV Cache 估算（每 Token 144 KiB）、参数说明、挂载 adapter、OpenAI SDK 流式调用、`vllm bench serve` 压测、上线检查 |
| 调用示例 | 本地 API 示例改用 `build_messages`，提示词与训练一致；仍由 `test_tutorial.py` 用临时 HTTP 服务执行 |
| GPU 上传包 | 新增 `tools/build_gpu_zip.py` 生成：31 个文件，约 3.88 MiB，不含权重；拒绝打包有输出的 Notebook |
| 回归测试 | **103 passed**（含微型 CPU 模型执行 Notebook 的 eval 与 loss 曲线格） |
| 电子书 | `mkdocs build --strict` 通过；57 个页面、28,387 处本地链接与资源检查通过；主导航改为 学习路线 → 概念 → GPU → 训练 → 部署 |

| 零基础可读性（2026-10-08） | 第 2 章重写为 8 节渐进概念课（模型目录、Token 实例、loss 数值直觉、LoRA 的 diff 类比、七个库分工）；第 6 章新增 nvidia-smi/驱动/CUDA 一节和 Jupyter 操作表；第 4 章治理内容折叠为进阶；Notebook 每个代码格新增“执行后你会看到”与逐参数表（BitsAndBytesConfig、generate、LoraConfig、SFTConfig）；Notebook 中的预期输出数值为估计，待 GPU 实测核对 |

本轮仍没有在 RTX 4090 上运行：72 步的实际耗时与显存、vLLM 0.21 的启动日志、`vllm bench serve` 的参数与结果都以 GPU 主机实测为准。

## 2026-09-29：小学科学探索，500 段训练对话

| 检查 | 实际结果 |
|---|---|
| 唯一可编辑数据源 | `data/dataset.json`：60 张事实卡；500 train、50 val、50 test；每段 2～3 轮 |
| 覆盖 | 五类各 100/10/10 段；train 的 1–2、3–4、5–6 年级分别为 167、167、166 段 |
| 生成方法 | 50 个训练主题各 10 条教学路径，逐主题撰写知识、误解、比喻边界、观察及追问，再组合完整对话；同主题部分回答复用，不是 500 个独立知识题 |
| 数据状态 | 全部 synthetic/draft；35 张卡记录参考入口阅读，25 张仍标 pending；所有逐项事实、儿童适读性和真实探索效果仍待审核 |
| 监督数量 | 1150 个 train 回答目标，val/test 各 115；训练只筛选 train |
| 参考卡独立性 | 事实改为独立摘要，未发现任何整条 assistant 目标原样包含在对应事实 text 中；这不等于排除了全部语义相似 |
| 真实分词与掩码 | 固定 Qwen tokenizer 和真实 TRL collator 检查 1150/1150 条通过，最长 553 Token；首条 408 Token，其中 22 个监督位置 |
| 数据快照 | 最终对应 `data/processed/science-v3`，9 个自动生成文件；v1/v2 是本轮整理过程的旧快照，正式使用以源数据重新 prepare 为准 |
| 输入与隔离 | grade_band 进入提示词；behavior/review/source_url 不进入；同主题及同事实不能跨 split；未来问答不能泄漏 |
| 评估协议 | B/C 保持相同输入；few-shot 优先覆盖三个学段且只取 train；盲评保留学段，bootstrap 按主题聚类，各评估集合仅 5 个独立主题 |
| Notebook 与教程 | 46 格/16 个代码格，输出为空，每格附解释；科学观察题、数据说明、正式对照、vLLM 与中文 API 示例同步 |
| 实验目录 | 新场景默认 `runs/science-001`，避免与原场景的 lesson/gpu 目录混淆 |
| 回归测试 | 101 passed，10 warnings；含微型 CPU LoRA 更新/保存/重载/合并及教程 API 请求和历史传递 |
| 发布边界 | 本轮没有运行真实 4B CUDA 训练或 vLLM，也没有儿童使用研究；本地检查不能证明 LoRA 比完整提示词基线更好 |

电子书、GPU ZIP 的最终构建与一致性检查随本轮交付完成。下方均为历史版本记录，旧场景和旧数量不代表当前数据。

## 2026-09-29：单文件数据与 D1～D7 使用说明

| 检查 | 实际结果 |
|---|---|
| 数据合并 | `data/dataset.json` 保存原来的 10 张事实卡、30 段对话、60 个参考回答和全部元信息；迁移前后对象完全一致，删除重复的两个源 JSONL |
| 主线入口 | Notebook 直接读取单文件、在内存中构造样本，不依赖 processed 目录；D1～D7 标出读取、筛选、上下文、目标、训练器、labels、保存位置 |
| 训练内容 | 用真实固定 Qwen tokenizer 执行 Notebook 生成样本格；60 条 prompt/completion/input_ids/completion_mask 全部与旧 seed-v2 一致。首条仍为 302 Token，其中 53 个监督位置 |
| 防泄漏 | 测试确认 val/test 不进入训练、当前目标和未来轮次不进入 prompt、历史正确保留；出处 URL 和审核字段不作为模型输入 |
| 回归测试 | **95 passed，0 skipped，10 warnings**；真实微型 CPU LoRA 训练、保存、重载、合并和教程 API 示例通过 |
| Notebook 格式 | nbformat 验证通过，46 格/16 个代码格；代码格下方均有解释，交付输出为空 |
| 实验复查 | 训练前保存内存中的数据到 runs/实验名/dataset.json，并在 manifest 记录该副本 SHA-256；日常只编辑 data/dataset.json |
| CLI | data/prepare 改为 `--input data/dataset.json`；进阶实验按需生成快照；历史 seed-v1/v2 和 runs 保留 |
| 上传包 | **31 文件，约 3.82 MiB**；只含一份源数据，不含 processed 中间文件；解压后验证 30 段对话与 60 个训练目标，文件与源码逐字节一致 |
| 轻量 CLI | 在不含 Torch 的 `.venv-data` 里用 `--input` 准备单文件数据，生成的 60 条样本与旧快照一致；随后真实 TRL collator 校验 mask/EOS/padding 通过 |
| 电子书 | 严格构建通过；54 个页面、5,987 处本地链接和资源检查通过；数据页展示一段完整对话及其引用的事实，旧双文件源码页已清除 |

10 条警告仍来自微型模型的 PEFT 配置路径及检查点时间排序。本轮没有执行 RTX 4090 的真实 4B CUDA 训练或 vLLM 服务；数据与本地代码验证不代表模型质量提升。

以下是此前版本的记录，旧数据入口与文件数量以本节为准。

## 2026-09-29：电子书、Notebook 与代码最终对应检查

| 检查 | 实际结果 |
|---|---|
| 完整回归 | **91 passed，0 skipped，10 warnings**；包含实际 Notebook 微型训练和教程 API 示例的本地 HTTP 检查 |
| 教程命令 | README、数据说明及正文中的 `lab.*` 命令均通过对应真实 CLI 解析器；每个保留的 CLI 都有可执行示例 |
| 入口与解释 | 第 13 章逐项对应现存模块；补齐进阶训练环境切换和 A/B/C 数据、提示词差异；Notebook 与第 10 章部署/调用代码逐块一致 |
| 源码同步 | 训练附录与实际两个训练模块一致；构建时清除旧的自动生成源码页及下载副本 |
| 清理 | 删除 `lab/client.py` 和对应旧测试；改为直接测试教材中的标准库 API 示例。保留仍用于正式对照、盲评、恢复和导出的 CLI |
| 下载版本 | 修正首次从 Hugging Face 下载可能取得新 revision 的问题；三个下载来源统一到教程固定版本，旧版本目录不会被悄悄改写 |
| 实际数据 | 固定 Qwen tokenizer 与 TRL collator 重新检查 seed-v2，60/60 条样本的监督范围、EOS 与 padding 通过 |
| GPU 上传包 | 已更新为 **38 文件，约 3.85 MiB**；解压后的冻结数据可读取 60 条训练样本，已移除旧客户端 |
| 电子书与交付包 | 严格构建通过；55 个页面、4,880 处本地链接与资源通过。GPU ZIP 中 37 个源文件、通用源码 ZIP 中 53 个文件逐字节对应当前文件；通用源码 ZIP 解压后可独立构建并通过链接检查 |
| 文档片段 | 32 个 Bash 块、3 个 Python 块、10 个 JSON 块语法通过；SSH 隧道的地址/端口占位值按文中要求替换后检查，没有连接远端 |

以上是本地可验证的对应关系。10 条警告来自微型模型的 PEFT 配置路径及检查点时间排序。没有连接 RTX 4090，不能据此宣称真实 4B 下载、CUDA/NF4 训练、质量评审或 vLLM 服务已通过；GitHub Pages 工作流也未在本轮发布。

以下记录对应此前版本，文件数量、入口和测试数量以本节为准。

## 2026-09-29：教学 Notebook 重写与国内下载

| 检查 | 实际结果 |
|---|---|
| 完整回归 | **51 passed，0 skipped，10 warnings**；本机端口测试在允许监听的环境运行 |
| Notebook 格式 | 官方 nbformat 校验通过；44 格，其中 15 个 Python 代码格；每格下方有用途与原因，交付输出为空 |
| Notebook 真实代码 | 提取实际代码格，使用微型 Qwen3 CPU 模型执行 LoRA 配置、TRL 监督校验、两步训练、保存、重新加载、生成、合并及重新加载完整模型；重载 logits 一致，合并 logits 在容差内一致 |
| 设备边界 | CPU 测试只替换 CUDA 设备/精度设置；没有用命令返回值模拟训练。显式固定 PEFT 加载设备，避免 CPU 合并被自动迁移到 GPU；CLI 合并同步修正 |
| 下载来源 | ModelScope 1.34.0 真实下载 9 个配置/分词器文件，逐个 SHA-256 与固定 HF revision 对齐；真实 Qwen tokenizer 指纹与 seed-v2 一致 |
| 权重可访问性 | 三个 safetensors 分片分别读取 64 字节，均返回 HTTP 206；Content-Range 总大小与冻结清单一致。没有下载完整 8 GB 权重 |
| 下载失败保护 | 测试部分本地缓存、同大小损坏文件、远端版本变化、分词器变动、版本锁保护、镜像 endpoint 与离线完整性检查 |
| 电子书 | 第 6 章负责环境和国内下载；第 7 章对应直接训练；第 10 章解释导出、vLLM 和本地单轮/多轮调用；严格构建与本地链接检查通过 |
| 上传包 | 39 文件，约 3.85 MiB；解压后 60 条冻结样本完整，Notebook/代码/下载锁及依赖同步，不含环境、缓存、完整权重或训练结果 |
| 清理 | 删除旧 Notebook 子进程辅助模块、可选流程开关和结果 ZIP 自动打包；保留正式评估、训练恢复等仍被教程使用的 CLI |

参考资料：[ModelScope 模型仓库](https://www.modelscope.cn/models/Qwen/Qwen3-4B-Instruct-2507)、[vLLM 0.21 GPU 安装](https://docs.vllm.ai/en/v0.21.0/getting_started/installation/gpu/)、[IPython 内核注册](https://ipython.readthedocs.io/en/stable/install/kernel_install.html)。用户提供的[知乎文章](https://zhuanlan.zhihu.com/p/689290456)触发安全验证，未能读取正文；镜像接口使用 SDK 和官方文档核对，没有声称已阅读全文。

10 条警告来自微型模型的 PEFT 配置路径及检查点时间排序。测试中发现 PEFT 默认自动分配设备会使 CPU 合并离开 CPU，已显式指定设备并通过完整回归。未连接用户的 RTX 4090；真实 4B 下载、CUDA/NF4 训练、效果评审和 vLLM 服务尚未在该主机执行。vLLM 使用独立环境，须先核对宿主机驱动对 CUDA 12.9 的支持。

以下是此前版本的历史记录，旧 Notebook 流程已由本次重写替代。

## 2026-09-29：GPU Notebook 与教程入口

| 检查 | 实际结果 |
|---|---|
| 完整回归测试 | **46 passed，0 skipped，6 warnings**；新增 Notebook 子进程中断清理、错误返回、输出保护、文件语法、下载一致性以及仅审计结果拒绝打包的检查 |
| Notebook 格式 | `notebooks/01_gpu_train.ipynb` 通过官方 nbformat 校验，33 个单元格（18 个代码格），交付文件的执行计数与输出均为空 |
| 单元格衔接 | 在本地真实 Jupyter 内核中，以模拟命令结果分别执行默认流程和全部可选流程；调用参数由真实 CLI 解析器验证，结果读取、回答 HTML 转义与 ZIP 备份通过。安装、下载和 GPU 运算均使用测试替身 |
| 电子书 | 第 6 章以 Notebook 为主要入口，第 7、10 章提供对应步骤；严格构建和本地链接检查通过，Notebook 下载文件与源文件一致 |
| GPU 上传包 | `llm-lora-tutorial-gpu.zip` 更新为 37 个文件、约 3.84 MiB；ZIP 完整性、解压导入、60 条冻结训练样本检查通过，包含 Notebook 及其执行辅助模块，不含完整权重或本地环境 |

根据用户截图复现了“只有 token-audit.json 仍可生成结果 ZIP”的问题，已修正第 13 步：先检查 A/B/C 回复、10 步训练记录和 adapter 文件，再允许打包。默认与可选流程重新通过本地内核模拟检查，GPU 上传包已更新；用户机器上的现有 Notebook 需要另行同步。

本轮没有连接用户的 RTX 4090，没有下载 4B 完整权重或执行真实 GPU 训练。Notebook 流程模拟与 CPU 微型训练回归只能验证本地可检查的部分，真实 CUDA、NF4、显存、耗时和训练效果仍以 GPU 上运行后的记录为准。6 条警告来自已有微型模型保存与检查点时间排序；HTTP 测试在允许本机临时端口的环境完成。

## 2026-09-26：数据与教程精简

| 检查 | 实际结果 |
|---|---|
| 完整回归测试 | **38 passed，0 skipped，6 warnings**；包含新旧数据包、缺失集合、数据浏览页及原有微型训练检查 |
| 轻量数据环境 | 新建 `.venv-data`，按 `requirements-data.txt` 安装 Transformers 4.57.3 与 Jinja2 3.1.6 及其依赖；确认未安装 torch、trl、peft、datasets、accelerate |
| 轻量环境准备数据 | 生成 30 段对话、60 条训练样本；5 个清单内文件与完整 CPU 环境生成的 `seed-v2` 逐字节一致 |
| 数据包结构 | 新版 `schema=2` 共 6 个文件；旧版 `seed-v1` 的 13 个文件保留，完整性检查通过 |
| 数据与监督 | 源数据内容保持原样；真实 Qwen tokenizer 与 TRL collator 检查 60/60 条通过，报告见 `runs/token-audit-v2.json` |
| 书中样例 | 故宫样例来自真实首条记录；`palace-01:0` 共 302 Token，249 个上下文位置、53 个监督位置 |
| 文档代码 | README、教程、数据说明的 30 个 Bash 块、2 个 Python 块、1 段嵌入 Python、10 个 JSON 块语法通过；训练代码附录与源码一致 |
| 电子书构建 | `mkdocs build --strict` 通过；51 个 HTML 页面、4192 个本地链接与资源检查通过 |
| 浏览器检查 | 桌面与 390px 手机宽度检查数据章节；长 JSON 自动换行，页面无横向溢出，EOS 字段显示完整；源数据页先显示用途与首条记录，默认折叠全量原稿 |

首次轻量运行发现聊天模板缺少 Jinja2，已补入依赖后重新准备成功。Transformers 提示没有模型计算库属于此环境的预期状态，分词与模板渲染均可用。HTTP 测试在允许本机临时端口的环境完成；沙箱内的监听限制未通过跳过测试处理。

6 条警告来自微型模型保存配置和检查点时间排序，与此前验证类型一致。此次只验证数据处理、CPU 训练机制和书籍网页；没有运行 4B GPU 训练，也没有审核通过候选对话。

## 2026-09-23：首次验证记录

以下保留首次执行记录。

## 环境

- 本地 Darwin 25.3.0 arm64，Python 3.11.15。
- 测试使用临时隔离环境 `/tmp/beijing-lora-env`，没有改系统 Python。
- Torch 2.8.0、Transformers 4.57.3、TRL 0.24.0、PEFT 0.18.0、Accelerate 1.11.0、Datasets 4.3.0、pytest 8.4.2。
- `uv pip check` 检查 52 个已安装包通过。这是 Mac CPU 环境，不含 bitsandbytes；不能据此声称 Linux CUDA 组合已联调。

## 已执行

| 检查 | 结果 |
|---|---|
| 完整 `python -m pytest -q tests` | **27 passed，0 skipped，6 warnings**，本次约 3.92 秒 |
| CLI 入口 | download/data/prepare/audit/train/infer/evaluate/merge/client 共 9 个 `--help` 成功 |
| Python 语法 | 全部 lab 与 tests 文件解析通过 |
| 教程静态检查 | 31 个 Bash 代码块语法、2 个完整 Python 代码块、1 个 JSON 示例及全部本地 Markdown 链接通过 |
| 原始数据 | 10 张卡、30 段对话、60 个目标回复，五类各 6 段；全部 draft |
| 真实 Qwen tokenizer | 固定 revision，下载配置与分词器，没有下载 4B 权重 |
| 准备数据 | seed-v1，train=30 段，60 条展开样本；val/test 为空 |
| 真实 TRL collator 审核 | 60/60 监督范围正确，EOS 恰好一个；两条不同长度一起检查 padding |
| Token 长度 | 总长度 285～382；目标监督长度 18～54，含 EOS；这是当前种子，不是正式数据估计 |
| 微型模型训练 | 随机初始化的极小 Qwen3，真实 Torch/PEFT/TRL 执行两步，只有 LoRA 参数改变 |
| 微型模型恢复与合并 | 总计划 4 步，第 2 步保存；恢复到 4；重载 logits 一致，合并 logits 在容差内一致 |
| HTTP 客户端 | 临时本地服务验证历史、正常响应、503、错误结构、非法 JSON；超时边界测试 |
| 评估与完整性 | 盲评配对、缺失评分、平局、按对话聚类；修改 adapter 配置使指纹改变；修改准备结果被拒绝 |

模型 revision：`cdbee75f17c01a7cc42f958dc650907174af0554`。

分词器指纹：`5cc703e3e68312a345b0aa88aed16fd37eb4ca2d871900402275514280124405`。

当前本地详细记录位于 `runs/token-audit.json` 与 `data/processed/seed-v1/manifest.json`；它们是可重建的生成文件。代码修改期间的中间 CPU 数据包已移到临时目录，最终 seed-v1 使用当前提示词代码重新构建。

## 测试中遇到的情况

- 最初 HTTP 测试受沙箱监听限制而失败；在允许本机端口的环境重跑后通过，没有删除或跳过测试。
- 修复了单段对话被显示为零宽度不确定性区间的问题：少于两段时返回空区间。
- adapter 指纹同时包含参数文件和配置，避免只改 alpha 仍通过冻结检查。
- 模型文件哈希采用流式读取，避免将多 GB 分片额外复制进 RAM。
- 6 条警告来自微型随机模型没有预训练配置路径，以及快速保存的检查点文件时间不可靠。检查点按数字步数排序后恢复成功；没有屏蔽这些警告。

## 尚未执行，不能声称通过

- 30 段数据的逐段人工/独立审核；目前没有 approved 对话。
- 30 张卡、150 段正式数据的扩充与审核。
- 4B 模型权重下载、Linux CUDA/bitsandbytes 联调、真实 QLoRA。
- GPU 10 步冒烟、20 条过拟合、1 Epoch 与 r8/r16 对照。
- 真实生成质量、事实支持率、盲评胜率、GPU 显存/耗时/成本。
- 真实 4B BF16 合并、vLLM 镜像拉取/启动、真实模型 API 与负载。

CPU 小模型测试不是 4B 实验的替代。后续完成一项，就附上相应日志/结果并更新 todo；没有实际运行的指标保持空缺。
