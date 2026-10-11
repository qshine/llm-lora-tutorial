# Hands-On LoRA Fine-Tuning: Exploring Children's Science "Why" Questions

This tutorial is for developers who **don't work on ML algorithms and want to learn LoRA fine-tuning from scratch**: backend, frontend, client, and data engineers alike. No machine-learning background is required. On a rented RTX 4090 you will do it all yourself: download an open-source model → fine-tune it with QLoRA → save and reload it → merge and export → deploy it with vLLM as an OpenAI-compatible API → call it and load test it from your own computer. The practice task is answering primary-school children's science questions in Chinese, which makes the before/after difference of training easy to see.

If you only want to read the book, use `./book_review.sh`. All commands are run from the project root.

## 0. Before You Start: What We Are Building and What You Need

This chapter involves no commands. When you finish it you should be able to answer four questions: what exactly we are building; how many steps the process has and which machine each step runs on; what you need to prepare in advance; and how the pieces fit together.

### 0.1 In One Sentence: "Onboarding" a General Model into a Specialized Service

We download an open-source general-purpose large model from the internet, **Qwen3-4B** (about 4 billion parameters, released by Alibaba's Tongyi Qianwen team). It can chat about anything, but we don't control how it talks: its answers may be too long, use words beyond a primary-school level, or keep asking questions back.

Next we **fine-tune** it with 600 demonstration conversations of the form "child's question → ideal answer", teaching it one fixed way of talking: answer first, use short sentences, correct a wrong guess first, and offer one small, safe observation. Finally we deploy the trained model as an HTTP service; any backend code that can call the OpenAI API can use it just by changing the address.

An analogy with hiring:

| Real-world hiring | What it corresponds to in this tutorial |
|---|---|
| Hire a new employee with broad knowledge | Download the open-source base model Qwen3-4B |
| Onboard them with a stack of "model reply examples" | SFT fine-tuning on demonstration conversations |
| The training lives in a small notebook and doesn't change what they already know | LoRA: the base model stays untouched; only an increment of about 130 MB is trained |
| After training, put them on the phones | Deploy the model as an API with vLLM |
| Spot-check their call quality and see how many calls they can handle a day | Test calls, blind review, load testing |

**The single most important boundary:** fine-tuning teaches "how to say it", not "what to know". The concrete scientific facts are put into the prompt by the caller on every request (this project calls them "fact cards"); the model's job is to explain them in a suitable way. If the base model already knows common facts, why do we still need fact cards, and why include them during training too? Section 2.2 answers exactly that.

### 0.2 The Whole Pipeline: Where Each Step Runs and What It Produces

```text
① 读懂数据      dataset.json → 1150 条训练样本              （第 2 章，只读）
      ↓
② 准备 GPU      训练环境 .venv-train + 8 GB 基座模型          （第 3 章，GPU 终端）
      ↓
③ 训练 LoRA     Notebook 跑 72 步 → 130 MB 的 adapter        （第 4 章，JupyterLab）
      ↓
④ 合并导出      基座 + adapter → 8 GB 的完整模型              （第 4 章，JupyterLab）
      ↓
⑤ vLLM 部署     完整模型 → OpenAI 兼容的 HTTP 服务             （第 5 章，GPU 终端）
      ↓
⑥ 调用与压测    本地电脑经 SSH 隧道调用，测延迟和吞吐          （第 5 章，本地终端）
      ↓
⑦ 评估值不值    和“原模型 + 好提示词”盲评对照                 （第 6 章，GPU 终端）
```

(Diagram: ① understand the data, dataset.json → 1150 training samples (chapter 2, read only) → ② prepare the GPU, .venv-train + 8 GB base model (chapter 3, GPU terminal) → ③ train LoRA, 72 steps in the notebook → 130 MB adapter (chapter 4, JupyterLab) → ④ merge and export, base + adapter → 8 GB full model (chapter 4, JupyterLab) → ⑤ deploy with vLLM as an OpenAI-compatible HTTP service (chapter 5, GPU terminal) → ⑥ call and load test from your computer through an SSH tunnel (chapter 5, local terminal) → ⑦ evaluate whether it was worth it, blind review against "base model + good prompt" (chapter 6, GPU terminal).)

| Step | Where | With what | Output | Size | Chapter |
|---|---|---|---|---|---|
| ① Understand the data | Any computer, just reading files | A browser or editor | Understanding how 600 conversations become 1150 training samples | — | Chapter 2 |
| ② Prepare the GPU | Terminal on the GPU host | pip, download script | The `.venv-train` training environment and the `models/qwen3-4b/` base model | Environment about 10 GB, base model 8 GB | Chapter 3 |
| ③ Train | JupyterLab on the GPU host | Run the notebook cell by cell | `adapter/` (the LoRA increment), `loss-curve.png` | About 130 MB | Chapter 4 |
| ④ Merge and export | The same notebook | PEFT | `merged-model/`, a complete model that runs on its own | About 8 GB | Chapters 4 and 5 |
| ⑤ Deploy | Terminal on the GPU host | vLLM | An OpenAI-compatible service listening on port 8000 | About 20 GB of VRAM | Chapter 5 |
| ⑥ Call and load test | Your own computer → SSH tunnel → GPU | Python standard library / OpenAI SDK, `vllm bench serve` | Model answers, latency and throughput numbers | — | Chapter 5 |
| ⑦ Evaluate | GPU host | Command-line scripts + human scoring | Evidence of "how much better LoRA is than just writing a prompt" | — | Chapter 6 |

Steps ①–⑥ are the main path, needing about 2–3 hours of GPU uptime. Step ⑦ answers "is fine-tuning worth it?"; it needs more GPU time and human scoring, and you can do it after the main path works.

### 0.3 Two Machines, Three Windows: How They Work Together

The whole process involves only two machines. The GPU host is rented by the hour and stopped when you're done; your own computer handles reading, packaging, and making calls.

```text
你的电脑（Mac / Linux / Windows 均可）
├── 浏览器：读这本书
├── 浏览器：打开平台提供的 JupyterLab ──HTTPS──→ GPU 主机的 JupyterLab
├── 终端 A：打包项目，用平台上传功能传到 GPU 主机
├── 终端 B：SSH 隧道，本地 18000 端口 ──SSH──→ GPU 主机 8000 端口
└── 终端 C：运行 Python 调用示例，请求发往本地 18000

云上 GPU 主机（Linux + RTX 4090）
├── JupyterLab：运行 Notebook，训练、保存、合并
├── 终端 1：装环境、下载模型、启动 vLLM 服务（监听 127.0.0.1:8000）
└── 终端 2：nvidia-smi 看显存、运行压测
```

(Diagram: your computer (Mac / Linux / Windows) runs a browser for this book; a browser tab with the platform's JupyterLab, connected over HTTPS to JupyterLab on the GPU host; terminal A to package the project and upload it with the platform's upload feature; terminal B for the SSH tunnel from local port 18000 to port 8000 on the GPU host; and terminal C to run the Python call examples against local port 18000. The cloud GPU host (Linux + RTX 4090) runs JupyterLab for the notebook (train, save, merge); terminal 1 to install environments, download the model, and start the vLLM service (listening on 127.0.0.1:8000); and terminal 2 for `nvidia-smi` VRAM checks and load testing.)

| Component | What it is | Why you need it |
|---|---|---|
| GPU cloud host | A Linux server rented by the hour, with one NVIDIA GPU | Training and inference are massive matrix computations, and a CPU is too slow; the 4-bit quantization library used for training only supports NVIDIA |
| Platform image | The system template you pick at startup, with Ubuntu, Python 3.11, and the GPU driver preinstalled | Saves you installing the OS and driver; only the platform can provide the driver |
| JupyterLab | An interactive Python editor in the browser, usually bundled by the platform | Training code runs one cell at a time, you see each cell's output, and on an error you rerun only that cell |
| Virtual environments `.venv-*` | Private Python dependency directories for the project | The library versions needed for training and for deployment conflict; installing them separately keeps them from overwriting each other |
| Kernel | The Python process behind Jupyter that actually runs the code | You must pick the kernel registered by `.venv-train` so the notebook can use the project's dependencies; VRAM is only released when the kernel is shut down |
| SSH tunnel | An encrypted channel that forwards a local port to a remote port | The model service listens only on `127.0.0.1` of the GPU host and is not exposed to the internet; locally you call it through the tunnel as if it were on your own machine |

### 0.4 What to Prepare in Advance

| Item | Requirements | Notes |
|---|---|---|
| Skills | Comfortable with a Linux terminal, can read Python, know HTTP and JSON | You do **not** need math, PyTorch, or machine learning; chapter 1 explains the concepts you need from scratch |
| GPU cloud account | Any GPU cloud platform with pay-as-you-go billing, with a balance in your account | You need to be able to start an RTX 4090 (24 GB VRAM) instance, and the platform should offer JupyterLab and SSH login. The author uses [Zhichuan Cloud (gpuhome.cc)](https://gpuhome.cc/auth/register?code=zt4f01m0); entering the invite code `zt4f01m0` at sign-up gets you a new-user voucher, see section 3.1 |
| Host configuration | 1 × RTX 4090, RAM ≥ 32 GB (≥ 64 GB recommended for merging), data disk ≥ 100 GB | Pick an Ubuntu 22 + Python 3.11 + CUDA 12.x image; details in section 3.1 |
| Budget | About 2–3 hours of uptime for the main path, plus 10–20 hours for the evaluation experiments | At Zhichuan Cloud's pay-as-you-go RTX 4090D rate of ¥1.25/hour, the main path costs about ¥3–4 and the evaluation experiments about ¥13–25; the estimation method is in section 7.1 |
| Your own computer | Internet access, a terminal, and an SSH client; install Python 3.11 if you want to run the checks or call examples locally | You do **not** need a local GPU, nor do you need to install CUDA yourself |
| Network | The GPU host can reach PyPI (or the platform's mirror) and ModelScope | Hosts in mainland China usually can't reach huggingface.co, so this tutorial downloads from ModelScope by default |
| This project's files | This repository, or the practice bundle from the e-book's "Source download" page | What you upload to the GPU is a slimmed-down bundle built with `tools/build_gpu_zip.py`, explained in section 3.3 |
| Time | 1–2 hours to read chapters 1–2; one continuous 2–3 hour GPU session | GPU time is billed by uptime, so read the first two chapters before starting the machine |

**What you don't need:** an OpenAI account or API key (the vLLM service uses the same interface format as OpenAI, but runs entirely on the machine you rent), a Hugging Face account (the model is public and downloaded from ModelScope), or your own training data (the project ships with a dataset).

### 0.5 Reading Path

The order of this book is the order of operations. Each chapter starts by stating what it does, what it needs, and what you get at the end.

| Chapter | Content | Where to read/do it | Required? |
|---|---|---|---|
| 0–1 | Goals, preparation, essential concepts | Read only | Required: chapter 1 is the dictionary for every number and parameter that follows |
| 2 | What the training data looks like and how it becomes training samples | Read only; optionally run the checks locally | Required: know what the model is actually learning |
| 3 | Rent a machine, install environments, download the model | GPU terminal | Required |
| 4 | Train in the notebook, watch the loss, compare answers, merge | GPU JupyterLab | Required |
| 5 | vLLM deployment, local calls, load testing | GPU terminal + local | Required |
| 6 | Comparison against "base model + good prompt", blind review | GPU terminal | After the main path works |
| 7 | Costs, troubleshooting table, final acceptance check | — | Look things up when you hit a problem; check against it at the end |
| 8, appendix | Automated tests, code map, full training code | — | Read when you want to change the code |

Readers who already know the concepts can start at chapter 3 and come back to chapter 1, or to the table under the corresponding notebook cell, whenever they meet an unfamiliar parameter.

### 0.6 The Practice Task: Exploring Science with Primary-School Children

Why not a common task such as customer support or code generation? Because "explaining science to primary-school children" has a distinctive answer style, so the change before and after fine-tuning is obvious at a glance; and scientific facts can be checked, which makes it easy to tell whether the model is making things up.

#### Why Science Exploration

When a child asks "why", what follows may be a wrong guess, not understanding, wanting to verify it hands-on, or just wanting the answer right away. This feedback lets us observe whether the model really adapts its explanation, instead of only comparing whether the wording sounds cute.

The goal: facts are correct, the child can understand, and the next step is worth exploring. Fun can come from a counterintuitive phenomenon, an analogy with clear limits, or a small safe observation; not every answer needs a story or an exaggerated tone. When the child says they're done, the conversation ends; longer conversations are not a measure of success.

#### Coverage

| Category | Data value | Example training topics |
|---|---|---|
| Astronomy and space | `space` | Day and night, seasons, moon phases, meteors |
| Earth and weather | `earth` | Clouds and rain, snowflakes, rocks, rivers |
| Life and nature | `life` | Seeds, roots, butterflies, food webs |
| Everyday physics | `physics` | Shadows, sound, buoyancy, dissolving |
| Engineering and tools | `engineering` | Arch bridges, levers, airplanes, solar cells |

Each category has 100 training conversations, plus 10 for validation and 10 for testing. This is a limited set of teaching samples, not a complete knowledge base of the primary-school curriculum. `grade_band` is split into grades 1–2, 3–4, and 5–6; the grade band is a starting point for the explanation, actual feedback takes priority, and we cannot assume that children in the same grade understand things equally well.

#### How the Prompt and Fine-Tuning Divide the Work

A single request contains four things, each responsible for one job:

| Component | What it governs | When it takes effect |
|---|---|---|
| System prompt (role prompt) | The role and basic rules: "You are a companion who explores science with primary-school children…" | Sent with every request |
| Fact cards | The scientific material this question may rely on | Selected per question and sent with every request |
| SFT demonstration conversations | Demonstrate how to handle misconceptions, not understanding, mismatched observations, and changing preferences | Used only during training; the learned behavior is stored in the adapter |
| LoRA | Restricts updates to a small subset of parameters; the base model is untouched | Decides at training time which parameters change |

The project's 10 demonstration paths are: correct a misconception, explain it another way, pick up a specific follow-up question, an optional observation, test a guess, go from simple to deeper, state the limits of an analogy, reconcile differing observations, follow up without quizzing, and end naturally. Some answers are reused within the same topic, so don't read 500 conversations as 500 independent science questions.

We do not promise that "a prompt can never replace fine-tuning". Chapter 6 compares group B, the base model with the full prompt plus three examples, against LoRA group C, which uses exactly the same input. On science topics not seen in training, we check whether C responds more consistently to the child's feedback. If there is no improvement, we honestly keep B.

### 0.7 What Exists Today

Updated: 2026-10-10.

| Item | Current status |
|---|---|
| Single data source | `data/dataset.json`: 60 fact cards, 600 synthetic conversations (500 train / 50 val / 50 test) |
| Training samples | The 500 train conversations expand into 1150 answer targets; the notebook runs a full epoch by default (about 72 steps) |
| Notebook | Tokenization, masking, QLoRA, loss curve, save, reload, comparison, BF16 merge, and finally vLLM deployment and calls |
| Deployment | vLLM serving the merged model or mounting the adapter directly; calls via the standard library / OpenAI SDK; load testing with `vllm bench serve` |
| Local verification | Data checks and tiny CPU training-mechanism tests, see the [verification record](verification.en.md) |
| Not yet verified on a real GPU | 4B training time and VRAM, the vLLM service, load-test numbers; trust what you measure on your GPU host |

### 0.8 The Limits of Our Conclusions (Stated Once for the Whole Book)

- All conversations are synthetic drafts (`review.status="draft"`) that have not been reviewed one by one by a human; the answers are on the short side and repetitive in structure. They are entirely sufficient for learning the **workflow and engineering**, but not for concluding that "LoRA is definitely better".
- Being able to generate answers only shows that the files and service work; a falling loss only shows that the model is more like the reference answers. Answer quality has to be judged by the human comparison in section 4.5 and the blind review in chapter 6.
- Fine-tuning changes "how to say it"; it does not automatically give the model new knowledge or real-time information. Facts are still put into the prompt by the caller.

**Acceptance check:** You can draw the "data → training → merge → deploy → call" pipeline and say which machine each step runs on and which files it produces; you can list what to have ready before starting the machine; and you can point out, in a conversation, "what the knowledge is" and "what changed in response to feedback".

## 1. Essential Concepts: Understanding Them Through Things Developers Already Know

This chapter has no math. The goal is that whenever a code cell runs later, you know what the numbers on screen mean. Each concept comes with an analogy from everyday development (databases, Git, web services, and so on); the analogies only help build intuition and are not rigorous definitions.

!!! abstract "Chapter at a glance"
    - **What this chapter does**: starting from "a model is a directory on disk", it covers tokens, inference, training and loss, SFT/LoRA/QLoRA, steps and epochs, the division of labor among seven libraries, and VRAM
    - **What you need**: nothing to install; any computer that can open a web page
    - **What you get**: you can match every parameter and number that appears in the notebook later; when you hit an error, you know which library to look at

The 8 sections follow the order "first there is a model, then how it reads text, how it answers, how it learns, how to save VRAM"; each section builds on the previous one, so read them in order.

### 1.1 A Model on Disk Is a Directory

Open `models/qwen3-4b/`: physically, a "model" is just these files. Note: the directory you download on your own computer with `--tokenizer-only` (section 2.8) contains only the tokenizer and config, about 15 MB, and **not** the three weight shards; the shards appear only after the full download on the GPU host (section 3.4). This is normal, not an interrupted download.

| File | What it is | Analogy |
|---|---|---|
| `config.json` | Structure definition: 36 layers, hidden size 2560, vocabulary of 151,936, etc. | A database's table-creation statements (schema) |
| `model-00001-of-00003.safetensors` to `model-00003-of-00003.safetensors` | The weights, about 4 billion numbers, 8.0 GB in BF16, split into 3 shards; only present after a full download | A database's data files |
| `model.safetensors.index.json` | Which weight lives in which shard | A shard routing table |
| `tokenizer.json`, `vocab.json`, `merges.txt` | The tokenizer: rules for text ↔ integer IDs | A serialization protocol |
| `tokenizer_config.json` | Contains `chat_template`: how system/user/assistant are assembled into one piece of text | A request message format |
| `generation_config.json` | Default sampling parameters | Server-side default configuration |
| `model-lock.json` | The version (commit) recorded by this tutorial | `package-lock.json` |

"Loading a model" means building the empty structure from `config.json`, then filling in the numbers from `safetensors` and placing them in VRAM. Later you will meet three kinds of directories: the **base model** (the one above), the **adapter** (only the LoRA increment, about 130 MB, which must be paired with the base model), and the **merged model** (the complete directory after combining the base and the increment, 8 GB again). vLLM loads either a base model or a merged model directory.

### 1.2 Tokens: Text as the Model Sees It

A model doesn't read characters; it reads integers. The tokenizer cuts text into pieces (tokens), and each piece maps to an ID in the vocabulary. Here is a sentence cut by this project's tokenizer:

```text
"为什么白天亮、晚上黑？"
→ ['为什么', '白天', '亮', '、', '晚上', '黑', '？']
→ [100678, 106772, 99538, 5373, 104030, 56652, 11319]

"The Earth rotates."
→ ['The', ' Earth', ' rotates', '.']
→ [785, 9237, 89059, 13]
```

(The Chinese sentence is "Why is it bright in the day and dark at night?")

Common words are a single token, while rare characters may be split into several; in English the space attaches to the following word. So the token count is not the character count; for Chinese, roughly 1 token ≈ 1.3–1.5 characters. Every "length limit of 4096" in this tutorial refers to tokens.

The vocabulary also contains a few special tokens that don't correspond to any text but act as structural markers. Qwen's conversation format (`chat_template`) assembles a request like this:

```text
<|im_start|>system
你是陪小学生探索科学的伙伴。……<|im_end|>
<|im_start|>user
为什么白天亮、晚上黑？<|im_end|>
<|im_start|>assistant
地球在不停转，我们住的地方转向太阳就是白天……<|im_end|>
```

(System: "You are a companion who explores science with primary-school children…"; user: "Why is it bright in the day and dark at night?"; assistant: "The Earth keeps spinning; when the place where we live turns toward the Sun, it is daytime…")

`<|im_end|>` (ID 151645) is the **end-of-sequence marker, EOS**: the model stops when it generates it. During training the model must learn to output it at the end of each answer; otherwise, after deployment, its answers won't stop.

### 1.3 Parameters and Inference: A Function with 4 Billion Coefficients

At its core a model is a function: input a sequence of token IDs, output "the probability that the next token is each entry in the vocabulary". This function has about 4 billion adjustable coefficients, called **parameters** (or weights): the numbers in `safetensors`.

**Inference** is calling this function repeatedly: give it the prompt, take the most probable token (greedy decoding, `do_sample=False`), append it to the end of the input, and call again, until it outputs EOS or reaches `max_new_tokens`. Inference doesn't change a single parameter.

`temperature=0` behaves like greedy decoding, always picking the most probable token, so the same input gives the same output, which is handy for comparisons; above 0 it samples randomly according to the probabilities, giving more varied but non-reproducible answers.

### 1.4 Training: Correcting Parameters Using the Error

**Training** works the other way around: show the model "context + reference answer", have it predict the next token at every position of the reference answer, and turn the gap between prediction and reference into a single number called the **loss**.

| Approximate loss | Meaning |
|---|---|
| About 3 or more | Every token of the reference answer "surprises" the model; random guessing would be ln(151936) ≈ 11.9 |
| 1–2 | The typical starting point of the base model with a good prompt on this project's reference answers |
| Around 0.5 | The model is already very close to the wording of the reference answers |
| Close to 0 | It has nearly memorized them word for word; if the validation loss is rising at the same time, that's overfitting |

Loss measures "how surprised the model is by the reference answer", not "a science knowledge score". It can only tell you whether the model is learning how the training data says things, not whether its answers are correct.

With the loss, you can compute for each parameter "which direction to nudge it to make the loss smaller": that is the **gradient** (computed by backpropagation). The **optimizer** (AdamW) updates the parameters according to the gradients, and the **learning rate** (`1e-4`) decides how large each step is: too large and it washes out the base model's abilities, too small and it doesn't learn. One round of "compute loss → compute gradients → update parameters" is called a **step**.

Analogy: the loss is like the number of failing CI tests, the gradient is like being told which way to turn each config setting, the learning rate is how much you dare to change at a time, and a step is like one commit.

### 1.5 SFT, LoRA, QLoRA: Three Independent Choices

The three terms answer three separate questions and can be combined independently:

**SFT (supervised fine-tuning) answers "what to teach with".** Show the model paired "input → expected output" examples, i.e. the conversation data from chapter 2. This project computes the loss only at the positions of "reference answer + EOS", not on the context (you will see this mask with your own eyes in section 2.5).

**LoRA answers "which parameters to change".** Full fine-tuning has to update 4 billion parameters, each with its own gradient and optimizer state, which doesn't fit on a single GPU, and every experimental version would need 8 GB of storage. LoRA leaves the original weights `W` completely untouched and adds two small matrices `A` and `B` beside them, training only those:

```text
实际使用的权重 = W（冻结） + (alpha / r) × B × A
```

(That is: effective weights = W (frozen) + (alpha / r) × B × A.)

A Git analogy: the base model is the main branch, LoRA is a very small diff, training only changes the diff, and the `adapter/` directory stores exactly that diff. "Merging" is like applying the diff to the main branch to produce a new complete copy of the files.

`r` (the rank) sets the width of the small matrices, i.e. the capacity of the diff. This project uses `r=16`; across 7 kinds of linear layers × 36 layers that adds up to about 33 million trainable parameters, 0.8% of the total, and an adapter file of about 130 MB. `alpha` is a scaling factor; this project always uses `alpha = 2 × r`.

**QLoRA answers "how to fit the frozen base model into VRAM".** Since the frozen weights never change anyway, they are compressed from BF16 (2 bytes per number) to 4 bits (0.5 bytes per number) in VRAM and temporarily restored to BF16 during computation. The 4B model's weights shrink from 8 GB to about 2.5 GB, which leaves a 24 GB card enough room for the intermediate results of training. `NF4` is the name of the compression format, and `double_quant` means also compressing the auxiliary parameters used by the compression. LoRA's small matrices themselves stay in BF16, uncompressed.

Further reading: [the original LoRA paper](https://arxiv.org/abs/2106.09685), [the original QLoRA paper](https://arxiv.org/abs/2305.14314).

### 1.6 Samples, Batches, Steps, Epochs

| Concept | Value in this project | Meaning |
|---|---|---|
| Sample | 1150 | One "context + one reference answer"; a 2-turn conversation expands into 2 samples |
| Micro-batch | 1 | Number of samples placed in VRAM for one forward/backward pass |
| Gradient accumulation | 16 | Accumulate the gradients of 16 samples before updating the parameters once; the effect is close to batch=16 while VRAM holds only 1 sample |
| Step | 72 | Number of optimizer updates = ⌈1150 ÷ 16⌉; the x-axis of the progress bar and the loss curve |
| Epoch | 1 | One pass over all samples |
| Checkpoint | Every 25 steps | A snapshot you can resume training from: adapter + optimizer state + progress |

Analogy: a sample is one message, gradient accumulation is collecting 16 messages before a batch write to the database, a step is one database write, and an epoch is consuming the whole queue once.

### 1.7 What Each of the Seven Libraries Does

The training environment installs seven libraries, and their names will show up in error messages:

| Library | Responsible for | Where you'll see it |
|---|---|---|
| **PyTorch** | Tensor computation, GPU scheduling, automatic gradients; all the other libraries are built on it | `torch.cuda.*`, VRAM statistics |
| **Transformers** | Building the model structure from `config.json`, loading weights, the tokenizer, the chat template, `generate()` | `AutoModelForCausalLM`, `AutoTokenizer` |
| **PEFT** | Adding LoRA layers to the model, freezing the original parameters, saving/loading the adapter, merging | `LoraConfig`, `get_peft_model`, `PeftModel` |
| **TRL** | A trainer tailored for SFT: computes loss according to the mask, manages steps/epochs/checkpoints | `SFTConfig`, `SFTTrainer` |
| **bitsandbytes** | The storage format and GPU compute kernels for 4-bit quantization | `BitsAndBytesConfig`, NVIDIA GPUs only |
| **accelerate** | Placing the model on the specified device, mixed precision; TRL depends on it internally | `device_map` |
| **datasets** | The data container the trainer requires | `Dataset.from_list` |

Deployment switches to another library: **vLLM** has its own inference engine and HTTP server and needs only the model directory, not PEFT or TRL from the seven libraries above.

How they call each other: Transformers loads the model → bitsandbytes compresses the weights to 4 bits → PEFT adds the LoRA layers → datasets holds the samples → TRL runs the training loop → PEFT saves the adapter → PEFT merges → vLLM deploys.

### 1.8 RAM, VRAM, and Disk

| Resource | Main use | Common misconception |
|---|---|---|
| CPU | Text processing, tokenization, loading and scheduling | More cores doesn't mean it's suitable for training large models |
| RAM (host memory) | Staging while loading weights, merging the model on the CPU | 100 GB of RAM can't replace 24 GB of VRAM: training computation happens on the GPU, so the data must be in VRAM |
| GPU | Massive matrix computation | The CUDA main path requires NVIDIA; Apple's MPS on the Mac is a different route that this tutorial doesn't support |
| VRAM | Weights, activations, gradients, optimizer state, the inference KV cache | An 8 GB model file doesn't mean 8 GB of VRAM: training adds intermediate results, inference adds the KV cache |
| Disk | Base model, environments, checkpoints, merged models | Every checkpoint includes optimizer state; every merge is another 8 GB |

The breakdown of training VRAM and actual readings are in section 4.3; the inference VRAM estimate is in section 5.3.

**Acceptance check:** You can say what each file in a model directory is; you can explain what a loss dropping from 2 to 0.6 means and why that alone doesn't show the answers got better; you can say why 1150 samples means 72 steps; and you can name which of the seven libraries handles LoRA and which handles the training loop.


## 2. Training Data: Where the Model Learns From, and What It Learns

The result of fine-tuning is determined almost entirely by the data: however the model ends up talking is exactly what the demonstrations you show it look like. So before you spend money renting a GPU, get a clear picture of the data first.

!!! abstract "Chapter at a glance"
    - **What this chapter does**: read one fact card and one conversation field by field; see how a conversation is split into training samples; see exactly what the model "sees and learns"
    - **What you need**: just a browser or editor; if you want to run the checks yourself, section 2.8 explains how to install a lightweight environment on your own computer (no GPU needed)
    - **What you get**: you can explain where the number 1150 comes from; you can point out which tokens in a sample contribute to the loss; you know how to make data changes take effect

You can remember the whole data flow in one sentence: **one JSON file → split into samples of "context + one reference answer" → turned into tokens by the chat template → loss computed only at the reference-answer positions.** The sections below follow that order.

### 2.1 Maintain just one dataset.json

Open [data/dataset.json](../data/dataset.json). All the reference material, the 500 training conversations and the evaluation conversations live in this one file. At the top level there are four sibling arrays: `facts` holds the reference fact cards, and `train`, `val` and `test` are three sets of complete conversations. Only `train` actually takes part in training; fact cards serve only as context, not as a learning target.

The three sets are split by science topic (`group_id`): any one topic appears in only one set, so the model never meets a topic on val/test that it already saw during training.

| Set | Science topics | Full conversations | Answer targets | Purpose |
|---|---:|---:|---:|---|
| train | 50 | 500 | 1150 | Update the LoRA parameters |
| val | 5 | 50 | 115 | Compare configurations, pick a candidate |
| test | 5 | 50 | 115 | Evaluate after the candidate is frozen |

> **Why do we need a val set at all? Isn't train plus test enough?**
>
> If you train once and just take the result, two sets are enough. The trouble starts when you **make choices based on test results**: train with r=16, look at test; not happy, switch to r=8, look at test again; then tweak the learning rate, change the prompt... and finally pick whichever version scores highest on test. At that point test no longer counts as "unseen data": the model parameters never touched it, but every one of your choices was guided by it, which is the same as using it to pick the answer, so the score you end up with is inflated.
>
> So keep "data used for making choices" separate from "data used for the final grade". A backend release analogy: train is the development environment, where the model updates its parameters again and again; val is staging, which you can look at as often as you like to compare and choose; test is real production traffic, which you look at only once after the candidate is settled, to produce the final report.
>
> In this project val supports these choices: computing validation loss every 12 steps during training to judge whether it is overfitting (section 4.5); comparing r16 and r8 (section 6.4); comparing group B, "base model + a good prompt", with group C, LoRA (sections 6.5 and 6.6); and finally choosing the candidate (section 6.7). Once chosen, you freeze it with `evaluate freeze`; only then does the code allow inference on test, and it does not let you use `--limit` to cherry-pick a subset of questions. If you look at test and then go back to change the model or the data, the next round needs a fresh test set.
>
> When you see "just a training set and a test set" elsewhere, it usually means they train only once and make no comparisons; or they use cross-validation, rotating a validation slice out of train (the val set is hidden inside train); or they simply use test as val, in which case the test score they report is optimistic.

Training grade-band distribution: 167 conversations for grades 1–2, 167 for grades 3–4, 166 for grades 5–6. Each topic has 10 conversation paths, each with 2–3 turns; they include explanations of the same knowledge at different grade bands as well as follow-ups that change depending on the child's different reactions.

```text
data/dataset.json
├── schema_version / description   文件版本与说明
├── facts[60]                      参考事实卡，只存一份，被对话按 id 引用
├── train[500]                     训练对话：唯一参与参数更新的部分
├── val[50]                        验证对话：比较配置、选候选、算 eval loss
└── test[50]                       测试对话：候选冻结后才使用

train / val / test 里的每段对话结构相同：
├── id / group_id                  对话编号 / 科学主题分组
├── grade_band                     1-2 / 3-4 / 5-6
├── fact_ids                       引用 facts 中的 id
└── turns[]                        user / assistant / behavior
```

(In English: `schema_version / description` are the file version and description; `facts[60]` are the reference fact cards, stored once and referenced by conversations via id; `train[500]` are the training conversations, the only part that updates parameters; `val[50]` are validation conversations for comparing configurations, picking a candidate and computing eval loss; `test[50]` are test conversations used only after the candidate is frozen. Every conversation in train / val / test has the same structure: `id / group_id` are the conversation id and science-topic group; `fact_ids` references ids in facts.)

Which set a conversation belongs to is determined solely by which array it sits in. Conversations no longer carry a `split` field, which avoids self-contradictions like "sits in val but says train". When the program reads the file (`lab.data.read_dataset`; in the notebook, `tag_splits`), it adds `split` to each conversation in memory, and all later filtering and cross-set checks use it; the source file stays unchanged. To move a conversation to another set, cut the whole conversation into the other array, and move the other conversations with the same `group_id` and the fact cards it references along with it.

JSON does not support `//` comments. The fields are explained here, and the notebook marks the places where they are actually used with D1–D7. Do not stuff the whole JSON file into the prompt, and do not pass the evaluation conversations to the trainer.

### 2.2 The first fact card

```json
{
  "id": "day-night",
  "category": "space",
  "topic": "昼夜",
  "text": "自转使地球表面各处交替朝向和背向太阳，形成主要的昼夜变化。太阳没有按昼夜关闭发光。地球还绕太阳公转；假定取消自转但保留公转，光照位置仍会随公转缓慢变化。",
  "source_url": "https://spaceplace.nasa.gov/seasons/en/",
  "checked_at": "2026-09-29",
  "source_review": "reference_read_claim_review_pending",
  "scope": "昼夜的入门解释；模型和比喻不能替代真实测量。",
  "kind": "fact",
  "reuse": "科学事实摘要和教学对话为原创中文表达；来源仅供事实核对，不复制来源的教材段落、题目或图像，不把网页正文纳入训练。链接不表示来源机构认可本数据或授予网页内容的训练许可。"
}
```

(Roughly: the topic is "day and night". `text`: "Rotation makes each place on Earth's surface alternately face toward and away from the Sun, producing the main day-night cycle. The Sun does not switch off its light at night. Earth also orbits the Sun; if rotation stopped but the orbit continued, the lit area would still shift slowly with the orbit." `scope`: "An introductory explanation of day and night; models and analogies cannot replace real measurement." `reuse`: the summaries and dialogues are original Chinese writing, sources are used only for fact-checking, and a link does not mean the source endorses this data or grants a training license.)

| Field | Meaning and use |
|---|---|
| `id` | Primary key, referenced by a conversation's `fact_ids` |
| `category/topic` | Category and topic, for lookup, statistics and review |
| `text` | Scientific material an answer can be based on; enters the context |
| `kind/scope` | Material type and scope of applicability; enters the context |
| `source_url`, `checked_at`, `source_review`, `reuse` | Source and review records, for humans only; **does not enter the model** |

Only the four fields `id`, `text`, `kind` and `scope` are put into the prompt. The other fields are for reviewers; you can skip them on a first read.

#### Why fact cards are needed: three common questions

Fine-tuning teaches "how to say it" (section 0.1). Beginners who get this far usually come up with three questions in a row.

**Q1: Primary-school science is common knowledge, and the base model already knows it. Why have fact cards at all?**

True, the base model knows most of this common knowledge. If the only question were "where does the knowledge come from", fact cards would not be necessary. We still want them, and the reason is not to add knowledge:

1. **So training learns only "how to say it".** The loss is computed on the reference answer, and the reference answer contains both phrasing and factual content. When the input has no material, the model can only memorize content like "turning toward the Sun" and "the side facing away from the light" into its weights, and training ends up learning "phrasing" and "knowledge" mixed together. When the material is right there in the input, the model can predict the content correctly just by following the material, so LoRA's limited capacity goes mainly to "short sentences, correct the misconception first, offer one safe small observation". The mechanism is explained in Q3.
2. **You can't inspect or edit the base model's knowledge, but you can inspect and edit fact cards.** You can't tell in advance which question a 4B model will get wrong (for example, attributing the seasons to distance from the Sun), and when it is wrong you can't open the weights and fix it. A fact card is a visible piece of text with a source and a review status; if it is wrong you fix one card, and the change takes effect on the next request. A backend analogy: business rules live in configuration or a database, not scattered through a compiled binary.
3. **Cards carry boundaries the base model won't respect on its own.** For example, the `text` of `day-night` states "if rotation stopped but the orbit continued, the lit area would still shift slowly with the orbit", and the follow-up question in the second turn of the reference conversation relies on it; `scope` states "models and analogies cannot replace real measurement"; `kind` distinguishes fact, legend and current, so a legend must not be told as fact.
4. **This is what real products look like, and it is the basis for evaluation.** In real business, what a model has to explain is often something it doesn't know at all: product specs, a particular school's textbook, today's data. The common approach is for the caller to look up the material and put it in the prompt while the model explains it clearly, which is what is usually called RAG (retrieval-augmented generation). This project uses fact cards to practice that pattern end to end. In blind review, graders also check each claim against the fact card to judge whether it is supported (section 6.6).

**Q2: Fact cards are in the training samples too, so does the model remember this knowledge after training?**

That's not the right way to think about it. Fact cards are the **input** of a training sample, not the **learning target**. In the 408-token sample in section 2.5, the fact card is among the 386 tokens with mask=0: the model can see it, but is not penalized for "not remembering it". The loss tests one thing only: given this material, how closely does the answer match the reference answer.

| | Learned after training? | Why |
|---|---|---|
| How to read the material and use it to structure an answer | Yes | This is what the loss pushes it to learn |
| The content and boundaries of the material itself | Not reliably | During training the material was always right in front of it; no need to memorize |

To be honest about one thing: the reference answers themselves contain factual content, which sits in the mask=1 region, so the model will incidentally pick up some common phrasings. But with only 1 epoch, seeing each sample once, that memory is shallow; it can't remember the full content and boundaries of the cards, and you have no way of knowing what it did remember or got wrong. That is why at deployment you **must pass the fact cards every time**, and why the call example in section 5.5 first uses `read_dataset` to load the facts.

To verify this yourself: after training, ask the same question twice, once with the fact card and once with the `CURRENT REFERENCE FACTS` section removed, and compare the accuracy and boundaries of the answers. The bigger the difference, the more the model relies on the material given on the spot rather than on memorized knowledge.

**Q3: If we don't rely on it to memorize knowledge, why include it during training? Can't we just pass it at deployment?**

No. Training is not about making the model **remember** the fact cards; it is about teaching it to **use** them, and to learn to use them it must see them during training. An analogy with training a new employee:

| | Method A: Q&A only | Method B: Q&A with the product manual alongside (this project) |
|---|---|---|
| Training material | Customer asks X → standard reply Y | Customer asks X + relevant manual page → standard reply Y |
| Habit formed | Hear the question, recite the answer from memory | Check the manual first, then explain it in the standard tone |
| Handed the manual on the job | Never practiced reading the manual, probably still recites old answers | Exactly the process they practiced |
| One entry in the manual is updated | Keeps giving the old answer | Follows the new manual |
| Faces a question never covered in training | Bluffs from vague memory, prone to making things up | Explains from the manual, doesn't make up what the manual doesn't say |

Here is the same sample from this project; the target answer is identical, only the input differs:

```text
不带事实卡：
  输入：为什么白天亮、晚上黑？
  目标：地球在不停转，我们住的地方转向太阳就是白天……

带事实卡：
  输入：[事实卡：自转使地球表面交替朝向和背向太阳……] 为什么白天亮、晚上黑？
  目标：地球在不停转，我们住的地方转向太阳就是白天……
```

(In English: "Without fact card — input: Why is it bright in the day and dark at night? Target: The Earth keeps turning; when the place we live turns toward the Sun, it is daytime…" and "With fact card — input: [Fact card: Rotation makes Earth's surface alternately face toward and away from the Sun…] Why is it bright in the day and dark at night? Target: the same.")

Without the card, the input contains no basis at all, yet the target is a definite scientific conclusion, so what the 1150 samples keep reinforcing is "hear a question, produce a definite conclusion out of thin air". With the card, everything in the target can be found in the input, so what gets reinforced is "take content from the given material and rewrite it in words a child can understand".

If you leave the cards out during training and add them only at deployment, four problems appear:

1. **The model never practiced reading material.** At deployment a big extra `CURRENT REFERENCE FACTS` block shows up; to the model it is unfamiliar input that may be ignored, and it answers from memory as before.
2. **Editing the material has no effect.** If you find a card is wrong, or want to add an applicability boundary to some card's `scope`, a Method A model won't change accordingly; you can only edit the training data and retrain.
3. **New topics are more likely to be made up.** Method A builds the habit of "producing conclusions out of thin air", so on topics not in the training data it still draws conclusions confidently.
4. **The input format doesn't match.** During training the prompt has no fact cards, at deployment it does; the way the model has learned to talk depends on what its training inputs looked like. This is also why section 5.5 stresses that "the prompt must match the one used in training".

In code terms, training is learning a function `answer = f(question, facts, grade_band, history)`. If training only provides `question`, what it learns is `f(question)`; passing an extra parameter `facts` at deployment that it has never used won't make it start using it.

You can train and answer without fact cards; the cost is everything above: mistakes you can't fix, a tendency toward overconfidence, and starting over whenever you move to a domain the base model doesn't know. Separating "facts" from "phrasing" is this project's design trade-off, not the only way to do it; its benefit ultimately has to be verified with the comparisons in chapter 6.

#### Where fact cards are used

**Why they are stored separately.** The 60 cards are referenced by the 600 conversations via `fact_ids` (one conversation can reference several cards), much like a foreign key in a database: each fact is edited in one place only, and every conversation that references it uses the new content the next time it is read.

**Why they are also isolated by set.** A card may only be referenced by conversations in one set (one of train/val/test). Otherwise the material behind the validation and test questions would already have been seen by the model during training, and the evaluation results would be optimistic.

| Stage | Code location | How fact cards are used |
|---|---|---|
| Load and validate (D1) | [data.py](../lab/data.py) `validate` | `id` is unique; all fields present; `source_url` is https; `kind` must be fact/legend/current; every `fact_ids` entry in a conversation must exist; the same card cannot be shared across train/val/test |
| Build the prompt (D3) | [common.py](../lab/common.py) `build_messages` | Fetches cards by `fact_ids`, keeps only `id/text/kind/scope`, and writes them as JSON after `CURRENT REFERENCE FACTS:` in the system prompt; source and review fields do not enter the model |
| Notebook training and validation samples | Notebook step 2, `prepare-data` cell | All 1150 train samples and 115 val samples go through `build_messages` and carry their own fact cards |
| Notebook observation questions | Notebook step 4, `cases` cell | Each of the three observation questions specifies a `fact_id`, and the same card is used to compare answers before and after training |
| CLI snapshot | [prepare.py](../lab/prepare.py) | Written as-is into the snapshot's `facts.jsonl`; few-shot examples record their `fact_ids` |
| A/B/C comparison inference | [infer.py](../lab/infer.py) | Reads `facts.jsonl` from the snapshot; all three groups get the same fact card for the current question (section 6.4) |
| Blind review | [evaluate.py](../lab/evaluate.py) `blind` | The score sheet includes `fact_ids`; graders fill in `fact_claims`/`supported` by checking against the fact card (section 6.6) |
| Calls after deployment | Call example in section 5.5 | `read_dataset` loads the facts, and `messages_for(fact_id, ...)` builds the same system prompt as in training |

#### Source and review fields

`source_url` is the entry point for a human to go back and check the source; the program does not go online and read it automatically. `checked_at` only means the reference entry point has been read; `source_review` set to `reference_read_claim_review_pending` means the individual claims are still awaiting review. `reuse` is the source attribution and a note on original rewriting; it does not mean the source organization endorses this data. Official science outreach from NASA, NOAA, USGS and others, and OpenStax textbooks, are the reference entry points; the body text is original Chinese writing, and the analogies, children's reactions and observation designs are synthetic demonstrations. A URL being present does not mean every sentence is supported by a source.

### 2.3 The first complete conversation

```json
{
  "id": "day-night-01",
  "group_id": "day-night",
  "category": "space",
  "topic": "昼夜",
  "grade_band": "1-2",
  "scenario": "repair_misconception",
  "fact_ids": [
    "day-night"
  ],
  "provenance": {
    "type": "synthetic",
    "author": "Codex",
    "date": "2026-09-29",
    "method": "按主题逐项撰写知识、误解、追问、比喻边界和安全观察，再组合为10种教学对话路径；不是儿童真实对话，部分主题内回答复用。"
  },
  "review": {
    "status": "draft",
    "reviewer": "",
    "reviewer_kind": "",
    "notes": "待逐段核查科学事实、来源支持、适读学段、观察可行性、近重复与上下文衔接；成人审阅不代表已验证儿童探索兴趣。"
  },
  "turns": [
    {
      "user": "为什么白天亮、晚上黑？",
      "assistant": "地球在不停转，我们住的地方转向太阳就是白天，转到背光的一面就是晚上。",
      "behavior": "explain_at_grade"
    },
    {
      "user": "太阳晚上关灯睡觉了吧？",
      "assistant": "太阳没有在晚上关灯。是我们所在的地面转到了背向太阳的一侧。还可以接着想：如果地球不转，就没有黑夜了吗？",
      "behavior": "repair_then_offer_specific_question"
    }
  ]
}
```

(The two turns in English: the child asks "Why is it bright in the day and dark at night?" and the assistant answers "The Earth keeps turning. When the place we live turns toward the Sun it is daytime; when it turns to the side away from the light, it is night." The child then asks "The Sun turns off its light and goes to sleep at night, right?" and the assistant answers "The Sun doesn't turn off its light at night. It's the ground we're on that has turned to the side facing away from the Sun. Here's something to think about next: if the Earth didn't turn, would there be no night at all?" `provenance.method` says the dialogues were written topic by topic and combined into 10 teaching paths, are not real children's conversations, and reuse some answers within a topic; `review.notes` lists what still needs human checking.)

The child first asks about day and night, then offers the guess that "the Sun goes to sleep". The reference answer first explains the phenomenon, then corrects the specific misconception, and leaves an optional question to keep thinking about. Don't just learn stock catchphrases like "你真棒" ("Great job!").

| Field | Does it enter the model, and where is it used |
|---|---|
| `id` | Generates sample ids such as `day-night-01:0`, for tracing |
| `group_id` | Groups a science topic together; all grade bands, paths and rewrites must be in the same split |
| (containing array) | This conversation is in the `train` array; on load it is tagged `split="train"`, and D2 filters to train only; not written into the prompt |
| `grade_band` | **D3 writes it into the system prompt as "小学学段" (primary-school grade band)**; it is visible input |
| `fact_ids` | D3 selects fact cards from this file, passing only id/text/kind/scope |
| `turns[].user` | The child's current question; earlier questions go into history |
| `turns[].assistant` | The current answer is the training target; earlier answers are context only |
| `turns[].behavior` | Lets a human see what behavior this turn demonstrates; **not passed to the model** |
| `category/topic/scenario` | Labels for statistics and review; we don't sneak "what the child got wrong" into the input |
| `provenance`, `review` | Synthesis method and review status, for humans only; all conversations are currently `draft` |

If you remember just one thing from this table: **the model sees only the system prompt (role + grade band + fact cards), the history and the current user turn; it learns only the current assistant turn.** Section 2.5 turns this sentence into real tokens and masks for you to see.

### 2.4 Expanding a conversation answer by answer

`day-night-01` has two turns, so it expands into two samples:

| Sample | Context | Target that contributes to the loss |
|---|---|---|
| `day-night-01:0` | Role + grade band + day-night material + "why is it bright in the day and dark at night" | 1st reference answer + EOS |
| `day-night-01:1` | Same role, grade band and material + turn 1 Q&A + the "Sun goes to sleep" guess | 2nd reference answer + EOS |

The first sample can't see the second turn. In the second sample, the earlier answer in the history does not contribute to this turn's loss. Of the 500 train conversations, 350 have two turns and 150 have three, which yields **1150 samples**. 1150 is neither a token count nor a number of optimizer steps.

**Does a two-turn conversation mean "training twice"?** It isn't the same content trained twice; the conversation contributes two independent samples, and each is trained once per epoch. After the 1150 samples go into the trainer they are shuffled and the parameters are updated every 16 samples, so `day-night-01:0` and `day-night-01:1` usually land in different steps. The 1st answer is part of the loss only in the 1st sample; in the 2nd sample it is just history.

**The same rule applies when there are more turns**, just like the full chat log of a session: sample k treats all of the previous k−1 Q&A turns as history and learns only the k-th answer. This matches how calls work at deployment: the server keeps no state, and the client sends the full history every time (section 5.5). There are two differences you should know about:

- During training the history contains the **reference answers**; at deployment it contains **what the model itself said**, which may differ from the reference answers. The multi-turn mode in section 6.5 specifically checks this difference.
- Later samples are longer, and the history is recomputed in every sample. This project's validation accepts only conversations of 2–4 turns, and anything over 4096 tokens raises an error right away instead of being silently truncated.

### 2.5 Masks: which tokens the model actually learns

Here is `day-night-01:0` rendered with this project's tokenizer (this is real output; a few lines of role rules in the middle of the system prompt are omitted). The whole sample is 408 tokens, and only the last 22 contribute to the loss:

```text
mask │ 渲染后的文本
─────┼──────────────────────────────────────────────────────────
  0  │ <|im_start|>system
  0  │ 你是陪小学生探索科学的伙伴。用中文，先回应孩子真正的问题，再按需要解释。
  0  │ ……（角色规则共 10 行）
  0  │ 小学学段：1-2 年级。
  0  │ CURRENT REFERENCE FACTS:
  0  │ [{"id": "day-night", "text": "自转使地球表面……", "kind": "fact",
  0  │   "scope": "昼夜的入门解释；……"}]<|im_end|>
  0  │ <|im_start|>user
  0  │ 为什么白天亮、晚上黑？<|im_end|>
  0  │ <|im_start|>assistant
  1  │ 地球在不停转，我们住的地方转向太阳就是白天，转到背光的一面就是晚上。<|im_end|>
```

(The header reads "mask │ rendered text". The system prompt begins "You are a companion who explores science with primary-school students. Use Chinese; first respond to the child's real question, then explain as needed.", followed by "(10 lines of role rules in total)" and "Primary-school grade band: grades 1-2.")

The 386 tokens with mask 0 are visible to the model but no loss is computed on them here; only the last 22 tokens with mask 1 are the learning target. The fact card is among those 386 tokens: the model doesn't memorize it, but it has to learn to answer while looking at it, so it must be included during training too (section 2.2, Q2 and Q3).

Split into tokens, the last line is: `地球` `在` `不停` `转` `，` `我们` `住` `的地方` `转向` `太阳` `就是` `白天` `，` `转` `到` `背` `光` `的一面` `就是` `晚上` `。` `<|im_end|>`. During training the model makes one guess of the "next token" at each of these 22 positions, and how wrong it is becomes the loss; the 386 positions before them only provide context.

Why not let the model learn the system and user parts too? Because after deployment that content is always supplied by the caller; the model only needs to learn "given input like this, how should I answer". If the user turn were included in the loss, the model would spend effort imitating children's questions, and might even learn to make up the next question by itself.

`rows` keeps `case_id/prompt/completion/input_ids/completion_mask` so you can inspect them easily. Actual training uses only the last two columns. The notebook prints the full prompt, target and real token length of the first row; don't reuse length numbers from older scenarios.

`completion_mask=0` marks context, and `=1` marks the current answer plus EOS. TRL sets the labels of context and padding positions to `-100`, so no loss is computed directly at those positions, but the model can still see the context when predicting the answer.

EOS is <code>&lt;&#124;im_end&#124;&gt;</code>; it appears exactly once per target so the model learns to stop. PAD is different from EOS. Exceeding 4096 tokens raises an error and the sample must be rewritten; the answer is never silently cut off.

### 2.6 Recap: seven steps from one JSON file to a parameter update

The previous five sections covered the data in the order "what the data looks like → how it is split → what the model sees and learns". Strung together, they form the seven steps below. The notebook's code comments mark with D1–D7 which line does each step; follow along during training and you'll know which line is doing which part of this chapter.

| Step | What it does | Where in this chapter |
|---|---|---|
| D1 Load | Read `dataset.json`, tag each conversation with its set, check fact-card references and cross-set isolation | Sections 2.1, 2.2 |
| D2 Split | Take train only, split each conversation into samples by answer, with history containing only earlier turns | Section 2.4 |
| D3 Build input | Role rules + grade band + fact cards + history + current question, assembled into one list of messages | Sections 2.2, 2.3 |
| D4 Mark the target | Apply the chat template, tokenize, and mark the current answer and EOS as mask=1 | Section 2.5 |
| D5 Hand to the trainer | Pass only the two columns `input_ids` and `completion_mask` to TRL | Section 2.5 |
| D6 Double-check | Check the labels the trainer actually generates: context and padding positions are all -100 (no loss) | Section 2.5 |
| D7 Keep a record | Save a copy of this run's data and its fingerprint; only then does `trainer.train()` actually update parameters | Chapter 4 |

### 2.7 Changing the data, or switching to your own

You don't need to change the data for the exercises; read this section when you want to start editing.

**Editing the existing data.** To change the phrasing, edit `assistant`. When adding a new reaction path, adjust the turns before and after it too, so the child never quotes something the model didn't actually say. If you change a fact card, re-check every conversation that references it. After editing, run the structure check first:

```bash
python -m lab.data --input data/dataset.json --allow-draft
```

It should print `{'train': 500, 'val': 50, 'test': 50}`. It only checks structure: all fields present, fact references exist, no topic crosses sets, at most one question mark per answer. It does not check whether the science is correct.

**Making changes take effect.** An adapter you have already trained does not change because you edited the JSON. In the notebook, set `RUN_DIR` to a new directory (for example `runs/science-002`) and re-run from step 2. Each training run directory automatically keeps a copy of `dataset.json` as a record of "which data this run used"; don't edit it in day-to-day work.

**Switching to your own data.** Common open-source SFT data formats such as Alpaca (instruction/output) and ShareGPT (conversations) are essentially the same as this project's: "a list of conversation messages + which one is the expected output", just laid out differently in JSON. To switch data you mainly change two things: organize your data into three sets of conversations, train/val/test; and adjust D3's `build_messages` as needed to decide what goes into the system prompt.

### 2.8 Optional: run the data checks on your own computer

Everything above can be understood just by reading. If you want to see the tokenization results with your own eyes before renting a GPU, or check your data after editing it, you can install a lightweight environment without PyTorch on your own computer (Mac or Linux, Python 3.11). It takes a few minutes and needs no graphics card:

```bash
python3.11 -m venv .venv-data
source .venv-data/bin/activate
python -m pip install -r requirements-data.txt
python -m lab.download --tokenizer-only
python -m lab.data --allow-draft
```

The last line should print `{'train': 500, 'val': 50, 'test': 50}`. `--tokenizer-only` downloads only the tokenizer and config (about 15 MB), not the 8 GB of weights, so it is normal for your local `models/qwen3-4b/` to have no weight shards. Transformers may warn `None of PyTorch, TensorFlow ... have been found`; locally you only use the tokenizer, so you can ignore it. Whenever you open a new terminal later, run `source .venv-data/bin/activate` first.


## 3. GPU Setup: Rent a Machine, Install the Environment, Download the Model

This is where you start spending money. Before you start the machine, make sure you have finished chapters 1–2 and prepared the upload bundle locally.

!!! abstract "Chapter at a glance"
    - **What this chapter does**: start an RTX 4090 cloud host; learn to read `nvidia-smi`; upload the project; build the Python environment for training; download the 8 GB base model from ModelScope and verify it; pick the right kernel in JupyterLab
    - **What you need**: an account and credit on a GPU platform; `llm-lora-tutorial-gpu.zip`, built on your local machine with `python3 tools/build_gpu_zip.py` (code, data and tokenizer only, a few MB)
    - **What you end up with**: the `.venv-train` environment passes its self-check; `models/qwen3-4b/` passes verification; the Notebook is open with the **LoRA · Python 3.11** kernel
    - **Rough time**: 30–60 minutes, mostly installing dependencies and downloading weights

The five steps in this chapter and the problem each one solves:

| Step | What problem it solves | What happens if you skip it |
|---|---|---|
| 3.1 Pick a machine | VRAM must fit the 4-bit base model plus training intermediates; the disk must fit the environments, the base model and the merged model | OOM during training, or the disk fills up halfway through the merge |
| 3.2 Read `nvidia-smi` | Confirms the GPU, driver version and free VRAM | The PyTorch you install later may be incompatible with the driver and report "CUDA not available" |
| 3.3 Upload + build the environment | Builds a project-private environment with pinned library versions and registers it as a Jupyter kernel | You end up on the image's older bundled libraries, and code behavior no longer matches the tutorial |
| 3.4 Download the base model | Gets model weights that are byte-identical to the ones this tutorial uses | A different version gives different answers and numbers; a missing shard only errors out at load time |
| 3.5 Open the Notebook | Picks the right kernel | The Notebook runs on the system Python, and imports fail or use the wrong versions |

### 3.1 GPU Host and Files

#### Where to rent: Zhichuan Cloud

All GPU work in this book was done on [Zhichuan Cloud (gpuhome.cc)](https://gpuhome.cc/auth/register?code=zt4f01m0). We recommend beginners simply use it, mainly because it is cheap and quick to get started with:

| Item | On Zhichuan Cloud |
|---|---|
| Price | RTX 4090D (24 GB) pay-as-you-go **¥1.25/hour**; ¥29 per day, ¥200 per week, ¥800 per month (prices on the page as of 2026-10-11; the price at order time is what counts). The main path of this book takes about 2–3 hours, **only three or four yuan** |
| New-user offer | Your first top-up of ¥1 or more gets a ¥5 voucher, which is roughly enough for one run through the main path |
| GPU models | RTX 3090 / 4090 / 4090D / 4090 Lite / 5090, A100, L40S and more, 1–8 GPUs, multiple regions |
| Images | Has the PyTorch base image used in this book, `2.7.1-CUDA12.6-Py3.11-Ubuntu22` |
| Usage | One-click JupyterLab, SSH and VSCode from the instance page; comes with a 50 GB data disk |

![Zhichuan Cloud rental page: RTX 4090D 24 GB, pay-as-you-go ¥1.25/hour, ¥29 per day, ¥200 per week, ¥800 per month](../images/gpuhome-rtx4090d-price-2026-10-11.png)

How to choose: for the GPU model pick any of **RTX 4090, 4090D or 4090 Lite**; all have 24 GB of VRAM, and none of the commands in this book need to change (the 4090D is the China-market version, with slightly less compute than the 4090 and the same VRAM). For the image pick **PyTorch → `2.7.1-CUDA12.6-Py3.11-Ubuntu22`**. For billing pick **pay-as-you-go**, and shut the machine down when you are done.

Two lessons from actually doing it; deal with these first after the machine starts:

- **Put the project on the data disk.** The instance's home directory `/root` is on a 30 GB system disk. Install the training environment, download the 8 GB model, then install vLLM, and it fills up. Use `df -h` to find the roughly 50 GB disk (on the author's instance it is mounted at `/root/rivermind-data`) and unzip the project there.
- **The official PyTorch index may be unreachable.** Sections 3.3 and 5.4 both give fallback commands that use a mirror inside China.

Sign-up link: **[https://gpuhome.cc/auth/register?code=zt4f01m0](https://gpuhome.cc/auth/register?code=zt4f01m0)**. Enter the invitation code **`zt4f01m0`** when you register to get the new-user voucher. This is the author's referral link: after your first top-up, the author also receives a platform voucher. Vouchers are issued manually by the platform; once you meet the conditions, contact customer support to claim yours. They are valid for three months.

#### The host configuration used here

This chapter follows the RTX 4090 cloud host we already chose and started. The configuration below comes from the launch screenshot saved on 2026-09-29; after logging in you still need to check the actual device and environment.

| Item | What we chose | What to watch for |
|---|---|---|
| GPU | 1× RTX 4090, 24 GB VRAM | This tutorial uses a single GPU; chapter 4 runs one full epoch by default (about 72 steps) |
| CPU / host RAM | 15 cores / 100 GB | Meets the 32 GB host RAM the tutorial recommends, and the 64 GB recommended for the later CPU merge |
| System disk | 30 GB SSD | Keep the project, training environment and model on the data disk where possible |
| Data disk | 50 GB SSD, no expansion enabled in the screenshot | For the full workflow, expanding to 100 GB is recommended; check the actual free space before downloading |
| Base image | `2.7.1-CUDA12.6-Py3.11-Ubuntu22` | Ubuntu 22, Python 3.11, with PyTorch 2.7.1 / CUDA 12.6 preinstalled |
| Billing | Pay-as-you-go | When you finish a session's experiments, stop idle instances according to the platform's rules and check storage charges |

The figure below shows the GPU and image configuration confirmed for this run:

![RTX 4090 24GB cloud host with the 2.7.1-CUDA12.6-Py3.11-Ubuntu22 image selected](../images/gpu-rtx4090-24gb-2026-09-29.png)

**The image is chosen; next, build the project's own training environment inside it.** The image comes with PyTorch 2.7.1 preinstalled, while this tutorial's code and dependencies are pinned to PyTorch 2.8.0. The next section creates `.venv-train` and installs `2.8.0 + cu126`, leaving the image's original environment untouched. The official install commands are at [PyTorch 2.8.0](https://pytorch.org/get-started/previous-versions/#v280).

The machine is running, but the driver, dependencies, training VRAM and run time on this host have not been verified yet. This chapter prepares the environment, the download and the kernel; chapter 4 looks at the base model in the Notebook and updates the LoRA parameters.

#### Choosing a different GPU

The table below refers to standard desktop cards and is meant for planning a Qwen3 4B, 4-bit QLoRA setup with at most 4096 tokens and one sample at a time. Actual usage and run time still have to be measured; when renting, check the full model name, per-GPU VRAM, and whether the GPU is dedicated to you.

| GPU model | VRAM per GPU | Recommendation |
|---|---|---|
| [RTX 4090](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/) | 24 GB GDDR6X | The model used here; use the `cu126` commands below as written |
| [RTX 3090](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/) | 24 GB GDDR6X | A fine choice if you already have one or it actually rents for less; use `cu126` |
| [RTX 5090](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/) | 32 GB GDDR7 | More VRAM headroom; when installing PyTorch, change the `cu126` at the end of the index URL to `cu128` |

The 5090 uses the Blackwell architecture and needs a host driver that supports it. The [bitsandbytes 0.48.2 support table](https://huggingface.co/docs/bitsandbytes/v0.48.2/installation) includes the `sm120` it needs in the CUDA 12.8 build. If you use a 5090, also change the CUDA version assertion in the check script below from `12.6` to `12.8`. This run uses a 4090, so just run the commands in the next section.
### 3.2 Read nvidia-smi First: Driver, CUDA and VRAM

The first thing to do after logging in to the GPU host is run `nvidia-smi`. Its output looks roughly like this:

```text
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.35.03              Driver Version: 560.35.03      CUDA Version: 12.6     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090        Off | 00000000:01:00.0 Off   |                  Off |
| 30%   35C    P8             20W / 450W  |       0MiB / 24564MiB  |      0%      Default |
+-----------------------------------------+------------------------+----------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

| Field | Meaning | What you should care about |
|---|---|---|
| `Driver Version` | Kernel driver version, set by the platform image; ordinary users can't change it | Determines the CUDA ceiling below |
| `CUDA Version` | **The highest CUDA version this driver supports**, not the installed CUDA | PyTorch's `cu126` needs it to be ≥ 12.6; vLLM 0.21's `cu129` needs ≥ 12.9 |
| `Memory-Usage` | Used / total VRAM | Should be close to 0 before training; watch how high it climbs during training; before starting vLLM, confirm the training kernel has released it |
| `GPU-Util` | GPU compute utilization | Should stay fairly high during training and load testing; sitting at 0 for long means the program is stuck on the CPU or waiting |
| `Processes` | Which processes are holding VRAM | Use it to find a Jupyter kernel you forgot to shut down |

**How the driver, CUDA and PyTorch relate:** the driver is installed in the operating system and tied to the GPU. CUDA is NVIDIA's compute library, and the PyTorch wheel ships the copy it needs (the `cu126` suffix on `torch==2.8.0` means "bundles the CUDA 12.6 runtime"), so you don't need to install the CUDA toolkit yourself. The only constraint is that the version the driver supports must not be lower than the version bundled in the wheel. The `CUDA12.6` in the image name refers to the preinstalled toolkit, which is a separate thing from the driver's ceiling; trust `nvidia-smi`.

An analogy: the driver is like the OS kernel version, the CUDA runtime is like a dependency an app bundles, and `nvidia-smi` is like `top` plus `free -h`.

VRAM and host RAM are two separate pools of memory. Model weights, training intermediates and the KV cache all have to live in VRAM; the host's 100 GB of RAM is only used for staging and for the merge on the CPU. When VRAM runs out, the error is `CUDA out of memory`, and the fix is to reduce usage (quantization, shorter samples, lower concurrency), not to add host RAM.

### 3.3 Upload the Project and Prepare the Training Kernel

Upload the bundle **`llm-lora-tutorial-gpu.zip`** to the GPU data disk using the platform's file upload feature. You generate this bundle in the local project root with the command below; regenerate it whenever the Notebook, code or data changes. The tokenizer in the bundle comes from your local `models/qwen3-4b/`, so before packing for the first time, run `lab.download --tokenizer-only` once as described in section 2.8:

```bash
python3 tools/build_gpu_zip.py
```

The packing script refuses a Notebook that still has run outputs, so what you upload is always a clean copy. After uploading, open a terminal where you uploaded it and unzip:

```bash
unzip llm-lora-tutorial-gpu.zip
cd llm-lora-tutorial
pwd
nvidia-smi
df -h .
```

Run every later GPU command from this project directory. The Notebook prints the actual absolute path; don't unzip a second copy of the project inside `notebooks/`. If you already have an older version of the project, unzip the new version into a different directory and keep the original `runs/` and model download directory; don't overwrite training results directly.

#### What's in the upload bundle

It is about 4 MB compressed and about 17 MB unzipped, 31 files in total. It only contains what is "needed for training and deployment on the GPU, but can't be downloaded online or must match this tutorial exactly" (the descriptions in the listing are in Chinese; the chapters referenced there explain each file):

```text
llm-lora-tutorial/
├── GPU_UPLOAD_README.md            GPU 上的操作速查：建环境 → 下权重 → 打开 Notebook
├── notebooks/01_gpu_train.ipynb    训练主入口，第 4 章逐格执行
├── data/
│   ├── dataset.json                唯一的数据文件：事实卡 + train/val/test 对话（第 2 章）
│   └── README.md                   数据逐字段说明
├── lab/                            Notebook 和命令行共用的 Python 代码
│   ├── common.py                   拼提示词、生成训练样本和掩码（第 2.5 节）
│   ├── data.py                     读入并校验 dataset.json、按回答拆样本
│   ├── download.py                 下载并校验基座权重（第 3.4 节）
│   ├── download-lock.json          12 个模型文件的大小和 SHA-256，下载后逐个核对
│   ├── training.py、train.py       训练器配置与命令行训练（第 4.6 节、第 6 章）
│   ├── prepare.py、audit.py        命令行快照与 Token 检查（第 4.6 节、第 6 章）
│   ├── infer.py、evaluate.py       A/B/C 对照推理与盲评（第 6 章）
│   └── merge.py                    命令行合并 adapter（第 5.1 节）
├── models/qwen3-4b/                基座的“说明部分”，没有权重
│   ├── config.json 等配置           模型结构、默认生成参数
│   ├── tokenizer.json、vocab.json、merges.txt、tokenizer_config.json   分词器和聊天模板
│   ├── model.safetensors.index.json   权重分片索引（分片本身还没下载）
│   ├── model-lock.json             本教程固定的模型版本号
│   └── LICENSE、README.md          模型许可证与模型卡
├── requirements-train.txt          训练依赖（Transformers、PEFT、TRL、bitsandbytes 等，版本固定）
├── requirements-download.txt       下载依赖（ModelScope；备用镜像用的 huggingface_hub 随 Transformers 安装）
├── requirements-notebook.txt       Jupyter 内核与画 loss 曲线的依赖
├── requirements-data.txt           数据检查依赖
└── pyproject.toml                  项目元数据
```

In short: `GPU_UPLOAD_README.md` is a quick reference for the GPU steps; `notebooks/01_gpu_train.ipynb` is the main training entry point (chapter 4); `data/` holds the single dataset file and its field guide; `lab/` is the Python code shared by the Notebook and the CLI (prompt building and masks in `common.py`, download and verification in `download.py` with the size and SHA-256 of 12 model files in `download-lock.json`, plus the CLI training, snapshot, inference, evaluation and merge scripts); `models/qwen3-4b/` is the base model's "descriptive part" (config, tokenizer and chat template, shard index, pinned version, license and model card) without the weights; the `requirements-*.txt` files pin the training, download, notebook and data-check dependencies.

| Not in the bundle | Why | Where it comes from |
|---|---|---|
| The 8 GB weight shards `model-0000X-of-00003.safetensors` | Too big and slow to upload; the GPU host downloads them faster from a mirror inside China | Downloaded on the GPU in section 3.4 and verified against `download-lock.json` |
| PyTorch and the other dependencies themselves | Tied to the GPU and CUDA version, so they must be installed on the GPU host | The pip commands below install them from `requirements-*.txt` |
| Training results `runs/`, snapshots `data/processed/` | These are artifacts produced on the GPU | Created by training in chapter 4 and by snapshot generation in section 4.6 |
| Automated tests `tests/`, e-book source | Not used on the GPU | The full project, or the e-book's "Source download" page (section 8.1) |

The tokenizer is included so the GPU host can check the model version before downloading weights; the download then only has to fill in the three weight shards. The Notebook reads `data/dataset.json` directly, so you don't need to run `lab.prepare` first. The 50 GB data disk has to hold the training environment, the original weights, the merged weights and the inference environment at the same time; for a full deployment, expanding to 100 GB is recommended.

Build the project environment in the GPU terminal:

```bash
python3.11 -m venv .venv-train
.venv-train/bin/python -m pip install --upgrade pip
.venv-train/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
.venv-train/bin/python -m pip install -r requirements-train.txt -r requirements-download.txt -r requirements-notebook.txt
.venv-train/bin/python -m pip check
.venv-train/bin/python -m ipykernel install --user --name lora-train --display-name "LoRA · Python 3.11"
```

Training uses pinned versions of Torch/Transformers/TRL/PEFT; ModelScope is only used for downloading; ipykernel lets the platform's Jupyter use this environment. If you already built `.venv-train` following an older version of the tutorial, you can reuse it: install the newly added dependencies and register the kernel. If the regular Python package index is slow, you can use the PyPI mirror the platform provides; the model download source and the pip package source are two different things.

**If you can't reach `download.pytorch.org` (you get `Network is unreachable` or endless retries):** the third command uses `--index-url` to temporarily point downloads at the official PyTorch index, which hosts inside China often can't reach. Switching only to a regular PyPI mirror (Tsinghua, Alibaba, etc.) does not fix this: they only carry the default torch 2.8.0 from PyPI (bundling CUDA 12.8), while the self-check below requires 12.6. Use Alibaba Cloud's PyTorch mirror instead, which has the same `2.8.0+cu126` package:

```bash
.venv-train/bin/python -m pip install 'torch==2.8.0+cu126' -f https://mirrors.aliyun.com/pytorch-wheels/cu126/
```

`-f` only tells pip to look on that page for torch itself; torch's dependencies such as `nvidia-*` and `triton` still come from your configured PyPI source. Writing the version as `2.8.0+cu126` keeps pip from installing the cu128 build from PyPI. Once it's installed, continue with the fourth command above (`requirements-*.txt`); the remaining steps are unchanged. To pin a PyPI source permanently, run `.venv-train/bin/python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple`, and use `pip config list` to view the current configuration.

```bash
.venv-train/bin/python - <<'CHECK'
import sys, torch, transformers, trl, peft, bitsandbytes
print("Python:", sys.executable)
print("Torch / CUDA:", torch.__version__, torch.version.cuda)
assert torch.__version__.split("+")[0] == "2.8.0"
assert torch.version.cuda == "12.6"
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
print("GPU:", torch.cuda.get_device_name(0))
print("依赖:", transformers.__version__, trl.__version__, peft.__version__, bitsandbytes.__version__)
CHECK
```

Only continue once this passes (the `依赖` line prints the dependency versions). If CUDA is not available, check the kernel, driver and wheel first; the image's preinstalled Torch 2.7.1 cannot stand in for this project's environment. The kernel registration follows the [official IPython documentation](https://ipython.readthedocs.io/en/stable/install/kernel_install.html).

### 3.4 Download the Full Weights on a GPU Host in China

By default we use Alibaba's **ModelScope**; the repository is [Qwen/Qwen3-4B-Instruct-2507](https://www.modelscope.cn/models/Qwen/Qwen3-4B-Instruct-2507). It has its own SDK; this is not a matter of pointing `HF_ENDPOINT` at the ModelScope homepage.

```bash
.venv-train/bin/python -m lab.download --source modelscope --output models/qwen3-4b
```

Do not add `--tokenizer-only`. This command keeps the `model-lock.json` and the already verified tokenizer from the upload bundle and fills in the three missing weight shards. If the network drops, rerun the same command; it keeps `.cache` and the version lock, and you don't need to prepare the data again.

[download.py](../lab/download.py) checks the remote file sizes and SHA-256 against [download-lock.json](../lab/download-lock.json), then checks the content actually downloaded. ModelScope fetches files from `master`, but they must be **byte-identical** to the Hugging Face revision pinned here; if the repository content has changed, it stops, so versions never get mixed. The metadata of all 12 files in the list has been checked against both sources; the three weight files total about 8.0 GB (decimal).

After downloading you can run an offline check on its own:

```bash
.venv-train/bin/python -m lab.download --check-only --output models/qwen3-4b
```

Only move on to training when you see **“校验通过：12 个文件；完整模型权重已就绪”** (“Verification passed: 12 files; full model weights are ready”) and the command exits normally. The directory should contain:

```text
models/qwen3-4b/
├── model-lock.json
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── model.safetensors.index.json
├── model-00001-of-00003.safetensors
├── model-00002-of-00003.safetensors
└── model-00003-of-00003.safetensors
```

Keep the other config, vocabulary and license files as well. Seeing only `model-lock.json` and the weight index, or a log line saying `Returning existing local_dir`, does not mean the weights were downloaded. The current downloader rejects such incomplete results.

#### Fallback: download through a Hugging Face mirror in China

If ModelScope is unreachable, you can explicitly choose `hf-mirror.com`:

```bash
.venv-train/bin/python -m lab.download --source hf-mirror --output models/qwen3-4b
.venv-train/bin/python -m lab.download --check-only --output models/qwen3-4b
```

The code passes the mirror address explicitly to the Hugging Face SDK, so you don't need to patch installed package source code or permanently change global environment variables, and it still uses the same version lock and file verification. This mirror is an independent third-party service, not Alibaba Cloud's official ModelScope; whether you can reach it depends on the GPU host's network. Based on the [Hugging Face download API](https://huggingface.co/docs/huggingface_hub/package_reference/file_download).

On the development machine, the ModelScope config/tokenizer download has been tested and passed verification, and the three large weight files also passed a download test of a few bytes. **The full 8 GB of weights has not yet been downloaded on this GPU host**; the full verification result on the host is what ultimately counts.

| Symptom | What to do |
|---|---|
| `huggingface.co` / `Network is unreachable` | Use an explicit `--source modelscope`, or the fallback `--source hf-mirror` |
| Missing weight shards | Rerun the full download in the same directory, without `--tokenizer-only` |
| Local file hash mismatch | Move the reported file aside as a backup and download again, keeping the version lock; don't skip the check |
| Remote file doesn't match the pinned version | Stop using that source and try another source for the pinned content; don't edit the hash to let it through |
| `No space left on device` | Check which partition the project is on and expand it or move to the data disk; don't delete experiment results you haven't backed up |

### 3.5 Open the Notebook and Start Training

In JupyterLab, open [01_gpu_train.en.ipynb](../notebooks/01_gpu_train.en.ipynb) and select the **LoRA · Python 3.11** kernel. If you don't see it, refresh Jupyter; don't pick the plain Python 3 kernel that comes with the image.

If this is your first time using Jupyter, these are the only things you need to know:

| Action | How |
|---|---|
| Run the current cell | Click into a code cell and press `Shift+Enter`; after it runs, focus jumps to the next cell |
| Tell whether it's running | `[*]` to the left of a code cell means it is running; a number like `[3]` means it finished as the third execution |
| See output | Output appears right below the code cell; long output is collapsed, click the blue bar on the left to expand it |
| Switch kernel | Menu Kernel → Change Kernel; the current kernel name is shown in the top right |
| Start over | Kernel → Restart Kernel clears all variables in memory; files on disk are not affected |
| Free VRAM | Kernel → Shutdown Kernel; closing the browser tab does not free it |

The Notebook uses straightforward Python training code. Below each code cell there are three parts: what it does, why, and what you'll see after running it. The third part shows the expected output and what each number means; your actual values will differ slightly. Cell 1 sets `ROOT` and `RUN_DIR`, with results going to `runs/science-001/` by default. When opened from the project or the `notebooks` directory it detects the path automatically; from anywhere else, fill in `ROOT` by hand.

Next, train cell by cell following chapter 4. The base model's answers are part of the Notebook too, so there's no need to run the old A/B commands separately.

## 4. Training and Validation: Your First QLoRA Run

!!! abstract "Chapter at a glance"
    - **What this chapter does**: run the Notebook cell by cell: see how the base model answers → add LoRA → train 72 steps → look at the loss curve → save, reload, compare answers → merge into a full model
    - **What you need**: everything in chapter 3 passed: the `.venv-train` self-check, base model verification, and the Notebook running on the **LoRA · Python 3.11** kernel
    - **What you end up with**: under `runs/science-001/`, the adapter, the loss curve, a before/after answer comparison, and `merged-model/`
    - **Rough time**: training itself takes from ten-odd to several tens of minutes (measure it yourself); with loading, generation and merging, the whole Notebook takes about 1 hour

This chapter uses the [GPU Notebook](../notebooks/01_gpu_train.en.ipynb) as the main entry point. `LoraConfig`, `SFTConfig`, `SFTTrainer`, `trainer.train()` and the save/reload code appear directly in the cells; the expected output of each cell and the parameter-by-parameter explanations are in the Notebook. This chapter covers the big picture: what each step does, how the training loop works inside, where the VRAM goes, where the artifacts end up, and how to read the results. The command-line version, [train.py](../lab/train.py), is used for resume from checkpoint in section 4.6 and for the comparison experiments in chapter 6.

### 4.1 What Each Step Does

| Notebook step | What runs | What you should check |
|---|---|---|
| 1–2 | Check the kernel, read dataset.json, build training/validation samples in memory, compute the step count | Python comes from `.venv-train`; 1150 train and 115 val samples; 72 planned steps |
| 3–4 | Load the base model in NF4 and generate answers to three fixed questions | Note the VRAM at this point; you can see the base model's answers |
| 5–6 | Add LoRA, configure TRL, inspect the actual `labels` | About 33 million trainable parameters (about 0.8%), all of them LoRA |
| 7 | `trainer.train()`, then plot the loss curve | Progress reaches 72/72; training loss goes down overall; the trend of the validation loss |
| 8 | Confirm the parameters were updated and save the adapter | The completion record, weights, config and training state have all been written |
| 9–10 | Release the training objects, reload the adapter from disk, compare answers again | The answer style has changed; judge by hand whether it's better |
| 11 | Merge in BF16 from the unquantized base model, reload the full model and generate | `merged-model/` loads on its own |
| 12–13 | Deploy vLLM from the GPU terminal, call it locally | Run in a terminal; see chapter 5 |

The three questions before and after training use the same role prompt, facts, NF4 precision and greedy decoding, so you can compare the effect of the weight changes.

#### What happens inside `trainer.train()`

Cell 7 is a single line, `trainer.train()`, but it ties together every concept from chapter 1. One step goes like this:

```text
重复 16 次（梯度累积）：
    取 1 条样本的 input_ids（约 400 个 Token）
    前向：NF4 基座临时还原成 BF16 参与计算，LoRA 的 A、B 叠加在 7 类线性层上
          → 每个位置输出“下一个 Token”的概率
    loss：只在 completion_mask = 1 的位置（参考回答 + EOS）比较预测和答案
    反向：算梯度，但只算到 LoRA 的 A、B 上；基座参数冻结，没有梯度
          → 梯度累加起来，不立刻更新
AdamW 按累加的梯度更新 LoRA 参数一次（学习率按 warmup + cosine 变化）
step + 1；每 12 步在 115 条 val 样本上只做前向、算一次验证 loss；每 25 步存一次检查点
```

In English: repeat 16 times (gradient accumulation): take one sample's `input_ids` (about 400 tokens); forward pass, with the NF4 base temporarily dequantized to BF16 for computation and LoRA's A and B added on top of 7 kinds of linear layers, so each position outputs a probability for "the next token"; loss compares prediction and answer only where `completion_mask = 1` (the reference answer + EOS); backward pass computes gradients only down to LoRA's A and B (the base parameters are frozen and get no gradient), and the gradients are accumulated rather than applied immediately. Then AdamW updates the LoRA parameters once with the accumulated gradients (learning rate follows warmup + cosine). Step + 1; every 12 steps, run a forward-only pass over the 115 val samples to compute the validation loss; every 25 steps, save a checkpoint.

After 72 steps, each of the 1150 samples has been seen exactly once (1 epoch). Throughout the whole process not a single one of the 4 billion base parameters changes; only about 33 million LoRA parameters do, which is why cell 8 specifically checks that "the LoRA tensors really changed".

### 4.2 Why These Settings

| Setting | Current value | Reason |
|---|---|---|
| Base model | Qwen3-4B-Instruct-2507, pinned revision | A fixed point of comparison |
| Training method | NF4 + BF16 compute + LoRA | Updates only a small number of parameters on a single 24 GB GPU |
| LoRA | rank 16, alpha 32, dropout 0.05 | A common starting point for practice |
| Target layers | q/k/v/o and gate/up/down projections | Covers the attention and feed-forward projections |
| Micro-batch / accumulation | 1 / 16 | Only one sample in VRAM at a time; accumulate 16 before each update |
| Optimizer / learning rate | AdamW / 1e-4 | Paired with a short warmup and a cosine schedule |
| Length / concatenation | 4096 / no packing | Never silently truncates targets, and keeps the supervision range easy to inspect |
| Amount of training | 1 epoch ≈ 72 steps | Every training sample is seen once and the before/after difference is visible; to just test the pipeline, change it to 10 |
| Validation | Validation loss every 12 steps | Uses topics that were not trained on to watch for overfitting |

What each config parameter means, why it has this value, and what happens if you change it are covered in the tables under the corresponding Notebook code cells (steps 3, 5 and 6), so they are not repeated here.

### 4.3 Watching the GPU During Training: Where the VRAM Goes

Before training starts, open a second GPU terminal:

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv -l 2
```

It prints one line every 2 seconds. Compare how the numbers change against the estimates below:

| Usage | Estimate | When it appears |
|---|---|---|
| CUDA context | About 0.5 GB | The first time the kernel uses the GPU |
| NF4 base weights | About 2.5–3 GB (4B × 0.5 bytes; a few layers such as the word embeddings stay in BF16) | After the model is loaded in step 3 |
| LoRA parameters + gradients + AdamW state | 33 million parameters × about 16 bytes ≈ 0.5 GB | Steps 5–7 |
| Activations | Vary with sample length; gradient checkpointing keeps them fairly small | During training, in each sample's forward/backward pass |

The longest sample in this dataset is about 550 tokens, so peak VRAM is usually far below 24 GB. After training, `training-complete.json` records PyTorch's `peak_allocated_gib` and `peak_reserved_gib`; the `nvidia-smi` numbers are somewhat larger because they also include the CUDA context and the caching pool. GPU utilization should stay fairly high during training; if it stays low for long, the bottleneck may be data loading or the CPU.

Don't use the VRAM of this batch of short samples to infer the cost of 4096-token samples. If you hit OOM, first shut down other kernels, then check sample lengths; if you really need to lower the limit, rewrite the overlong samples rather than cutting off the reference answers.

### 4.4 Where to Confirm That Training Has Finished

If the project is at `/root/llm-lora-tutorial`, the default artifacts are in the locations below; for a project on the data disk, go by the absolute path the Notebook prints:

```text
/root/llm-lora-tutorial/runs/science-001/
├── training-complete.json       # steps, loss, number of changed parameters, run time, peak VRAM, adapter fingerprint
├── trainer_state.json          # training log and global_step
├── manifest.json               # base model version, data fingerprint, training config
├── dataset.json                # copy of the data actually read in this run; don't edit it day to day
├── loss-curve.png              # training/validation loss curve
├── adapter/
│   ├── adapter_model.safetensors   # LoRA weights, about 130 MB
│   ├── adapter_config.json
│   └── manifest.json
├── checkpoint-50/              # training checkpoint (includes optimizer state, training can resume from it)
├── checkpoint-72/
├── before-answers.json
├── answer-review.json           # before/after answers and the review to fill in
├── inference-check.json         # the reloaded adapter can generate
├── merged-model/                # full BF16 model, about 8 GB
└── merged-inference-check.json
```

`training-complete.json` only exists once step 8 finishes. You should see `final_step=expected_steps=72`, a finite loss, `changed_lora_tensors>0`, and the adapter's two core files non-empty. Note that the adapter is only about 130 MB while the merged full model is about 8 GB: that's where LoRA saves storage.

### 4.5 Read the Loss Curve, Then Compare Answers

How to read `loss-curve.png`: training loss going down overall means the model is fitting the reference answers; validation loss going down at the same time means this way of answering also works on topics it hasn't seen; training loss still going down while validation loss starts rising is overfitting. If neither line moves, check the learning rate, the trainable parameters and the mask first.

The figure below is a real curve from running one full epoch with this chapter's default configuration on an RTX 4090 (24 GB) on 2026-10-11; it took about 26 minutes:

![Real loss curve for 1 epoch on an RTX 4090: training loss drops quickly from about 6 to around 2.5 and then declines slowly; validation loss falls from 2.51 to 1.96 and then levels off](../images/loss-curve-rtx4090-2026-10-11.png)

| Step | 12 | 24 | 36 | 48 | 60 | 72 |
|---|---:|---:|---:|---:|---:|---:|
| Validation loss | 2.51 | 2.19 | 2.03 | 1.98 | 1.96 | 1.96 |

Reading this curve section by section:

- **First 10 steps: training loss plunges from about 6 to around 2.5.** The way the base model is used to talking is very different from these short, conversational reference answers, so at first it is very "surprised"; LoRA learns the format and tone first. During the first few steps the learning rate is still warming up, which is why steps 1–3 barely go down.
- **Steps 10–48: both lines slowly go down together.** This is the phase where it really learns "how to say it". The validation loss is going down too, which means what it learned also applies to the 5 topics it hasn't seen.
- **After step 48: validation loss levels off at 1.96.** One epoch has basically learned as much as it can; training longer probably wouldn't do any better on the validation set.
- **Ignore the jagged blue line.** Each point is just the average of one step's 16 samples, so it jumps when it hits an easy or hard batch; look at the overall trend.
- **The blue line ending slightly below the orange line (about 1.5 vs 1.96) is normal.** The validation set covers topics it hasn't seen, and the answers contain a lot of topic-specific content; learning only the speaking style can't push that part down.

**Compare an abnormal curve:** if you run the training cell again without restarting the kernel, it keeps training the already trained model, which amounts to training for several extra epochs. It then looks like this: training loss gets pushed down to around 0.5, while validation loss climbs back up from 2.0 to 2.3. The training set keeps getting better while the validation set keeps getting worse: that is overfitting by memorizing the training data. Before retraining, always do **Kernel → Restart Kernel** first, then run again from cell 1.

Step 9 recreates the base model and loads the adapter from disk; step 10 prints the answers before and after training. After 1 epoch you will usually see the answers get shorter and structured more like the training data. Open `answer-review.json` and judge each question: are the facts correct, does it follow requirements like “一句话” (“one sentence”) and “不追问” (“no follow-up questions”), is the Chinese right for the grade band, are the exploration suggestions concrete and safe; then fill in which answer you prefer and why.

The three observation questions come from training topics, so they only show "what style it learned". To answer "is LoRA better than the base model with a good prompt", use the validation-set blind review in chapter 6.

### 4.6 Reruns, Small-Sample Overfitting, and Resume from Checkpoint

To retrain from scratch, restart the kernel, change `RUN_DIR` to a new directory such as `runs/science-002`, and run the cells one by one again. Keep the old directory. The save cell can be used to check the logs, but don't casually rerun `trainer.train()` after training has finished.

The two exercises below are not in the Notebook; they use the command-line scripts.

**First, know this: the command-line scripts read a "snapshot".** The Notebook reads `dataset.json` directly; the command-line scripts (this section, section 5.2 and chapter 6) read a read-only snapshot generated by `lab.prepare`. A database analogy: `dataset.json` is the editable source table, and the snapshot is a read-only export with checksums that every experiment result can be traced back to. The snapshot records fingerprints of the data, the tokenizer and the prompt code; if any of them changes, later commands refuse to use the old snapshot instead of quietly mixing them. The snapshot directory contains the train/val/test conversations, `facts.jsonl`, `fewshots.json`, `manifest.json` and the tokenized samples, all generated automatically; don't edit them by hand. A non-empty output directory is refused; just pick a different version name.

The exercises: **small-sample overfitting** trains 20 samples for 100 steps to confirm the training pipeline can "memorize" the reference answers; it's the first step when debugging "loss doesn't go down". **Resume from checkpoint** shows how, after training is interrupted, to continue from a checkpoint (which includes optimizer state and progress) and finish the same training plan. Run these in the GPU terminal: generate the snapshot first, and keep the outputs in their own directories. Don't use the CLI's `--resume` directly on a Notebook directory; the two entry points record their configuration differently.

```bash
# Only needed the first time you use the CLI; skip if it already exists and the data hasn't changed.
.venv-train/bin/python -m lab.prepare --input data/dataset.json --allow-draft --output data/processed/science-v3

# Checks whether a small batch of samples can learn the reference answers; it can't be used to judge generalization.
.venv-train/bin/python -m lab.train --data data/processed/science-v3 \
  --output runs/overfit-20 --allow-draft --limit 20 --max-steps 100

# The same plan: 10 steps in total, stop at step 5 first, then resume the remaining 5.
.venv-train/bin/python -m lab.train --data data/processed/science-v3 \
  --output runs/resume-demo --allow-draft --max-steps 10 --stop-after 5
.venv-train/bin/python -m lab.train --data data/processed/science-v3 \
  --output runs/resume-demo --allow-draft --max-steps 10 --resume runs/resume-demo/checkpoint-5
```

Run these extra experiments as needed; on your first pass, finish training, reloading, answer evaluation and deployment first.


## 5. Deployment and API Calls: Exporting the Model, vLLM Serving, and Load Testing

What training gives you is just files on disk. This chapter turns them into an HTTP service that any backend code can call, and measures how much concurrency it can handle.

!!! abstract "Chapter at a glance"
    - **What this chapter does**: tells an adapter apart from a full model; explains how vLLM uses VRAM; deploys the merged model; calls it from your own computer through an SSH tunnel (single-turn, multi-turn, streaming); switches to the "base model + mounted adapter" deployment and compares the two; load tests
    - **What you need**: `runs/science-001/merged-model/` and `adapter/` produced in chapter 4; the notebook kernel already **Shut down** (to free VRAM); a local computer that can SSH into the GPU host
    - **What you get**: a running OpenAI-compatible service; answers fetched by calls from your local machine; a filled-in latency and throughput table for concurrency 1/4/16
    - **Rough time**: about 10 minutes to install vLLM; about 30–60 minutes to deploy, call, and load test

An API request leaves your computer and comes back with an answer after passing through these stages:

```text
本地 Python 发出请求
  → 本地 127.0.0.1:18000
  → SSH 隧道
  → GPU 主机 127.0.0.1:8000 上的 vLLM
      ① 用模型目录里的 chat_template 把 messages 拼成一段文本（和训练时相同）
      ② 分词成 Token，进入调度队列，和其他请求拼成同一批
      ③ 模型逐个生成 Token，KV Cache 保存已处理部分，直到 EOS 或 max_tokens
      ④ Token 解码回文字，一次返回 JSON，或按 stream=True 分段推送
  → 原路返回：回答文本 + finish_reason
```

(In English: local Python sends the request → local 127.0.0.1:18000 → SSH tunnel → vLLM on the GPU host at 127.0.0.1:8000, which ① uses the chat_template in the model directory to join the messages into one text (same as in training), ② tokenizes it, puts it in the scheduling queue and batches it with other requests, ③ generates tokens one by one, with the KV Cache holding the already-processed part, until EOS or max_tokens, ④ decodes the tokens back to text and returns JSON in one piece, or pushes it in chunks with stream=True → back along the same path: answer text + finish_reason.)

Step ① explains why calls must use the same prompt structure as training: vLLM uses the very same chat template, so as long as the messages are the same, the model sees the same tokens it saw during training.

### 5.1 Telling an Adapter Apart from a Full Model

The default tutorial result `runs/science-001/adapter/` is a delta: it needs the same version of `models/qwen3-4b/`. Notebook steps 9–10 release the training objects, reload the base model and the adapter from disk, and generate three answers. Look at `inference-check.json` and `answer-review.json`; a successful load still needs a human to judge the output quality.

Step 11 loads a fresh **unquantized BF16 base model** on the CPU, merges the LoRA into it, saves it to `runs/science-001/merged-model/`, and then reloads that directory to generate one answer. vLLM deployment uses the complete directory, including the config, tokenizer, weight shards, and index. Before deploying, make sure you have looked at the answer comparison in section 4.5 and have a rough idea of how this model answers.

When you need to export the adapter of another experiment, run [merge.py](../lab/merge.py) in a terminal:

```bash
.venv-train/bin/python -m lab.merge --model models/qwen3-4b \
  --adapter runs/science-001/adapter --output runs/science-001/merged-model
```

**Skip this command if the notebook already merged.** The merge directory must be empty; the standalone adapter and the base model version lock are kept.

### 5.2 Optional: Checking Before and After the Merge at the Same Precision

This section uses the command-line snapshot introduced in section 4.6. If you have not prepared it, first run `lab.prepare --input runs/science-001/dataset.json` with the parameters from section 4.6 to generate `data/processed/science-v3`, so that it uses the data copy from this training run; if the directory already exists, confirm the contents match, and if you changed the source data, use a different output directory.

The NF4 answers before the merge and the BF16 answers after the merge may differ. To check whether the merge introduced anything abnormal, compare a freshly loaded BF16 base model + adapter against the BF16 full model, keeping the prompt and decoding configuration identical:

```bash
.venv-train/bin/python -m lab.infer --model models/qwen3-4b --data data/processed/science-v3 \
  --variant C --adapter runs/science-001/adapter --precision bf16 \
  --split train --limit 3 --output runs/science-001/bf16-adapter.jsonl
.venv-train/bin/python -m lab.infer --model runs/science-001/merged-model --data data/processed/science-v3 \
  --variant C --merged --precision bf16 \
  --split train --limit 3 --output runs/science-001/bf16-merged.jsonl
```

This is only a sanity check of saving/merging; for a formal candidate, switch to the reviewed validation set and the candidate paths. Look at the facts, instruction following, and any abnormal repetition or truncation, and record the differences; do not treat similar answers on the training set as a quality improvement. The serving framework can also introduce numerical differences, so check again with the same fixed cases before going live.

### 5.3 What vLLM Does on the GPU

Think of vLLM as an **application server that lives in VRAM**: at startup it loads the model weights onto the GPU and exposes HTTP endpoints in the same format as OpenAI (`/v1/chat/completions`, `/v1/models`). So any backend code that calls the OpenAI API can call your own model by changing `base_url`.

It is much faster than `model.generate()` in the notebook, mainly thanks to two things:

| Mechanism | Analogy | Effect |
|---|---|---|
| Continuous batching | Requests join the same processing batch as they arrive, like a shared thread pool | Many requests generate at once and the GPU never sits idle; as soon as one request finishes, the next one is swapped in |
| PagedAttention managing the KV Cache | Like an operating system allocating memory in pages instead of reserving the maximum space for each request | A request takes only as much VRAM as the tokens it actually uses, so more requests fit at the same time |

The **KV Cache** holds intermediate attention results saved for every already-processed token during generation, so the model does not have to recompute all the preceding text for each new character. It grows with "context length × number of requests processed at once", and is the biggest consumer of inference VRAM after the weights.

#### Work Out the VRAM Yourself First, Then Compare with the Log

Qwen3-4B's architecture parameters are in `models/qwen3-4b/config.json`: 36 layers, 8 KV heads, 128 dimensions per head. BF16 uses 2 bytes per number, so:

```text
每个 Token 的 KV Cache = 2（K 和 V）× 36 层 × 8 头 × 128 维 × 2 字节
                       = 147,456 字节 ≈ 144 KiB
一条 4096 Token 的满长请求 ≈ 4096 × 144 KiB ≈ 576 MiB
```

(That is: KV Cache per token = 2 (K and V) × 36 layers × 8 heads × 128 dims × 2 bytes = 147,456 bytes ≈ 144 KiB; one full-length 4096-token request ≈ 4096 × 144 KiB ≈ 576 MiB.)

| VRAM budget (`--gpu-memory-utilization 0.85`) | Estimate |
|---|---|
| Total available to vLLM: 24 GB × 0.85 | ≈ 20.4 GiB |
| BF16 weights: about 4 billion parameters × 2 bytes | ≈ 7.5 GiB |
| Runtime overhead such as activations and CUDA Graphs | ≈ 1–2 GiB |
| Everything left goes to the KV Cache | ≈ 11 GiB ≈ 70–80 thousand tokens ≈ 18–20 full-length requests |

The `GPU KV cache size: ... tokens` and `Maximum concurrency for 4096 tokens per request: ...x` lines in the startup log are what vLLM actually computed; write them down and compare them with the table above. If they differ a lot, another process (for example, a training kernel you forgot to shut down) is usually holding VRAM.

The startup parameters that appear here, such as `--gpu-memory-utilization`, `--max-model-len`, and `--max-num-seqs`, are explained one by one below the launch command in section 5.4.

### 5.4 Deploying the Merged Full Model

Run the following in a **terminal on the GPU host**, not in a Python cell. First use the Jupyter menu **Shutdown Kernel** to free the training VRAM, then change to the project directory. A separate `.venv-serve` keeps vLLM from overwriting the training dependencies.

```bash
cd /root/llm-lora-tutorial  # if the project is on a data disk, use the actual path instead
nvidia-smi
python3.11 -m venv .venv-serve
.venv-serve/bin/python -m pip install --upgrade pip
.venv-serve/bin/python -m pip install 'vllm==0.21.0' --extra-index-url https://download.pytorch.org/whl/cu129
.venv-serve/bin/python -m pip check
```

This vLLM version is built against CUDA 12.9 by default. The CUDA Version shown in the top-right corner of `nvidia-smi` is the highest version the driver supports; if it is below 12.9, first upgrade the driver through the platform or switch to a different image. CUDA 12.6 in an image's name is not the driver's upper limit. Source: [vLLM 0.21 GPU installation](https://docs.vllm.ai/en/v0.21.0/getting_started/installation/gpu/).

**If you cannot reach `download.pytorch.org`:** vLLM 0.21 depends on `torch==2.11.0`, `torchaudio==2.11.0`, and `torchvision==0.26.0`, and the command above fetches their CUDA 12.9 builds from the official PyTorch index. If a host in mainland China cannot reach it, get the same packages from the Alibaba Cloud PyTorch mirror instead:

```bash
.venv-serve/bin/python -m pip install 'vllm==0.21.0' -f https://mirrors.aliyun.com/pytorch-wheels/cu129/
.venv-serve/bin/python -m pip check
```

After installing, confirm with `.venv-serve/bin/python -c "import torch; print(torch.__version__, torch.version.cuda)"`; it should print `2.11.0+cu129 12.9`.

```bash
HF_HUB_OFFLINE=1 .venv-serve/bin/vllm serve "$PWD/runs/science-001/merged-model" \
  --served-model-name science-explorer-lora \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 4096 \
  --gpu-memory-utilization 0.85 --max-num-seqs 4 \
  --generation-config vllm
```

#### What This Launch Command Says, Piece by Piece

If you have ever started a web service, this command will look familiar. It is the same kind of thing as starting a Python web app with uvicorn:

```text
uvicorn app:app       --host 127.0.0.1 --port 8000 --workers 4
vllm serve <模型目录>  --host 127.0.0.1 --port 8000 --max-num-seqs 4 ……
```

(`<模型目录>` means `<model directory>`.)

The only difference is that what vLLM "deploys" is not a piece of business code but a model directory, so there are a few extra parameters about the model and VRAM. First, get to know the three shell idioms in the command:

| Syntax | Meaning |
|---|---|
| `HF_HUB_OFFLINE=1` at the start of the line | An environment variable that applies only to this one command; equivalent to `export` followed by running the command, but it does not affect later commands in the terminal |
| `$PWD` | The absolute path of the current directory. So you must `cd` into the project directory before running it |
| `\` at the end of a line | Line continuation: tells the shell the next line is still the same command. There must be no space after the `\` |

Below, the parameters are explained one by one in the order "who runs it → what it loads → what it is called externally → where it listens → precision and length → VRAM and concurrency → default generation parameters":

| Parameter | Value in this tutorial | Meaning | Why this value / what happens if you change it |
|---|---|---|---|
| `HF_HUB_OFFLINE=1` | — | Read only local files; do not access the Hugging Face website | The model is already local; hosts in mainland China often hang when accessing huggingface.co, and without it startup may hang |
| `.venv-serve/bin/vllm serve` | — | Start an HTTP service using the vLLM in the inference-only environment | Use `.venv-serve`, not `.venv-train`: the two environments have different torch versions, and mixing them makes them overwrite each other |
| First argument: model directory | `runs/.../merged-model` | The model to load. You pass a **directory**, just like `from_pretrained`; vLLM reads the `config.json`, tokenizer, and weight shards inside it by itself | Change it to `models/qwen3-4b` to deploy the original model; section 5.6 demonstrates another way: load the original model, then mount the adapter |
| `--served-model-name` | `science-explorer-lora` | The model name published externally; the `"model"` field in client requests must use it, and `/v1/models` lists it too | It is just an alias and has nothing to do with which weights are loaded. Think of a service's public domain name, which has nothing to do with the directory name on the machine. A wrong name in a request returns 404 |
| `--host` | `127.0.0.1` | The network interface address to listen on. `127.0.0.1` means only connections from this machine are accepted | Not exposed to the public internet; outside access goes only through the SSH tunnel in section 5.5. Changing it to `0.0.0.0` opens it on all interfaces, and then you must add authentication (section 5.8) |
| `--port` | `8000` | The port to listen on | If the port is taken, pick another, e.g. 8001, and change the later `curl` commands and SSH tunnel to match |
| `--dtype` | `bfloat16` | The numeric precision for weights and computation | Matches the precision saved during the merge in chapter 4. Serving is not quantized, so the weights take about 7.5 GiB of VRAM, not the 2.5 GiB they took during training |
| `--max-model-len` | `4096` | The maximum number of tokens for one request, **input and output combined** | Same as the training limit. Longer requests get HTTP 400 right away. Raising it allows longer conversations, but each full-length request takes more KV Cache, so fewer requests fit at once |
| `--gpu-memory-utilization` | `0.85` | The fraction of the whole card's VRAM that vLLM **reserves all at once** at startup, for weights and KV Cache | That is why `nvidia-smi` shows about 85% in use right after startup with no requests at all: it is preallocation, not a leak. Raising it fits more KV Cache; the 15% left over is for the CUDA context and other processes. If startup fails for lack of VRAM, lower it first |
| `--max-num-seqs` | `4` | The maximum number of requests generating together at any moment | Like a web service's worker count or connection pool size: extra requests queue and wait. Raising it increases total throughput but makes each request slower; the load test in section 5.7 actually compares 4 and 16 |
| `--generation-config` | `vllm` | Generation parameters not given in the request (`temperature`, etc.) use vLLM's own defaults instead of the values in the model directory's `generation_config.json` | That file holds Qwen's default random sampling (`temperature 0.7`, etc.). This tutorial explicitly passes `temperature=0` in every request, and this parameter guarantees the defaults in the model directory cannot quietly affect it |

**How the three VRAM-related parameters work together:** `--gpu-memory-utilization` decides how much VRAM vLLM can use in total (24 GB × 0.85 ≈ 20 GiB); after subtracting the weights and runtime overhead, the remaining ~11 GiB all goes to the KV Cache. `--max-model-len` decides the most KV Cache one request can take (4096 tokens ≈ 576 MiB); `--max-num-seqs` decides how many can run at once at most. `Maximum concurrency` in the startup log is "how many full-length requests the total KV Cache can hold"; section 5.3 has the full estimate.

**What this command leaves out and so uses the defaults for:** only 1 GPU (`--tensor-parallel-size 1`); no API key check (`--api-key`, section 5.8); the endpoints are the OpenAI-compatible `/v1/chat/completions` and so on. See all parameters with `.venv-serve/bin/vllm serve --help`.

Change `science-001` in the path to the name of the experiment you just trained. `science-explorer-lora` is the model name used in API requests; the weights actually loaded come from `merged-model/`. In the startup log, find the two lines `GPU KV cache size` and `Maximum concurrency`; section 5.3 of the tutorial uses them to check the VRAM estimate. Once `Application startup complete` appears, check from another terminal on the GPU host:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

On success, `/health` returns 200 with no body; `/v1/models` returns something like:

```json
{"object": "list", "data": [{"id": "science-explorer-lora", "object": "model", "max_model_len": 4096, ...}]}
```

Even with no requests, about 85% of VRAM is already in use: at startup vLLM reserves the weights and KV Cache all at once according to `--gpu-memory-utilization`. This is preallocation, not a leak. Keep the service process running.

`HF_HUB_OFFLINE=1` makes vLLM read only the local directory and stay offline, so it does not hang trying to reach huggingface.co from a host in mainland China. If startup fails, check in this order: whether the training kernel is shut down (are there other Python processes in `nvidia-smi`), the driver's CUDA version, and whether the model directory is complete (it has `config.json`, the tokenizer, and all `*.safetensors`). If VRAM runs out, first lower `--max-model-len` or `--gpu-memory-utilization`; do not "fix" the service by switching the model path back to the original model.

### 5.5 Calling from Your Local Computer: Three Single-Turn Questions and One Multi-Turn Question

First set up an SSH tunnel in a **terminal on your local computer**. Replace the placeholders (`<平台SSH端口>` = the platform's SSH port, `<平台SSH地址>` = the platform's SSH address) with the actual values your cloud platform gives you, and keep this terminal open. This step is for calling the service, not for uploading files.

```bash
ssh -p <平台SSH端口> -N -L 18000:127.0.0.1:8000 root@<平台SSH地址>
```

The tunnel forwards local `127.0.0.1:18000` to `127.0.0.1:8000` on the GPU host. The service listens only on the GPU host's loopback address and is not exposed to the public internet; this is the simplest form of access control. In another local terminal, run `curl --fail http://127.0.0.1:18000/v1/models`. If the platform only offers web Jupyter and no SSH forwarding, first enable forwarding on the platform or use its authenticated port proxy; the Jupyter file-browser URL is not the model API address.

#### Calling Directly with curl First

Run all the following commands in **another terminal on your local computer**, keeping the tunnel terminal open. Locally you only need `curl`: macOS and Linux ship with it; on Windows use Git Bash or WSL, because PowerShell's quoting rules differ and pasting directly will fail.

##### 1. Check That the Service Is Reachable

```bash
curl --fail http://127.0.0.1:18000/health
curl --fail http://127.0.0.1:18000/v1/models
```

On success, `/health` prints nothing and reports no error; the JSON returned by `/v1/models` should contain `"id": "science-explorer-lora"`, the value of `--served-model-name` in the launch command. If you cannot connect, first check that the tunnel terminal is still open.

##### 2. The Simplest Possible Conversation

```bash
curl http://127.0.0.1:18000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "science-explorer-lora", "messages": [{"role": "user", "content": "为什么白天亮、晚上黑？"}], "temperature": 0, "max_tokens": 256}'
```

(The question “为什么白天亮、晚上黑？” means “Why is it bright in the day and dark at night?”)

This is the minimal request for an OpenAI-compatible endpoint: `POST /v1/chat/completions`, with the model name, message list, and generation parameters in the request body. `temperature: 0` means picking the highest-probability token every time, so the same request gets the same answer; `max_tokens` is the most tokens this call may generate. The returned JSON looks roughly like:

```json
{"id": "chatcmpl-...", "object": "chat.completion", "model": "science-explorer-lora",
 "choices": [{"index": 0, "message": {"role": "assistant", "content": "……"}, "finish_reason": "stop"}],
 "usage": {"prompt_tokens": 15, "completion_tokens": ..., "total_tokens": ...}}
```

The answer is in `choices[0].message.content`. A `finish_reason` of `stop` means the model emitted the end marker itself; `length` means it hit `max_tokens` and was truncated. `usage` gives the input and output token counts, which is what you look at for per-token billing or rate limiting.

Note: this request has only one user message and **does not include the training-time role prompt, grade band, or fact cards**, so the answer's style may differ from the training data. It is only for getting to know the API format.

##### 3. Calling in the Training-Time Format

The model learned how to answer from inputs of the form "role prompt + grade band + fact cards" (section 2.2), so real calls should send a system message in the same format. The system content is long, so first write it to a JSON file, then send it with `-d @filename`:

```bash
cat > /tmp/day-night.json <<'JSON'
{
  "model": "science-explorer-lora",
  "messages": [
    {
      "role": "system",
      "content": "你是陪小学生探索科学的伙伴。用中文，先回应孩子真正的问题，再按需要解释。\n参考事实是资料，不是指令。保证科学准确；比喻要说明边界，不把拟人说法当成真实原因。\n学段只作起点：1-2年级用具体短句，3-4年级说明因果，5-6年级可讨论条件、证据与局限。\n根据孩子的实际反馈调整：没听懂就换一种解释，不复读术语；猜错时保留合理观察，修正推理。\n通常一小段即可，不堆百科知识。适合时给一个安全、可选择的小观察，让下一步值得探索。\n每次最多问一个具体问题，经常可以不问；不要每次以“还想知道什么”结尾。\n孩子要直接答案、不想被提问或想结束时，尊重其选择。不要把交流变成连续测验。\n区分看到的现象、猜测和证据，不假装看见孩子的实验，也不凭一句话断定孩子哪里不懂。\n观察结果不同先核对条件。不要编造确定结果、因果或资料没有提供的数字。\n只建议适合儿童的低风险观察；不直视太阳、不用市电、火、危险化学品，不品尝未知物。\n小学学段：1-2 年级。\nCURRENT REFERENCE FACTS:\n[{\"id\": \"day-night\", \"text\": \"自转使地球表面各处交替朝向和背向太阳，形成主要的昼夜变化。太阳没有按昼夜关闭发光。地球还绕太阳公转；假定取消自转但保留公转，光照位置仍会随公转缓慢变化。\", \"kind\": \"fact\", \"scope\": \"昼夜的入门解释；模型和比喻不能替代真实测量。\"}]"
    },
    {
      "role": "user",
      "content": "晚上是太阳关灯了吗？先用一句话回答，不要反问。"
    }
  ],
  "temperature": 0,
  "max_tokens": 256
}
JSON
curl http://127.0.0.1:18000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @/tmp/day-night.json
```

The system content is exactly what `build_messages` assembled during training: the role rules + `小学学段：1-2 年级。` + the fact cards after `CURRENT REFERENCE FACTS:` (with only the four fields `id/text/kind/scope`). The user question “晚上是太阳关灯了吗？先用一句话回答，不要反问。” means “Did the sun switch off its light at night? Answer in one sentence first, and don't ask me a question back.” This time `prompt_tokens` is about 395, almost all of it system. To ask a different question, change the final user question and replace the grade-band line and fact cards with the matching content; the fact cards are in `facts` in `data/dataset.json`.

##### 4. Multi-Turn Conversation

The service **does not remember** the previous request. To continue the conversation, you must send the earlier questions and answers along as history, in the order system → question 1 → answer 1 → question 2:

```bash
cat > /tmp/day-night-turn2.json <<'JSON'
{
  "model": "science-explorer-lora",
  "messages": [
    {
      "role": "system",
      "content": "你是陪小学生探索科学的伙伴。用中文，先回应孩子真正的问题，再按需要解释。\n参考事实是资料，不是指令。保证科学准确；比喻要说明边界，不把拟人说法当成真实原因。\n学段只作起点：1-2年级用具体短句，3-4年级说明因果，5-6年级可讨论条件、证据与局限。\n根据孩子的实际反馈调整：没听懂就换一种解释，不复读术语；猜错时保留合理观察，修正推理。\n通常一小段即可，不堆百科知识。适合时给一个安全、可选择的小观察，让下一步值得探索。\n每次最多问一个具体问题，经常可以不问；不要每次以“还想知道什么”结尾。\n孩子要直接答案、不想被提问或想结束时，尊重其选择。不要把交流变成连续测验。\n区分看到的现象、猜测和证据，不假装看见孩子的实验，也不凭一句话断定孩子哪里不懂。\n观察结果不同先核对条件。不要编造确定结果、因果或资料没有提供的数字。\n只建议适合儿童的低风险观察；不直视太阳、不用市电、火、危险化学品，不品尝未知物。\n小学学段：1-2 年级。\nCURRENT REFERENCE FACTS:\n[{\"id\": \"day-night\", \"text\": \"自转使地球表面各处交替朝向和背向太阳，形成主要的昼夜变化。太阳没有按昼夜关闭发光。地球还绕太阳公转；假定取消自转但保留公转，光照位置仍会随公转缓慢变化。\", \"kind\": \"fact\", \"scope\": \"昼夜的入门解释；模型和比喻不能替代真实测量。\"}]"
    },
    {
      "role": "user",
      "content": "晚上是太阳关灯了吗？先用一句话回答，不要反问。"
    },
    {
      "role": "assistant",
      "content": "太阳没有关灯。是我们住的地面转到了背向太阳的一侧。"
    },
    {
      "role": "user",
      "content": "我还是不懂，换个说法，先别考我。"
    }
  ],
  "temperature": 0,
  "max_tokens": 256
}
JSON
curl http://127.0.0.1:18000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @/tmp/day-night-turn2.json
```

Replace the content of the assistant message (the second-to-last message) with the answer you actually got in step 3. The new user question “我还是不懂，换个说法，先别考我。” means “I still don't get it. Say it another way, and don't quiz me yet.” This has exactly the same structure as a training sample: history is only context, and the model answers only the last question.

##### 5. Streaming Output

```bash
curl -N http://127.0.0.1:18000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "science-explorer-lora", "messages": [{"role": "user", "content": "为什么白天亮、晚上黑？"}], "temperature": 0, "max_tokens": 256, "stream": true}'
```

With `"stream": true`, the service pushes each small piece as soon as it is generated; `curl -N` makes curl print what it receives immediately instead of buffering it. The output is in Server-Sent Events format: each line starts with `data:`, and the last line is `data: [DONE]`:

```text
data: {"id":"chatcmpl-...","object":"chat.completion.chunk",...,"choices":[{"index":0,"delta":{"content":"地球"},...}]}
data: {"id":"chatcmpl-...","object":"chat.completion.chunk",...,"choices":[{"index":0,"delta":{"content":"在"},...}]}
...
data: [DONE]
```

Each line's `delta.content` is a newly generated small piece; the frontend concatenates them in order to get the "display while generating" effect.

#### Batch Calls with Python

Run the following Python example from **the project root on your local computer**; it needs only the Python standard library and this repository's code. It reuses the training-time `build_messages`, so the system prompt, grade band, and fact-card format sent to the service are exactly the same as in the training samples. You need neither a local GPU nor an OpenAI account.

```python
import json
from urllib.request import Request, urlopen

from lab.common import build_messages
from lab.data import read_dataset

url = "http://127.0.0.1:18000/v1/chat/completions"
MODEL = "science-explorer-lora"
_, facts = read_dataset("data/dataset.json")


def ask(messages):
    body = {"model": MODEL, "messages": messages, "temperature": 0, "max_tokens": 256}
    request = Request(url, data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=120) as response:
        choice = json.load(response)["choices"][0]
    print("finish_reason:", choice["finish_reason"])
    return choice["message"]["content"]


def messages_for(fact_id, grade, user, history=()):
    # Same as training: role prompt + grade band + fact cards + history + current question.
    case = {"fact_ids": [fact_id], "grade_band": grade, "history": list(history), "user": user}
    return build_messages(facts, case)


questions = [
    ("day-night", "1-2", "晚上是太阳关灯了吗？先用一句话回答，不要反问。"),
    ("buoyancy", "3-4", "为什么小铁块沉，大铁船却浮着？我猜是大东西更容易浮。"),
    ("moon-phase", "5-6", "月亮变弯是被地球影子挡住了吗？我想安全地观察一下。"),
]
for fact_id, grade, question in questions:
    print("问题:", question)
    print("回答:", ask(messages_for(fact_id, grade, question)), "\n")

# Multi-turn: the service does not remember the previous turn, so you must pass the actually generated answer back as history.
first_question = "轮船那么重，为什么不会沉？"
first = ask(messages_for("buoyancy", "3-4", first_question))
history = [{"role": "user", "content": first_question},
           {"role": "assistant", "content": first}]
print("多轮回答:", ask(messages_for("buoyancy", "3-4", "我还是不懂排开水，换个简单说法，先别考我。", history)))
```

(The script's printed labels 问题 / 回答 / 多轮回答 mean "Question" / "Answer" / "Multi-turn answer". The questions are: “为什么小铁块沉，大铁船却浮着？我猜是大东西更容易浮。” (“Why does a small piece of iron sink while a big iron ship floats? I guess big things float more easily.”), “月亮变弯是被地球影子挡住了吗？我想安全地观察一下。” (“Does the moon look curved because Earth's shadow blocks it? I want to observe it safely.”), “轮船那么重，为什么不会沉？” (“Ships are so heavy, why don't they sink?”), and the follow-up “我还是不懂排开水，换个简单说法，先别考我。” (“I still don't get displacing water. Say it more simply, and don't quiz me yet.”))

For each question it first prints `finish_reason: stop`, then the answer, like this:

```text
问题: 晚上是太阳关灯了吗？先用一句话回答，不要反问。
finish_reason: stop
回答: 太阳没有关灯。是我们住的地面转到了背向太阳的一侧。
```

(The answer means: “The sun didn't switch off. The ground we live on has turned to the side facing away from the sun.”)

`stop` means the model emitted EOS itself and ended normally; `length` means the output hit `max_tokens` and was truncated, so it cannot be treated as a complete answer. The first request is a second or two slower (vLLM warming up); after that a single question usually returns within 1 second.

Check, in order: does it correct the wrong guess; can it switch to an explanation a child can understand; is the suggested observation safe and concrete; does it respect "don't quiz me".

The prompt must match the one used in training: the model learned how to answer from inputs of the form "role prompt + grade band + fact cards", and switching to a very different prompt may make results noticeably worse. The caller is responsible for passing in the fact cards; fine-tuning does not give the model new knowledge or real-time information.

#### Using the OpenAI SDK and Streaming Output

In backend projects it is more common to use the official OpenAI SDK directly. Install it locally with `python3 -m pip install openai`, and simply point `base_url` at the tunnel address; vLLM does not check keys by default, so `api_key` can be any non-empty string:

```python
from openai import OpenAI

from lab.common import build_messages
from lab.data import read_dataset

client = OpenAI(base_url="http://127.0.0.1:18000/v1", api_key="EMPTY")
_, facts = read_dataset("data/dataset.json")
messages = build_messages(facts, {"fact_ids": ["seasons"], "grade_band": "3-4", "history": [],
                                  "user": "为什么夏天热、冬天冷？"})

# stream=True: the service pushes each small piece as it is generated, so the frontend can display while generating.
stream = client.chat.completions.create(model="science-explorer-lora", messages=messages,
                                        temperature=0, max_tokens=256, stream=True)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

(The question “为什么夏天热、冬天冷？” means “Why is summer hot and winter cold?”)

Streaming output uses Server-Sent Events: multiple JSON chunks pushed one after another in the same HTTP response. The wait users feel is mainly "how long until the first token appears" (TTFT), not the total time of the whole answer; section 5.7 measures this metric.

### 5.6 Without Merging: Mounting the Adapter Directly and Comparing with the Base Model in the Same Service

Merging is not the only way to deploy. vLLM can load the **original base model** and then mount the LoRA adapter as an extra "model name". In the same service, requests for `qwen3-4b-base` go to the base model, and requests for `science-lora` go to the base model + LoRA. First press `Ctrl+C` to stop the service from section 5.4 (the port and VRAM cannot be given to two services at once), then start:

```bash
HF_HUB_OFFLINE=1 .venv-serve/bin/vllm serve "$PWD/models/qwen3-4b" \
  --served-model-name qwen3-4b-base \
  --enable-lora --lora-modules science-lora="$PWD/runs/science-001/adapter" \
  --max-lora-rank 16 \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 4096 \
  --gpu-memory-utilization 0.85 --max-num-seqs 4 \
  --generation-config vllm
```

`/v1/models` now lists two names. Change `MODEL` in the section 5.5 example to `qwen3-4b-base` and then `science-lora` and run it once with each: that is a "before vs after fine-tuning" comparison under the same prompt. `--max-lora-rank` must be no smaller than the rank used in training (16 in this tutorial).

| | Merged deployment (5.4) | Mounted adapter (5.6) |
|---|---|---|
| Disk | A full model of about 8 GB per version | One base model + an adapter of about 130 MB per version |
| Inference speed | Same as the base model | One extra small-matrix computation per layer, usually slightly slower |
| Multiple fine-tuned versions | One service and one set of VRAM per version | One service mounts several adapters at once, sharing the base model's VRAM |
| Updating the model | Re-merge and restart | Swap the adapter directory and restart the service |
| Best for | A single version, lowest latency | A/B comparison, multi-tenant, gradual multi-version rollout |

### 5.7 Load Testing: Concurrency, Latency, and Throughput

Before going live you need to know "how many requests one 4090 can handle". vLLM ships with a load-testing tool; run it in **another terminal** on the GPU host (keep the service running; this example uses the merged-model service from section 5.4):

```bash
.venv-serve/bin/vllm bench serve --backend openai-chat \
  --base-url http://127.0.0.1:8000 --endpoint /v1/chat/completions \
  --model science-explorer-lora --tokenizer "$PWD/runs/science-001/merged-model" \
  --dataset-name random --random-input-len 512 --random-output-len 128 \
  --num-prompts 64 --max-concurrency 1
```

The `random` dataset generates random inputs of the given length; it is only for measuring performance, and the answer contents are ignored. Parameter names follow `vllm bench serve --help` on your machine. Change `--max-concurrency` to 1, 4, and 16 in turn, run each once, and fill in the table:

| Client concurrency | Request throughput (req/s) | Output token throughput (tok/s) | Mean TTFT (ms) | Mean TPOT (ms) |
|---:|---:|---:|---:|---:|
| 1 | | | | |
| 4 | | | | |
| 16 | | | | |

| Metric | Meaning | Analogy |
|---|---|---|
| TTFT (Time To First Token) | From sending the request to receiving the first token | Time to first byte |
| TPOT (Time Per Output Token) | The average gap between each subsequent token | Push interval of a streaming response |
| Output token throughput | Total tokens generated per second | QPS, in tokens |

What you should see: going from concurrency 1 to 4, total throughput rises clearly while per-request TPOT gets only slightly slower; that is the benefit of continuous batching. At concurrency 16, the server-side `--max-num-seqs 4` caps the number of requests generating at once, the extra requests queue, and TTFT gets clearly longer. Stop the service, restart it with `--max-num-seqs` changed to 16, run concurrency 16 again, and compare throughput and TTFT. This is the trade-off between "throughput" and "per-request latency".

### 5.8 What Is Still Missing Before Going Live

The service in this tutorial listens only on `127.0.0.1` and is accessed through an SSH tunnel, which is fine for practice. To let other people use it, you need at least the following:

| Area | What to do |
|---|---|
| Authentication | Add `--api-key <密钥>` (`<密钥>` = your key) at startup, and have clients send it in `Authorization: Bearer <密钥>`; or put it behind a gateway that already has authentication. Never listen on `0.0.0.0` without authentication |
| Monitoring | vLLM serves Prometheus-format `/metrics` on the same port (`curl http://127.0.0.1:8000/metrics`), including queued requests, KV Cache usage, token throughput, and more; you can plug it into your existing monitoring |
| Process management | Host it with systemd, Docker, or the platform's service feature so it restarts automatically on crashes; loading the model takes tens of seconds, and health checks use `/health` |
| Client | Set timeouts; retry a limited number of times only on connection failures and 5xx; HTTP 400 usually means the input exceeds `--max-model-len`, and retrying won't help |
| Cost | VRAM stays occupied even when the service is idle, so stop pay-as-you-go instances when not in use |


## 6. Evaluation: Does Fine-Tuning Actually Beat Prompting?

This chapter answers a very practical engineering question: how much better is spending GPU time on fine-tuning than just writing a good prompt? You can run this **exploratory comparison** directly on the draft data to practice the experimental method; before reporting conclusions to anyone else, finish reviewing the data first.

!!! abstract "Chapter at a glance"
    - **What you do**: have three groups of models (base model + simple prompt, base model + good prompt and examples, LoRA) answer the same set of validation questions; compare two ranks; score with a blind review; touch the test set only after a candidate has been chosen
    - **What you need**: the `.venv-train` and base model from chapter 3; a few hours of continuous GPU time; at least one rater (ideally not the person who trained the model)
    - **What you get**: an experiment record with raw outputs, scores, and uncertainty intervals that can answer "is fine-tuning worth it?"
    - **When to do it**: after the main path of chapters 4–5 runs end to end. This chapter takes roughly 10–20 hours of GPU time (including several inference passes and two training runs), plus separate time for human scoring

Why not just look at the loss? Loss only measures "how closely it resembles the reference answer". The reference answers themselves may be wordy or contain mistakes and gaps, and the base model with a good prompt may already speak well enough. To judge whether fine-tuning adds value, you must take **topics it was never trained on**, have "good prompt" and "good prompt + LoRA" answer **exactly the same inputs**, and then have a person score them without knowing which is which. This is the same thing as running an A/B experiment on a backend: control your variables, test blind, and look at confidence intervals.

The chapter has two halves: 6.1–6.4 prepare the data and generate the three groups of answers, and 6.5–6.8 score, freeze, and draw conclusions. All commands run from the project root on the GPU host.

### 6.1 Current Data Size and Review Requirements

There are already 500/50/50 episodes, so there is no need to expand to the size of the old version again. Each category has 100/10/10 episodes, with 60 fact cards in total. "formal" is the fixed quota for this teaching experiment, not a general minimum requirement for LoRA.

All variants of the same topic share one group_id, and fact references must not cross splits either. The validation and test sets each have only 5 independent topics, so even though each has 50 episodes, they cannot be treated as 50 unrelated knowledge tests.

**What counts as review.** A formal review checks every episode: does the source support the facts, is there excessive anthropomorphism, can a child understand it, are the limits of each analogy clear, is the observation safe and feasible, and does it respect a child who doesn't want to be asked questions or wants to stop? Review happens directly in `dataset.json`, with version control keeping the history. Only entries that have truly been reviewed get `review.status` changed to `approved`, with the actual `reviewer`, `reviewer_kind`, and `notes` filled in; do not bulk-change the status to pass it off as review. Programs can check structure, but not near-duplicates, template-like phrasing, or scientific errors: the 10 path types deliberately repeat teaching behaviors, the same topic also reuses some answers, and 50 train topics are not 500 independent topics; val/test are isolated from the training topics, but still share basic concepts such as light, force, and living things.

With `--allow-draft`, draft data is enough to complete the validation-set comparison in sections 6.1–6.6 of this chapter, and the output manifest records `draft: true`. Freezing a candidate and the test set (section 6.7) still require reviewed data, so that draft results are not mistaken for formal conclusions.

### 6.2 How to Keep Improving the Data

First read all 10 paths for each topic and check whether each piece of feedback from the child causes the corresponding change. If unexplained terminology still appears for lower grades, rewrite it; the same analogy must not repeatedly stand in for a concrete explanation; and one observation cannot prove every conclusion unrelated to it.

Use the validation set to find failure types, then add training demonstrations for them; avoid changing the data based on test answers. Group new topics first, then assign them to splits. The current ten path types are structured synthetic data, not real interactions with children; if you are able to, collect anonymous feedback with informed guardian consent, and separately evaluate understanding and voluntary exploration rather than making "more rounds of chat" the only goal.

### 6.3 Validation and Preparation

Open a terminal in the project root on the GPU host and first activate the training environment installed in chapter 3; the following commands need PyTorch and TRL:

```bash
source .venv-train/bin/activate

python -m lab.data \
  --input data/dataset.json --allow-draft --formal

python -m lab.prepare \
  --input data/dataset.json \
  --tokenizer models/qwen3-4b --allow-draft --formal --output data/processed/experiment-v1

python -m lab.audit --data data/processed/experiment-v1 \
  --output runs/formal-token-audit.json
```

`--formal` checks the 500/50/50 episodes, the 60 fact cards, and the 100/10/10 quota per category; `--allow-draft` allows unreviewed entries. The two are independent checks: after a real review, drop `--allow-draft` and you get the formal data version (use a different output directory, for example `experiment-v2`).

Automatic checks: quotas, review status, source IDs, overlap of complete conversation groups, identical user conversations, control characters, non-empty targets, and number of questions asked. Near-duplicates, semantic errors, unsupported claims, and so on still need review; a program cannot judge whether a fact is true from a URL alone.

### 6.4 Three Baselines and Two Ranks

| Group | Weights | Prompt and examples |
|---|---|---|
| A | Base model | Basic prompt, no examples |
| B | Base model | Full role prompt + 3 few-shot examples selected from the training set |
| C | Base model + this LoRA | Exactly the same prompt and 3 few-shot examples as B |

The 3 few-shot examples for B and C are picked by `lab.prepare`, one episode each from near the beginning, middle, and end of train (covering all three grade bands where possible), taking only the first question-answer round, and written into the snapshot's `fewshots.json`. Few-shot examples are used only at inference time; SFT training samples do not include examples, to avoid pasting a sample's own answer into its input ahead of time.

All three groups use the same fact cards for the current question, NF4 precision, and greedy decoding. By default they read `val`, and the reference answers are used only for later review; fixed-history mode uses the previous rounds' reference replies as context. The B/C setup in this section and the Notebook's three no-few-shot observation questions are different protocols; do not mix their scores.

First run A and B on the validation set:

```bash
python -m lab.infer --data data/processed/experiment-v1 --variant A \
  --output runs/val-A.jsonl
python -m lab.infer --data data/processed/experiment-v1 --variant B \
  --output runs/val-B.jsonl
```

Train r=16, 1 epoch:

```bash
python -m lab.train --data data/processed/experiment-v1 \
  --output runs/formal-r16 --allow-draft --epochs 1 --rank 16
python -m lab.infer --data data/processed/experiment-v1 --variant C \
  --adapter runs/formal-r16/adapter --output runs/val-C-r16.jsonl
```

Change only the rank and its corresponding alpha, starting again from the base model:

```bash
python -m lab.train --data data/processed/experiment-v1 \
  --output runs/formal-r8 --allow-draft --epochs 1 --rank 8
python -m lab.infer --data data/processed/experiment-v1 --variant C \
  --adapter runs/formal-r8/adapter --output runs/val-C-r8.jsonl
```

The code always sets alpha=2×rank, so the scaling factor is 2 in both runs. Do not continue training r8 from the r16 adapter and then call it an independent comparison.

With 1150 training samples and an effective batch of ≈16, 1 epoch is about 72 optimizer updates, the same amount of training as the Notebook default. Group B already has a good prompt and few-shot examples, so it is perfectly normal for group C to show no clear advantage, and that is a valuable conclusion in itself: it means prompting is enough for this task.

**Acceptance:** both runs use identical data, prompts, generation parameters, and base model; you can compare validation results, training time, VRAM, and adapter size; do not pick the "better exploration companion" from training loss alone.

When you open a new terminal, first run `source .venv-train/bin/activate`, and make sure the data snapshot and candidate models from sections 6.3–6.4 have been generated.

### 6.5 Fixed History First, Then Continuous Conversation

Fixed history (the default, `--mode fixed`) uses the reference history from the data. The hash of each B/C input must be identical, which makes it good for pinpointing wording differences in a particular round.

Continuous conversation uses the model's own answers as the subsequent history:

```bash
python -m lab.infer --data data/processed/experiment-v1 --variant B \
  --mode rollout --output runs/val-B-rollout.jsonl
python -m lab.infer --data data/processed/experiment-v1 --variant C \
  --adapter runs/formal-r16/adapter --mode rollout --output runs/val-C-rollout.jsonl
```

Both groups use the same user script, but each model generates its own subsequent history, so later prompt hashes may differ. Some preset follow-up questions may not fit the generated answer; record such cases when scoring. A real person asking follow-ups based on the answers is a different kind of evaluation.

### 6.6 Human Blind Review

Full program: [evaluate.py](../lab/evaluate.py).

```bash
python -m lab.evaluate blind --base runs/val-B.jsonl \
  --tuned runs/val-C-r16.jsonl --output runs/blind-val-r16
```

This generates `ratings.jsonl`, `key.json`, and a protocol. When scoring, look only at the ratings and the fact cards, not the key; reveal which side is LoRA only afterwards. You can ask someone who was not involved in training to score, to reduce the author's bias toward the candidate.

For each item, fill in `winner` as X/Y/tie and write a concrete reason in `reason`; fill in `fact_claims_X/Y` and `supported_X/Y` separately. If there are no checkable facts, fill in 0, and do not treat that as "100% correct". The issues field records problems such as factual errors, Chinese above the grade band or piled-up jargon, excessive questioning, ignoring preferences, wrong attribution, and pretending to see the child's observation.

| Dimension | 0 points | 1 point | 2 points |
|---|---|---|---|
| Facts | Key claims unsupported/wrong | Partly unclear or too broad in scope | All checkable claims supported, unknowns stated clearly |
| Grade-band fit | Beyond the child's level, piled-up jargon | Understandable, but explanation doesn't adjust to feedback | Concrete and concise, re-explains when not understood |
| Exploration value | Clichés, fabrication, or forced questions | Has a point of interest, but the next step is vague | Concrete, safe, optional observation or follow-up |
| Interaction | Ignores an explicit choice | Basically responsive but repetitive | Respects choices, good pacing |
| Evidence awareness | Treats guesses or analogies as facts | Has qualifiers, but not enough | Distinguishes observation, explanation, and analogy; admits insufficient information |

The rubric helps you decide the winner, and per-dimension scores can go in reason; it is not automatic keyword scoring. An answer with a serious factual error should not win on fun alone, even if it is more "fun".

```bash
python -m lab.evaluate score --ratings runs/blind-val-r16/ratings.jsonl \
  --key runs/blind-val-r16/key.json --output runs/val-r16-score.json
```

A win counts 1, a tie 0.5, a loss 0. The script bootstraps by group_id science topic, to avoid treating the ten paths and multiple rounds of the same topic as independent samples; it reports the number of rounds, conversations, and topics. With fewer than 2 topics it outputs no interval. Five validation/test topics each is still very few, and the intervals cannot support broad conclusions. Small-sample intervals are only exploratory results and cannot remove rater bias.

Fact support rate = claims supported by sources / checkable claims; the script does not verify these claims automatically. If not all fact-review fields are filled in, it marks `factual_review_complete=false`, and you must not treat a partial ratio as a complete quality conclusion.

### 6.7 Use the Test Set Only After Freezing

This section needs reviewed, formal data (rerun prepare from section 6.3 without `--allow-draft`); `freeze` and test-set inference reject draft data. Choose the model based on the validation set. The following uses r16 as an example; if r8 fits better, change the subsequent adapter arguments to that candidate throughout. The choice is an outcome of this learning experiment, not something the tutorial assumes in advance.

```bash
python -m lab.evaluate freeze --data data/processed/experiment-v1 \
  --adapter runs/formal-r16/adapter --output runs/frozen-candidate.json

python -m lab.infer --data data/processed/experiment-v1 --split test --variant B \
  --freeze runs/frozen-candidate.json --output runs/test-B.jsonl
python -m lab.infer --data data/processed/experiment-v1 --split test --variant C \
  --adapter runs/formal-r16/adapter --freeze runs/frozen-candidate.json \
  --output runs/test-C.jsonl

python -m lab.evaluate blind --base runs/test-B.jsonl \
  --tuned runs/test-C.jsonl --output runs/blind-test
```

Run continuous conversation on the test set the same way, with `--mode rollout` and new output files. On test, using `--limit` to cherry-pick easy questions is forbidden. The freeze file records the data hash, the base model, a combined fingerprint of the adapter weights and config, the inference code, and the decoding parameters.

The test file is still local, so the experimenter needs to hold back the test questions until the model is frozen, to avoid writing training answers based on them. If you look at the test results and then keep tuning, the next round needs a new, independent test set.

### 6.8 How to Draw Conclusions

Write a one-page experiment record in this order: training configuration and cost → B/C comparison of facts and blind review → representative successes and failures → uncertainty → next steps.

Choose the next step based on the validation results: when the difference is uncertain, add independent evaluation; when a certain kind of behavior goes wrong, add matching training examples; when the LoRA gain is insufficient, keep the prompt baseline. Conclusions must consider both factual errors and stylistic preferences.

Adult preferences and model self-evaluation cannot directly prove that children are more willing to explore. The real goal also requires observing whether a child understands the core reason, whether they spontaneously ask related new questions, and whether they are willing to choose a safe observation, while respecting a refusal to participate.

**Acceptance:** save the complete scores, raw outputs, and failure classifications, so that others can re-check what input each comparison used.

## 7. Cost, Troubleshooting, and Acceptance

You don't need to read this chapter in order: before starting the machine, read 7.1 to estimate the budget; when you hit an error, look in 7.2; when everything is done, check against 7.3.

### 7.1 Measure Speed First, Then Estimate Cost

First check the provider's hourly price, storage charges, and the billing rules after stopping an instance, then set a budget for this experiment. The calculation below uses an assumed price of US$0.74/hour; the actual price is whatever is quoted when you create the instance.

```text
GPU 费用 = 小时单价 × 开机计费时长
总成本 = GPU 费用 + 存储 + 其他服务 + 数据制作/审核投入
训练时间估算 ≈ 预热后每个 optimizer step 耗时 × 预计更新步数
```

(In English: GPU cost = hourly price × billed running hours; total cost = GPU cost + storage + other services + data creation/review effort; estimated training time ≈ time per optimizer step after warm-up × expected number of update steps.)

The main path (environment setup, download, 1-epoch training, merge, deployment, load test) is expected to take 2–3 hours of machine time; adding the comparison experiments and review from chapter 6, it may take 10–20 hours. At the example price that is about US$1.5–2.2 and US$7.4–14.8 respectively. In practice you also need to account for setup, transfer, idle, and storage costs.

The Notebook's `training-complete.json` (the CLI's `metrics-step-N.json`) records training time and peak allocated/reserved VRAM. PyTorch's statistics do not cover other processes or all driver overhead, so cross-check against `nvidia-smi` and your bill. Update the budget with the measured time.

### 7.2 Troubleshooting Table

| Symptom | Check first | What not to do |
|---|---|---|
| CUDA unavailable | Host, driver, Torch wheel, current virtual environment | Reinstall CUDA on a Mac over and over |
| Missing Python module | Python path, activated environment, install result | Randomly upgrade libraries in multiple environments |
| Tokenizer works but model fails | Whether only the tokenizer was downloaded, whether the weights are complete | Always assume insufficient VRAM |
| Length limit exceeded | Token length of the facts, history, and target | Silently truncate the reference answer |
| Low loss but poor results | Target leakage, near-duplicates, which adapter is actually loaded | Just add more epochs |
| Loss doesn't decrease | Mask, trainable parameters, learning rate, data | Immediately switch GPUs |
| Outputs thinking/role markers | Model variant, template, training target, stop token | Blindly strip special text in the backend |
| Repeats/doesn't stop | EOS supervision, duplicated data, overfitting, output limit | Only filter out repeated sentences |
| OOM | Other processes, length, batch, KV cache, quantization | Treat RAM as equivalent to VRAM |
| Resume rejected | Whether the data/code/environment/model changed | Bypass the manifest check |
| Makes up stories more | Whether fiction is written as fact, missing samples of honest responses | Use fun scores to cover factual errors |
| Merge degradation | Revision, unquantized base model, template, precision | Attribute all NF4/BF16 differences to LoRA |
| API timeout | Service health, model loading, context, concurrency, logs | Retry forever |
| API returns 400 | Whether input + `max_tokens` exceeds `--max-model-len`, whether the `model` name matches `/v1/models` | Keep retrying the same request |
| Latency spikes during load test | Whether concurrency exceeds `--max-num-seqs`, whether the KV cache is full (check `/metrics` and logs) | Look only at the average, not the queueing |
| vLLM fails to start | Inference environment, driver, model directory, GPU in use by other processes | Overwrite the training environment's dependencies |

### 7.3 Final Learning Checklist

- Can explain why this project uses SFT+QLoRA.
- Can show a sample's full context and its actual supervision target.
- Completed 1 epoch of training, can read the loss curve, checked the completion record, and saved and reloaded the adapter.
- Can name the components of VRAM during training and match them against `nvidia-smi` readings.
- Completed the small-sample overfitting and resume-from-checkpoint exercises in section 4.6.
- Completed 1 epoch and one rank comparison, and saved a reviewable checklist.
- Can distinguish in-training performance, validation results, and frozen test results.
- Can report a gain, a regression, or "uncertain", and explain the limits of the evidence.
- Can merge, export, start vLLM, and make multi-turn and streaming calls.
- Can estimate KV cache capacity and compare it with the vLLM startup log; can explain `--gpu-memory-utilization`, `--max-model-len`, and `--max-num-seqs`.
- Can compare the base model and LoRA in the same service by mounting an adapter, and explain its trade-offs versus merged deployment.
- Ran load tests at different concurrency levels, filled in the TTFT, TPOT, and throughput table, and can explain the trade-off between throughput and latency.

Record the gain or regression of LoRA relative to the prompt baseline, with data and outputs attached so others can re-check.

## 8. Developer Reference: Automated Tests and Code Map

Readers who only do the exercises and don't change code can skip this chapter. If you want to change the data processing, the training code, or the tutorial itself, read this first: this project uses automated tests to tie "what the docs say" to "what the code actually does".

### 8.1 Automated Tests

After changing data-processing or training code, run the regression tests in the CPU environment. This environment includes PyTorch, TRL, and PEFT, for verifying the training mechanics on a tiny randomly initialized model; you can skip it for the data exercises in chapter 2.

```bash
python3.11 -m venv .venv-cpu
source .venv-cpu/bin/activate
python -m pip install -r requirements-cpu.txt
python -m pytest -q
```

The tests need the complete project source including `tests/`; the slimmed-down GPU upload bundle from chapter 3 does not include the developer tests. In the complete project you can also run the tests with `.venv-train`. The tiny-model tests don't need any weights downloaded.

| Layer | Test/command | What it proves |
|---|---|---|
| Data | `test_data.py`, `test_prepare.py`, `test_bundle.py`, `test_notebook.py` | Single-file loading, draft gate, group and future-round isolation, non-empty targets, snapshot compatibility |
| Template | `test_render.py`, `lab.audit` | Target span, token boundaries, EOS, overflow, real collator padding |
| Tiny training | `test_training.py`, `test_notebook.py` | The CLI training components and the Notebook's actual code can update, save, reload, and merge LoRA |
| Evaluation | `test_eval.py` | Pairing, missing scores, ties, conversation-clustered statistics |
| Docs and code | `test_tutorial.py`, `test_book.py` | Tutorial commands are accepted by the real CLI; deployment examples and the appendix match the source; pages and the download bundle work |
| API examples | `test_tutorial.py` | Runs the book's call examples directly, checking model name, address, parameters, and multi-turn history passing |
| Download | `test_download.py` | Every download source uses the same revision; missing/corrupted weights are rejected |
| GPU model | Commands from chapters 4 and 5 | Real QLoRA training/resume, adapter and merged output |
| vLLM integration | `/health`, `/v1/models`, the Python examples in chapter 5, `vllm bench serve` | The real service is callable; single-turn/multi-turn/streaming calls and concurrent load tests complete |

The API example tests use a temporary local HTTP server and cannot serve as evidence of vLLM inference. GPU and service checks run in their respective environments; items not run stay unchecked in todo.

Keep the following acceptance evidence: raw/expanded data and review records; token reports and test output; model SHA, dependencies, driver, and training config; checkpoints, adapter, and resume steps; A/B/C raw outputs and scores; merge manifest, inference environment versions, API responses.

### 8.2 Code Reading Map and Rationale

For the main path, read the Notebook first; look up the supporting modules by responsibility:

| Where in the tutorial | Corresponding code | Purpose |
|---|---|---|
| Section 2.8, chapter 3 | [download.py](../lab/download.py), [download-lock.json](../lab/download-lock.json) | Download a pinned revision and check integrity |
| Chapter 2 | [dataset.json](../data/dataset.json), [data.py](../lab/data.py), [prepare.py](../lab/prepare.py), [common.py](../lab/common.py) | Validate facts and conversations, build prompts, generate tokenized samples |
| Chapters 2 and 6 | [audit.py](../lab/audit.py) | Check the real collator's supervision span and padding |
| Chapter 4 | [01_gpu_train.en.ipynb](../notebooks/01_gpu_train.en.ipynb) | 1-epoch training, loss curve, artifact checks, reloaded answers, and merge |
| Section 4.6, chapter 6, appendix | [train.py](../lab/train.py), [training.py](../lab/training.py) | Command-line training, formal parameter comparison, and resume from checkpoint |
| Chapter 6 | [infer.py](../lab/infer.py), [evaluate.py](../lab/evaluate.py) | A/B/C inference, blind-review statistics, and candidate freezing |
| Chapter 5 | [merge.py](../lab/merge.py), the last two sections of the Notebook | Standalone export, vLLM deployment and adapter mounting, API calls, load testing |

The Notebook shows the training details; the CLI is kept for repeated experiments and resuming, and each entry point has its own chapters. The API examples call HTTP or the OpenAI SDK directly, and load testing uses vLLM's built-in `vllm bench serve`, so no separate client is maintained.

| Decision | Reason and source |
|---|---|
| Fixed text-only 4B | Compare experiments on the same base model; [Qwen model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) |
| Explicit completion mask | The supervision span can be inspected; [TRL 0.24](https://huggingface.co/docs/trl/v0.24.0/en/sft_trainer) |
| NF4+LoRA | Train a small number of parameters on a single GPU; [PEFT quantization](https://huggingface.co/docs/peft/v0.18.0/en/developer_guides/quantization) |
| Separate inference environment | Avoid dependencies overwriting each other; [vLLM GPU installation](https://docs.vllm.ai/en/v0.21.0/getting_started/installation/gpu/) |
| Train facts and expression separately | This project's experimental design; the gain is validated by comparison |

Fact cards record links to scientific sources. When reviewing, check each claim, its conditions of applicability, and the limits of each analogy one by one, paying special attention to sources still marked pending; a web page being reachable does not mean it supports every generated conversation.


## Appendix A: Full Core Training Code

The code below matches the source files. When making changes, the files under `lab/` are the source of truth. The code comments are kept exactly as they appear in the source files (some are in Chinese).

### lab/training.py

```python
"""The TRL path is shared by the CUDA lesson and the tiny CPU regression test."""
from trl import SFTConfig, SFTTrainer
from transformers import TrainerCallback


class StopAfter(TrainerCallback):
    """Deliberately interrupt a planned run without changing its LR schedule."""
    def __init__(self, step):
        self.step = step

    def on_step_end(self, args, state, control, **kwargs):
        if self.step and state.global_step >= self.step:
            control.should_save = True
            control.should_training_stop = True
        return control


def make_trainer(model, tokenizer, train, val, output, *, cpu=False, epochs=1,
                 max_steps=-1, accum=16, lr=1e-4, max_length=4096, seed=42, stop_after=0):
    args = SFTConfig(
        output_dir=output, num_train_epochs=epochs, max_steps=max_steps,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=accum, learning_rate=lr,
        warmup_ratio=0.05, lr_scheduler_type="cosine", optim="adamw_torch",
        max_grad_norm=1.0, bf16=not cpu, fp16=False, use_cpu=cpu,
        gradient_checkpointing=not cpu,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=max_length, packing=False, completion_only_loss=True,
        assistant_only_loss=False, eos_token=tokenizer.eos_token,
        eval_strategy="epoch" if val is not None else "no",
        save_strategy="steps", save_steps=5, save_total_limit=2,
        logging_steps=1, report_to="none", prediction_loss_only=True,
        seed=seed, data_seed=seed, dataloader_num_workers=0,
        dataloader_pin_memory=not cpu, push_to_hub=False)
    return SFTTrainer(model=model, args=args, processing_class=tokenizer,
                      train_dataset=train, eval_dataset=val,
                      callbacks=[StopAfter(stop_after)])


def check_supervision(trainer, rows, eos_id, count=10):
    # D6：上下文和 padding 的 labels 应为 -100，当前参考回答及 EOS 才参与损失。
    for i, row in enumerate(rows[:count]):
        item = trainer.train_dataset[i]
        batch = trainer.data_collator([item])
        actual = batch["labels"][0].tolist()
        expected = [token if mask else -100 for token, mask in
                    zip(row["input_ids"], row["completion_mask"])]
        if actual != expected or -100 not in actual or not any(x != -100 for x in actual):
            raise ValueError(f"supervision mismatch at row {i}")
        supervised = [x for x in actual if x != -100]
        if supervised[-1] != eos_id or supervised.count(eos_id) != 1:
            raise ValueError("supervision must include one final EOS")
```

### lab/train.py

```python
"""Single NVIDIA GPU QLoRA lesson. Run `python -m lab.train --help` first."""
import argparse
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

from lab.common import sha, tokenizer_signature, verify_bundle, write_json
from lab.data import read_sft


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/qwen3-4b")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--accum", type=int, default=16)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-draft", action="store_true")
    p.add_argument("--resume")
    p.add_argument("--stop-after", type=int, default=0)
    a = p.parse_args()
    if min(a.rank, a.accum, a.max_length) <= 0 or a.lr <= 0 or a.epochs <= 0:
        raise ValueError("positive training settings required")
    if a.max_steps == 0 or a.max_steps < -1 or a.limit < 0 or a.stop_after < 0:
        raise ValueError("invalid step/limit settings")
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from lab.training import make_trainer, check_supervision
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("main lesson requires a BF16-capable NVIDIA CUDA GPU")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("this lesson uses one process and one GPU")
    folder, out = Path(a.data), Path(a.output)
    bundle = verify_bundle(folder)
    if bundle["draft"] and not a.allow_draft:
        raise ValueError("draft data requires --allow-draft; results are exploratory, not formal claims")
    if not bundle["formal"] and a.max_steps <= 0:
        raise ValueError("epoch experiments require the formal 600-episode science bundle")
    if out.exists() and any(out.iterdir()) and not a.resume:
        raise ValueError("output nonempty: choose new directory or resume")
    lock = json.loads((Path(a.model)/"model-lock.json").read_text())
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True, trust_remote_code=False)
    if tokenizer_signature(tok) != bundle["tokenizer_sha256"]:
        raise ValueError("tokenizer differs from prepared data")
    tok.padding_side = "right"
    # CLI 读取 lab.prepare 从 dataset.json 生成的快照；Notebook 直接在内存中生成同样的 rows。
    rows = read_sft(folder, "train", bundle)
    if a.limit:
        rows = rows[:a.limit]
    val_rows = read_sft(folder, "val", bundle)
    if not rows:
        raise ValueError("training set is empty")
    for row in [*rows, *val_rows]:
        if len(row["input_ids"]) > a.max_length:
            raise ValueError("sample length exceeds training limit; rewrite and prepare again")
    packages = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "trl", "peft", "bitsandbytes", "accelerate", "datasets")}
    config = {k: v for k,v in vars(a).items() if k not in {"resume", "stop_after", "output", "model", "data"}}
    contract = {"config": config, "base_lock": lock, "data_manifest_sha256": sha(folder/"manifest.json"),
                "code": {f: sha(Path(__file__).with_name(f)) for f in ("train.py", "training.py", "common.py")},
                "packages": packages}
    resume_step = 0
    if a.resume:
        previous = json.loads((out/"manifest.json").read_text())
        if previous["contract"] != contract:
            raise ValueError("resume configuration/data/code/environment differs")
        checkpoint = Path(a.resume).resolve()
        if checkpoint.parent != out.resolve():
            raise ValueError("resume checkpoint must belong to this output directory")
        for f in ("trainer_state.json", "optimizer.pt", "scheduler.pt"):
            if not (checkpoint/f).is_file():
                raise ValueError(f"not a resumable training checkpoint: missing {f}")
        resume_step = json.loads((checkpoint/"trainer_state.json").read_text())["global_step"]
    set_seed(a.seed)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=a.rank, lora_alpha=2*a.rank, lora_dropout=0.05,
        bias="none", target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    trainable = {n: p for n,p in model.named_parameters() if p.requires_grad}
    if not trainable or any("lora_" not in n for n in trainable):
        raise ValueError("unexpected trainable parameters")
    initial = {n: p.detach().cpu().clone() for n,p in trainable.items()}
    def dataset(items):
        # D5：只将 Token 与监督掩码交给训练器，case_id/prompt 等保留用于人工核对。
        return Dataset.from_list([{k: row[k] for k in ("input_ids", "completion_mask")} for row in items])
    trainer = make_trainer(model, tok, dataset(rows), dataset(val_rows) if val_rows else None, str(out),
        epochs=a.epochs, max_steps=a.max_steps, accum=a.accum, lr=a.lr,
        max_length=a.max_length, seed=a.seed, stop_after=a.stop_after)
    check_supervision(trainer, rows, tok.eos_token_id)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"contract": contract, "draft": bundle["draft"], "gpu": torch.cuda.get_device_name(0),
                "cuda": torch.version.cuda, "supervised_rows": len(rows),
                "trainable_parameters": sum(p.numel() for p in trainable.values())}
    write_json(out/"manifest.json", manifest)
    (out/"dependencies.txt").write_text("\n".join(sorted(
        f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())) + "\n")
    driver = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
    (out/"nvidia-smi.txt").write_text(driver.stdout + driver.stderr)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    result = trainer.train(resume_from_checkpoint=a.resume)
    if trainer.state.global_step <= resume_step:
        raise RuntimeError("no new optimizer step; inspect resume/max-steps")
    if not math.isfinite(result.training_loss) or any(not torch.isfinite(p).all() for p in trainable.values()):
        raise RuntimeError("non-finite loss or adapter parameters; inspect data and precision")
    changed = [n for n,p in trainable.items() if not torch.equal(initial[n], p.detach().cpu())]
    if not changed:
        raise RuntimeError("no LoRA parameter changed")
    trainer.save_model(str(out/"adapter"))
    tok.save_pretrained(out/"adapter")
    shutil.copyfile(out/"manifest.json", out/"adapter"/"manifest.json")
    trainer.save_state()
    metrics = dict(result.metrics, elapsed_seconds=time.monotonic()-started,
                   peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                   peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                   changed_lora_tensors=len(changed), resumed_from_step=resume_step,
                   final_step=trainer.state.global_step)
    if val_rows:
        metrics.update(trainer.evaluate())
    write_json(out/f"metrics-step-{trainer.state.global_step}.json", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
```
