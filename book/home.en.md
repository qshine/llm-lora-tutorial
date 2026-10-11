# Hands-On Guide to LoRA Fine-Tuning and vLLM Deployment

## About the author

<div class="author-card" markdown>

**qqling**: a senior developer who has worked at both big tech companies and startups; a seasoned AI enthusiast focused on putting LLMs and agents into production. This guide records the complete process of getting LoRA fine-tuning and deployment working from scratch; the training curves, errors, and troubleshooting in the book all come from real runs.

- X: [@qqinling](https://x.com/qqinling)
- WeChat Official Account: qqling的AI小院
- Xiaohongshu (RED): AI_qqling

<div class="author-qr">
<figure>
<img src="assets/author-wechat-qrcode.jpg" alt="QR code for the WeChat Official Account “qqling的AI小院”" width="170">
<figcaption>WeChat Official Account: qqling的AI小院</figcaption>
</figure>
<figure>
<img src="assets/author-xiaohongshu-qrcode.jpg" alt="QR code for the Xiaohongshu (RED) account “AI_qqling”" width="170">
<figcaption>Xiaohongshu (RED): AI_qqling</figcaption>
</figure>
</div>

</div>

## Renting a GPU: Zhichuan Cloud, which the author uses

All the training, merging, vLLM deployment, and load testing in this book were actually run on GPUs rented from [Zhichuan Cloud](https://gpuhome.cc/auth/register?code=zt4f01m0). Overall it is easy to use, and its biggest advantage is that it is **cheap**.

Sign-up link: **[https://gpuhome.cc/auth/register?code=zt4f01m0](https://gpuhome.cc/auth/register?code=zt4f01m0)**

Enter the invite code **`zt4f01m0`** when signing up to receive a new-user voucher. Note: this is the author's referral link, and after your first top-up the author also receives a voucher from the platform; vouchers are currently issued manually by the platform, so once you meet the conditions, contact platform customer support to claim yours. They are valid for three months.

Prices:

| RTX 4090D · 24 GB VRAM | Price (as shown on the page on 2026-10-11; the price at order time applies) |
|---|---|
| **Pay-as-you-go** | **¥1.25 / hour** |
| Daily / weekly / monthly plan | ¥29 / ¥200 / ¥800 |
| Included | 22-core vCPU, 100 GB RAM, 30 GB system disk + 50 GB data disk |

![Zhichuan Cloud rental page: RTX 4090D 24 GB, pay-as-you-go ¥1.25/hour, daily ¥29, weekly ¥200, monthly ¥800](assets/gpuhome-rtx4090d-price-2026-10-11.png)

At this price, running through the book's main path (about 2–3 hours of uptime) **costs only three or four yuan**. New users who top up at least ¥1 for the first time get a ¥5 voucher, which is roughly enough to run the whole main path once.

- **Wide range of models**: RTX 3090, 4090, 4090D, 5090, A100, L40S and more, with 1–8 GPUs per instance and many regions, so it is easy to switch to another data center when stock runs out.
- **Ready out of the box**: there is the PyTorch base image used in this book (`2.7.1-CUDA12.6-Py3.11-Ubuntu22`), and the instance page opens JupyterLab, SSH, and VSCode with one click.
- **Flexible billing**: pay-as-you-go, daily, weekly, monthly, and overnight plans are all available; when you shut the instance down, GPU charges stop.

## What this book walks you through

It is written for **developers who do not work on algorithms and want to learn LoRA fine-tuning systematically from scratch**: backend, frontend, client, data, or test developers are all welcome. No machine learning background is needed; you just need to be comfortable with a terminal and able to read Python. On a single RTX 4090 you will go through, hands-on: download an open-source model → QLoRA fine-tuning → save and reload → merge and export → deploy with vLLM as an OpenAI-compatible API → call it and load test it from your own computer. Each step explains where the VRAM goes, where the files are, and how to confirm the step really succeeded.

## Why practice with "primary-school science Q&A"

A child asks “月亮怎么变弯了” (“Why does the moon look curved?”), then after the answer says “我没听懂” (“I didn't understand”), or guesses “是不是被地球挡住了” (“Is it blocked by the Earth?”). Conversations like these have a distinctive style, the changes before and after fine-tuning are easy to see, and it is easy for you to judge whether the model is making things up.

`data/dataset.json` holds 60 fact cards and 600 synthetic conversations: **500 for training, 50 for validation, 50 for testing**, all of them drafts awaiting review — enough for learning the workflow.

## Reading order

Goals and preparation → essential concepts → training data → GPU preparation → Notebook training (1 epoch by default, about 72 steps) → deployment and calls → evaluation. Reading in table-of-contents order is the same as the order of operations; each chapter opens by stating what it does, what you need to prepare, and what you get when you finish. Cost, troubleshooting, and developer reference come at the end, for when you run into problems.

It is fine if you have no GPU or machine learning background: chapter 0 first explains the big picture and what you need to prepare, and chapter 1 explains every concept used with analogies developers already know (databases, Git, web services).

When training is done, you will have a training-complete record, a loss curve, an adapter, a deployable full model, a running vLLM service, and latency and throughput numbers you measured yourself.
