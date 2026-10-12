# 宣传短片

约 64 秒的项目宣传片，两个画幅：

| 文件 | 用途 |
|---|---|
| `lora-promo-16x9.mp4` | 横屏 1920×1080：B 站、YouTube、X、公众号 |
| `lora-promo-9x16.mp4` | 竖屏 1080×1920：小红书、视频号、抖音 |
| `cover-16x9.jpg` / `cover-9x16.jpg` | 对应封面 |

H.264 + AAC，30 fps，各约 11 MB。片中文字已覆盖全部信息，静音也能看懂。

## 片中内容与出处

| 时间 | 内容 | 出处 |
|---|---|---|
| 0–7 s | 开场：写给会写代码、没训练过模型的开发者 | README 读者定位 |
| 7–13 s | 书名与“在一张 RTX 4090 上，从零走通一次” | README |
| 13–23 s | 六步全链路：下载 → QLoRA → 保存重载 → BF16 合并 → vLLM → 调用压测 | 教程第 3–5 章 |
| 23–30 s | 训练数据样例（昼夜，1–2 年级）与 60 / 500·50·50 / 1150 | `data/dataset.json` |
| 30–40 s | 2026-10-11 RTX 4090 实测：72 步、26 分 19 秒、loss 曲线、过拟合教训 | `tasks/verification.md`、`images/loss-curve-rtx4090-2026-10-11.png` |
| 40–47 s | ¥1.25/小时，主线约三四块钱（按开机 2～3 小时估算） | 电子书首页价格 |
| 47–53 s | 不夸大：微调不一定胜过提示词、数据是草稿、命令由测试核对 | README、教程第 6 章 |
| 53–64 s | 片尾：网址、二维码（指向电子书）、GitHub、作者 | — |

loss 曲线逐步数值是从实测截图读出的近似值；首末值和验证 loss 取自验证记录原值。

## 重新生成

画面是 `index.html` 里由时间驱动的动画（`render(t)`），Playwright 逐帧截图；配乐由 `music.py` 用 numpy 合成（72 BPM，每个镜头切换都落在小节线上），不含第三方音频素材。

```bash
python3 make_qr.py                 # 片尾二维码
python3 getfonts.py                # 从 Google Fonts 取 Noto Serif SC / Noto Sans SC / JetBrains Mono 子集到 fonts/
python3 music.py music.wav         # 合成配乐（需 numpy）
node stills.js 1920 1080 11.5 35   # 可选：先看几张静帧
bash render_all.sh                 # 逐帧渲染横、竖两版（需 Node 版 playwright 与 ffmpeg）
bash mux.sh                        # 拼接、加细颗粒、混入配乐
```

改了片中文字后要重新运行 `getfonts.py`，因为字体子集只包含当前用到的字。
