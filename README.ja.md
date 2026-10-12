<div align="center">

# LoRA ファインチューニングと vLLM デプロイ実践ガイド

**小学校の理科探究パートナー · 1 枚の RTX 4090 で QLoRA ファインチューニング、マージ、vLLM デプロイをゼロから通しで体験**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3110/)
[![Model: Qwen3-4B](https://img.shields.io/badge/Model-Qwen3--4B--Instruct--2507-8C432E)](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
[![GPU: RTX 4090](https://img.shields.io/badge/GPU-RTX%204090%20·%2024%20GB-76B900?logo=nvidia&logoColor=white)](tasks/plan.ja.md)
[![電子書籍](https://img.shields.io/badge/電子書籍-オンラインで読む-A7492D)](https://qshine.github.io/llm-lora-tutorial/ja/)

[简体中文](README.md) · [English](README.en.md) · **日本語** · [한국어](README.ko.md)

**[📖 電子書籍をオンラインで読む](https://qshine.github.io/llm-lora-tutorial/ja/)**（ページ右上で 中文 / English / 日本語 / 한국어 を切り替えられます）

</div>

---

## 概要

本書は、**機械学習を専門としておらず、LoRA ファインチューニングをゼロから学びたい開発者**のために書かれています。バックエンド、フロントエンド、クライアント、データエンジニアのいずれでもかまわず、機械学習の知識は必要ありません。なお、学習データセットは中国語のままです（ファインチューニングのタスクが中国語で子どもの理科の質問に答えることだからです）。1 枚の RTX 4090 上で、次の流れを自分の手で進めます。

```text
QLoRA fine-tuning → save and reload → BF16 merge → vLLM as an OpenAI-compatible API → local calls and load testing
```

そして、各ステップで GPU メモリが何に使われるのか、各ファイルがどこに置かれるのか、そのステップが本当に成功したことをどう確認するのかを学びます。練習タスクは、小学生の理科の質問に中国語で答えることです。会話のスタイルに特徴があるので、ファインチューニング前後の変化が見えやすくなっています。

## 特徴

| | |
|---|---|
| 📖 **完全な電子書籍** | 本文 9 章 + 付録 5 つ。概念、データ、GPU のレンタルから、デプロイ、評価、トラブルシューティングまでを扱います。中文、English、日本語、한국어 をワンクリックで切り替えられます |
| 🧪 **そのまま動くノートブック** | `LoraConfig`、`SFTTrainer`、`trainer.train()` を直接見せます。各セルの下に、何をするのか、なぜそうするのか、何が表示されるのかを書いています |
| 🗂️ **データは 1 ファイル** | `data/dataset.json` に 60 枚のファクトカードと、学習用・検証用・テスト用の会話 500 / 50 / 50 件が入っており、トピックごとに分離されています |
| 🚀 **学習からデプロイまで** | マージとエクスポート、vLLM でのサービング、adapter のマウント、OpenAI SDK でのストリーミング呼び出し、`vllm bench serve` での負荷テスト |
| ✅ **テストで保護** | チュートリアル内のコマンド、付録のコード、デプロイ例は、実際のコードと突き合わせてテストで検証しています |

## クイックスタート

### 1. 電子書籍を読む

オンラインで読む：<https://qshine.github.io/llm-lora-tutorial/ja/>（各ページの右上で 中文 / English / 日本語 / 한국어 を切り替えられます）。ローカルでプレビューすることもできます。

```bash
./book_review.sh
```

このスクリプトは閲覧用の環境を準備し、<http://127.0.0.1:8000> を開きます。初回の実行には Python 3.11 とネットワーク接続が必要です。ポートが使用中の場合は `./book_review.sh 8001` を使い、終了するには `Ctrl+C` を押します。読むだけなら GPU もモデルの重みも不要です。

### 2. データをチェックする

データの形式だけをチェックします（PyTorch は不要です）。

```bash
python3.11 -m lab.data --input data/dataset.json --allow-draft
```

### 3. GPU 上でセルごとに学習する

`python3 tools/build_gpu_zip.py` を実行して `llm-lora-tutorial-gpu.zip` を作成し、GPU ホストのデータディスクにアップロードして展開します。チュートリアルの第 3 章に従って環境を構築し、ModelScope から完全なモデルをダウンロードして検証したら、[01_gpu_train.en.ipynb](notebooks/01_gpu_train.en.ipynb) を開き、**LoRA · Python 3.11** カーネルを選択してください。これはノートブックの英語版で、中国語版の [01_gpu_train.ipynb](notebooks/01_gpu_train.ipynb) と同じコードを実行します。日本語版のノートブックはありません。

このノートブックは RTX 4090 / 24 GB を想定しています：ベースモデルの回答 → LoRA の設定 → 1 エポックの学習（約 72 ステップ）→ loss 曲線 → 保存と再読み込み → 人によるレビュー → BF16 マージと進み、最後に vLLM のデプロイとローカルからの呼び出し例で終わります。デフォルトの出力先は `runs/science-001/` です。

| 出力 | 意味 |
|---|---|
| `training-complete.json` | 学習が完了したことを示します |
| `adapter/` | LoRA の差分重み |
| `merged-model/` | vLLM 用の完全なモデル |
| `answer-review.json` | 学習前後の回答。人によるレビュー待ちです |

## データ

管理するのは [data/dataset.json](data/dataset.json) だけです。`facts` には 60 枚の参考用ファクトカードが入り、並列する `train`、`val`、`test` 配列には学習用 500 件、検証用 50 件、テスト用 50 件の会話が入っています。ファクトカードは `fact_ids` を通じて各質問のシステムプロンプトに挿入され、回答の根拠になります。学習ターゲットではありません。学習用の split はターンごとに展開され、1150 個の回答ターゲットになります。出典、カテゴリ、レビュー記録もすべて同じファイルに入っています。

[データガイド](data/README.ja.md)では、昼と夜についての完全な例を通して、各フィールドの意味、どのフィールドがモデルに届き、どのフィールドがレビュー用のメタデータにすぎないのかを説明しています。ノートブック内のコメント D1～D7 は、読み込み、フィルタリング、コンテキストの組み立て、トークン化、trainer の入口、loss の範囲、保存場所を示しています。

ノートブックはこの JSON を直接読み込み、メモリ上で学習サンプルを組み立てるので、先に `lab.prepare` を実行する必要はありません。学習時には実際の入力のコピーが自動的に `runs/science-001/dataset.json` に保存されます。新しく凍結したスナップショットが必要なのはコマンドラインでの実験だけです（チュートリアルの 4.6 節を参照）。

> [!NOTE]
> 会話はすべて合成されたドラフトで、1～6 年生を対象とし、1～2、3～4、5～6 の学年帯に分けられています。学習用は 50 トピックで各 10 パス、val/test はトピック単位で分離されています。事実、出典、読みやすさはまだ 1 件ずつレビューする必要があります。ファインチューニングが、完全なプロンプトを使うベースラインを上回る保証はありません。データセットは中国語のままです。ファインチューニングのタスクそのものが、子どもの理科の質問に中国語で答えることだからです。

## プロジェクト構成

| パス | 役割 |
|---|---|
| `tasks/plan.md` (`plan.en.md`, …) | 実践チュートリアル（電子書籍の本文） |
| `notebooks/01_gpu_train.ipynb` (`.en.ipynb`) | GPU 学習のメインの入口。チュートリアルの第 4～5 章に対応します |
| `data/dataset.json` | 唯一のデータソース |
| `lab/data.py`, `prepare.py`, `common.py` | 元データの検証、Q&A の展開、プロンプトと学習サンプルの組み立て |
| `lab/audit.py` | 実際の TRL collator で教師信号のターゲットを確認します |
| `lab/download.py`, `train.py`, `training.py` | モデルのリビジョンの固定、学習、再開 |
| `lab/infer.py`, `evaluate.py` | 比較用の回答の生成、ブラインド評価、結果の集計 |
| `lab/merge.py` | 単独でのマージとエクスポート |
| `book/`, `mkdocs*.yml` | 多言語電子書籍のビルド |
| `tests/` | データ、学習、ノートブック、チュートリアル、電子書籍のテスト |

`models/`、`runs/`、`data/processed/` にはローカルの成果物が置かれ、バージョン管理の対象外です。

## その他のドキュメント

- [実践チュートリアル](tasks/plan.ja.md)：例、コマンド、フィールドの意味、結果の読み方。
- [学習チェックリスト](tasks/todo.ja.md)：コードのチェックと、まだ行っていない学習実験を区別しています。
- [検証ログ](tasks/verification.ja.md)：実際に実行したチェックとその結果。
- [電子書籍のメンテナンス](book/README.md)（中国語）：ビルド、リンクチェック、公開。

## ライセンス

このプロジェクトは [MIT License](LICENSE) のもとで公開されています。ベースモデル [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) はこのリポジトリには含まれていません。そのモデル自身のライセンスに従って使用してください。
