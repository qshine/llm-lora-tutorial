<div align="center">

# LoRA 파인튜닝과 vLLM 배포 실전 가이드

**초등학교 과학 탐구 친구 · RTX 4090 한 장으로 처음부터 해 보는 QLoRA 파인튜닝, 병합, vLLM 배포**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3110/)
[![Model: Qwen3-4B](https://img.shields.io/badge/Model-Qwen3--4B--Instruct--2507-8C432E)](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
[![GPU: RTX 4090](https://img.shields.io/badge/GPU-RTX%204090%20·%2024%20GB-76B900?logo=nvidia&logoColor=white)](tasks/plan.ko.md)
[![전자책](https://img.shields.io/badge/전자책-온라인으로%20읽기-A7492D)](https://qshine.github.io/llm-lora-tutorial/ko/)

[简体中文](README.md) · [English](README.en.md) · [日本語](README.ja.md) · **한국어**

**[📖 전자책 온라인으로 읽기](https://qshine.github.io/llm-lora-tutorial/ko/)** (페이지 오른쪽 위에서 中文 / English / 日本語 / 한국어 전환)

</div>

---

## 개요

**ML 업무를 하지 않지만 LoRA 파인튜닝을 처음부터 배우고 싶은 개발자**를 위해 썼습니다. 백엔드, 프런트엔드, 클라이언트, 데이터 엔지니어 누구나 환영하며 머신러닝 배경지식은 필요 없습니다. RTX 4090 한 장으로 다음 과정을 직접 따라 합니다.

```text
QLoRA fine-tuning → save and reload → BF16 merge → vLLM as an OpenAI-compatible API → local calls and load testing
```

그리고 각 단계에서 GPU 메모리가 어디에 쓰이는지, 각 파일이 어디에 생기는지, 그 단계가 정말 성공했는지 어떻게 확인하는지를 배웁니다. 연습 과제는 초등학생의 과학 질문에 중국어로 답하는 것입니다. 대화 스타일이 뚜렷해서 파인튜닝 전후의 변화가 쉽게 보입니다. 이 과제 때문에 한국어판에서도 데이터셋은 중국어 그대로 둡니다.

## 특징

| | |
|---|---|
| 📖 **완전한 전자책** | 본문 9장 + 부록 5개. 개념, 데이터, GPU 대여부터 배포, 평가, 문제 해결까지 다루며, 中文, English, 日本語, 한국어를 클릭 한 번으로 전환할 수 있습니다 |
| 🧪 **바로 실행할 수 있는 노트북** | `LoraConfig`, `SFTTrainer`, `trainer.train()`을 그대로 보여 주며, 모든 셀 뒤에 무엇을 하는지, 왜 하는지, 무엇을 보게 되는지 적혀 있습니다 |
| 🗂️ **데이터 파일 하나** | `data/dataset.json`에 사실 카드 60장과 학습·검증·테스트 대화 500 / 50 / 50개가 주제별로 격리되어 들어 있습니다 |
| 🚀 **학습부터 배포까지** | 병합과 내보내기, vLLM 서빙, 어댑터 마운트, OpenAI SDK 스트리밍 호출, `vllm bench serve` 부하 테스트 |
| ✅ **테스트로 보호** | 튜토리얼의 명령어, 부록 코드, 배포 예제를 실제 코드와 대조해 검사합니다 |

## 빠른 시작

### 1. 전자책 읽기

온라인으로 읽기: <https://qshine.github.io/llm-lora-tutorial/ko/> (모든 페이지 오른쪽 위에서 中文 / English / 日本語 / 한국어를 전환할 수 있습니다). 로컬에서 미리 볼 수도 있습니다.

```bash
./book_review.sh
```

스크립트가 읽기 환경을 준비하고 <http://127.0.0.1:8000>을 엽니다. 처음 실행할 때는 Python 3.11과 네트워크 연결이 필요합니다. 포트가 사용 중이면 `./book_review.sh 8001`을 쓰고, `Ctrl+C`로 중지합니다. 읽기만 할 때는 GPU나 모델 가중치가 필요 없습니다.

### 2. 데이터 확인

데이터 형식만 확인합니다(PyTorch 불필요).

```bash
python3.11 -m lab.data --input data/dataset.json --allow-draft
```

### 3. GPU에서 셀 단위로 학습

`python3 tools/build_gpu_zip.py`를 실행해 `llm-lora-tutorial-gpu.zip`을 만들고, GPU 호스트의 데이터 디스크에 업로드해 압축을 풉니다. 튜토리얼 3장을 따라 환경을 준비하고 ModelScope에서 전체 모델을 내려받아 검증한 다음, [01_gpu_train.en.ipynb](notebooks/01_gpu_train.en.ipynb)를 열고 **LoRA · Python 3.11** 커널을 선택합니다. 이 노트북은 [01_gpu_train.ipynb](notebooks/01_gpu_train.ipynb)의 영어판으로, 중국어판과 완전히 같은 코드를 실행하며 설명과 주석만 영어입니다. 한국어판 노트북은 없습니다.

노트북은 RTX 4090 / 24 GB를 기준으로 합니다: 베이스 모델의 답변 → LoRA 설정 → 1 에폭 학습(약 72스텝) → loss 곡선 → 저장과 다시 불러오기 → 사람의 검토 → BF16 병합, 마지막으로 vLLM 배포와 로컬 호출 예제가 이어집니다. 기본 산출물은 `runs/science-001/`에 저장됩니다.

| 산출물 | 의미 |
|---|---|
| `training-complete.json` | 학습 완료 기록 |
| `adapter/` | LoRA 델타 가중치 |
| `merged-model/` | vLLM에서 사용할 전체 모델 |
| `answer-review.json` | 학습 전후 답변 비교, 사람의 검토 대기 |

## 데이터

관리하는 파일은 [data/dataset.json](data/dataset.json) 하나뿐입니다. `facts`에는 참고용 사실 카드 60장이, 나란히 있는 `train`, `val`, `test` 배열에는 각각 학습 대화 500개, 검증 대화 50개, 테스트 대화 50개가 들어 있습니다. 사실 카드는 `fact_ids`를 통해 각 질문의 시스템 프롬프트에 답변의 근거로 삽입되며, 학습 대상이 아닙니다. 학습 세트는 턴 단위로 펼쳐져 1150개의 답변 타깃이 됩니다. 출처, 분류, 검토 기록도 모두 같은 파일에 있습니다.

[데이터 가이드](data/README.ko.md)는 낮과 밤에 관한 완전한 예시를 따라가며 각 필드의 의미, 어떤 필드가 모델에 들어가고 어떤 필드가 검토용 메타데이터일 뿐인지를 설명합니다. 노트북의 D1~D7 주석은 로딩, 필터링, 컨텍스트 구성, 토큰화, 트레이너 진입점, loss 범위, 저장 위치를 표시합니다.

노트북은 이 JSON을 직접 읽어 메모리에서 학습 샘플을 만들므로, `lab.prepare`를 먼저 실행할 필요가 없습니다. 학습할 때 실제 입력의 사본이 자동으로 `runs/science-001/dataset.json`에 남습니다. 새로 동결한 스냅샷이 필요한 것은 명령줄 실험뿐입니다(튜토리얼 4.6절 참고).

> [!NOTE]
> 모든 대화는 합성된 초안이며 초등학교 1~6학년을 1~2, 3~4, 5~6의 학년대로 나누어 다룹니다. 학습 주제 50개에 각각 경로 10개가 있고, val/test는 주제 단위로 격리되어 있습니다. 사실, 출처, 가독성은 여전히 항목별 검토가 필요합니다. 파인튜닝이 완전한 프롬프트를 쓰는 베이스라인보다 낫다는 보장은 없습니다. 파인튜닝 과제 자체가 아이들의 과학 질문에 중국어로 답하는 것이므로 데이터셋은 중국어 그대로 둡니다.

## 프로젝트 구조

| 경로 | 역할 |
|---|---|
| `tasks/plan.md` (`plan.en.md`, …) | 실전 튜토리얼(전자책 본문) |
| `notebooks/01_gpu_train.ipynb` (`.en.ipynb`) | GPU 학습의 메인 진입점, 튜토리얼 4~5장에 해당 |
| `data/dataset.json` | 유일한 데이터 소스 |
| `lab/data.py`, `prepare.py`, `common.py` | 원본 데이터 검증, 질의응답 펼치기, 프롬프트와 학습 샘플 구성 |
| `lab/audit.py` | 실제 TRL collator로 감독 타깃 확인 |
| `lab/download.py`, `train.py`, `training.py` | 모델 리비전 고정, 학습, 재개 |
| `lab/infer.py`, `evaluate.py` | 비교 답변 생성, 블라인드 평가, 결과 통계 |
| `lab/merge.py` | 독립 실행형 병합과 내보내기 |
| `book/`, `mkdocs*.yml` | 다국어 전자책 빌드 |
| `tests/` | 데이터, 학습, 노트북, 튜토리얼, 전자책 테스트 |

`models/`, `runs/`, `data/processed/`에는 로컬 산출물이 저장되며 버전 관리 대상이 아닙니다.

## 더 많은 문서

- [실전 튜토리얼](tasks/plan.ko.md): 예제, 명령어, 필드의 의미, 결과를 읽는 방법.
- [학습 체크리스트](tasks/todo.ko.md): 코드 검사와 아직 해야 할 학습 실험을 구분합니다.
- [검증 기록](tasks/verification.ko.md): 실제로 실행한 검사와 그 결과.
- [전자책 유지 관리](book/README.md)(중국어): 빌드, 링크 검사, 배포 방법.

## 저자 소개

**qqling**: 대기업과 스타트업을 모두 거친 시니어 개발자이자, 대규모 언어 모델(LLM)과 에이전트를 실제 서비스에 적용하는 데 집중하는 베테랑 AI 애호가입니다. 이 가이드는 LoRA 파인튜닝과 배포를 처음부터 끝까지 직접 해 본 전체 과정을 기록한 것이며, 책에 나오는 학습 곡선, 오류, 문제 해결 과정은 모두 실제로 실행한 결과입니다.

- X: [@qqinling](https://x.com/qqinling)
- 위챗 공식 계정: qqling的AI小院
- 샤오훙슈: AI_qqling

<table>
  <tr>
    <td align="center"><img src="images/author-wechat-qrcode.jpg" alt="위챗 공식 계정 “qqling的AI小院” QR 코드" width="150"><br><sub>위챗 공식 계정: qqling的AI小院</sub></td>
    <td align="center"><img src="images/author-xiaohongshu-qrcode.jpg" alt="샤오훙슈 계정 “AI_qqling” QR 코드" width="150"><br><sub>샤오훙슈: AI_qqling</sub></td>
  </tr>
</table>

💻 **GPU 대여:** 이 책의 모든 작업은 중국 본토용 GPU 클라우드 [Zhichuan Cloud](https://gpuhome.cc/auth/register?code=zt4f01m0)에서 빌린 GPU로 실행했습니다(저자의 초대 링크, 초대 코드 `zt4f01m0`. 요금은 [전자책 첫 페이지](https://qshine.github.io/llm-lora-tutorial/ko/) 참고).

## 라이선스

이 프로젝트는 [MIT License](LICENSE)로 배포됩니다. 베이스 모델 [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)은 이 저장소와 함께 배포되지 않으므로, 해당 모델의 라이선스에 따라 사용하세요.
