# Single-File Science Conversation Data

## 1 Maintain only one dataset.json

Open [data/dataset.json](dataset.json). All the reference material, the 500 training conversations, and the evaluation conversations live in this one file. At the top level there are four sibling arrays: `facts` holds the reference fact cards, and `train`, `val`, and `test` are three groups of complete conversations. Only `train` actually takes part in training; fact cards serve only as context, not as learning targets.

| Set | Science topics | Complete conversations | Answer targets | Purpose |
|---|---:|---:|---:|---|
| train | 50 | 500 | 1150 | Update the LoRA parameters |
| val | 5 | 50 | 115 | Compare configurations, pick a candidate |
| test | 5 | 50 | 115 | Evaluate after the candidate is frozen |

Grade-band distribution in training: 167 conversations for grades 1–2, 167 for grades 3–4, and 166 for grades 5–6. Each topic has 10 conversation paths, each with 2–3 turns; there are explanations of the same knowledge at different grade bands, as well as follow-up changes caused by different feedback.

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

(Gloss of the tree above: `schema_version / description` is the file version and description; `facts[60]` are the reference fact cards, stored once and referenced by id from conversations; `train[500]` are the training conversations, the only part that updates parameters; `val[50]` are the validation conversations for comparing configurations, picking a candidate, and computing eval loss; `test[50]` are the test conversations, used only after the candidate is frozen. Every conversation in train / val / test has the same structure: `id / group_id` is the conversation ID / science-topic group, `grade_band` is 1-2 / 3-4 / 5-6, `fact_ids` references ids in `facts`, and `turns[]` holds user / assistant / behavior.)

Which set a conversation belongs to is determined solely by the array it sits in. Conversations no longer carry a `split` field, which avoids contradictions like "sits in val but says train". When the program reads the file (`lab.data.read_dataset`; in the Notebook it is `tag_splits`), it adds `split` to each conversation in memory, and later filtering and cross-set checks all use it; the source file stays unchanged. To move a conversation to another set, cut the whole conversation into the other array; the other conversations with the same `group_id` and the fact cards it references must move along with it.

JSON does not support `//` comments. The fields are explained here, and the Notebook uses D1–D7 to mark where they are actually used. Do not stuff the whole JSON into the prompt, and do not pass the evaluation conversations to the trainer.

## 2 The first fact card

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

| Field | Meaning and use |
|---|---|
| `id` | Primary key, referenced by a conversation's `fact_ids` |
| `category/topic` | Category and topic, for search, statistics, and review |
| `text` | Science material the answer can rely on; enters the context |
| `kind/scope` | Material type and scope of applicability; enters the context |
| `source_url` | Entry point for a human to look up the source; the program does not go online to read it |
| `checked_at/source_review` | `pending` means awaiting verification; the date only means the reference entry point has been read, and `reference_read_claim_review_pending` means the individual claims are still awaiting review |
| `reuse` | Source attribution and a note on original rewriting; does not mean the source institution endorses this data |

Official popular-science material from NASA, NOAA, USGS, and others, plus OpenStax textbooks, serve as reference entry points. The text is an original Chinese write-up; the analogies, children's feedback, and observation designs are synthetic demonstrations and cannot be claimed to come entirely from these sources. A URL being present does not mean every sentence is supported by the source; each claim needs to be checked.

### Why fact cards are needed: three common questions

Fine-tuning teaches "how to say it" (tutorial section 0.1), and at this point beginners usually come up with three questions in a row.

**Q1: Primary-school science is common knowledge that the base model already knows. Why do we still need fact cards?**

True, the base model knows most of this common knowledge. If the only question were "where does the knowledge come from", fact cards would not be necessary. We still want them, and the reason is not to supply knowledge:

1. **So that training learns only "how to say it".** The loss is computed on the reference answer, and the reference answer contains both phrasing and factual content. When there is no material in the input, the model can only memorize content like "turning toward the sun" or "the side facing away from the light" into its weights, and training mixes "phrasing" and "knowledge" together; when the material is right there in the input, the model can predict the content correctly just by following the material, so LoRA's limited capacity is spent mainly on "short sentences, correct the misconception first, offer a safe little observation". See Q3 for the principle.
2. **The base model's knowledge can be neither inspected nor corrected; fact cards can.** You cannot tell in advance which question a 4B model will get wrong (for example, attributing the seasons to distance from the sun), and when it is wrong you cannot open the weights and fix it. A fact card is a visible piece of text with a source and a review status; if it is wrong, you fix one card and it takes effect on the next request. A backend analogy: business rules belong in configuration or a database, not scattered inside a compiled binary.
3. **Cards carry boundaries the base model would not respect on its own.** For example, the `text` of `day-night` states “假定取消自转但保留公转，光照位置仍会随公转缓慢变化” (“if rotation were removed but revolution kept, the lit position would still change slowly with the revolution”), and the follow-up question in the second turn of the reference conversation is based on it; `scope` states “模型和比喻不能替代真实测量” (“models and analogies cannot replace real measurement”); `kind` distinguishes fact, legend, and current, so a legend must not be told as fact.
4. **This is what real products look like, and it is the basis for evaluation.** In business settings, what the model has to explain is often something it simply does not know: product specs, a particular school's textbook, today's data. The common approach is for the caller to look up the material and put it in the prompt while the model is responsible for explaining it clearly — what is usually called RAG (retrieval-augmented generation). This project uses fact cards to practice that pattern end to end. During blind review, raters also check each claim against the fact cards to judge whether it is supported (tutorial section 6.6).

**Q2: Fact cards are also in the training samples, so does the model remember this knowledge after training?**

That is not the right way to think about it. A fact card is part of a training sample's **input**, not its **learning target**. In the 408-token sample from tutorial section 2.5, the fact card sits among the 386 tokens with mask=0: the model can see it, but it is not penalized for "not remembering it". The loss tests only one thing: given this material, how closely does the answer match the reference answer.

| | Learned after training? | Why |
|---|---|---|
| How to read the material and how to organize an answer with it | Yes | This is what the loss pushes it to learn |
| The content and boundaries of the material itself | Not reliably | During training the material is always right in front of it, so there is no need to memorize it |

To be honest about one point: the reference answers themselves contain factual content, which is in the mask=1 region, so the model will incidentally remember some common phrasings. But with only 1 epoch of training, seeing each sample once, the memory is shallow; it cannot remember the full content and boundaries of the cards, and you have even less way of knowing what it remembered and what it remembered wrong. So at deployment you **must pass the fact cards every time**; this is why the call example in tutorial section 5.5 first reads the facts with `read_dataset`.

If you want to see this for yourself: after training, ask the same question twice, once with the fact card and once with the `CURRENT REFERENCE FACTS` section removed, and compare the accuracy and boundaries of the answers. The more obvious the difference, the more the model relies on the material given on the spot rather than on memorized knowledge.

**Q3: If we are not relying on it to memorize knowledge, why include it during training? Can't we just pass it at deployment?**

No. Training is not about making the model **remember** the fact cards but about teaching it to **use** them; to learn to use them, it has to see them during training. Think of onboarding a new employee:

| | Method A: Q&A only | Method B: Q&A with the product manual alongside (this project) |
|---|---|---|
| Training material | Customer asks X → standard reply Y | Customer asks X + relevant manual page → standard reply Y |
| Habit it builds | Hear a question, recite the answer from memory | Look up the manual first, then explain in the standard tone |
| Handed the manual on the job | Never practiced reading the manual, probably still recites old answers | Exactly the workflow it practiced |
| One entry in the manual is updated | Still says the old answer | Follows the new manual |
| A question never covered in training | Answers from vague impressions, prone to making things up | Explains from the manual and does not make up what the manual does not say |

Translated to this project, here is the same sample; the target answer is exactly the same, only the input differs:

```text
不带事实卡：
  输入：为什么白天亮、晚上黑？
  目标：地球在不停转，我们住的地方转向太阳就是白天……

带事实卡：
  输入：[事实卡：自转使地球表面交替朝向和背向太阳……] 为什么白天亮、晚上黑？
  目标：地球在不停转，我们住的地方转向太阳就是白天……
```

(Gloss: “不带事实卡” = without a fact card, “带事实卡” = with a fact card; “输入” = input, “目标” = target. The question is “Why is it bright during the day and dark at night?”, the fact card says “Rotation makes each part of Earth's surface alternately face toward and away from the sun…”, and the target answer is “The Earth keeps turning; when the place where we live turns toward the sun it is daytime…”.)

Without the card, the input contains no basis at all, yet the target is a definite scientific conclusion, so 1150 samples repeatedly reinforce "hear a question, produce a confident conclusion out of thin air"; with the card, everything in the target can be found in the input, so what gets reinforced is "take the content from the given material and rewrite it in words a child can understand".

If you leave the cards out during training and add them only at deployment, four problems appear:

1. **The model never practiced reading material.** At deployment an extra large `CURRENT REFERENCE FACTS` section appears; to the model it is unfamiliar input, may be ignored, and the model answers from memory as usual.
2. **Changing the material has no effect.** If you find a card is wrong, or want to add a scope boundary to a card's `scope`, a Method A model will not change accordingly; you can only change the training data and retrain.
3. **New topics are more prone to fabrication.** Method A builds the habit of "producing conclusions out of thin air", so on topics not in the training data it still draws conclusions confidently.
4. **The input format does not match.** During training the prompt has no fact cards, at deployment it does, and the way the model speaks depends on what the input looked like during training; this is also why tutorial section 5.5 stresses that "the prompt must match the one used in training".

In code terms, training learns a function `answer = f(question, facts, grade_band, history)`. If you give only `question` during training, what it learns is `f(question)`; passing an extra parameter `facts` at deployment that it has never used will not make it start using it automatically.

You can train and answer without fact cards; the cost is everything above: mistakes cannot be corrected, the model tends to be overconfident, and moving to a domain the base model does not understand means starting over. Separating "facts" from "phrasing" is a design trade-off of this project, not the only way to do it; its benefit ultimately has to be verified by the comparison in tutorial chapter 6.

### Where fact cards are used

**Why they are stored separately.** The 60 cards are referenced by the 600 conversations via `fact_ids` (one conversation can reference several cards), similar to a foreign key in a database: a fact is changed in only one place, and every conversation that references it uses the new content the next time it is read.

**Why they must also be isolated by set.** A card can only be referenced by conversations in the same set (one of train/val/test). Otherwise the material that validation and test questions rely on would already have been seen by the model during training, and the evaluation results would be too optimistic.

| Stage | Code location | How fact cards are used |
|---|---|---|
| Load-time validation (D1) | [data.py](../lab/data.py) `validate` | `id` is unique; all fields present; `source_url` is https; `kind` can only be fact/legend/current; every `fact_ids` entry in a conversation must exist; the same card cannot span train/val/test |
| Building the prompt (D3) | [common.py](../lab/common.py) `build_messages` | Fetch cards by `fact_ids`, keep only `id/text/kind/scope`, and write them as JSON after `CURRENT REFERENCE FACTS:` in the system prompt; source and review fields do not enter the model |
| Notebook training and validation samples | Notebook step 2, `prepare-data` cell | All 1150 train samples and 115 val samples go through `build_messages` with their own fact cards |
| Notebook observation questions | Notebook step 4, `cases` cell | Each of the three observation questions specifies one `fact_id`, and the same card is used to compare answers before and after training |
| CLI snapshot | [prepare.py](../lab/prepare.py) | Written as-is to the snapshot's `facts.jsonl`; few-shot examples record their `fact_ids` |
| A/B/C comparison inference | [infer.py](../lab/infer.py) | Reads `facts.jsonl` from the snapshot; all three groups attach the same fact card for the current question (section 6.4) |
| Blind review | [evaluate.py](../lab/evaluate.py) `blind` | The scoring sheet includes `fact_ids`; raters fill in `fact_claims`/`supported` by checking against the fact cards (section 6.6) |
| Calls after deployment | Call example in section 5.5 | `read_dataset` reads the facts, and `messages_for(fact_id, ...)` builds the same system prompt as in training |

## 3 The first complete conversation

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

The child first asks about day and night, then offers the guess that "the sun goes to sleep". The reference answer first explains the phenomenon, then corrects the specific misconception, and leaves an optional question to keep thinking about. Do not let the model learn only fixed catchphrases like “你真棒” (“You're awesome”).

| Field | Does it enter the model, and where is it used |
|---|---|
| `id` | Generates sample IDs such as `day-night-01:0` for tracking |
| `group_id` | Groups the same science topic; all grade bands, paths, and rewrites must be in the same split |
| (containing array) | This conversation is in the `train` array; on load it is tagged `split="train"`, and D2 filters only train; not written into the prompt |
| `grade_band` | **D3 writes it into the system prompt as the “小学学段” (primary-school grade band)**; it is a visible input |
| `fact_ids` | D3 selects fact cards from this file, passing only id/text/kind/scope |
| `turns[].user` | The child's current question; earlier questions go into history |
| `turns[].assistant` | The current answer is the training target; earlier answers serve only as context |
| `turns[].behavior` | For humans to see what behavior this turn demonstrates; **not passed to the model** |
| `category/topic/scenario` | Statistics and review labels; they do not sneak "what the child got wrong" into the input |
| `provenance` | Records the synthesis method, author, and date; these are not real children's conversations |
| `review` | All draft; after review, fill in the actual reviewer, reviewer type, and change notes |

## 4 How to read, edit, and review

Find the `topic` first, then read `fact_ids`, `grade_band`, and the whole `turns`. To change the wording, edit the assistant turn; to add a realistic feedback path, adjust the context at the same time so the child does not quote something the model never said. After updating a fact, re-check every conversation that references it.

```bash
python -m lab.data --input data/dataset.json --allow-draft
```

It should print `{'train': 500, 'val': 50, 'test': 50}`. This is a structural check, not a scientific review. A formal review covers: whether the source supports the fact, whether there is excessive anthropomorphism, whether a child can understand it, whether the boundaries of analogies are clear, whether observations are safe and feasible, and whether the model respects a child who stops asking questions or wants to end.

Review can be done directly in this one file, with version control keeping the history. Only entries that have truly been reviewed get `review.status="approved"` with the actual `reviewer/reviewer_kind/notes` filled in; do not bulk-change the status to pretend a review happened. Fact cards should record their actual verification status as well. Afterwards, rerun the validation without `--allow-draft`.

The 10 kinds of structure in the data deliberately repeat teaching behaviors, and within a topic some knowledge answers are reused; you still need to check manually for semantic near-duplicates and a templated feel. The 50 train topics are not 500 independent topics. val/test are isolated from the training topics but still share basic concepts such as light, force, and life, so knowledge is not absolutely isolated.

Revision on 2026-10-04 (still draft): removed the half-width spaces left over from stitching sentences together in 662 answers, so the model no longer learns seams like “。 下一句” (a period followed by a space before the next sentence); the 60 answers for “多讲一点原因” (“tell me more about why”, `increase_depth_when_requested`) were changed to "mechanism + analogy + the limits of the analogy", so they really are deeper than in the previous round; the 60 answers for “观察不一样” (“my observation was different”, `ask_for_evidence_without_blame`) dropped the stock phrase “我们刚才讨论的是……” (“What we just discussed was…”) and now check the conditions first; the 60 “invite exploration” answers now add “还可以想想：” (“You could also think about:”) before the follow-up question. Answers are still fairly short overall (about 51 characters on average) and still need human review.

## 5 How changes take effect

After saving the JSON, give the Notebook a new `RUN_DIR`, rerun D1 loading and D2–D5 tokenization, and then start training. Rows already in memory and adapters already saved do not change automatically. The `dataset.json` in a run directory is the input snapshot for that run, not a second copy of the source data to maintain day to day.


## 2 D1–D7: where the data is actually used

| Comment | Location and actual behavior |
|---|---|
| D1 | Notebook step 2 reads `DATA_FILE`, tags the conversations in train/val/test with their split, and validates fact-card references and cross-set isolation |
| D2 | Selects only train; `expand_episode` keeps the past history turn by turn |
| D3 | `build_messages` assembles the role, **grade_band**, selected facts, history, and the current user turn |
| D4 | `render_pair` applies the chat template and marks the current assistant answer plus EOS as the target |
| D5 | Hands only the two columns `input_ids/completion_mask` to `SFTTrainer` |
| D6 | Checks the labels TRL actually produces; context and padding must be -100 |
| D7 | Saves a copy and fingerprint of this run's data; only `trainer.train()` actually updates the parameters |

```text
单一 dataset.json → train 对话 → 当前问题 + 已发生的历史
                                  ↓
            角色 + 学段 + 对应事实 + history + user
                                  ↓
                       prompt / 当前 assistant + EOS
                                  ↓
                   input_ids + completion_mask
                                  ↓
               labels → loss → 反向传播 → LoRA 参数更新
```

(Gloss: single dataset.json → train conversations → current question + history so far → role + grade band + matching facts + history + user → prompt / current assistant + EOS → input_ids + completion_mask → labels → loss → backpropagation → LoRA parameter update.)

## 3 Five visible fields

`rows` keeps `case_id/prompt/completion/input_ids/completion_mask` so you can inspect them. Actual training only receives the last two columns. The Notebook prints the full prompt, the target, and the real token length of the first row; do not reuse length numbers from older scenarios.

`completion_mask=0` marks context, and `=1` marks the current answer plus EOS. TRL sets the labels of context and padding to `-100`, so no loss is computed directly at those positions, but the model can still see the context when predicting the answer.

EOS is <code>&lt;&#124;im_end&#124;&gt;</code>, and it appears exactly once per target so the model learns to stop; PAD is different from EOS. Anything over 4096 tokens raises an error and the sample has to be rewritten; the answer is never silently truncated.


Per-item scientific sources are in facts[].source_url; fact review and child-readability review are not yet complete.
