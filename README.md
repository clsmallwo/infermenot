# MCQ Inference Model

This project downloads public English multiple-choice question datasets, normalizes them into `A/B/C/D` choices with the right answer letter, trains a lightweight answer-choice scorer, and exposes a CLI for new questions.

## Data Sources

The downloader uses public datasets from Hugging Face:

- `allenai/ai2_arc` with `ARC-Easy` and `ARC-Challenge`
- `allenai/openbookqa`
- `allenai/sciq`

SciQ does not ship with letters, so the downloader deterministically shuffles the correct answer plus three distractors into `A/B/C/D`.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Train

```bash
python3 download_data.py
python3 train_model.py
```

This writes:

- `data/mcqs.jsonl`: normalized questions
- `models/mcq_inference.joblib`: trained inference model
- `models/metrics.json`: validation/test accuracy and per-source breakdowns

## Larger V2 Data And Faster Training

Build the larger mixed-choice dataset:

```bash
python3 download_data.py \
  --extended \
  --v2 \
  --extended-train-limit-per-source 5000 \
  --out data/mcqs_v2_5k_each.jsonl
```

This adds MMLU, RACE, MedMCQA, CommonsenseQA, QASC, and HellaSwag. The current v2 file contains `98,885` normalized MCQs, including 4-, 5-, and 8-choice questions.

Fast iteration recipe:

```bash
python3 train_choice_classifier.py \
  --data data/mcqs_v2_5k_each.jsonl \
  --base-model models/deberta_mcq_classifier \
  --epochs 1 \
  --batch-size 16 \
  --max-train-per-source 1500 \
  --eval-limit 1000 \
  --freeze-lower-layers 8 \
  --gradient-accumulation-steps 2 \
  --out models/deberta_mcq_v2_fast \
  --metrics-out models/deberta_mcq_v2_fast_metrics.json
```

Fuller training recipe:

```bash
python3 train_choice_classifier.py \
  --data data/mcqs_v2_5k_each.jsonl \
  --base-model models/deberta_mcq_classifier \
  --epochs 1 \
  --batch-size 16 \
  --freeze-lower-layers 4 \
  --gradient-accumulation-steps 2 \
  --out models/deberta_mcq_v2 \
  --metrics-out models/deberta_mcq_v2_metrics.json
```

The v2 trainer groups batches by answer-choice shape, so it can train on variable-choice examples without padding fake choices. It also supports lower-layer freezing, gradient accumulation, source balancing, and capped evaluation to reduce iteration time.

## AP-Oriented Training

Build the broader AP-oriented dataset:

```bash
python3 download_data.py \
  --extended \
  --v2 \
  --ap-sources \
  --extended-train-limit-per-source 5000 \
  --out data/mcqs_ap_ready.jsonl
```

This keeps the existing public MCQ sources, samples capped large sources deterministically instead of taking biased prefixes, and adds text-only CK-12/TQA and ScienceQA examples. The current file contains `122,976` normalized MCQs with 2-8 answer choices.

The current AP-oriented checkpoint was trained with:

```bash
python3 train_choice_classifier.py \
  --data data/mcqs_ap_ready.jsonl \
  --base-model models/deberta_mcq_classifier_extended_aligned \
  --epochs 1 \
  --batch-size 8 \
  --learning-rate 8e-6 \
  --max-length 256 \
  --max-train-per-source 4000 \
  --eval-limit-per-source 600 \
  --gradient-accumulation-steps 4 \
  --freeze-lower-layers 2 \
  --warmup-ratio 0.08 \
  --out models/deberta_mcq_ap_ready \
  --metrics-out models/deberta_mcq_ap_ready_metrics.json
```

It writes `models/deberta_mcq_ap_ready`, which is auto-selected by the classifier runtime when present. The AP-readiness evaluator can be rerun with:

```bash
python3 eval_ap_readiness.py \
  --model models/deberta_mcq_ap_ready \
  --out models/deberta_mcq_ap_ready_ap_readiness_metrics.json
```

Current AP-readiness accuracy is `46.0%` over `10,214` held-out AP-like questions, up from `45.4%` for the previous aligned checkpoint. Science is stronger than math: held-out SciQ is `77.8%`, upper-grade ScienceQA is `62.4%`, high-school biology is `41.3%`, and high-school math is still only `20.4%`. This is a better local checkpoint, but not yet a reliable AP exam solver.

## Super Massive Trainable Data

The current no-cap combined dataset is:

- `data/mcqs_supermassive_trainable.jsonl`
- `555,004` normalized MCQs: `496,762` train, `29,706` validation, `28,536` test
- Choice shapes from 2 through 8 options, grouped automatically by `train_choice_classifier.py`
- Includes a `10 MB` synthetic song info and music history supplement under `synthetic/music_song_history_v1`

Rebuild the public base with:

```bash
python3 download_data.py \
  --extended \
  --v2 \
  --ap-sources \
  --out data/mcqs_supermassive_trainable.jsonl
```

Then add the music supplement:

```bash
python3 generate_music_mcqs.py \
  --out data/music_song_history_10mb.jsonl \
  --target-mb 10
cat data/music_song_history_10mb.jsonl >> data/mcqs_supermassive_trainable.jsonl
```

Full local training is intentionally conservative on memory:

```bash
python3 train_choice_classifier.py \
  --data data/mcqs_supermassive_trainable.jsonl \
  --base-model models/deberta_mcq_classifier_extended_aligned \
  --epochs 1 \
  --batch-size 8 \
  --gradient-accumulation-steps 4 \
  --freeze-lower-layers 8 \
  --eval-limit-per-source 1000 \
  --out models/deberta_mcq_supermassive_trainable \
  --metrics-out models/deberta_mcq_supermassive_trainable_metrics.json
```

A faster first pass can add `--max-train-per-source 25000`, which keeps the source mix broad while reducing the one-epoch run size.

## Predict

```bash
python3 predict.py \
  --question "Which process lets plants make food using sunlight?" \
  --a "evaporation" \
  --b "photosynthesis" \
  --c "condensation" \
  --d "erosion"
```

The model scores each answer independently against the question and returns the highest-scoring choice. It is a baseline model, not a guarantee of correctness, but it should perform above random on questions similar to the public datasets it trained on.

## Stronger Transformer Predictor

For better than the lightweight baseline, use the UnifiedQA predictor:

```bash
python3 evaluate_transformer.py --split test
python3 predict_transformer.py \
  --question "Which process lets plants make food using sunlight?" \
  --a "evaporation" \
  --b "photosynthesis" \
  --c "condensation" \
  --d "erosion"
```

This downloads `allenai/unifiedqa-t5-base` on first use and runs it locally after that. It is slower and larger than the TF-IDF model, but it has pretrained question-answering knowledge and is the better default when answer quality matters.

## Fine-Tune The Transformer

To train directly on the downloaded public MCQs:

```bash
python3 finetune_transformer.py
python3 evaluate_transformer.py --model models/unifiedqa_mcq_finetuned --split test
python3 predict_transformer.py --model models/unifiedqa_mcq_finetuned ...
```

## Supervised Choice Classifier

The strongest local training path is a real four-choice classifier:

```bash
python3 train_choice_classifier.py
python3 predict_classifier.py \
  --question "Which process lets plants make food using sunlight?" \
  --a "evaporation" \
  --b "photosynthesis" \
  --c "condensation" \
  --d "erosion" \
  --e "cellular respiration"
```

The checked-in trained classifier under `models/deberta_mcq_classifier` was trained on 3,000 public MCQs and evaluated on the full held-out test split:

- Validation accuracy: `53.3%`
- Test accuracy: `51.5%`
- Random A/B/C/D baseline: `25.0%`

## Web UI

Run the local web interface:

```bash
python3 webui.py
```

Then open `http://127.0.0.1:8000`. The app loads `models/deberta_mcq_classifier` once and uses Apple Metal through PyTorch MPS when available.

After each inference, the result panel also checks the local training split for overlapping question phrases and displays the closest matching training excerpt when one is found.

## First Grade Benchmark

The objective portions of the public Kolbe End of First Grade Assessment can be rerun with:

```bash
python3 eval_first_grade.py --json-out models/first_grade_eval_latest.json
```

The current deployed inference path scores `39/39 = 100.0%` on that converted benchmark. The improvement comes from first-grade deterministic reasoning guards for arithmetic, sequences, simple fractions, clocks, ordering, capitalization, and sentence punctuation before falling back to the neural MCQ classifier.
