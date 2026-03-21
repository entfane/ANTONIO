# Classifier Verification

Verifies that all points within a computed hyper-rectangle are classified above a given threshold.

## Usage

```bash
python src/classifier_verification.py \
  --model <model_id> \
  --dataset <dataset_id_or_path> \
  --split <split> \
  --input-col <col> \
  --threshold <threshold> \
  --pooling <first|last>
```

## Arguments

| Argument | Short | Required | Default | Description |
|---|---|---|---|---|
| `--model` | `-m` | Yes | — | HuggingFace model ID |
| `--dataset` | `-d` | Yes | — | HuggingFace dataset ID or path to local `.jsonl` file |
| `--split` | `-s` | Yes | — | Dataset split (e.g. `train`, `test`) |
| `--threshold` | `-t` | Yes | — | Classification threshold (e.g. `0.5`) |
| `--input-col` | `-i` | Yes | — | Column name to use as input text |
| `--pooling` | `-p` | Yes | — | Pooling strategy: `first` for encoder models, `last` for decoder models |
| `--output-col` | `-o` | No | `None` | Column name to use as output label |
| `--batch-size` | `-b` | No | `2` | Batch size for embedding extraction |
| `--max-len` | `-l` | No | `128` | Max token length for tokenizer |

## Pooling Strategy

Controls which hidden state is used as the sentence embedding:

| Value | Use for | Reasoning |
|---|---|---|
| `first` | Encoder models (BERT, RoBERTa, DeBERTa, ...) | Takes the `[CLS]` token at position 0, which aggregates the full sequence |
| `last` | Decoder models (GPT-2, LLaMA, Mistral, ...) | Takes the last non-padding token, the only one that has attended to the full context |

## Examples

**Encoder model with HuggingFace dataset:**
```bash
python src/classifier_verification.py \
  --model "bert-base-uncased" \
  --dataset "glue" \
  --split "train" \
  --input-col "sentence" \
  --output-col "label" \
  --threshold 0.5 \
  --pooling first
```

**Decoder model with local JSONL file:**
```bash
python src/classifier_verification.py \
  --model "entfane/gpt2_constitutional_classifier" \
  --dataset datasets/toy/test.jsonl \
  --split train \
  --input-col "input" \
  --output-col "output" \
  --threshold 0.2 \
  --pooling last
```

## Local JSONL Format

Each line should be a valid JSON object with at minimum an input column:

```json
{"input": "What a great day", "output": "For sure"}
{"input": "Can you recommend a good book?", "output": "Sure! I loved The Alchemist recently."}
```

## Output

- **UNSAT** — every point inside the hyper-rectangle is classified `>= threshold`
- **SAT** — there exists a point inside the hyper-rectangle classified `< threshold`