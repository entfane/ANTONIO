# Classifier Verification

Verifies that all points within a computed hyper-rectangle are classified above a given threshold.

## Usage

```bash
python src/classifier_verification.py \
  --model <model_id> \
  --dataset <dataset_id_or_path> \
  --split <split> \
  --input-col <col> \
  --threshold <threshold>
```

## Arguments

| Argument | Short | Required | Default | Description |
|---|---|---|---|---|
| `--model` | `-m` | Yes | — | HuggingFace model ID |
| `--dataset` | `-d` | Yes | — | HuggingFace dataset ID or path to local `.jsonl` file |
| `--split` | `-s` | Yes | — | Dataset split (e.g. `train`, `test`) |
| `--threshold` | `-t` | Yes | — | Classification threshold (e.g. `0.5`) |
| `--input-col` | `-i` | Yes | — | Column name to use as input text |
| `--output-col` | `-o` | No | `None` | Column name to use as output label |
| `--batch-size` | `-b` | No | `2` | Batch size for embedding extraction |
| `--max-len` | `-l` | No | `128` | Max token length for tokenizer |

## Examples

**HuggingFace dataset:**
```bash
python src/classifier_verification.py \
  --model "bert-base-uncased" \
  --dataset "glue" \
  --split "train" \
  --input-col "sentence" \
  --output-col "label" \
  --threshold 0.5
```

**Local JSONL file:**
```bash
python src/classifier_verification.py \
  --model "entfane/gpt2_constitutional_classifier" \
  --dataset datasets/toy/test.jsonl \
  --split train \
  --input-col "input" \
  --output-col "output" \
  --threshold 0.2
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