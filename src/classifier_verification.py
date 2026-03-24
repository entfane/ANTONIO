import argparse
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from data import load_align_mat
from hyperrectangles import calculate_hyperrectangle
from hyper_rect_extraction import compute_hyperrectangles
from verifier import Verifier
from datasets import load_dataset
import os


def get_classifier_head(model):
    linear_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]
    
    _, head = linear_layers[-1]
    
    weights = head.weight
    bias = head.bias
    
    return weights, bias


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify classifier"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g. 'bert-base-uncased')"
    )
    parser.add_argument(
        "--pooling", "-p",
        type=str,
        choices=["first", "last"],
        required=True,
        help="Pooling strategy: 'first' for encoder models ([CLS]), 'last' for decoder models"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        help="HuggingFace dataset ID (e.g. 'glue')"
    )
    parser.add_argument(
        "--split", "-s",
        type=str,
        required=True,
        help="HuggingFace dataset split (e.g. 'train')"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        required=True,
        help="Classification threshold (e.g. 0.5)"
    )
    parser.add_argument("--input-col",  "-i", type=str,   required=True,  help="Dataset column to use as input text")
    parser.add_argument("--output-col", "-o", type=str, default=None, help="Dataset column to use as output label (optional)")
    parser.add_argument("--batch-size", "-b", type=int,   default=2,     help="Batch size for embedding extraction (default: 2)")
    parser.add_argument("--max-len",    "-l", type=int,   default=128,    help="Max token length for tokenizer (default: 128)")
    parser.add_argument("--use-single-hyper-rectangle", type = bool, default=False, help="Boolean, whether to use a single hyper-rectangle or use multiple eps-cubes (default: single hyper-rectangle)")
    parser.add_argument("--eps", type=float, default=0.05, help="Epsilon value for hyper-cubes (default: 0.05)")
    args = parser.parse_args()

    HF_MODEL    = args.model
    DATASET_NAME = args.dataset
    DATASET_SPLIT = args.split
    THRESHOLD   = args.threshold
    INPUT_COL = args.input_col
    OUTPUT_COL = args.output_col
    BATCH_SIZE = args.batch_size
    MAX_LEN = args.max_len
    POOLING = args.pooling
    SINGLE_HYPER_RECTANGLE = args.use_single_hyper_rectangle
    EPS = args.eps

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    classifier = AutoModelForSequenceClassification.from_pretrained(HF_MODEL, device_map = "auto", num_labels = 1)
    classifier.eval()
    if DATASET_NAME.endswith(".jsonl") or os.path.isfile(DATASET_NAME):
        dataset = load_dataset("json", data_files={DATASET_SPLIT: DATASET_NAME}, split=DATASET_SPLIT)
    else:
        dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

    verifier = Verifier(POOLING)
    embeddings = verifier.extract_embeddings(dataset, classifier, tokenizer, INPUT_COL, OUTPUT_COL, BATCH_SIZE, MAX_LEN)

    align_mat  = load_align_mat(DATASET_NAME, HF_MODEL, embeddings, False)
    
    embeddings = embeddings @ align_mat
    if SINGLE_HYPER_RECTANGLE:
        hyperrectangles = [calculate_hyperrectangle(embeddings)]
    else:
        hyperrectangles = compute_hyperrectangles(embeddings, min_cluster_size=5)
    weights, bias = get_classifier_head(classifier)
    weights = weights.squeeze().detach().cpu().float().numpy() @ align_mat
    bias = bias.squeeze().detach().cpu().float().numpy() if bias is not None else None

    result = verifier.verify(hyperrectangles, weights, bias, THRESHOLD, align_mat)
    print(result)
    if result == verifier.UNSAT:
        print(f"Everything inside the hyper-rectangle classified >= {THRESHOLD}")
    else:
        print(f"There exists a point within the hyper-rectangle which is classified < {THRESHOLD}")