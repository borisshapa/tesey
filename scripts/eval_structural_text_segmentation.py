import argparse
import os

import transformers
import tqdm
import wandb
from sklearn import metrics

from src import utils


def _configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="dennlinger/roberta-cls-consec",
        help="the name of pretrained hf model from the article "
        "Structural Text Segmentation of Legal Documents.",
    )
    parser.add_argument(
        "--test-dataset",
        type=str,
        default="data/test.jsonl",
        help="path to test dataset",
    )
    parser.add_argument(
        "--max-len", type=int, default=512, help="limit on input length"
    )
    return parser


def main(args: argparse.Namespace):
    wandb.init(
        project=os.getenv(utils.WANDB_PROJECT_ENV),
        entity=os.getenv(utils.WANDB_ENTITY_ENV),
    )
    pipe = transformers.pipeline("text-classification", model=args.model)
    tokenizer_kwargs = {
        "padding": True,
        "truncation": True,
        "max_length": 512,
    }
    data = utils.load_jsonl(args.test_dataset)

    ground_truth = []
    predictions = []

    for document in tqdm.tqdm(data):
        ground_truth.extend([item[1] for item in document])
        document_predictions = [1]
        for i in range(len(document) - 1):
            sent1 = document[i][0]
            sent2 = document[i + 1][0]
            res = pipe(f"{sent1} [SEP] {sent2}", **tokenizer_kwargs)[0]
            document_predictions.append(int(res["label"] == "LABEL_0"))
        predictions.extend(document_predictions)

    log_dict = {
        "accuracy": metrics.accuracy_score(ground_truth, predictions),
        "precision": metrics.precision_score(ground_truth, predictions),
        "recall": metrics.recall_score(ground_truth, predictions),
        "f1": metrics.f1_score(ground_truth, predictions),
    }
    wandb.log(log_dict)
    print("\n".join([f"{key}: {value}" for key, value in log_dict.items()]))


if __name__ == "__main__":
    args = _configure_parser().parse_args()
    main(args)
