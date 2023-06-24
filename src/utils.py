import os
import random
from typing import Any
import loguru

import numpy as np
import torch
import wandb
from sklearn import metrics
from torch import utils
import yaml
from torch import nn
import ujson

from src import seg_datasets, seg_model

WANDB_PROJECT_ENV = "WANDB_PROJECT"
WANDB_ENTITY_ENV = "WANDB_ENTITY"


def mean_pooling(
    model_output: tuple[torch.Tensor, ...], attention_mask: torch.Tensor
) -> torch.Tensor:
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def load_jsonl(file_path: str) -> list[Any]:
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(ujson.loads(line))
    return data


def save_jsonl(obj: list[Any], file_path: str):
    with open(file_path, "w", encoding="utf-8") as jsonl_file:
        for item in obj:
            ujson.dump(item, jsonl_file, ensure_ascii=False)
            jsonl_file.write("\n")


def load_yaml(file_path: str) -> dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as yaml_file:
        data = yaml.safe_load(yaml_file)
    return data


def eval_model(
    model: seg_model.BertOverSentences,
    dataloader: torch.utils.data.DataLoader,
    device: str,
):
    predictions = []
    ground_truth = []

    model.eval()
    for batch in dataloader:
        batch = dict_to_device(batch, device)
        output = model(
            inputs_embeds=batch["sentence_embeddings"],
            attention_mask=batch["attention_mask"],
        ).logits.flatten(end_dim=1)
        labels = batch["labels"].flatten()

        mask = labels != -100
        output = output[mask]
        labels = labels[mask]

        predictions.append(output.argmax(dim=1).detach().cpu())
        ground_truth.append(labels.detach().cpu())

    predictions_np = torch.cat(predictions).numpy()
    ground_truth_np = torch.cat(ground_truth).numpy()

    log_dict = {
        "accuracy": metrics.accuracy_score(ground_truth_np, predictions_np),
        "precision": metrics.precision_score(ground_truth_np, predictions_np),
        "recall": metrics.recall_score(ground_truth_np, predictions_np),
        "f1": metrics.f1_score(ground_truth_np, predictions_np),
    }
    model.train()
    return log_dict


def collate_function(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    sentence_embeddings = [item["sentence_embeddings"] for item in batch]
    labels = [item["labels"] for item in batch]

    padded_sentence_embeddings = nn.utils.rnn.pad_sequence(
        sentence_embeddings, batch_first=True
    )
    padded_labels = nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )

    bs, seq_len, _ = padded_sentence_embeddings.shape

    attention_mask = torch.zeros((bs, seq_len), dtype=torch.float)
    for i, seq in enumerate(sentence_embeddings):
        attention_mask[i, : len(seq)] = 1

    return {
        "sentence_embeddings": padded_sentence_embeddings,
        "attention_mask": attention_mask,
        "labels": padded_labels,
    }


def set_deterministic_mode(seed):
    loguru.logger.info("Setting deterministic mode | seed = {}", seed)
    _set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.deterministic = True
    torch.backends.benchmark = False

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dataloader(
    data_path: str, batch_size: int, max_len: int
) -> torch.utils.data.DataLoader:
    dataset = seg_datasets.SentenceEmbeddingDataset(
        data_path=data_path, max_len=max_len
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_function
    )


def dict_to_device(d: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    for key, value in d.items():
        d[key] = value.to(device)
    return d
