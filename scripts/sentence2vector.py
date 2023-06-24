import argparse
import os

import loguru
import torch
import transformers
import ujson
import wandb
import tqdm

from src import utils


def _configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="data/train.jsonl",
        help="path to dataset with sentences",
    )
    parser.add_argument(
        "--sentence-bert",
        type=str,
        default="ai-forever/sbert_large_nlu_ru",
        help="path or name of hf model used for getting sentence vector",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=512,
        help="upper bound for sentence's tokens count",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda/cpu/None")
    parser.add_argument(
        "--save-to",
        type=str,
        default="data/sentence_embeddings_train.jsonl",
        help="where to save sentence embeddings",
    )
    return parser


def main(args: argparse.Namespace):
    wandb.init(
        project=os.getenv(utils.WANDB_PROJECT_ENV),
        entity=os.getenv(utils.WANDB_ENTITY_ENV),
    )
    device = args.device or "cuda" if torch.cuda.is_available() else "cpu"

    data = utils.load_jsonl(args.data)

    loguru.logger.info("Loading model {}", args.sentence_bert)
    model = transformers.AutoModel.from_pretrained(args.sentence_bert).to(device)
    loguru.logger.info("Loading tokenizer {}", args.sentence_bert)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.sentence_bert)

    with open(args.save_to, "w", encoding="utf-8") as file:
        for document in tqdm.tqdm(data):
            if len(document) == 0:
                continue

            labels = [item[1] for item in document]
            sentences = [item[0] for item in document]

            sentence_embeddings = []
            for sentence in sentences:
                try:
                    tokenized_sentence = tokenizer(
                        sentence,
                        truncation=True,
                        max_length=args.max_len,
                        return_tensors="pt",
                    )
                    input_ids = tokenized_sentence["input_ids"].to(device)
                    attention_mask = tokenized_sentence["attention_mask"].to(device)
                    model_output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    sentence_embeddings.append(
                        utils.mean_pooling(model_output, attention_mask).detach().cpu()
                    )
                except Exception as e:
                    loguru.logger.warning(
                        "Failed during tokenizing the sentence '{}': {}", sentence, e
                    )
            ujson.dump(
                {
                    "sentence_embeddings": torch.cat(sentence_embeddings).tolist(),
                    "labels": labels,
                },
                file,
                ensure_ascii=False,
            )
            file.write("\n")
    wandb.save(args.save_to)


if __name__ == "__main__":
    args = _configure_parser().parse_args()
    main(args)
