import argparse
import os

import torch.utils.data
import transformers
import wandb
from torch import nn

from src import utils, seg_model


def _configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="path to yaml config"
    )
    return parser


def main(args: argparse.Namespace):
    wandb.init(
        project=os.getenv(utils.WANDB_PROJECT_ENV),
        entity=os.getenv(utils.WANDB_ENTITY_ENV),
    )
    config = utils.load_yaml(args.config)
    model_config = config["model"]
    data_config = config["data"]
    train_config = config["train"]

    device = (
        train_config.get("device") or "cuda" if torch.cuda.is_available() else "cpu"
    )

    segmentation_bert = transformers.BertForTokenClassification.from_pretrained(
        model_config["segmentation_bert"]
    )
    model = seg_model.BertOverSentences(segmentation_bert, data_config["emb_size"]).to(
        device
    )

    train_dataloader = utils.get_dataloader(
        data_path=data_config["train"],
        batch_size=train_config["batch_size"],
        max_len=train_config["max_len"],
    )
    val_dataloader = utils.get_dataloader(
        data_path=data_config["val"],
        batch_size=train_config["batch_size"],
        max_len=train_config["max_len"],
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config["lr"])

    model.train()
    step = 0
    for epoch in range(1, train_config["epochs"] + 1):
        for batch in train_dataloader:
            step += 1

            batch = utils.dict_to_device(batch, device)

            output = model(
                inputs_embeds=batch["sentence_embeddings"],
                attention_mask=batch["attention_mask"],
            ).logits
            loss = criterion(output.flatten(end_dim=1), batch["labels"].flatten())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({"train/loss": loss.item()})

            if step % train_config["log_every"] == 0:
                log_dict = utils.eval_model(model, val_dataloader, device)
                log_dict = {f"val/{key}": value for key, value in log_dict.items()}
                wandb.log(log_dict)

    model.eval()
    test_dataloader = utils.get_dataloader(
        data_path=data_config["test"],
        batch_size=train_config["batch_size"],
        max_len=train_config["max_len"],
    )
    log_dict = utils.eval_model(model, test_dataloader, device)
    log_dict = {f"test/{key}": value for key, value in log_dict.items()}
    wandb.log(log_dict)


if __name__ == "__main__":
    args = _configure_parser().parse_args()
    main(args)
