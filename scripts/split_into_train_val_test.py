import argparse
import os
import random

from src import utils


def _configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/data.jsonl")
    parser.add_argument(
        "--val", type=float, default=0.1, help="the proportion of the validation subset"
    )
    parser.add_argument(
        "--test", type=float, default=0.1, help="the proportion of the test subset"
    )
    parser.add_argument(
        "--save-to",
        type=str,
        default="data",
        help="directory where to save the partition",
    )
    parser.add_argument("--seed", type=str, default=21, help="random seed")
    return parser


def main(args: argparse.Namespace):
    utils.set_deterministic_mode(args.seed)

    data = utils.load_jsonl(args.data)
    random.shuffle(data)

    data_len = len(data)

    val_size = int(data_len * args.val)
    test_size = int(data_len * args.test)
    train_size = data_len - val_size - test_size

    partition = {
        "train": data[:train_size],
        "val": data[train_size : train_size + val_size],
        "test": data[train_size + val_size :],
    }
    for key, subset in partition.items():
        utils.save_jsonl(subset, os.path.join(args.save_to, f"{key}.jsonl"))


if __name__ == "__main__":
    args = _configure_parser().parse_args()
    main(args)
