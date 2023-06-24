import argparse
import os.path

from src import utils


def _configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="data/test.jsonl",
        help="path to jsonl file convert to txt files",
    )
    parser.add_argument(
        "--save-to",
        type=str,
        default="data/test_inputs",
        help="directory where to save txt fiels",
    )
    return parser


def main(args: argparse.Namespace):
    if not os.path.exists(args.save_to):
        os.makedirs(args.save_to, exist_ok=True)

    data = utils.load_jsonl(args.data)
    for ind, document in enumerate(data):
        sentences = [item[0] for item in document]
        with open(os.path.join(args.save_to, f"{ind}.txt"), "w+", encoding="utf-8") as txt_file:
            for sentence in sentences:
                txt_file.write(sentence)
                txt_file.write("\n")


if __name__ == "__main__":
    args = _configure_parser().parse_args()
    main(args)
