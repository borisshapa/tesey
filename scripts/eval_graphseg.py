import argparse
import os.path

from sklearn import metrics

from src import utils


def _configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-dataset",
        type=str,
        default="data/test.jsonl",
        help="path to test jsonl file",
    )
    parser.add_argument(
        "--graphseg-output",
        type=str,
        default="data/test_outputs",
        help="path to directory with labeled documents",
    )
    return parser


def main(args: argparse.Namespace):
    data = utils.load_jsonl(args.test_dataset)
    ground_truth = []
    predictions = []

    for ind, document in enumerate(data):
        sentences = [item[0] for item in document]
        labels = [item[1] for item in document]
        ground_truth.extend(labels)

        pred = []
        next_label = 1

        sent_ind = 0
        with open(
            os.path.join(args.graphseg_output, f"{ind}.txt"), "r", encoding="utf-8"
        ) as txt_file:
            lines = list(txt_file)
            line_ind = 0
            if len(lines) == 0:
                pred.append(next_label)

            while line_ind < len(lines):
                if lines[line_ind].strip() == "==========":
                    line_ind += 1
                    next_label = 1
                    continue

                cur_sent = sentences[sent_ind]
                while cur_sent:
                    line = lines[line_ind].strip()
                    cur_sent = cur_sent.removeprefix(line).strip()
                    line_ind += 1
                sent_ind += 1

                pred.append(next_label)
                next_label = 0
        predictions.extend(pred)

    log_dict = {
        "accuracy": metrics.accuracy_score(ground_truth, predictions),
        "precision": metrics.precision_score(ground_truth, predictions),
        "recall": metrics.recall_score(ground_truth, predictions),
        "f1": metrics.f1_score(ground_truth, predictions),
    }
    print("\n".join([f"{key}: {value}" for key, value in log_dict.items()]))


if __name__ == "__main__":
    args = _configure_parser().parse_args()
    main(args)
