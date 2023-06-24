from typing import Any, Iterator

import torch
import ujson
from torch.utils import data


class SentenceEmbeddingDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_path: str, max_len: int = None):
        self.data_path = data_path
        self.max_len = max_len if max_len is not None else torch.iinfo(torch.int32).max

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        with open(self.data_path, "r", encoding="utf-8") as file:
            for item in file:
                json_obj = ujson.loads(item)
                sentence_embeddings_t = torch.FloatTensor(
                    json_obj["sentence_embeddings"]
                )
                labels_t = torch.LongTensor(json_obj["labels"])

                sentence_count = sentence_embeddings_t.shape[0]
                if sentence_count > self.max_len:
                    chunk_count = (sentence_count + self.max_len - 1) // self.max_len
                    chunk_size = (sentence_count + chunk_count - 1) // chunk_count
                    sentence_embeddings_chunks = torch.split(
                        sentence_embeddings_t, chunk_size
                    )
                    labels_chunks = torch.split(labels_t, chunk_size)
                    yield from [
                        {"sentence_embeddings": se, "labels": l}
                        for se, l in zip(sentence_embeddings_chunks, labels_chunks)
                    ]
                else:
                    yield {
                        "sentence_embeddings": sentence_embeddings_t,
                        "labels": labels_t,
                    }
