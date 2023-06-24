import torch
import transformers
from torch import nn


class BertOverSentences(nn.Module):
    def __init__(
        self,
        hf_model: transformers.PreTrainedModel,
        sentence_embedding_size: int,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model = hf_model
        emb_size = self.model.config.hidden_size
        self.linear = nn.Linear(sentence_embedding_size, emb_size)

    def forward(
        self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        embeddings = self.linear(inputs_embeds)
        return self.model(inputs_embeds=embeddings, attention_mask=attention_mask)
