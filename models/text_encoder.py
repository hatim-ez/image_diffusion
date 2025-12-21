from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import CLIPTextModel


class CLIPTextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        max_length: int = 77,
        vocab_size: Optional[int] = None,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        self.model = CLIPTextModel.from_pretrained(model_name)
        if vocab_size and vocab_size != self.model.text_model.embeddings.token_embedding.num_embeddings:
            self.model.resize_token_embeddings(vocab_size)
        self.max_length = max_length
        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return {"embeds": outputs.last_hidden_state, "pooled": outputs.pooler_output}
