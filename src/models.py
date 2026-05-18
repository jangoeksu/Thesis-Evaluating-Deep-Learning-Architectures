from __future__ import annotations

import torch
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    PreTrainedModel,
)

from configs.experiment_settings import (
    CNN_CONFIG,
    EXPERIMENT_CONFIG,
    ROBERTA_CONFIG,
)


class TextCNN(nn.Module):
    """
    Convolutional neural network for multi-class news classification.

    The CNN uses an embedding layer that is trained from scratch on the
    experiment datasets. RoBERTa, in contrast, starts from large-scale
    pretrained transformer weights. The comparison therefore reflects a
    practical contrast between a lightweight non-pretrained neural model and
    a pretrained transformer, rather than a perfectly symmetric architecture
    comparison.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = CNN_CONFIG["embedding_dim"],
        num_filters: int = CNN_CONFIG["num_filters"],
        kernel_sizes: tuple[int, ...] = CNN_CONFIG["kernel_sizes"],
        dropout: float = CNN_CONFIG["dropout"],
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        self.convolutions = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=num_filters,
                    kernel_size=kernel_size,
                )
                for kernel_size in kernel_sizes
            ]
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            in_features=num_filters * len(kernel_sizes),
            out_features=num_classes,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Run the CNN forward pass and return unnormalized class logits.
        """
        embedded = self.embedding(input_ids)
        embedded = embedded.permute(0, 2, 1)

        convolved = [
            torch.relu(convolution(embedded))
            for convolution in self.convolutions
        ]

        pooled = [
            torch.max(feature_map, dim=2).values
            for feature_map in convolved
        ]

        concatenated = torch.cat(pooled, dim=1)
        dropped = self.dropout(concatenated)

        return self.classifier(dropped)


def initialize_roberta_model(
    num_classes: int,
) -> PreTrainedModel:
    """
    Load the configured RoBERTa checkpoint for sequence classification.
    """
    pretrained_kwargs = {
        "num_labels": num_classes,
    }

    if ROBERTA_CONFIG["revision"] is not None:
        pretrained_kwargs["revision"] = ROBERTA_CONFIG["revision"]

    model = AutoModelForSequenceClassification.from_pretrained(
        EXPERIMENT_CONFIG["roberta_checkpoint"],
        **pretrained_kwargs,
    )

    return model
