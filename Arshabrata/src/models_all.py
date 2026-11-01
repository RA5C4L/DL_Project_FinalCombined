import torch
from torch import nn


class TfidfMLP(nn.Module):
    """Report-specified 256-unit ReLU MLP with dropout for dense TF-IDF batches."""
    def __init__(self, input_dim: int, dropout: float = 0.5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 1)
        )

    def forward(self, features):
        return self.network(features).squeeze(1)


class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, filters: int = 128, dropout: float = 0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convolutions = nn.ModuleList(
            [nn.Conv1d(embedding_dim, filters, width) for width in (3, 4, 5)]
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(filters * 3, 1)

    def forward(self, tokens, lengths=None):
        embedded = self.embedding(tokens).transpose(1, 2)
        pooled = [torch.amax(torch.relu(conv(embedded)), dim=2) for conv in self.convolutions]
        return self.output(self.dropout(torch.cat(pooled, dim=1))).squeeze(1)
