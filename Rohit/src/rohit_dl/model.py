import torch
from torch import nn


class TextCNN(nn.Module):
    """Embedding -> parallel valid convolutions -> ReLU -> masked max -> dropout -> logits."""
    def __init__(self, vocab_size, embedding_dim=128, num_filters=128,
                 kernel_sizes=(3, 4, 5), dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(embedding_dim, num_filters, k) for k in kernel_sizes)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters*len(kernel_sizes), 2)

    def forward(self, ids):
        embeddings = self.embedding(ids).transpose(1, 2)
        lengths = ids.ne(0).sum(1)
        pooled = []
        for conv in self.convs:
            values = torch.relu(conv(embeddings))
            # Short inputs use one padded window; trailing PAD-only windows never win pooling.
            valid_count = (lengths-conv.kernel_size[0]+1).clamp(min=1)
            mask = torch.arange(values.shape[2], device=ids.device)[None, :] < valid_count[:, None]
            values = values.masked_fill(~mask[:, None, :], -torch.inf)
            pooled.append(values.amax(dim=2))
        return self.classifier(self.dropout(torch.cat(pooled, dim=1)))
