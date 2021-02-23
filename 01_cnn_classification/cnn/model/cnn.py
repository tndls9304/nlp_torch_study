import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1d(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx, channel_size=1):
        super().__init__()
        self.channel_size = channel_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if channel_size == 2:
            self.embedding2 = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.convs = nn.ModuleList([
                                    nn.Conv1d(in_channels=embedding_dim,
                                              out_channels=n_filters,
                                              kernel_size=fs,
                                              padding=fs-1)
                                    for fs in filter_sizes
                                    ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [batch size, sent len]
        embedded = self.embedding(text)
        # embedded = [batch size, sent len, emb dim]
        embedded = embedded.permute(0, 2, 1)
        # embedded = [batch size, emb dim, sent len]
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        if self.channel_size == 2:
            embedded2 = self.embedding2(text)
            embedded2 = embedded2.permute(0, 2, 1)
            conved2 = [F.relu(conv(embedded2)) for conv in self.convs]
            conved = [conved[i] + conved2[i] for i in range(len(conved))]
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]
        return self.fc(cat)
