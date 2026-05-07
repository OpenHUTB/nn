import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def weights_init(m):
    """Xavier uniform initialization for Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class WordEmbedding(nn.Module):
    def __init__(self, vocab_length, embedding_dim, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_length, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.embedding(x))


class PoemLSTM(nn.Module):
    def __init__(self, vocab_len, embedding_dim=128, hidden_dim=256,
                 num_layers=2, dropout=0.2, embedding_dropout=0.1):
        super().__init__()
        self.vocab_len = vocab_len
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.word_embedding = WordEmbedding(vocab_len, embedding_dim, embedding_dropout)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Linear(hidden_dim, vocab_len)
        self.dropout = nn.Dropout(dropout)

        self.apply(weights_init)

    def forward(self, x, hidden=None):
        """
        Args:
            x: (batch, seq_len) token indices
            hidden: tuple (h0, c0) or None
        Returns:
            logits: (batch, seq_len, vocab_len)
            hidden: (h_n, c_n) for next step
        """
        embed = self.word_embedding(x)  # (batch, seq_len, embed_dim)

        if hidden is None:
            hidden = self._init_hidden(x.size(0), x.device)

        output, hidden = self.lstm(embed, hidden)  # (batch, seq_len, hidden_dim)

        output = self.dropout(output)
        logits = self.fc(output)  # (batch, seq_len, vocab_len)

        return logits, hidden

    def _init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h0, c0)

    def generate(self, start_token_idx, end_token_idx, word_int_map, vocabularies,
                 max_len=60, temperature=1.0, device='cpu'):
        """Generate a poem given a start token."""
        self.eval()
        poem_indices = [start_token_idx]
        hidden = None

        with torch.no_grad():
            for _ in range(max_len):
                x = torch.tensor([[poem_indices[-1]]], dtype=torch.long, device=device)
                logits, hidden = self(x, hidden)
                logits = logits[0, -1, :] / temperature
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                idx = np.random.choice(len(probs), p=probs)
                if idx == end_token_idx:
                    break
                poem_indices.append(idx)

        return poem_indices
