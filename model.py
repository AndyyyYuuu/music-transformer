import torch
from torch import nn, Tensor
import numpy as np
import math


class Composer(torch.nn.Module):
    def __init__(self, num_notes, emb_size, num_heads, hidden_size, num_layers, dropout_chance=0.2):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout_chance = dropout_chance
        self.pos_encoder = PositionalEncoding(emb_size, dropout_chance)
        encoder_layers = nn.TransformerEncoderLayer(emb_size, num_heads, hidden_size, dropout_chance)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.embedding = nn.Embedding(num_notes, emb_size)
        self.emb_size = emb_size
        self.linear = nn.Linear(emb_size, num_notes)

        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-init_range, init_range)


    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(self.emb_size)
        x = self.pos_encoder(x)
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(len(x))  # .to(device)
        output = self.transformer_encoder(x, mask)
        output = self.linear(output)
        output = output[:, -1, :]
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout_chance, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_chance)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2) * (-math.log(10000.0)/emb_size))
        pe = torch.zeros(max_len, 1, emb_size)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x += self.pe[:x.size(0)]
        return self.dropout(x)
