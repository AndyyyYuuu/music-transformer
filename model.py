import torch
import numpy as np


class Composer(torch.nn.Module):
    def __init__(self, num_notes, layers, hidden_size=256, dropout_chance=0.2):
        super().__init__()

        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=layers, batch_first=True, dropout=dropout_chance)
        self.dropout_chance = dropout_chance
        self.dropout = torch.nn.Dropout(dropout_chance)
        self.linear = torch.nn.Linear(hidden_size, num_notes)
        self.layers = layers
        self.hidden_size = hidden_size

    def forward(self, x):
        x, _ = self.lstm(x)
        # take only the last output
        x = x[:, -1, :]
        # produce output
        x = self.linear(self.dropout(x))
        return x