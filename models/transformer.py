import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:,:x.size(1)]


class Transformer(nn.Module):
    def __init__(self, dims: int, hdims: int, heads: int, layers: int, classes: int):
        super().__init__()
        self.classes = classes

        self.emb = nn.Embedding(classes, dims)
        self.pos_emb = PositionalEncoding(dims)
        layer = nn.TransformerEncoderLayer(
            d_model=dims, 
            nhead=heads, 
            dim_feedforward=hdims, 
            dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(layer, layers)
        self.postnorm = nn.LayerNorm(dims)
        self.cls_head = nn.Linear(dims, classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, tokens, labels):
        tokens = self.pos_emb(self.emb(tokens))
        a = self.transformer(tokens)
        a = self.postnorm(a)

        pred = self.cls_head(a)
        loss = self.criterion(pred.reshape(-1, self.classes), labels.reshape(-1))
        acc = (pred.argmax(dim=-1) == labels).float().mean()
        return loss, acc
