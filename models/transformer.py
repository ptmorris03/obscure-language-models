import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, dims: int, hdims: int, heads: int, layers: int, classes: int):
        super().__init__()
        self.classes = classes

        self.emb = nn.Embedding(classes, dims)
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
        tokens = self.emb(tokens)
        a = self.transformer(tokens)
        a = self.postnorm(a)

        pred = self.cls_head(a)
        loss = self.criterion(pred.reshape(-1, self.classes), labels.reshape(-1))
        acc = (pred.argmax(dim=-1) == labels).float().mean()
        return loss, acc
