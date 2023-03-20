import torch
import torch.nn as nn
import torch.nn.functional as F


class Bind(nn.Module):
    def __init__(self, dims: int, hdims: int):
        super().__init__()

        self.prenorm = nn.LayerNorm(dims)
        self.linear1 = nn.Linear(dims * 2, hdims * 2)
        self.linear2 = nn.Linear(hdims, dims * 2)

    def forward(self, a, b):
        ab = torch.stack([a, b], dim=-2)
        ab = self.prenorm(ab)
        ab = ab.reshape(*ab.shape[:-2], -1)
        h = self.linear1(ab)
        h1, h2 = h.chunk(2, dim=-1)
        h = F.gelu(h1) * h2
        h = self.linear2(h)
        h, out = h.chunk(2, dim=-1)
        h = h + a + b
        return h, out

class BindRNN(nn.Module):
    def __init__(self, dims: int, hdims: int, classes: int):
        super().__init__()

        self.emb = nn.Embedding(classes, dims)
        self.start_token = nn.Parameter(torch.randn(1, dims))
        self.bind = Bind(dims, hdims)
        self.postnorm = nn.LayerNorm(dims)
        self.cls_head = nn.Linear(dims, classes)

    def forward(self, tokens):
        tokens = self.emb(tokens)
        a = self.start_token.expand(tokens.shape[0], -1)
        out_list = []
        for i in range(tokens.shape[-2]):
            b = tokens[...,i,:]
            a, out = self.bind(a, b)
            out_list.append(out)
        out_list = torch.stack(out_list, dim=-2)
        out = self.postnorm(out_list)
        return self.cls_head(out)
