import torch
import torch.nn as nn
import torch.nn.functional as F


class TAGLayer(nn.Module):
    def __init__(self, dim=512, nhead=8):
        super().__init__()
        self.nhead = nhead
        self.head_dim = dim // nhead
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x, adj):
        C = x.shape[0]
        
        q = self.q(x).view(C, self.nhead, self.head_dim).transpose(0,1)
        k = self.k(x).view(C, self.nhead, self.head_dim).transpose(0,1)
        v = self.v(x).view(C, self.nhead, self.head_dim).transpose(0,1)

        # 核心：adj 作为软偏置，不屏蔽任何类别
        attn = (q @ k.transpose(-2,-1)) * self.scale + adj.unsqueeze(0)
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(0,1).reshape(C, -1)
        x = self.norm1(x + self.proj(out))
        x = self.norm2(x + self.ffn(x))
        return x

class TAG_GraphTransformer(nn.Module):
    def __init__(self, dim=768, nhead=8, num_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([TAGLayer(dim, nhead) for _ in range(num_layers)])

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
        return x

