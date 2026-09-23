import torch
import torch.nn as nn


class LUTQueryCrossAttention(nn.Module):
    """
    Cross-attention fusion: a learned, fixed-size set of "LUT queries" attends
    over a variable-size set of scene tokens (DINOv2 patch tokens + semantic
    mask tokens) and produces a fixed-size fused representation.
    """

    def __init__(self, dim, num_queries):
        super().__init__()
        self.lut_queries = nn.Parameter(torch.randn(num_queries, dim))
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)

    def forward(self, scene_tokens):
        d = scene_tokens.shape[-1]
        Q = self.W_q(self.norm_q(self.lut_queries))
        K = self.W_k(self.norm_kv(scene_tokens))
        V = self.W_v(self.norm_kv(scene_tokens))
        scores = Q @ K.T / (d ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        output = weights @ V
        return output, weights