import torch
import torch.nn as nn

class LUTQueryCrossAttention(nn.Module):
    def __init__(self, dim, num_queries):
        super().__init__()
        # the learned "LUT query" set -- NOT derived from any image, just learned numbers
        self.lut_queries = nn.Parameter(torch.randn(num_queries, dim))

        # learned projections: transform raw tokens into "good" queries/keys/values
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)

    def forward(self, scene_tokens):
        # scene_tokens: [num_tokens, dim] -- the patch tokens + mask tokens for ONE image
        d = scene_tokens.shape[-1]

        Q = self.W_q(self.lut_queries)     # [num_queries, dim]
        K = self.W_k(scene_tokens)         # [num_tokens, dim]
        V = self.W_v(scene_tokens)         # [num_tokens, dim]

        scores = Q @ K.T / (d ** 0.5)      # [num_queries, num_tokens] -- every query vs every token
        weights = torch.softmax(scores, dim=-1)  # softmax over the TOKEN axis, one weight-set per query

        output = weights @ V               # [num_queries, dim]
        return output, weights

# sanity check on random, correctly-SHAPED fake data (not real tokens yet)
dim = 384          # matches DINOv2's embedding size
num_queries = 8    # just a placeholder number for now, we'll decide the real value later
num_scene_tokens = 20  # e.g. 256 patch tokens + a handful of mask tokens, shrunk for this quick test

module = LUTQueryCrossAttention(dim=dim, num_queries=num_queries)
fake_scene_tokens = torch.randn(num_scene_tokens, dim)

output, weights = module(fake_scene_tokens)
print("output shape:", output.shape)     # expect [8, 384]
print("weights shape:", weights.shape)   # expect [8, 20]
print("each query's weights sum to 1:", weights.sum(dim=-1))