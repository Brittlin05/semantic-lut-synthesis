import numpy as np
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel, SegformerForSemanticSegmentation
from PIL import Image


class LUTQueryCrossAttention(nn.Module):
    def __init__(self, dim, num_queries):
        super().__init__()
        self.lut_queries = nn.Parameter(torch.randn(num_queries, dim))
        self.norm_q = nn.LayerNorm(dim)      # normalize queries before projecting
        self.norm_kv = nn.LayerNorm(dim)     # normalize scene tokens before projecting
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

class LUTQueryCrossAttention(nn.Module):
    def __init__(self, dim, num_queries):
        super().__init__()
        self.lut_queries = nn.Parameter(torch.randn(num_queries, dim))
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)

    def forward(self, scene_tokens):
        d = scene_tokens.shape[-1]
        Q = self.W_q(self.lut_queries)
        K = self.W_k(scene_tokens)
        V = self.W_v(scene_tokens)
        scores = Q @ K.T / (d ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        output = weights @ V
        return output, weights

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
dinov2 = AutoModel.from_pretrained("facebook/dinov2-small").to(device)
dinov2.eval()
seg_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
id2label = seg_model.config.id2label

fname = "frame_004.jpg"  # the busy park scene, has 10 mask tokens -- good test case

# 1. real patch tokens for this image
img = Image.open(f"data/frames/{fname}").convert("RGB")
inputs = processor(images=img, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = dinov2(**inputs)
patch_tokens = outputs.last_hidden_state[0, 1:, :]  # [256, 384]

# 2. real mask tokens, already built and saved in Phase 2's previous step
mask_data = np.load(f"data/mask_tokens/{fname.replace('.jpg', '.npz')}")
mask_tokens = torch.tensor(mask_data["tokens"], dtype=torch.float32).to(device)  # [K, 384]
mask_class_names = [id2label[c] for c in mask_data["class_ids"]]

# 3. build labels for every token, so we can interpret attention weights afterward
patch_labels = [f"patch ({i // 16},{i % 16})" for i in range(patch_tokens.shape[0])]
mask_labels = [f"mask token: {name}" for name in mask_class_names]
all_labels = patch_labels + mask_labels

# 4. concatenate patch tokens + mask tokens into one set of keys/values
scene_tokens = torch.cat([patch_tokens, mask_tokens], dim=0)  # [256 + K, 384]
print("scene_tokens shape:", scene_tokens.shape)

# 5. run through the (still untrained) cross-attention module
module = LUTQueryCrossAttention(dim=384, num_queries=8).to(device)
output, weights = module(scene_tokens)
print("output shape:", output.shape)

# 6. interpretability check: what is each query attending to most, right now?
weights = weights.detach().cpu().numpy()
for q in range(weights.shape[0]):
    top3 = np.argsort(weights[q])[::-1][:3]
    print(f"query {q} top attended tokens:", [(all_labels[i], round(float(weights[q][i]), 3)) for i in top3])

    