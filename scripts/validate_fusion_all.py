import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from fusion_module import LUTQueryCrossAttention
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
dinov2 = AutoModel.from_pretrained("facebook/dinov2-small").to(device)
dinov2.eval()

module = LUTQueryCrossAttention(dim=384, num_queries=8).to(device)

token_counts = []
errors = 0

mask_files = sorted(os.listdir("data/mask_tokens"))
for mask_fname in mask_files:
    fname = mask_fname.replace(".npz", ".jpg")
    try:
        img = Image.open(f"data/frames/{fname}").convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = dinov2(**inputs)
        patch_tokens = outputs.last_hidden_state[0, 1:, :]

        mask_data = np.load(f"data/mask_tokens/{mask_fname}")
        mask_tokens = torch.tensor(mask_data["tokens"], dtype=torch.float32).to(device)

        scene_tokens = torch.cat([patch_tokens, mask_tokens], dim=0)
        output, weights = module(scene_tokens)

        assert output.shape == (8, 384), f"bad output shape: {output.shape}"
        assert not torch.isnan(output).any(), "NaN in output!"
        assert torch.allclose(weights.sum(dim=-1), torch.ones(8, device=device), atol=1e-4), "weights don't sum to 1!"

        token_counts.append(scene_tokens.shape[0])
    except Exception as e:
        print(f"FAILED on {fname}: {e}")
        errors += 1

print(f"\nprocessed {len(mask_files)} images, {errors} errors")
print(f"scene token count range: min={min(token_counts)}, max={max(token_counts)}, mean={np.mean(token_counts):.1f}")