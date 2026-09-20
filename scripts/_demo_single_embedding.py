import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print("using device:", device)

# the processor does resizing + normalization to match what DINOv2 was trained on
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
model = AutoModel.from_pretrained("facebook/dinov2-small").to(device)
model.eval()  # inference mode: disables dropout etc. (frozen, we never call .backward())

img = Image.open("data/frames/frame_000.jpg").convert("RGB")

inputs = processor(images=img, return_tensors="pt").to(device)
with torch.no_grad():  # frozen model: don't bother tracking gradients, saves memory/time
    outputs = model(**inputs)

print("last_hidden_state shape:", outputs.last_hidden_state.shape)  # [batch, tokens, dim]
print("pooler_output shape:", outputs.pooler_output.shape)          # [batch, dim] -- the CLS-based summary vector

cls_embedding = outputs.last_hidden_state[:, 0, :]  # token 0 is the CLS token
print("cls token (manual) shape:", cls_embedding.shape)
print("matches pooler_output?", torch.allclose(cls_embedding, outputs.pooler_output, atol=1e-4))
