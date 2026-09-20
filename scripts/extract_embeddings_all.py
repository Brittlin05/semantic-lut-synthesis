import csv
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print("using device:", device)

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
model = AutoModel.from_pretrained("facebook/dinov2-small").to(device)
model.eval()

with open("data/labels.csv", newline="") as f:
    rows = list(csv.DictReader(f))

filenames = []
embeddings = []

for row in rows:
    img = Image.open(f"data/frames/{row['filename']}").convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_vector = outputs.pooler_output[0].cpu().numpy()

    embeddings.append(cls_vector)
    filenames.append(row["filename"])

X = np.stack(embeddings)
filenames = np.array(filenames)

print("X shape:", X.shape)
np.save("data/X_all.npy", X)
np.save("data/filenames_all.npy", filenames)
print("saved data/X_all.npy and data/filenames_all.npy")