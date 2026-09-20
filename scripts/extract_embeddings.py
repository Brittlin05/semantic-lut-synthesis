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

# 1. Read labels.csv and keep only rows where location isn't "unknown"
rows = []
with open("data/labels.csv", newline="") as f:
    for row in csv.DictReader(f):
        if row["location"] != "unknown":
            rows.append(row)

print(f"using {len(rows)} labeled images (skipped 'unknown' location)")

# 2. Loop over each image, get its DINOv2 embedding
embeddings = []
labels = []

for row in rows:
    img = Image.open(f"data/frames/{row['filename']}").convert("RGB")

    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    cls_vector = outputs.pooler_output[0]      # shape [384], still a GPU tensor
    cls_vector = cls_vector.cpu().numpy()       # move to CPU, convert to plain numpy array

    embeddings.append(cls_vector)
    labels.append(row["location"])              # "outdoor" or "indoor"

# 3. Stack the list of 384-length vectors into one [N, 384] array
X = np.stack(embeddings)
y = np.array(labels)

print("X shape:", X.shape)
print("y shape:", y.shape)

# 4. Save so we never have to re-run DINOv2 again for this data
np.save("data/X_location.npy", X)
np.save("data/y_location.npy", y)
print("saved data/X_location.npy and data/y_location.npy")