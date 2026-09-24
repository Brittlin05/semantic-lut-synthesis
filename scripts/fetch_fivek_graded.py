import io
import os
import pyarrow.parquet as pq
from PIL import Image

os.makedirs("data/frames_graded", exist_ok=True)

table = pq.read_table("data/raw_parquet/test_shard0.parquet")
rows = table.to_pylist()

for i, row in enumerate(rows):
    img_bytes = row["augmented"]["bytes"]   # the expert-retouched version, unused until now
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img.save(f"data/frames_graded/frame_{i:03d}.jpg", quality=95)

print(f"saved {len(rows)} expert-retouched images to data/frames_graded/")