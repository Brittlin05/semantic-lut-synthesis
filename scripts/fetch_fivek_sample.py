"""
Extract images + scene labels from the downloaded FiveK parquet shard.

Each row has:
  - original:  the flat/unretouched rendering (our LOG stand-in)
  - augmented: the expert-C retouched rendering (not used in Phase 1)
  - location / time / light / subject: integer label codes

We decode 'original' into a JPG per image and write a labels.csv mapping
filename -> human-readable label names (not the raw integer codes).
"""
import io
import csv
import pyarrow.parquet as pq
from PIL import Image

# these name lists come from the dataset's ClassLabel schema (ordered by index)
LOCATION_NAMES = ["outdoor", "indoor", "unknown"]
TIME_NAMES = ["day", "unknown", "dusk", "night"]
LIGHT_NAMES = ["sun_sky", "artificial", "unknown", "mixed"]
SUBJECT_NAMES = ["people", "man_made", "nature", "unknown", "animals", "abstract"]

SRC = "data/raw_parquet/test_shard0.parquet"
OUT_DIR = "data/frames"
LABELS_CSV = "data/labels.csv"

table = pq.read_table(SRC)
rows = table.to_pylist()
print(f"loaded {len(rows)} rows")

with open(LABELS_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "location", "time", "light", "subject"])

    for i, row in enumerate(rows):
        img_bytes = row["original"]["bytes"]
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        filename = f"frame_{i:03d}.jpg"
        img.save(f"{OUT_DIR}/{filename}", quality=95)

        writer.writerow([
            filename,
            LOCATION_NAMES[row["location"]],
            TIME_NAMES[row["time"]],
            LIGHT_NAMES[row["light"]],
            SUBJECT_NAMES[row["subject"]],
        ])

print(f"saved {len(rows)} images to {OUT_DIR}/ and labels to {LABELS_CSV}")
