import os
import numpy as np
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print("using device:", device)

processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512").to(device)
model.eval()

id2label = model.config.id2label

os.makedirs("data/masks", exist_ok=True)
os.makedirs("data/mask_overlays", exist_ok=True)

frame_files = sorted(os.listdir("data/frames"))[:10]  # just first 10 for now, to sanity-check

for fname in frame_files:
    img = Image.open(f"data/frames/{fname}").convert("RGB")

    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # collapse 150 channels -> 1 class-id per pixel WHILE STILL SMALL (before any upsampling)
    logits = outputs.logits  # [1, 150, h_small, w_small]
    class_map_small = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)  # [h_small, w_small]

    present_classes = sorted(set(class_map_small.flatten().tolist()))
    print(f"{fname}: classes present = {[id2label[c] for c in present_classes]}")

    np.save(f"data/masks/{fname.replace('.jpg', '.npy')}", class_map_small)

    # for a human-viewable overlay only: resize the now-single-channel mask up with plain
    # nearest-neighbor image resizing (cheap -- no GPU, no 150-channel blowup)
    mask_img_full = Image.fromarray(class_map_small).resize(img.size, resample=Image.NEAREST)
    class_map_full = np.array(mask_img_full)

    rng = np.random.RandomState(0)
    palette = rng.randint(0, 255, size=(150, 3), dtype=np.uint8)
    color_mask = palette[class_map_full]
    overlay = (0.5 * np.array(img) + 0.5 * color_mask).astype(np.uint8)
    Image.fromarray(overlay).save(f"data/mask_overlays/{fname}")

print("done — check data/mask_overlays/ for visual sanity check")