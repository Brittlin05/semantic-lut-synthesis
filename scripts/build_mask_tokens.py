import os
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print("using device:", device)

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
model = AutoModel.from_pretrained("facebook/dinov2-small").to(device)
model.eval()

os.makedirs("data/mask_tokens", exist_ok=True)

mask_files = sorted(os.listdir("data/masks"))  # only the frames we've segmented so far

for mask_fname in mask_files:
    fname = mask_fname.replace(".npy", ".jpg")
    img = Image.open(f"data/frames/{fname}").convert("RGB")

    # 1. get DINOv2's 16x16 grid of patch tokens (one 384-vector per small image region)
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    patch_tokens = outputs.last_hidden_state[0, 1:, :]  # drop the CLS token, keep the 256 patch tokens
    grid_size = int(patch_tokens.shape[0] ** 0.5)         # 16
    patch_grid = patch_tokens.reshape(grid_size, grid_size, -1).cpu().numpy()  # [16, 16, 384]

    # 2. shrink the segmentation mask down to the SAME 16x16 grid, so patch (r,c)
    #    and mask cell (r,c) refer to the same square region of the image
    class_map_small = np.load(f"data/masks/{mask_fname}")
    mask_img = Image.fromarray(class_map_small).resize((grid_size, grid_size), resample=Image.NEAREST)
    class_grid = np.array(mask_img)  # [16, 16]

    # 3. group patches by class label, average each group
    class_ids = sorted(set(class_grid.flatten().tolist()))
    tokens = []
    for class_id in class_ids:
        matches = (class_grid == class_id)         # [16, 16] boolean: True where this class appears
        matching_patches = patch_grid[matches]      # [num_matching, 384] -- just this class's patches
        tokens.append(matching_patches.mean(axis=0))  # average them into one 384-vector

    tokens = np.stack(tokens)              # [num_classes_present, 384]
    class_ids = np.array(class_ids)

    np.savez(f"data/mask_tokens/{fname.replace('.jpg', '.npz')}", class_ids=class_ids, tokens=tokens)
    print(f"{fname}: {len(class_ids)} mask tokens -> shape {tokens.shape}")

print("done")