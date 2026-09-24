import os
import numpy as np
from PIL import Image

LUT_SIZE = 17

os.makedirs("data/target_luts", exist_ok=True)

frame_files = sorted(os.listdir("data/frames"))
skipped = []

for fname in frame_files:
    img_in_pil = Image.open(f"data/frames/{fname}").convert("RGB")
    img_out_pil = Image.open(f"data/frames_graded/{fname}").convert("RGB")

    w_in, h_in = img_in_pil.size
    w_out, h_out = img_out_pil.size
    ratio_in, ratio_out = w_in / h_in, w_out / h_out

    if abs(ratio_in - ratio_out) > 0.05:
        # orientation mismatch (e.g. one is portrait, one landscape) -- can't
        # safely align pixels, so skip rather than risk corrupted training data
        skipped.append((fname, img_in_pil.size, img_out_pil.size))
        continue

    # same aspect ratio, just possibly a slightly different pixel size --
    # safe to resize the graded image onto the original's exact grid
    img_out_pil = img_out_pil.resize((w_in, h_in))

    img_in = np.array(img_in_pil) / 255.0
    img_out = np.array(img_out_pil) / 255.0

    indices = np.round(img_in * (LUT_SIZE - 1)).astype(int)

    lut_sum = np.zeros((LUT_SIZE, LUT_SIZE, LUT_SIZE, 3))
    lut_count = np.zeros((LUT_SIZE, LUT_SIZE, LUT_SIZE))

    r_idx, g_idx, b_idx = indices[..., 0].flatten(), indices[..., 1].flatten(), indices[..., 2].flatten()
    out_flat = img_out.reshape(-1, 3)

    np.add.at(lut_sum, (r_idx, g_idx, b_idx), out_flat)
    np.add.at(lut_count, (r_idx, g_idx, b_idx), 1)

    identity = np.stack(np.meshgrid(
        np.linspace(0, 1, LUT_SIZE),
        np.linspace(0, 1, LUT_SIZE),
        np.linspace(0, 1, LUT_SIZE),
        indexing="ij",
    ), axis=-1)[..., [1, 0, 2]]

    lut = identity.copy()
    has_data = lut_count > 0
    lut[has_data] = lut_sum[has_data] / lut_count[has_data, None]

    coverage = has_data.sum() / has_data.size
    np.save(f"data/target_luts/{fname.replace('.jpg', '.npy')}", lut.astype(np.float32))
    print(f"{fname}: {coverage:.1%} grid coverage")

print(f"\ndone -- {len(skipped)} skipped due to orientation mismatch:")
for s in skipped:
    print(" ", s)