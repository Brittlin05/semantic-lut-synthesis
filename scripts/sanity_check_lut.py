import numpy as np
from PIL import Image

LUT_SIZE = 17

def apply_lut(img_array, lut):
    indices = np.round(img_array * (LUT_SIZE - 1)).astype(int)
    r_idx, g_idx, b_idx = indices[..., 0], indices[..., 1], indices[..., 2]
    return lut[r_idx, g_idx, b_idx]

for fname in ["frame_004.jpg", "frame_064.jpg", "frame_063.jpg"]:
    lut = np.load(f"data/target_luts/{fname.replace('.jpg', '.npy')}")
    img_in = np.array(Image.open(f"data/frames/{fname}").convert("RGB")) / 255.0

    reconstructed = apply_lut(img_in, lut)
    reconstructed_img = (np.clip(reconstructed, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(reconstructed_img).save(f"data/target_luts_{fname.replace('.jpg', '_reconstructed.jpg')}")

print("done")
