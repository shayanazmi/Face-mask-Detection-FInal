"""
preprocess_images.py
Normalizes all images to a fixed size (224×224) while preserving aspect ratio.
Adds padding to maintain center alignment.
"""

import os
import cv2
import numpy as np


# NEW official project base directory
BASE_DIR = r"D:\college\sem4\project\Face mask Detection FInal"

# Path to the split dataset (train/valid/test)
SPLIT_PATH = os.path.join(BASE_DIR, "data", "split")

# Categories remain the same
CATEGORIES = ['with_mask', 'without_mask']


def normalize_aspect_ratio(root_path, target_size=(224, 224), padding_color=(0, 0, 0), force=False):
    marker_file = os.path.join(root_path, ".aspect_fixed")

    if os.path.exists(marker_file) and not force:
        print("Aspect ratio normalization already done. Skipping...")
        return

    tgt_w, tgt_h = target_size
    print(f"Normalizing images to {target_size} ...")

    for subset in ['train', 'valid', 'test']:
        subset_dir = os.path.join(root_path, subset)

        for category in CATEGORIES:
            cat_dir = os.path.join(subset_dir, category)
            if not os.path.exists(cat_dir):
                continue

            for fname in os.listdir(cat_dir):
                fpath = os.path.join(cat_dir, fname)

                try:
                    img = cv2.imread(fpath)
                    if img is None:
                        continue

                    h, w = img.shape[:2]
                    if (w, h) == (tgt_w, tgt_h):
                        continue

                    scale = min(tgt_w / w, tgt_h / h)
                    new_w, new_h = int(w * scale), int(h * scale)

                    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
                    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

                    canvas = np.full((tgt_h, tgt_w, 3), padding_color, dtype=np.uint8)
                    x_offset = (tgt_w - new_w) // 2
                    y_offset = (tgt_h - new_h) // 2

                    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

                    cv2.imwrite(fpath, canvas)

                except Exception as e:
                    print(f"Skipping {fpath}: {e}")

    open(marker_file, 'w').close()
    print("Aspect ratio normalization complete.")


if __name__ == "__main__":
    normalize_aspect_ratio(SPLIT_PATH)