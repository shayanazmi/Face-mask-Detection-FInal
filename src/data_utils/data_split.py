"""
dataset_split.py
Splits the raw dataset into Train / Validation / Test (70/15/15).
Logic unchanged from original.
"""

import os
import shutil
import random

RAW_PATH = "D:/college/sem4/project/face_mask_detection/Covid-19-PIS.v2i.folder2/train"
BASE_DIR = r"D:\college\sem4\project\Face mask Detection FInal"

SPLIT_PATH = os.path.join(BASE_DIR, "data", "split")

CATEGORIES = ['with_mask', 'without_mask']


def split_dataset():
    if os.path.exists(SPLIT_PATH):
        print("Dataset is already split. Skipping...")
        return

    for category in CATEGORIES:
        os.makedirs(os.path.join(SPLIT_PATH, 'train', category), exist_ok=True)
        os.makedirs(os.path.join(SPLIT_PATH, 'valid', category), exist_ok=True)
        os.makedirs(os.path.join(SPLIT_PATH, 'test', category), exist_ok=True)

        src_dir = os.path.join(RAW_PATH, category)
        images = os.listdir(src_dir)

        random.seed(42)
        random.shuffle(images)

        train_end = int(len(images) * 0.7)
        val_end = train_end + int(len(images) * 0.15)

        for idx, img in enumerate(images):
            src_path = os.path.join(src_dir, img)

            if idx < train_end:
                dst_path = os.path.join(SPLIT_PATH, 'train', category, img)
            elif idx < val_end:
                dst_path = os.path.join(SPLIT_PATH, 'valid', category, img)
            else:
                dst_path = os.path.join(SPLIT_PATH, 'test', category, img)

            shutil.copy(src_path, dst_path)

    print("Dataset split completed successfully (70/15/15).")


if __name__ == "__main__":
    split_dataset()