"""
check_duplicates.py

Utility script for detecting duplicate or near-duplicate images
between two dataset folders using the CNN method from imagededup.

This script does NOT alter the dataset; it only identifies duplicates.
"""

import os
from imagededup.methods import CNN


def find_duplicates(folder1: str, folder2: str, threshold: float = 0.95):
    """
    Find duplicate or near-duplicate images between two folders
    based on cosine similarity of CNN embeddings.

    Parameters
    ----------
    folder1 : str
        Path to the first image folder.
    folder2 : str
        Path to the second image folder.
    threshold : float, optional
        Similarity threshold above which images are considered duplicates.
        Default is 0.95.

    Returns
    -------
    list of tuple
        List of (image_from_folder1, image_from_folder2) duplicates.
    """

    cnn = CNN()

    # Encode images from each folder
    encodings1 = cnn.encode_images(image_dir=folder1)
    encodings2 = cnn.encode_images(image_dir=folder2)

    duplicates = []
    for img1, emb1 in encodings1.items():
        for img2, emb2 in encodings2.items():
            similarity = cnn.compute_similarity(emb1, emb2)

            if similarity > threshold:
                duplicates.append((img1, img2))

    return duplicates


if __name__ == "__main__":
    # Example usage — replace with your dataset paths
    folder_with_mask = r"D:\college\sem4\project\face_mask_detection\Covid-19-PIS.v2i.folder2\train\with_mask"
    folder_without_mask = r"D:\college\sem4\project\face_mask_detection\Covid-19-PIS.v2i.folder2\train\without_mask"

    duplicates_found = find_duplicates(folder_with_mask, folder_without_mask)

    print("Duplicates found:", duplicates_found)