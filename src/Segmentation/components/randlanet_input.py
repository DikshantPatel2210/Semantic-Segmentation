import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from src.Segmentation import logger
from src.Segmentation.entity.config_entity import RandLANetInputConfig

# =========================================
# Utility to split a large point cloud into smaller sub-patches
# =========================================
def generate_sub_patches(coords, colors, normals, strength, labels, max_points=20000):
    """
    Splits a single chunk of points into smaller sub-patches for training.
    Args:
        coords: (N,3) coordinates of points
        colors: (N,3) color features
        normals: (N,3) normal vectors
        strength: (N,1) intensity/strength feature
        labels: (N,) segmentation labels
        max_points: maximum points per sub-patch
    Returns:
        List of dictionaries, each containing a sub-patch
    """
    N = coords.shape[0]
    sub_patches = []

    # If the chunk is already small enough, keep it as a single patch
    if N <= max_points:
        sub_patches.append({
            "coords": coords,
            "colors": colors,
            "normals": normals,
            "strength": strength,
            "labels": labels
        })
    else:
        # Split into multiple random sub-patches
        n_sub = (N + max_points - 1) // max_points  # ceil division
        for _ in range(n_sub):
            idx = np.random.choice(N, max_points, replace=False)
            sub_patches.append({
                "coords": coords[idx],
                "colors": colors[idx],
                "normals": normals[idx],
                "strength": strength[idx],
                "labels": labels[idx]
            })
    return sub_patches


# =========================================
# Dataset class for RandLANet sub-patches
# =========================================
class RandLANetPatchDataset(Dataset):
    """
    Generates sub-patches from pre-existing chunks on-the-fly.
    Each sub-patch contains coordinates, color, normals, strength, and labels.
    """
    def __init__(self, root_dir, use_norm=True, max_points=20000):
        # Load all folders/chunks in the root directory
        self.folders = sorted([d for d in glob.glob(os.path.join(root_dir, "*")) if os.path.isdir(d)])
        self.use_norm = use_norm
        self.max_points = max_points
        self.sub_patches = []

        for folder in self.folders:
            # Choose normalized or raw coordinates
            coord_file = "coord_norm.npy" if use_norm else "coord.npy"
            coords = np.load(os.path.join(folder, coord_file)).astype(np.float32)
            colors = np.load(os.path.join(folder, "color.npy")).astype(np.float32)
            normals = np.load(os.path.join(folder, "normal.npy")).astype(np.float32)
            strength = np.load(os.path.join(folder, "strength.npy")).astype(np.float32).reshape(-1, 1)
            labels = np.load(os.path.join(folder, "segment.npy")).astype(np.int64)

            if coords.shape[0] == 0:
                logger.warning(f"⚠️ Empty chunk in folder {folder}, skipping...")
                continue

            # Normalize colors to [0,1] if in [0,255] range
            if colors.max() > 1.5:
                colors = colors / 255.0

            # Normalize strength to [-1,1] to match neural network input
            s_min, s_max = strength.min(), strength.max()
            if s_max != s_min:
                strength = 2 * (strength - s_min) / (s_max - s_min) - 1
            else:
                strength = np.zeros_like(strength)

            # Ensure coordinates and normals are in shape [N,3]
            if coords.shape[1] != 3:
                coords = coords.T
            if normals.shape[1] != 3:
                normals = normals.T

            # Generate sub-patches for large chunks
            patches = generate_sub_patches(coords, colors, normals, strength, labels, self.max_points)
            self.sub_patches.extend(patches)

    def __len__(self):
        return len(self.sub_patches)

    def __getitem__(self, idx):
        """
        Returns a single sub-patch for training:
        - 'xyz': coordinates
        - 'feats': concatenated features (coords + color + normals + strength)
        - 'labels': segmentation labels
        """
        patch = self.sub_patches[idx]
        coords = patch["coords"]
        colors = patch["colors"]
        normals = patch["normals"]
        strength = patch["strength"]
        labels = patch["labels"]

        feats = np.concatenate([coords, colors, normals, strength], axis=1).astype(np.float32)
        return {
            "xyz": torch.from_numpy(coords),
            "feats": torch.from_numpy(feats),
            "labels": torch.from_numpy(labels)
        }


# =========================================
# RandLANet Input Pipeline
# =========================================
class RandLANetInputPipeline:
    """
    Prepares RandLANet datasets on-the-fly for training and testing.
    Does not save processed sub-patches to disk.
    """
    def __init__(self, params: RandLANetInputConfig):
        self.params = params
        self.datasets = {}

    def prepare_datasets(self):
        """
        Loads chunks and generates sub-patches for train/test splits.
        Returns a dictionary of PyTorch datasets that can be used directly for training.
        """
        dataset_root = self.params.base_dir
        if not dataset_root.exists():
            logger.error(f"❌ Base directory does not exist: {dataset_root}")
            return

        for split in ["train", "test"]:
            split_dir = dataset_root / split / "chunks"  # Use chunks as input
            if not split_dir.exists():
                logger.warning(f"⚠️ Split directory not found: {split_dir}")
                continue

            # Initialize dataset for this split
            dataset = RandLANetPatchDataset(
                split_dir, 
                use_norm=self.params.use_norm, 
                max_points=self.params.max_points
            )
            self.datasets[split] = dataset
            logger.info(f"✅ Prepared {len(dataset)} sub-patches for '{split}' split")

        return self.datasets
