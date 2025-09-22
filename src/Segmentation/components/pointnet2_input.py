import os
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from src.Segmentation.utils.common import create_directories
from src.Segmentation import logger
from src.Segmentation.entity.config_entity import Pointnet2InputConfig

# ======================================================
# Helper Functions
# ======================================================

def farthest_point_sample_gpu(xyz, npoint):
    """
    Perform Farthest Point Sampling (FPS) on a point cloud to downsample points.

    Args:
        xyz: (N, 3) tensor of point coordinates
        npoint: number of points to sample

    Returns:
        sampled_idx: indices of sampled points
    """
    N, _ = xyz.shape
    device = xyz.device
    sampled_idx = torch.zeros(npoint, dtype=torch.long, device=device)
    distances = torch.ones(N, device=device) * 1e10  # initialize distances to inf
    farthest = torch.randint(0, N, (1,), device=device).item()

    for i in range(npoint):
        sampled_idx[i] = farthest
        centroid = xyz[farthest].unsqueeze(0)
        dist = torch.sum((xyz - centroid) ** 2, dim=1)
        distances = torch.minimum(distances, dist)  # update min distance
        farthest = torch.argmax(distances).item()   # pick next farthest point

    return sampled_idx

def augment_pointcloud(coords, features):
    """
    Apply random rotation and jitter to point cloud for data augmentation.

    Args:
        coords: (N, 3) numpy array of coordinates
        features: (N, F) point features (colors/normals/etc.)

    Returns:
        augmented coords and features
    """
    theta = np.random.uniform(0, 2 * np.pi)
    rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    coords = coords @ rot_matrix.T                     # rotate around Z-axis
    coords += np.random.normal(0, 0.01, coords.shape) # jitter
    return coords, features

# ======================================================
# Dataset for NPZ chunks
# ======================================================
class PointCloudChunkNPZDataset(Dataset):
    """
    Simple Dataset wrapper for pre-chunked .npz point cloud files.
    Can optionally return raw numpy points for testing/inference.
    """
    def __init__(self, chunk_dir, return_raw_points=False):
        self.files = sorted([f for f in os.listdir(chunk_dir) if f.endswith(".npz")])
        self.chunk_dir = chunk_dir
        self.return_raw_points = return_raw_points

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.chunk_dir, self.files[idx])
        data = np.load(path)
        coords = data['coords']
        features = data['features']
        labels = data['labels']

        points = np.hstack([coords, features])  # combine coords + features
        tensor_points = torch.tensor(points, dtype=torch.float32)
        tensor_labels = torch.tensor(labels, dtype=torch.long)

        if self.return_raw_points:
            return tensor_points, tensor_labels, points
        else:
            return tensor_points, tensor_labels

# ======================================================
# Dataset for dynamic chunking from raw patches
# ======================================================
class PointCloudChunkDataset(Dataset):
    """
    Dataset that loads point cloud patches and splits them into
    fixed-size chunks for training/evaluation.
    """
    def __init__(self, root_dir, chunk_size=4096, augment=False, device=torch.device("cpu")):
        self.root_dir = root_dir
        self.chunk_size = chunk_size
        self.augment = augment
        self.device = device
        self.chunks = []
        self.chunk_names = []
        self._load_and_chunk()  # load all patches and generate chunks

    def _load_patch(self, patch_path):
        """
        Load a patch from disk and assemble features.

        Returns:
            coords, features, labels
        """
        coords = np.load(os.path.join(patch_path, "coord_norm.npy")).astype(np.float32)
        colors = np.load(os.path.join(patch_path, "color.npy")).astype(np.float32)
        normals = np.load(os.path.join(patch_path, "normal.npy")).astype(np.float32)
        labels = np.load(os.path.join(patch_path, "segment.npy")).astype(np.int64)
        strength = np.load(os.path.join(patch_path, "strength.npy")).astype(np.float32)
        features = np.concatenate([colors, normals, strength[:, np.newaxis]], axis=1)
        return coords, features, labels

    def _chunk_patch(self, coords, features, labels, patch_name):
        """
        Split a single patch into fixed-size chunks using FPS.

        Returns:
            list of chunk dicts, list of chunk names
        """
        N = coords.shape[0]
        chunks, names = [], []
        remaining_idx = np.arange(N)
        coords_tensor = torch.tensor(coords, dtype=torch.float32, device=self.device)
        features_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)
        chunk_idx = 0

        while len(remaining_idx) > 0:
            # Sample indices
            if len(remaining_idx) < self.chunk_size:
                sampled_idx = torch.tensor(
                    np.random.choice(remaining_idx, self.chunk_size, replace=True),
                    device=self.device, dtype=torch.long
                )
            else:
                sampled_idx_local = farthest_point_sample_gpu(coords_tensor[remaining_idx], self.chunk_size)
                sampled_idx = torch.tensor(remaining_idx, device=self.device)[sampled_idx_local]

            # Extract chunk
            chunk_coords = coords_tensor[sampled_idx]
            chunk_features = features_tensor[sampled_idx]
            chunk_labels = labels_tensor[sampled_idx]

            chunks.append({"coords_norms": chunk_coords, "features": chunk_features, "labels": chunk_labels})
            names.append(f"{patch_name}_chunk_{chunk_idx}")
            chunk_idx += 1

            # Remove used indices
            remaining_idx = np.setdiff1d(remaining_idx, sampled_idx.cpu().numpy())

        return chunks, names

    def _load_and_chunk(self):
        """
        Load all patches in root_dir and split into chunks.
        """
        logger.info(f"📂 Loading patches from {self.root_dir}")
        area_folders = sorted([f for f in os.listdir(self.root_dir)
                               if os.path.isdir(os.path.join(self.root_dir, f))])
        total_chunks = 0
        for area in area_folders:
            coords, features, labels = self._load_patch(os.path.join(self.root_dir, area))
            chunks, names = self._chunk_patch(coords, features, labels, area)
            self.chunks.extend(chunks)
            self.chunk_names.extend(names)
            total_chunks += len(chunks)
            logger.info(f"✅ Processed {area}: {len(chunks)} chunks")
        logger.info(f"📊 Total chunks created: {total_chunks}")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        coords = chunk["coords_norms"]
        features = chunk["features"]
        labels = chunk["labels"]

        # Apply augmentation if enabled
        if self.augment:
            coords_np, features_np = coords.cpu().numpy(), features.cpu().numpy()
            coords_np, features_np = augment_pointcloud(coords_np, features_np)
            coords = torch.tensor(coords_np, dtype=torch.float32, device=self.device)
            features = torch.tensor(features_np, dtype=torch.float32, device=self.device)

        return torch.cat([coords, features], dim=1), labels

# ======================================================
# Pipeline for generating PointNet2 inputs
# ======================================================
class Pointnet2InputPipeline:
    """
    Pipeline to create fixed-size chunks of point clouds for PointNet2 input.
    Saves them as .npz files per split (train/test).
    """
    def __init__(self, params: Pointnet2InputConfig, device=torch.device("cpu")):
        self.params = params
        self.device = device

    def run(self):
        if not self.params.enabled:
            logger.info("⏭️ pointnet2 input chunking is disabled. Skipping...")
            return

        logger.info("🚀 Starting pointnet2 input Chunking Pipeline...")
        dataset_root = self.params.base_dir
        splits_to_process = []

        # Detect train/test directories
        for split in ["train", "test"]:
            split_dir = dataset_root / split / "chunks"
            if split_dir.exists() and any(split_dir.iterdir()):
                save_dir = dataset_root / split / "same_size_chunks"
                splits_to_process.append((split, split_dir, save_dir))
                logger.info(f"✅ Found '{split}' directory: {split_dir}")
            else:
                logger.warning(f"⚠️ '{split}' directory not found or empty: {split_dir}")

        if not splits_to_process:
            logger.error("❌ No valid dataset directories found. Exiting...")
            return

        # Process each split
        for split, input_dir, save_dir in splits_to_process:
            logger.info(f"📂 Processing '{split}' split...")
            create_directories([save_dir])
            dataset = PointCloudChunkDataset(input_dir, self.params.chunk_size, self.params.augment, device=self.device)
            area_to_chunks = defaultdict(list)

            for i, chunk_name in enumerate(dataset.chunk_names):
                area = chunk_name.split("_chunk_")[0]
                area_to_chunks[area].append((i, chunk_name))

            # Save each chunk as compressed npz
            for area, chunk_list in area_to_chunks.items():
                for idx, chunk_name in chunk_list:
                    new_filename = f"{chunk_name}.npz"
                    filepath = save_dir / new_filename
                    chunk = dataset.chunks[idx]
                    np.savez_compressed(filepath,
                                        coords=chunk['coords_norms'].cpu().numpy(),
                                        features=chunk['features'].cpu().numpy(),
                                        labels=chunk['labels'].cpu().numpy())

            # Save list of chunk names for reference
            np.save(save_dir / "chunk_names.npy", np.array(dataset.chunk_names))
            logger.info(f"💾 Saved all same-size pointnet2 input for '{split}' split to {save_dir}")

        logger.info("🎯 Pointnet2 input chunking pipeline completed successfully!")
