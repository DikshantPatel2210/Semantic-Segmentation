
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.Segmentation import logger
from src.Segmentation.entity.config_entity import NormalizationConfig


# ======================================================
# Class for Normalizing Point Cloud Data
# ======================================================
class Normalization:
    def __init__(self, params: NormalizationConfig):
        """
        Initialize Normalization class with configuration parameters.

        Args:
            params (NormalizationConfig): Contains attributes like:
                - coord_method, color_method, intensity_method
                - targets: which data to normalize ["coords", "color", "intensity"]
                - base_dir: root directory of dataset
                - vis_dir: directory to save visualizations
                - enabled: whether normalization is active
        """
        self.params = params
        # Ensure visualization directory exists
        self.params.vis_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------
    # Normalize coordinates (generic)
    # --------------------------
    def normalize_points(self, points: np.ndarray) -> np.ndarray:
        """Normalize coordinates based on the chosen method (minmax, zscore, custom)."""
        method = self.params.coord_method.lower()
        if method == "minmax":
            min_vals = points.min(axis=0)
            max_vals = points.max(axis=0)
            return (points - min_vals) / (max_vals - min_vals + 1e-8)
        elif method == "zscore":
            mean = points.mean(axis=0)
            std = points.std(axis=0)
            return (points - mean) / (std + 1e-8)
        elif method == "custom":
            coords_norm, _, _ = self.custom_normalize(points)
            return coords_norm
        else:
            logger.warning(f"⚠️ Unknown coord normalization method '{method}'. Skipping.")
            return points

    # --------------------------
    # Custom normalization for coordinates
    # --------------------------
    def custom_normalize(self, coords: np.ndarray):
        """
        Perform custom normalization:
            - Center XY coordinates
            - Shift Z to start at zero
            - Scale by maximum range across all axes
        """
        min_vals = coords.min(axis=0)
        max_vals = coords.max(axis=0)
        center_xy = (min_vals[:2] + max_vals[:2]) / 2.0
        coords_centered = coords.copy()
        coords_centered[:, :2] -= center_xy  # Center XY
        z_min = coords_centered[:, 2].min()
        coords_centered[:, 2] -= z_min       # Shift Z
        max_range = np.abs(coords_centered).max()
        coords_norm = coords_centered / max_range if max_range > 0 else coords_centered
        return coords_norm, np.hstack([center_xy, z_min]), max_range

    # --------------------------
    # Normalize other arrays (color/intensity)
    # --------------------------
    def normalize_array(self, data: np.ndarray, method: str) -> np.ndarray:
        """
        Generic normalization for color or intensity arrays.

        Args:
            data (np.ndarray): Input array
            method (str): "minmax" or "zscore"

        Returns:
            np.ndarray: Normalized array
        """
        method = method.lower()
        if method == "minmax":
            min_vals = data.min(axis=0)
            max_vals = data.max(axis=0)
            return (data - min_vals) / (max_vals - min_vals + 1e-8)
        elif method == "zscore":
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            return (data - mean) / (std + 1e-8)
        else:
            logger.warning(f"⚠️ Unknown method '{method}', skipping normalization")
            return data

    # --------------------------
    # Visualize coordinates before and after normalization
    # --------------------------
    def visualize_coords(self, before: np.ndarray, after: np.ndarray, room_name: str):
        """
        Scatter plot 3D before and after normalization for visual inspection.

        Args:
            before (np.ndarray): Original coordinates
            after (np.ndarray): Normalized coordinates
            room_name (str): Room identifier for saving figure
        """
        sample_n = min(20000, before.shape[0])
        idx = np.random.choice(before.shape[0], size=sample_n, replace=False)
        before_sample = before[idx]
        after_sample = after[idx]

        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.scatter(before_sample[:, 0], before_sample[:, 1], before_sample[:, 2], s=1, c='blue')
        ax1.set_title(f"{room_name} - Before Normalization")

        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.scatter(after_sample[:, 0], after_sample[:, 1], after_sample[:, 2], s=1, c='red')
        ax2.set_title(f"{room_name} - After Normalization")

        plt.tight_layout()
        vis_file = self.params.vis_dir / f"{room_name}_coord_norm.png"
        plt.savefig(vis_file)
        plt.close(fig)
        logger.info(f"📸 Visualization saved at: {vis_file}")

    # --------------------------
    # Process a single room
    # --------------------------
    def process_room(self, room_dir: Path):
        """
        Normalize coordinates, color, and intensity for a single room.
        Visualization is only done for coordinates.
        """
        logger.info(f"📏 Processing room: {room_dir.name}")

        # --- COORDINATES ---
        if "coords" in self.params.targets:
            coord_file = room_dir / "coord.npy"
            if coord_file.exists():
                coords = np.load(coord_file)
                coords_norm = (
                    self.custom_normalize(coords)[0]
                    if self.params.coord_method.lower() == "custom"
                    else self.normalize_array(coords, self.params.coord_method)
                )
                norm_file = room_dir / "coord_norm.npy"
                np.save(norm_file, coords_norm)
                logger.info(f"✅ Normalized coords saved at: {norm_file} using method: {self.params.coord_method}")
                self.visualize_coords(coords, coords_norm, room_dir.name)
            else:
                logger.warning(f"⚠️ Coord file not found: {coord_file}")
        else:
            logger.info(f"ℹ Skipping coords normalization (not in targets)")

        # --- COLOR ---
        if "color" in self.params.targets:
            color_file = room_dir / "color.npy"
            if color_file.exists():
                color = np.load(color_file)
                color_norm = self.normalize_array(color, self.params.color_method)
                np.save(color_file, color_norm)
                logger.info(f"✅ Normalized color saved at: {color_file} using method: {self.params.color_method}")
            else:
                logger.warning(f"⚠️ Color file not found: {color_file}")
        else:
            logger.info(f"ℹ Skipping color normalization (not in targets)")

        # --- INTENSITY ---
        if "intensity" in self.params.targets:
            strength_file = room_dir / "strength.npy"
            if strength_file.exists():
                strength = np.load(strength_file)
                strength_norm = self.normalize_array(strength, self.params.intensity_method)
                np.save(strength_file, strength_norm)
                logger.info(f"✅ Normalized intensity saved at: {strength_file} using method: {self.params.intensity_method}")
            else:
                logger.warning(f"⚠️ Strength file not found: {strength_file}")
        else:
            logger.info(f"ℹ Skipping intensity normalization (not in targets)")

    # --------------------------
    # Run normalization pipeline for all splits
    # --------------------------
    def run(self):
        """
        Run normalization for all rooms in all splits (train/test).
        Skips rooms not in target_files or missing coord.npy.
        """
        if not self.params.enabled:
            logger.info("⏭️ Normalization is disabled. Skipping...")
            return

        for split in ["train", "test"]:
            split_dir = self.params.base_dir / split
            if not split_dir.exists():
                logger.warning(f"⚠️ Input directory does not exist: {split_dir}")
                continue

            for room_dir in split_dir.iterdir():
                if not room_dir.is_dir():
                    continue  # Skip files

                # Skip known non-room directories
                if room_dir.name.lower() in ["chunks", "visualizations"] or room_dir.name.startswith("."):
                    logger.info(f"⏭️ Skipping helper/hidden folder: {room_dir.name}")
                    continue

                # Ensure room has coord.npy
                coord_file = room_dir / "coord.npy"
                if not coord_file.exists():
                    logger.info(f"⏭️ Skipping {room_dir.name} (no coord.npy found)")
                    continue

                # Process only target rooms or all
                if "all" in self.params.target_files or room_dir.name in self.params.target_files:
                    self.process_room(room_dir)
                else:
                    logger.info(f"⏭️ Skipping {room_dir.name} (not in target list)")
