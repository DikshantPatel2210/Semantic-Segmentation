import numpy as np
from plyfile import PlyData, PlyElement
from src.Segmentation import logger
from src.Segmentation.constants import *
from src.Segmentation.utils.common import read_yaml, create_directories
from src.Segmentation.entity.config_entity import RotationConfig

# ===============================================
# Rotation Pipeline
# ===============================================
class RotationPipeline:
    """
    Pipeline to rotate 3D point cloud data (PLY files) along specified axes.
    Features:
    - Rotate points around X, Y, Z axes by a configurable angle
    - Process all PLY files or selected target files in train/test splits
    - Save rotated PLY files in an output directory
    """
    def __init__(self, params: RotationConfig):
        self.params = params

    # -------------------------
    # Rotate 3D points around selected axes
    # -------------------------
    def rotate_points(self, points: np.ndarray) -> np.ndarray:
        """
        Rotate points by self.params.angle degrees along axes in self.params.direction
        Returns a new numpy array of rotated points
        """
        angle_rad = np.deg2rad(self.params.angle)  # Convert angle to radians
        rotated_points = points.copy()  # Avoid modifying original array

        for axis in self.params.direction:
            # Create rotation matrix per axis
            if axis.lower() == "x":
                R = np.array([[1, 0, 0],
                              [0, np.cos(angle_rad), -np.sin(angle_rad)],
                              [0, np.sin(angle_rad), np.cos(angle_rad)]])
            elif axis.lower() == "y":
                R = np.array([[np.cos(angle_rad), 0, np.sin(angle_rad)],
                              [0, 1, 0],
                              [-np.sin(angle_rad), 0, np.cos(angle_rad)]])
            elif axis.lower() == "z":
                R = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                              [np.sin(angle_rad), np.cos(angle_rad), 0],
                              [0, 0, 1]])
            else:
                logger.warning(f"⚠️ Unknown rotation axis '{axis}', skipping.")
                continue

            rotated_points = rotated_points @ R.T  # Apply rotation

        return rotated_points

    # -------------------------
    # Process a single PLY file
    # -------------------------
    def process_ply(self, ply_file: Path, split: str):
        """
        Rotates a single PLY file and saves the output
        - ply_file: Path to the input PLY
        - split: "train" or "test" split
        """
        logger.info(f"🔄 Rotating file: {ply_file}")
        plydata = PlyData.read(str(ply_file))
        vertex_data = plydata['vertex'].data

        # Extract XYZ points as float32 array
        points = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T.astype(np.float32)
        rotated_points = self.rotate_points(points)  # Rotate points

        # Copy vertex data to avoid overwriting original
        rotated_vertex = vertex_data.copy()
        rotated_vertex['x'] = rotated_points[:, 0]
        rotated_vertex['y'] = rotated_points[:, 1]
        rotated_vertex['z'] = rotated_points[:, 2]

        # Prepare output directory for the current split
        output_split_dir = self.params.output_dir / split
        create_directories([output_split_dir])

        # Save rotated PLY file
        output_file = output_split_dir / ply_file.name
        PlyData([PlyElement.describe(rotated_vertex, 'vertex')], text=True).write(str(output_file))
        logger.info(f"✅ Rotated file saved at: {output_file}")

    # -------------------------
    # Run pipeline for all splits
    # -------------------------
    def run(self):
        """
        Executes the rotation pipeline for all train/test splits
        Skips files if rotation is disabled in config or target_files filtering
        """
        if not self.params.enabled:
            logger.info("⏭️ Rotation is disabled in config.yaml. Skipping.")
            return

        for split in ["train", "test"]:
            input_dir = self.params.base_dir / split
            if not input_dir.exists():
                logger.warning(f"⚠️ Input directory does not exist: {input_dir}")
                continue

            ply_files = list(input_dir.glob("*.ply"))
            if not ply_files:
                logger.warning(f"⚠️ No PLY files found in {input_dir}")
                continue

            for ply_file in ply_files:
                # Process only if in target_files list or if "all" is selected
                if "all" in self.params.target_files or ply_file.name in self.params.target_files:
                    self.process_ply(ply_file, split)
                else:
                    logger.info(f"⏭️ Skipping {ply_file} (not in target list)")

        logger.info("🎯 Rotation pipeline completed successfully!")
