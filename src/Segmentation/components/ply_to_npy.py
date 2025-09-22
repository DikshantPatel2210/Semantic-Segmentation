from pathlib import Path
import numpy as np
from plyfile import PlyData
import os
import re
from src.Segmentation import logger
from src.Segmentation.utils.common import save_numpy, create_directories
from src.Segmentation.entity.config_entity import PLYProcessorConfig


# ======================================================
# Class for Reading, Processing, and Saving PLY Files
# ======================================================
class PLYProcessor:
    def __init__(self, config: PLYProcessorConfig):
        """
        Initialize the PLYProcessor with configuration parameters.

        Args:
            config (PLYProcessorConfig): Configuration object containing:
                - base_dir: input directory containing PLY files
                - output_dir: directory to save processed numpy arrays
                - save_normals: whether to save normal vectors if available
        """
        self.base_dir = config.base_dir
        self.output_dir = config.output_dir
        self.save_normals = config.save_normals

    # --------------------------
    # Helper function: Extract room number from filename
    # --------------------------
    @staticmethod
    def extract_room_number(filename: str) -> str:
        """
        Extract the first sequence of digits in the filename as room number.
        If no digits are found, return the filename without extension.
        """
        match = re.search(r'\d+', os.path.basename(filename))
        return match.group(0) if match else os.path.splitext(os.path.basename(filename))[0]

    # --------------------------
    # Process a single PLY file
    # --------------------------
    def process_ply_file(self, ply_file: Path):
        """
        Read a PLY file and extract coordinates, colors, intensity, labels, and normals.

        Returns:
            points (np.ndarray): Nx3 coordinates
            colors (np.ndarray): Nx3 color values
            scalar_field (np.ndarray): intensity values
            label (np.ndarray): label IDs
            normals (np.ndarray or None): Nx3 normals if available
        """
        plydata = PlyData.read(str(ply_file))
        data = plydata['vertex'].data

        # Log available properties
        logger.info(f"📋 Available vertex properties: {data.dtype.names}")

        # Extract mandatory fields
        x, y, z = data['x'], data['y'], data['z']
        red, green, blue = data['red'], data['green'], data['blue']

        # Extract optional fields
        scalar_field = data['intensity'] if 'intensity' in data.dtype.names else np.zeros_like(x)
        label = data['label_id'] if 'label_id' in data.dtype.names else np.zeros_like(x)

        # Extract normals if available
        normals = None
        if all(k in data.dtype.names for k in ['nx', 'ny', 'nz']):
            normals = np.vstack((data['nx'], data['ny'], data['nz'])).T.astype(np.float32)
            if self.save_normals:
                logger.info(f"✅ Normals found! Shape: {normals.shape}")
        else:
            logger.warning("❌ Normals not found in this file.")

        # Convert all data to numpy arrays
        points = np.vstack((x, y, z)).T.astype(np.float32)
        colors = np.vstack((red, green, blue)).T.astype(np.uint8)
        scalar_field = scalar_field.astype(np.float32)
        label = label.astype(np.int64)

        return points, colors, scalar_field, label, normals

    # --------------------------
    # Save processed numpy arrays
    # --------------------------
    def save_processed_data(self, output_dir: Path, points, colors, scalar_field, label, normals=None):
        """
        Save processed numpy arrays to disk in the output directory.
        Creates directory if it does not exist.
        """
        create_directories([output_dir])
        save_numpy(points, output_dir / "coord.npy")
        save_numpy(colors, output_dir / "color.npy")
        save_numpy(scalar_field, output_dir / "strength.npy")
        save_numpy(label, output_dir / "segment.npy")
        if normals is not None and self.save_normals:
            save_numpy(normals, output_dir / "normal.npy")

    # --------------------------
    # Main processing loop for all PLY files
    # --------------------------
    def run(self):
        """
        Iterate through train/test splits, process each PLY file,
        and save numpy arrays with coordinates, colors, labels, etc.
        """
        logger.info(f"📁 Using base output directory: {self.output_dir}")

        for split in ["train", "test"]:
            input_dir = self.base_dir / split
            if not input_dir.exists():
                logger.warning(f"⚠️ Input directory does not exist: {input_dir}")
                continue

            ply_files = list(input_dir.glob("*.ply"))
            logger.info(f"Processing {len(ply_files)} files in '{input_dir}'...")

            for ply_file in ply_files:
                logger.info("=" * 60)
                logger.info(f"📂 Processing File: {ply_file}")

                # Extract room number
                room_number = self.extract_room_number(str(ply_file))
                logger.info(f"🏠 Room number detected: {room_number}")

                # Read PLY file and extract data
                points, colors, scalar_field, label, normals = self.process_ply_file(ply_file)

                # Log detailed info
                logger.info(f"✅ Points shape: {points.shape}")
                logger.info(f"🎨 Colors shape: {colors.shape}")
                logger.info(f"📊 Scalar field shape: {scalar_field.shape}")
                logger.info(f"🏷️ Labels shape: {label.shape}")

                # Count points per unique label
                unique_labels, counts = np.unique(label, return_counts=True)
                logger.info(f"🏷️ Unique labels in Room {room_number}: {unique_labels}")
                for ul, c in zip(unique_labels, counts):
                    logger.info(f"    Label {ul}: {c} points")

                # Save processed data to output directory
                output_split_dir = self.output_dir / split
                room_output_dir = output_split_dir / f"{room_number}_room"
                self.save_processed_data(room_output_dir, points, colors, scalar_field, label, normals)

                logger.info(f"💾 Saved processed data for Room {room_number} in '{room_output_dir}/' ✅\n\n")
