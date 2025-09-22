
from pathlib import Path
import numpy as np
from src.Segmentation import logger
from src.Segmentation.constants import *
from src.Segmentation.utils.common import read_yaml, create_directories
from src.Segmentation.entity.config_entity import LabelMappingConfig
import json


# ======================================================
# Class to handle remapping of semantic labels in dataset
# ======================================================
class LabelMapping:
    def __init__(self, params: LabelMappingConfig):
        """
        Initialize the LabelMapping class with configuration parameters.

        Args:
            params (LabelMappingConfig): Configuration object with attributes such as:
                - base_dir: root dataset directory
                - splits: list of splits to process (train/test)
                - target_files: list of target rooms or ["all"]
                - output_dir: directory to save label mapping JSON files
                - enabled: whether label mapping is enabled
        """
        self.params = params
        # Ensure output directory exists
        create_directories([self.params.output_dir])

    # -------------------------------
    # Remap labels for a single room
    # -------------------------------
    def remap_labels(self, segment_file: Path, room_name: str):
        """
        Remap the labels in a segment.npy file to consecutive integers starting from 0.

        Args:
            segment_file (Path): Path to segment.npy file of the room
            room_name (str): Name of the room (used for JSON mapping file)

        Returns:
            dict: Mapping from old labels to new labels
        """
        labels = np.load(segment_file)  # Load labels
        unique_labels = np.unique(labels)  # Identify unique labels

        # Create mapping: old_label -> new_label (0..N-1)
        label_map = {int(old_label): int(new_label) for new_label, old_label in enumerate(unique_labels)}

        # Apply remapping to the array
        remapped_labels = np.vectorize(label_map.get)(labels)
        np.save(segment_file, remapped_labels)  # Overwrite segment file
        logger.info(f"✅ Remapped labels saved to '{segment_file}'")

        # Save mapping to JSON for reference
        mapping_file = self.params.output_dir / f"{room_name}_label_map.json"
        with open(mapping_file, "w") as f:
            json.dump(label_map, f, indent=4)
        logger.info(f"📄 Label mapping saved at: {mapping_file}")

        return label_map

    # -------------------------------
    # Run label mapping pipeline
    # -------------------------------
    def run(self):
        """
        Run the label mapping pipeline for all specified splits and target rooms.
        Skips processing if the feature is disabled or directories/files are missing.
        """
        if not self.params.enabled:
            logger.info("⏭️ Label mapping is disabled. Skipping...")
            return

        # Process each split (train/test)
        for split in self.params.splits:
            split_dir = self.params.base_dir / split
            if not split_dir.exists():
                logger.warning(f"⚠️ Split directory does not exist: {split_dir}")
                continue

            # Iterate through each room directory
            for room_folder in split_dir.iterdir():
                # Only process directories
                if not room_folder.is_dir():
                    continue

                # Skip system/helper directories
                if room_folder.name.startswith(".") or room_folder.name.lower() in ["chunks", "same_size_chunks"]:
                    logger.info(f"⏭️ Skipping helper/system folder: {room_folder}")
                    continue

                segment_file = room_folder / "segment.npy"
                if not segment_file.exists():
                    logger.warning(f"⚠️ segment.npy not found in {room_folder}")
                    continue

                # Process only target files or all rooms
                if "all" in self.params.target_files or room_folder.name in self.params.target_files:
                    self.remap_labels(segment_file, room_folder.name)
                else:
                    logger.info(f"⏭️ Skipping {room_folder.name} (not in target list)")

        logger.info("🎯 Label Mapping pipeline completed successfully!")
