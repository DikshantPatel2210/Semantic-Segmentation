
import numpy as np
from src.Segmentation import logger
from src.Segmentation.utils.common import  create_directories
from src.Segmentation.entity.config_entity import ChunkingConfig
from concurrent.futures import ThreadPoolExecutor

# ======================================================
# Class to handle chunking of point cloud data for rooms
# ======================================================
class Chunking:
    def __init__(self, params: ChunkingConfig):
        """
        Initialize the Chunking class with configuration parameters.

        Args:
            params (ChunkingConfig): Configuration object with attributes such as:
                - base_dir: root directory of the dataset
                - splits: list of splits to process (train/test)
                - chunk_range: size of each chunk in x and y
                - chunk_stride: stride to move the chunk window
                - chunk_minimum_size: minimum number of points per chunk
                - grid_size: optional grid subsampling size
                - num_workers: number of threads to use for processing
                - enabled: whether chunking is enabled
        """
        self.params = params

    # -------------------------------
    # Process a single room
    # -------------------------------
    def chunk_room(self, room_name, split_name):
        messages = [f"Processing room: {room_name} in {split_name} split"]

        room_path = self.params.base_dir / split_name / room_name

        # -------------------------------
        # Load necessary numpy feature files
        # -------------------------------
        try:
            coord = np.load(room_path / "coord.npy")
            coord_norm = np.load(room_path / "coord_norm.npy")
            color = np.load(room_path / "color.npy")
            normal = np.load(room_path / "normal.npy")
            segment = np.load(room_path / "segment.npy")
            strength = np.load(room_path / "strength.npy")
        except Exception as e:
            messages.append(f"❌ Failed to load data for {room_name}: {e}")
            return messages

        working_coord = coord.copy()

        # -------------------------------
        # Apply optional grid subsampling
        # -------------------------------
        if self.params.grid_size is not None:
            grid_coord = np.floor(working_coord / self.params.grid_size).astype(np.int64)
            _, unique_idx = np.unique(grid_coord, axis=0, return_index=True)

            # Subsample all features using unique indices
            coord = coord[unique_idx]
            coord_norm = coord_norm[unique_idx]
            color = color[unique_idx]
            normal = normal[unique_idx]
            segment = segment[unique_idx]
            strength = strength[unique_idx]
            working_coord = coord
            messages.append(f"✅ Applied grid subsampling with grid_size={self.params.grid_size}, kept {len(coord)} points")

        # -------------------------------
        # Compute bounding box for chunks
        # -------------------------------
        x_min, y_min = working_coord[:, :2].min(axis=0)
        x_max, y_max = working_coord[:, :2].max(axis=0)

        # Generate grid positions for chunks
        x_grid = np.arange(x_min, x_max - self.params.chunk_range[0] + self.params.chunk_stride[0],
                           self.params.chunk_stride[0])
        y_grid = np.arange(y_min, y_max - self.params.chunk_range[1] + self.params.chunk_stride[1],
                           self.params.chunk_stride[1])
        # Create all possible chunk start positions
        chunks = np.array(np.meshgrid(x_grid, y_grid, indexing="ij")).T.reshape(-1, 2)

        chunk_idx = 0
        for chunk_start in chunks:
            x0, y0 = chunk_start
            # Mask points that fall inside the current chunk
            mask = (
                (working_coord[:, 0] >= x0) & (working_coord[:, 0] < x0 + self.params.chunk_range[0]) &
                (working_coord[:, 1] >= y0) & (working_coord[:, 1] < y0 + self.params.chunk_range[1])
            )

            # Skip chunks with insufficient points
            if np.sum(mask) < self.params.chunk_minimum_size:
                continue

            # Create folder for the chunk and save features
            chunk_name = f"{room_name}_{chunk_idx}"
            chunk_folder = self.params.base_dir / split_name / "chunks" / chunk_name
            create_directories([chunk_folder])

            np.save(chunk_folder / "coord.npy", coord[mask])
            np.save(chunk_folder / "coord_norm.npy", coord_norm[mask])
            np.save(chunk_folder / "color.npy", color[mask])
            np.save(chunk_folder / "normal.npy", normal[mask])
            np.save(chunk_folder / "segment.npy", segment[mask])
            np.save(chunk_folder / "strength.npy", strength[mask])

            messages.append(f"💾 Saved chunk {chunk_idx} for {room_name} ({np.sum(mask)} points)")
            chunk_idx += 1

        # Log if no chunks were created
        if chunk_idx == 0:
            messages.append(f"⚠️ No chunks created for {room_name} (not enough points).")

        return messages

    # -------------------------------
    # Process all rooms in a split
    # -------------------------------
    def process_split(self, split_name):
        split_dir = self.params.base_dir / split_name
        if not split_dir.exists():
            logger.warning(f"⚠️ Split directory does not exist: {split_dir}")
            return

        # Ignore system or helper directories
        ignore_dirs = {"chunks", "same_size_chunks", "visualizations"}
        room_list = [d.name for d in split_dir.iterdir()
                     if d.is_dir() and not d.name.startswith(".") and d.name not in ignore_dirs]

        logger.info(f"📦 Found {len(room_list)} valid room(s) in {split_name}: {room_list}\n")

        num_workers = self.params.num_workers or min(len(room_list), 4)

        from itertools import repeat
        # Use ThreadPoolExecutor for parallel chunking
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for room_messages in executor.map(self.chunk_room, room_list, [split_name]*len(room_list)):
                for msg in room_messages:
                    logger.info(msg)
                logger.info("")  # empty line between rooms

    # -------------------------------
    # Main entry point to run chunking pipeline
    # -------------------------------
    def run(self):
        if not self.params.enabled:
            logger.info("⏭️ Chunking is disabled. Skipping...")
            return

        # Process each split (train/test)
        for split in self.params.splits:
            self.process_split(split)

        logger.info("🎯 Chunking pipeline completed successfully!")
