
from src.Segmentation.entity.config_entity import DirectoryStructureConfig
from src.Segmentation import logger
from src.Segmentation.utils.common import read_yaml, create_directories

# ======================================================
# Class for validating dataset directory structure
# ======================================================
class DirectoryStructureValidator:
    def __init__(self, config: DirectoryStructureConfig):
        """
        Initialize the validator with a DirectoryStructureConfig object.

        Args:
            config (DirectoryStructureConfig): Configuration containing root, train, and test directories.
        """
        self.config = config

    def validate(self):
        """
        Validate dataset structure and optionally create missing directories.

        Steps:
        1. Check if root, train, and test directories exist. If not, create them.
        2. Verify that .ply files exist in train and test directories.
        3. Log warnings for created directories and errors for missing data.
        """
        try:
            # List of directories to check for existence
            dirs_to_check = [self.config.root_dir, self.config.train_dir, self.config.test_dir]

            # Ensure each directory exists, create if missing
            for d in dirs_to_check:
                if not d.exists():
                    create_directories([d])  # Use utility function to create directory
                    logger.warning(f"⚠️ Directory created: {d} (was missing)")

            # Check for .ply files in train and test directories
            for d in [self.config.train_dir, self.config.test_dir]:
                ply_files = list(d.glob("*.ply"))  # List all .ply files
                if not ply_files:
                    message = (
                        f"❌ No .ply files found in {d}.\n"
                        "Please add your dataset (.ply files).\n"
                        "Expected structure:\n"
                        f"{self.config.root_dir}/\n"
                        "    ├── train/ -> must contain .ply files\n"
                        "    └── test/  -> must contain .ply files"
                    )
                    logger.error(message)
                    raise FileNotFoundError(message)  # Raise exception if dataset missing

            # Log success if all checks pass
            logger.info("✅ Dataset structure validated successfully!")

        except Exception as e:
            # Log any exception encountered during validation
            logger.exception("❌ Error during dataset validation")
            raise
