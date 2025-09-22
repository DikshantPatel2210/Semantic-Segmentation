# ==========================================
# preprocessing.py
# ==========================================
import argparse
from pathlib import Path
from src.Segmentation.components.rotation import RotationPipeline
from src.Segmentation.components.normalization import Normalization
from src.Segmentation.config.configuration import ConfigurationManager
from src.Segmentation import logger

def main():
    parser = argparse.ArgumentParser(
        description="Preprocessing: Apply Rotation and/or Normalization on dataset"
    )

    # -------------------------
    # Rotation arguments
    # -------------------------
    parser.add_argument("--rotate", action="store_true", help="Apply rotation")
    parser.add_argument("-axis", type=str, choices=["x", "y", "z"], default="z",
                        help="Rotation axis (x, y, z)")
    parser.add_argument("-angle", type=float, default=90, help="Rotation angle in degrees")
    parser.add_argument("--rooms_rotate", type=str, default="all",
                        help="Comma-separated rooms to rotate (default: all)")

    # -------------------------
    # Normalization arguments
    # -------------------------
    parser.add_argument("--normalize", action="store_true", help="Apply normalization")
    parser.add_argument("-target", type=str, default="coord", choices=["coord"],
                        help="Target file for normalization")
    parser.add_argument("-method", type=str, default="custom",
                        choices=["minmax", "zscore", "custom"], help="Normalization method")
    parser.add_argument("--rooms_norm", type=str, default="all",
                        help="Comma-separated rooms to normalize (default: all)")

    # -------------------------
    # Config argument
    # -------------------------
    parser.add_argument("--config", "-c", type=str, default="config/config.yaml",
                        help="Path to config.yaml")

    args = parser.parse_args()

    logger.info("========== Starting Preprocessing ==========")

    # -------------------------
    # Load configuration
    # -------------------------
    config_manager = ConfigurationManager(config_filepath=Path(args.config))

    # -------------------------
    # Rotation Stage
    # -------------------------
    if args.rotate:
        logger.info(f">>>>> Rotation Stage (Axis={args.axis}, Angle={args.angle}) <<<<<")
        rotation_config = config_manager.rotation_config()

        # Override axis/angle and rooms if CLI provided
        rotation_config = rotation_config.__class__(
            enabled=rotation_config.enabled,
            direction=[args.axis],
            angle=args.angle,
            target_files=args.rooms_rotate.split(",") if args.rooms_rotate != "all" else ["all"],
            base_dir=rotation_config.base_dir,
            output_dir=rotation_config.output_dir
        )

        rotation_pipeline = RotationPipeline(config=rotation_config)
        rotation_pipeline.run()
        logger.info(">>>> Finished Rotation Stage <<<<")

    # -------------------------
    # Normalization Stage
    # -------------------------
    if args.normalize:
        logger.info(f">>>>> Normalization Stage (Method={args.method}) <<<<<")
        normalization_config = config_manager.normalization_config()

        # Override method and target rooms dynamically
        norm_rooms = args.rooms_norm.split(",") if args.rooms_norm != "all" else ["all"]
        normalization_config = normalization_config.__class__(
            enabled=normalization_config.enabled,
            method=args.method,
            target_files=norm_rooms,
            base_dir=normalization_config.base_dir,
            vis_dir=normalization_config.vis_dir
        )

        normalizer = Normalization(config=normalization_config)
        normalizer.run()
        logger.info(">>>> Finished Normalization Stage <<<<")

    logger.info("========== Preprocessing Finished ==========")


if __name__ == "__main__":
    main()
