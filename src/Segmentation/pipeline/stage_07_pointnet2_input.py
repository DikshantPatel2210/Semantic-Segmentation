# ======================================================
# Same-Size Chunking Processing Pipeline
# ======================================================
from pathlib import Path
import torch
from src.Segmentation import logger
from src.Segmentation.config.configuration import ConfigurationManager
from src.Segmentation.entity.config_entity import Pointnet2InputConfig
from src.Segmentation.components.pointnet2_input import Pointnet2InputPipeline, PointCloudChunkDataset

STAGE_NAME = "PointNet++ Input Preparation Stage"

class Pointnet2InputProcessingPipeline:
    def main(self):
        # --------------------------
        # Load configuration
        # --------------------------
        params_path = Path("params.yaml")
        config_manager = ConfigurationManager(params_path)
        chunking_config: Pointnet2InputConfig = config_manager.pointnet2_input_config()

        # --------------------------
        # Detect device
        # --------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # --------------------------
        # Initialize and run chunking
        # --------------------------
        chunking_pipeline = Pointnet2InputPipeline(params=chunking_config, device=device)
        chunking_pipeline.run()

# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>> starting {STAGE_NAME} <<<<<<<<<<<")
        pipeline = Pointnet2InputProcessingPipeline()
        pipeline.main()
        logger.info(f">>>>>>>>> finished {STAGE_NAME} <<<<<<<<<<<<")
    except Exception as e:
        logger.exception(f"❌ Error during {STAGE_NAME}: {e}")
        raise e
