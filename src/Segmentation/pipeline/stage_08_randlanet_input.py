# ======================================================
# RandLANet Input Preparation Pipeline
# ======================================================
from pathlib import Path
import torch
from src.Segmentation import logger
from src.Segmentation.config.configuration import ConfigurationManager
from src.Segmentation.entity.config_entity import RandLANetInputConfig
from src.Segmentation.components.randlanet_input import RandLANetInputPipeline

STAGE_NAME = "RandLANet Input Preparation Stage"

class RandLANetInputProcessingPipeline:
    def __init__(self):
        # --------------------------
        # Load configuration
        # --------------------------

        params_path = Path("params.yaml")
        self.params_manager = ConfigurationManager( params_path)
        self.cfg: RandLANetInputConfig = self.params_manager.randlanet_input_config()

    def main(self):
        # --------------------------
        # Initialize and prepare datasets
        # --------------------------
        pipeline = RandLANetInputPipeline(self.cfg)
        datasets = pipeline.prepare_datasets()

        # --------------------------
        # Example: inspect first train patch
        # --------------------------
        if "train" in datasets and len(datasets["train"]) > 0:
            patch = datasets["train"][0]
            logger.info(f"Patch features shape: {patch['feats'].shape}")
            logger.info(f"Patch coords shape: {patch['xyz'].shape}")
            logger.info(f"Patch labels shape: {patch['labels'].shape}")

        return datasets

# --------------------------
# Run the pipeline
# --------------------------
if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>> starting {STAGE_NAME} <<<<<<<<<<<")
        pipeline = RandLANetInputProcessingPipeline()
        datasets = pipeline.main()
        logger.info(f">>>>>>>>> finished {STAGE_NAME} <<<<<<<<<<<<")
    except Exception as e:
        logger.exception(f"❌ Error during {STAGE_NAME}: {e}")
        raise e
