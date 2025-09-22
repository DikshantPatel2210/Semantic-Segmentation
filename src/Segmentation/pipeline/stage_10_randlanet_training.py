# ======================================================
# RandLaNet Training Pipeline
# ======================================================
from src.Segmentation import logger
from src.Segmentation.config.configuration import ConfigurationManager
from src.Segmentation.components.randlanet_training import RandlaNetTrainer

STAGE_NAME = "RandLaNet Training Stage"

class RandLaNetTrainingPipeline:
    def __init__(self):
        # --------------------------
        # Load configuration and initialize trainer
        # --------------------------
        self.config_manager = ConfigurationManager()
        self.trainer = RandlaNetTrainer()
        logger.info(f"[INFO] RandLaNetTrainingPipeline initialized.")

    def main(self):
        # --------------------------
        # Start training (prints inside component will show)
        # --------------------------
        logger.info(f"[INFO] Starting training...")
        self.trainer.train()
        logger.info(f"[INFO] Training finished.")

# --------------------------
# Run the pipeline
# --------------------------
if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>> starting {STAGE_NAME} <<<<<<<<<<<")
        pipeline = RandLaNetTrainingPipeline()
        pipeline.main()
        logger.info(f">>>>>>>>> finished {STAGE_NAME} <<<<<<<<<<<<")
    except Exception as e:
        logger.exception(f"❌ Error during {STAGE_NAME}: {e}")
        raise e
