# ======================================================
# PointNet2 Training Pipeline
# ======================================================
import os
from src.Segmentation import logger
from src.Segmentation.config.configuration import ConfigurationManager
from src.Segmentation.components.pointnet2_training import PointNet2TrainingStage  # your existing pipeline

STAGE_NAME = "PointNet2 Training Stage"

class PointNet2TrainingPipeline:
    def __init__(self):
        # --------------------------
        # Load configuration and initialize training pipeline
        # --------------------------
        self.config_manager = ConfigurationManager()
        self.train_dir = self.config_manager.get_training_config().train_dir
        self.pipeline = PointNet2TrainingStage(train_dir=self.train_dir)
        logger.info(f"[INFO] PointNet2TrainingStage initialized.")

    def main(self):
        # --------------------------
        # Start training (prints and logging inside pipeline)
        # --------------------------
        logger.info(f"[INFO] Starting training...")
        self.pipeline.main()
        logger.info(f"[INFO] Training finished.")

# ===================== Run Pipeline =====================
if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>> starting {STAGE_NAME} <<<<<<<<<<<")
        stage = PointNet2TrainingPipeline()
        stage.main()
        logger.info(f">>>>>>>>> finished {STAGE_NAME} <<<<<<<<<<<<")
    except Exception as e:
        logger.exception(f"❌ Error during {STAGE_NAME}: {e}")
        raise e
