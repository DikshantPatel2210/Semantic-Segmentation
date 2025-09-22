# ======================================================
# PointNet++ Testing Pipeline
# ======================================================
from pathlib import Path
from src.Segmentation import logger
from src.Segmentation.config.configuration import ConfigurationManager
from src.Segmentation.entity.config_entity import PointNet2TestConfig
from src.Segmentation.components.pointnet2_testing import PointNetTester

STAGE_NAME = "PointNet++ Testing Stage"


class Pointnet2TestingPipeline:
    def __init__(self, config_path: str, model_type: str):
        self.config_path = Path(config_path)
        self.model_type = model_type.lower()
        self.config_manager = ConfigurationManager(self.config_path)

        if self.model_type != "pointnet2":
            raise ValueError(f"❌ Unknown model_type: {model_type}")

    def main(self):
        # --------------------------
        # Load configuration
        # --------------------------
        logger.info("📄 Loading test configuration...")
        test_cfg: PointNet2TestConfig = self.config_manager.get_test_pointnet2_config()

        # --------------------------
        # Initialize tester
        # --------------------------
        logger.info("🛠 Initializing PointNet++ tester...")
        tester = PointNetTester(test_cfg)

        # --------------------------
        # Run testing
        # --------------------------
        logger.info(f"🚀 Running test pipeline for model: {test_cfg.model_type}")
        merged_file = tester.run()  # returns merged CSV path
        logger.info(f"✅ Testing finished. Merged CSV saved at: {merged_file}")

        return merged_file


# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>> Starting {STAGE_NAME} <<<<<<<<<<<")
        pipeline = Pointnet2TestingPipeline(config_path="params.yaml", model_type="pointnet2")
        merged_file = pipeline.main()
        logger.info(f"✅ Finished {STAGE_NAME} successfully")
    except Exception as e:
        logger.exception(f"❌ Error during {STAGE_NAME}: {e}")
        raise e
