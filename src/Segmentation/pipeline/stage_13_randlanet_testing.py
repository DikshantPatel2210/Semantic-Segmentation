# ======================================================
# RandLaNet Testing Pipeline
# ======================================================
from src.Segmentation import logger
from pathlib import Path
from src.Segmentation.config.configuration import ConfigurationManager
from src.Segmentation.entity.config_entity import RandLANetTestConfig
from src.Segmentation.components.randlanet_testing import RandlaNetTester

STAGE_NAME = "RandLaNet Testing Stage"

class RandLaNetTestingPipeline:
    def __init__(self, params_path: str):
        # --------------------------
        # Load configuration
        # --------------------------
        params_path = Path(params_path)  # ensure Path object
        self.params_manager = ConfigurationManager(params_path)
        self.test_config: RandLANetTestConfig = self.params_manager.get_randlanet_test_config()
        class_names = self.params_manager.get_class_names()  # load from YAML

        # --------------------------
        # Initialize tester
        # --------------------------
        self.tester = RandlaNetTester(self.test_config, class_names=class_names)
        logger.info(f"[INFO] RandLaNetTestingPipeline initialized for model: {self.test_config.model_type}")

    def main(self):
        # --------------------------
        # Run testing
        # --------------------------
        logger.info(f"[INFO] Starting testing for {self.test_config.model_type}...")
        merged_file, metrics_file, cm_file = self.tester.test()
        logger.info(f"[INFO] Testing finished.")
        logger.info(f"   - Merged predictions: {merged_file}")
        logger.info(f"   - Metrics CSV: {metrics_file}")
        logger.info(f"   - Confusion matrix PNG: {cm_file}")
        return merged_file, metrics_file, cm_file


# --------------------------
# Run the pipeline
# --------------------------
if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>> starting {STAGE_NAME} <<<<<<<<<<<")
        PARAMS_PATH = "params.yaml"  # Using params.yaml
        pipeline = RandLaNetTestingPipeline(PARAMS_PATH)
        pipeline.main()
        logger.info(f">>>>>>>>> finished {STAGE_NAME} <<<<<<<<<<<<")
    except Exception as e:
        logger.exception(f"❌ Error during {STAGE_NAME}: {e}")
        raise e
