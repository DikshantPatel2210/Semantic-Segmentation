from src.Segmentation import logger
from src.Segmentation.config.configuration import ConfigurationManager
from src.Segmentation.entity.config_entity import RotationConfig
from src.Segmentation.components.rotation import RotationPipeline  # Assuming you saved it as rotation.py

STAGE_NAME = "Rotation Stage"

class RotationProcessingPipeline:
    def __init__(self):
        pass

    def main(self):
        # Step 1: Load configuration
        config_manager = ConfigurationManager()
        rotation_config: RotationConfig = config_manager.rotation_config()

        # Step 2: Initialize RotationPipeline and run
        rotation_pipeline = RotationPipeline(params=rotation_config)
        rotation_pipeline.run()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>> starting {STAGE_NAME} <<<<<<<<<<<")
        pipeline = RotationProcessingPipeline()
        pipeline.main()
        logger.info(f">>>>>>>>> finished {STAGE_NAME} <<<<<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
