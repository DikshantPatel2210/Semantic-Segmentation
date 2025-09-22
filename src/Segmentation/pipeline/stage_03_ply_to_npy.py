from src.Segmentation import logger
from src.Segmentation.config.configuration import ConfigurationManager
from src.Segmentation.entity.config_entity import PLYProcessorConfig
from src.Segmentation.components.ply_to_npy import PLYProcessor  # assuming you put PLYProcessor in components
STAGE_NAME = "PLY Processing"

class PLYProcessingPipeline:
    def __init__(self):
        pass

    def main(self):
        # Step 1: Load configuration
        config_manager = ConfigurationManager()
        ply_config: PLYProcessorConfig = config_manager.ply_processor_config()

        # Step 2: Initialize processor and run
        processor = PLYProcessor(config=ply_config)
        processor.run()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>> starting {STAGE_NAME} <<<<<<<<<<<")
        pipeline = PLYProcessingPipeline()
        pipeline.main()
        logger.info(f">>>>>>>>> finished {STAGE_NAME} <<<<<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
