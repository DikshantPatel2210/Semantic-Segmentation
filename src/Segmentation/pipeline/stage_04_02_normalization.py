# ======================================================
# Normalization Processing Pipeline
# ======================================================
from src.Segmentation import logger
from src.Segmentation.config.configuration import ConfigurationManager
from src.Segmentation.entity.config_entity import NormalizationConfig
from src.Segmentation.components.normalization import Normalization


STAGE_NAME = "Normalization Stage"

class NormalizationProcessingPipeline:
    def main(self):
        config_manager = ConfigurationManager()
        normalization_config: NormalizationConfig = config_manager.normalization_config()
        normalizer = Normalization(params=normalization_config)
        normalizer.run()

# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>> starting {STAGE_NAME} <<<<<<<<<<<")
        pipeline = NormalizationProcessingPipeline()
        pipeline.main()
        logger.info(f">>>>>>>>> finished {STAGE_NAME} <<<<<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
