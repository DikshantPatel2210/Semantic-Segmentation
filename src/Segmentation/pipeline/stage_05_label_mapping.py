# ======================================================
# Label Mapping Processing Pipeline
# ======================================================
from src.Segmentation import logger
from src.Segmentation.config.configuration import ConfigurationManager
from src.Segmentation.entity.config_entity import LabelMappingConfig
from src.Segmentation.components.label_mapping import LabelMapping

STAGE_NAME = "Label Mapping Stage"

class LabelMappingProcessingPipeline:
    def main(self):
        # Load configuration
        params_manager = ConfigurationManager()
        label_mapping_config: LabelMappingConfig = params_manager.label_mapping_config()

        # Initialize and run label mapping
        label_mapper = LabelMapping(params=label_mapping_config)
        label_mapper.run()

# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>> starting {STAGE_NAME} <<<<<<<<<<<")
        pipeline = LabelMappingProcessingPipeline()
        pipeline.main()
        logger.info(f">>>>>>>>> finished {STAGE_NAME} <<<<<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
