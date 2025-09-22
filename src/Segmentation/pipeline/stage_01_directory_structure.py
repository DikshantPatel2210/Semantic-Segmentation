from src.Segmentation import logger
from src.Segmentation.components.directory_structure import DirectoryStructureValidator
from src.Segmentation.config.configuration import ConfigurationManager


STAGE_NAME = "Directory Structure Validation"

class DirectoryValidatorPipeline:
    def __init__(self):
        pass

    def main(self):
        # Step 1: Load configuration
        config_manager = ConfigurationManager()
        directory_structure_config = config_manager.directory_structure()

        # Step 2: Validate directories
        validator = DirectoryStructureValidator(config=directory_structure_config)
        validator.validate()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>> Starting {STAGE_NAME} <<<<<<<<<<<")
        pipeline = DirectoryValidatorPipeline()
        pipeline.main()
        logger.info(f">>>>>>>>> Finished {STAGE_NAME} <<<<<<<<<<<<")
    except Exception as e:
        logger.exception(f"❌ Error during {STAGE_NAME}: {e}")
        raise