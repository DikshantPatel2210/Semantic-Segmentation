from src.Segmentation import logger
from src.Segmentation.config.configuration import ConfigurationManager
from src.Segmentation.components.gpu_checker import GPUChecker
from src.Segmentation.entity.config_entity import GPUCheckerConfig
STAGE_NAME = "GPU Checker"

class GPUCheckerPipeline:
    def __init__(self):
        pass

    def main(self):
        # Step 1: Load configuration
        config_manager = ConfigurationManager()
        gpu_checker_config: GPUCheckerConfig = config_manager.gpu_checker()

        # Step 2: Run GPU checks
        gpu_checker = GPUChecker(config=gpu_checker_config)
        gpu_checker.check_torch_version()
        gpu_checker.check_gpu_availability()



if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>> starting {STAGE_NAME} <<<<<<<<<<<")
        pipeline = GPUCheckerPipeline()
        pipeline.main()
        logger.info(f">>>>>>>>> finished {STAGE_NAME} <<<<<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e

