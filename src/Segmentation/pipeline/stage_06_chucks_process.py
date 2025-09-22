# ======================================================
# Chunking Processing Pipeline
# ======================================================
from src.Segmentation import logger
from src.Segmentation.config.configuration import ConfigurationManager
from src.Segmentation.entity.config_entity import ChunkingConfig
from src.Segmentation.components.chucks_process import Chunking
from pathlib import Path

STAGE_NAME = "Chunking Stage"

class ChunkingProcessingPipeline:
    def main(self):
        # --------------------------
        # Load configuration
        # --------------------------

        params_path = Path("params.yaml")
        config_manager = ConfigurationManager(params_path)
        chunking_config: ChunkingConfig = config_manager.chunking_config()

        # --------------------------
        # Initialize and run chunking
        # --------------------------
        chunking_pipeline = Chunking(params=chunking_config)
        chunking_pipeline.run()

# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>> starting {STAGE_NAME} <<<<<<<<<<<")
        pipeline = ChunkingProcessingPipeline()
        pipeline.main()
        logger.info(f">>>>>>>>> finished {STAGE_NAME} <<<<<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
