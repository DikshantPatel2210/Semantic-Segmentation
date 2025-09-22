from src.Segmentation import logger
from src.Segmentation.pipeline.stage_01_directory_structure import DirectoryValidatorPipeline
from src.Segmentation.pipeline.stage_02_check_gpu_connection import GPUCheckerPipeline
from src.Segmentation.pipeline.stage_03_ply_to_npy import PLYProcessingPipeline
from src.Segmentation.pipeline.stage_04_01_rotation import RotationProcessingPipeline
from src.Segmentation.pipeline.stage_04_02_normalization import NormalizationProcessingPipeline
from src.Segmentation.pipeline.stage_05_label_mapping import LabelMappingProcessingPipeline
from src.Segmentation.pipeline.stage_06_chucks_process import ChunkingProcessingPipeline
from src.Segmentation.pipeline.stage_07_pointnet2_input import Pointnet2InputProcessingPipeline
from src.Segmentation.pipeline.stage_08_randlanet_input import RandLANetInputProcessingPipeline
from src.Segmentation.pipeline.stage_09_randlanet import RandLANetForwardPipeline
from src.Segmentation.pipeline.stage_10_randlanet_training import RandLaNetTrainingPipeline
from src.Segmentation.pipeline.stage_11_pointnet2 import PointNet2ForwardPipeline
from src.Segmentation.pipeline.stage_12_pointnet2_training import PointNet2TrainingPipeline
from src.Segmentation.pipeline.stage_13_randlanet_testing import RandLaNetTestingPipeline
from src.Segmentation.pipeline.stage_14_pointnet2_testing import Pointnet2TestingPipeline

if __name__ == "__main__":
    # ============================
    # Stage 1: Directory Validation
    # ============================
    # Uncomment to run
    STAGE_NAME = "Directory Structure Validation"
    try:
        logger.info(f">>>>>>>>> Starting {STAGE_NAME} <<<<<<<<<<<")
        obj = DirectoryValidatorPipeline()
        obj.main()
        logger.info(f">>>>>>>>> Finished {STAGE_NAME} <<<<<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e

    # ============================
    # Stage 2: GPU Checker
    # ============================
    # Uncomment to run
    STAGE_NAME = "GPU Checker"
    try:
        logger.info(f">>>>>>>>> Starting {STAGE_NAME} <<<<<<<<<<<")
        obj = GPUCheckerPipeline()
        obj.main()
        logger.info(f">>>>>>>>> Finished {STAGE_NAME} <<<<<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e

    # ============================
    # Stage 3-7: Data Preprocessing
    # ============================

    # ----------------------------
    # Stage 3: PLY to NPY
    # ----------------------------
    STAGE_NAME = "PLY to NPY Processing"
    try:
        logger.info(f">>>>>>>>> Starting {STAGE_NAME} <<<<<<<<<<<")
        pipeline = PLYProcessingPipeline()
        pipeline.main()
    except Exception as e:
        logger.exception(f"❌ Error during {STAGE_NAME}: {e}")
        raise e

    # ----------------------------
    # Stage 4: Rotation
    # ----------------------------
    STAGE_NAME = "Rotation"
    try:
        logger.info(f">>>>>>>>> Starting {STAGE_NAME} <<<<<<<<<<<")
        pipeline = RotationProcessingPipeline()
        pipeline.main()
        logger.info(f">>>>>>>>> Finished {STAGE_NAME} <<<<<<<<<<<<")
    except Exception as e:
        logger.exception(f"❌ Error during {STAGE_NAME}: {e}")
        raise e

    # ----------------------------
    # Stage 5: Normalization
    # ----------------------------
    STAGE_NAME = "Normalization Stage"
    try:
        logger.info(f">>>>>>>>> Starting {STAGE_NAME} <<<<<<<<<<<")
        pipeline = NormalizationProcessingPipeline()
        pipeline.main()
        logger.info(f">>>>>>>>> Finished {STAGE_NAME} <<<<<<<<<<<<")
    except Exception as e:
       logger.exception(f"❌ Error during {STAGE_NAME}: {e}")
       raise e

    # ----------------------------
    # Stage 6: Label Mapping
    # ----------------------------
    STAGE_NAME = "Label Mapping Stage"
    try:
       logger.info(f">>>>>>>>> Starting {STAGE_NAME} <<<<<<<<<<<")
       pipeline = LabelMappingProcessingPipeline()
       pipeline.main()
       logger.info(f">>>>>>>>> Finished {STAGE_NAME} <<<<<<<<<<<<")
    except Exception as e:
       logger.exception(f"❌ Error during {STAGE_NAME}: {e}")
       raise e

    # ----------------------------
    # Stage 7: Chunking
    # ----------------------------
    STAGE_NAME = "Chunking Stage"
    try:
        logger.info(f">>>>>>>>> Starting {STAGE_NAME} <<<<<<<<<<<")
        pipeline = ChunkingProcessingPipeline()
        pipeline.main()
        logger.info(f">>>>>>>>> Finished {STAGE_NAME} <<<<<<<<<<<<")
    except Exception as e:
        logger.exception(f"❌ Error during {STAGE_NAME}: {e}")
        raise e

    # ----------------------------
    # Stage 7.1: PointNet2 Input Preparation
    # ----------------------------
    STAGE_NAME = "PointNet++ Input Preparation Stage"
    try:
        logger.info(f">>>>>>>>> Starting {STAGE_NAME} <<<<<<<<<<<")
        pipeline = Pointnet2InputProcessingPipeline()
        pipeline.main()
        logger.info(f">>>>>>>>> Finished {STAGE_NAME} <<<<<<<<<<<<")
    except Exception as e:
        logger.exception(f"❌ Error during {STAGE_NAME}: {e}")
        raise e

    # ----------------------------
    # Stage 7.2: RandLANet Input Preparation
    # ----------------------------
    STAGE_NAME = "RandLANet Input Preparation Stage"
    try:
        logger.info(f">>>>>>>>> Starting {STAGE_NAME} <<<<<<<<<<<")
        pipeline = RandLANetInputProcessingPipeline()
        datasets = pipeline.main()
        logger.info(f">>>>>>>>> Finished {STAGE_NAME} <<<<<<<<<<<<")
    except Exception as e:
        logger.exception(f"❌ Error during {STAGE_NAME}: {e}")
        raise e

    # ============================
    # Stage 9: RandLANet Forward Pass
    # ============================
    STAGE_NAME = "RandLANet Forward Pass Stage"
    try:
        logger.info(f">>>>>>>>> starting {STAGE_NAME} <<<<<<<<<<<")
        pipeline = RandLANetForwardPipeline()
        out = pipeline.main()
        logger.info(f">>>>>>>>> finished {STAGE_NAME} <<<<<<<<<<<<")
    except Exception as e:
        logger.exception(f"❌ Error during {STAGE_NAME}: {e}")
        raise e

    # ============================
    # Stage 10: RandLANet Training
    # ============================
    # Uncomment to run
    STAGE_NAME = "RandLaNet Training Stage"
    try:
         logger.info(f">>>>>>>>> starting {STAGE_NAME} <<<<<<<<<<<")
         pipeline = RandLaNetTrainingPipeline()
         pipeline.main()
         logger.info(f">>>>>>>>> finished {STAGE_NAME} <<<<<<<<<<<<")
    except Exception as e:
         logger.exception(f"❌ Error during {STAGE_NAME}: {e}")
         raise e

    # ============================
    # Stage 11: PointNet++ Forward Pass
    # ============================
    STAGE_NAME = "PointNet++ Forward Pass Stage"

    try:
        logger.info(f">>>>>>> Starting {STAGE_NAME} <<<<<<<")
        pipeline = PointNet2ForwardPipeline()
        out = pipeline.main()
        logger.info(f"✅ Finished {STAGE_NAME} successfully")
    except Exception:
        logger.exception(f"❌ Error during {STAGE_NAME}")
        raise  # Re-raise the exception to propagate it

    # ============================
    # Stage 12: PointNet++ Training
    # ============================
    # Uncomment to run
    STAGE_NAME = "PointNet2 Training Stage"
    try:
        logger.info(f">>>>>>>>> starting {STAGE_NAME} <<<<<<<<<<<")
        stage = PointNet2TrainingPipeline()
        stage.main()
        logger.info(f">>>>>>>>> finished {STAGE_NAME} <<<<<<<<<<<<")
    except Exception as e:
        logger.exception(f"❌ Error during {STAGE_NAME}: {e}")
        raise e

    # ============================
    # Stage 13: RandLaNet Testing
    # ============================
    STAGE_NAME = "RandLaNet Testing Stage"

    try:
        logger.info(f">>>>>>>>> starting {STAGE_NAME} <<<<<<<<<<<")
        PARAMS_PATH = "params.yaml"  # Using params.yaml
        pipeline = RandLaNetTestingPipeline(PARAMS_PATH)
        pipeline.main()
        logger.info(f">>>>>>>>> finished {STAGE_NAME} <<<<<<<<<<<<")
    except Exception as e:
        logger.exception(f"❌ Error during {STAGE_NAME}: {e}")
        raise e

# ============================
# Stage 14: PointNet++ Testing
# ============================
    STAGE_NAME = "PointNet++ Testing Stage"
    try:
        logger.info(f">>>>>>> Starting {STAGE_NAME} <<<<<<<")
        # Initialize the testing pipeline
        pipeline = Pointnet2TestingPipeline(model_type="pointnet2")
        # Run the testing pipeline and get the merged CSV output
        merged_file = pipeline.main()
        logger.info(f"✅ Finished {STAGE_NAME} successfully")
        logger.info(f"Output CSV file: {merged_file}")

    except Exception:
        logger.exception(f"❌ Error during {STAGE_NAME}")
        raise


