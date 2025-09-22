# ======================================================
# PointNet++ Forward Pass Pipeline
# ======================================================
from pathlib import Path
import torch
from src.Segmentation import logger
from src.Segmentation.config.configuration import ConfigurationManager
from src.Segmentation.entity.config_entity import PointNet2ModelConfig
from src.Segmentation.components.pointnet2 import PointNet2SemSeg5
STAGE_NAME = "PointNet++ Forward Pass Stage"

class PointNet2ForwardPipeline:
    def main(self):
        # --------------------------
        # Load configuration
        # --------------------------
        config_path = Path("config/config.yaml")
        params_path = Path("params.yaml")
        config_manager = ConfigurationManager(config_path, params_path)
        pointnet2_cfg: PointNet2ModelConfig = config_manager.get_pointnet2_model_config()

        # --------------------------
        # Set device and seed
        # --------------------------
        torch.manual_seed(42)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # --------------------------
        # Create dummy input data
        # --------------------------
        B, N = 2, 2048  # batch size, number of points
        xyz = torch.rand(B, 3, N).to(device)         # Shape: [B, 3, N]
        features = torch.rand(B, pointnet2_cfg.in_channel_sa1, N).to(device)  # Shape: [B, in_channel, N]

        # --------------------------
        # Initialize PointNet++ model
        # --------------------------
        model = PointNet2SemSeg5(pointnet2_cfg).to(device)
        model.eval()

        # --------------------------
        # Forward pass
        # --------------------------
        with torch.no_grad():
            out = model(xyz, features)

        logger.info(f"✅ Forward pass output shape: {out.shape}")  # [B, num_classes, N]
        return out


# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>> starting {STAGE_NAME} <<<<<<<<<<<")
        pipeline = PointNet2ForwardPipeline()
        out = pipeline.main()
        logger.info(f"✅ Finished {STAGE_NAME} successfully")
    except Exception as e:
        logger.exception(f"❌ Error during {STAGE_NAME}: {e}")
        raise e
