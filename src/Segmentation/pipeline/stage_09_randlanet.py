# ======================================================
# RandLANet Forward Pipeline
# ======================================================
from pathlib import Path
import torch
from src.Segmentation import logger
from src.Segmentation.config.configuration import ConfigurationManager
from src.Segmentation.entity.config_entity import RandlaNetConfig
from src.Segmentation.components.randlanet import build_pyramid_gpu_batch, RandlaNet, random_sample

STAGE_NAME = "RandLANet Forward Pass Stage"

class RandLANetForwardPipeline:
    def main(self):
        # --------------------------
        # Load configuration
        # --------------------------
        config_path = Path("config/config.yaml")
        params_path = Path("params.yaml")
        config_manager = ConfigurationManager(config_path, params_path)
        randlanet_cfg: RandlaNetConfig = config_manager.get_randlanet_config()

        # --------------------------
        # Set device and seed
        # --------------------------
        torch.manual_seed(randlanet_cfg.seed)
        device = randlanet_cfg.device

        # --------------------------
        # Create dummy input features
        # --------------------------
        B, N0 = 2, 2000  # batch size, number of points
        feats = torch.rand(B, N0, 10).to(device)

        # --------------------------
        # Build pyramid for the dummy batch
        # --------------------------
        xyz_list, neigh_list, sub_list, interp_list = build_pyramid_gpu_batch(
            feats[:, :, :3],
            k=randlanet_cfg.k,
            n_layers=randlanet_cfg.n_layers,
            ratios=randlanet_cfg.ratios
        )

        inputs = {
            'features': feats,
            'xyz': xyz_list,
            'neigh_idx': neigh_list,
            'sub_idx': sub_list,
            'interp_idx': interp_list
        }

        # --------------------------
        # Initialize RandLA-Net model
        # --------------------------
        model = RandlaNet(
            d_out=randlanet_cfg.d_out,
            n_layers=randlanet_cfg.n_layers,
            n_classes=randlanet_cfg.n_classes
        ).to(device)

        # --------------------------
        # Forward pass
        # --------------------------
        out = model(inputs)
        logger.info(f"Forward pass output shape: {out.shape}")
        return out


# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>> starting {STAGE_NAME} <<<<<<<<<<<")
        pipeline = RandLANetForwardPipeline()
        out = pipeline.main()
        logger.info(f">>>>>>>>> finished {STAGE_NAME} <<<<<<<<<<<<")
    except Exception as e:
        logger.exception(f"❌ Error during {STAGE_NAME}: {e}")
        raise e
