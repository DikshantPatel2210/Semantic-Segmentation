from src.Segmentation.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.Segmentation.utils.common import read_yaml, create_directories
from src.Segmentation.entity.config_entity import (DirectoryStructureConfig,GPUCheckerConfig,
                                                   PLYProcessorConfig, RotationConfig,NormalizationConfig,
                                                   LabelMappingConfig, ChunkingConfig, Pointnet2InputConfig,
                                                   RandLANetInputConfig,RandlaNetConfig,RandlaNetTrainConfig,
                                                   PointNet2ModelConfig, PointNet2TrainingConfig,RandLANetTestConfig,
                                                   PointNet2TestConfig
                                                   )
import yaml
from src.Segmentation import logger
from pathlib import Path
import torch
import numpy as np


"""
======================================================
ConfigurationManager
======================================================

This module defines the `ConfigurationManager` class, which is responsible 
for loading, parsing, and providing access to all configuration parameters 
from `config.yaml` and `params.yaml` for the Semantic Segmentation project. 
"""

class ConfigurationManager:
    def __init__(self,
                 config_filepath=CONFIG_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH):
        # Load YAML files
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        logger.info(f"YAML files loaded successfully: {config_filepath}, {params_filepath}")

    def directory_structure(self) -> DirectoryStructureConfig:
        """Return directory structure config"""
        try:
            cfg = self.config.Directory_structure
            return DirectoryStructureConfig(
                root_dir=Path(cfg.base_dir),
                train_dir=Path(cfg.train_dir),
                test_dir=Path(cfg.test_dir)
            )
        except Exception as e:
            logger.exception(f"❌ Failed to read directory structure from config.yaml: {e}")
            raise

    def gpu_checker(self) -> GPUCheckerConfig:
        """
        Returns a GPUCheckerConfig object based on config.yaml
        """
        try:
            config = self.config.GPU_checker

            gpu_checker_config = GPUCheckerConfig(
                fail_on_no_gpu=config.get("fail_on_no_gpu", False),
                required_min_cuda_version=config.get("required_min_cuda_version"),
                required_min_pytorch_version=config.get("required_min_pytorch_version")
            )
            logger.info("✅ GPU checker config loaded successfully")
            return gpu_checker_config

        except Exception as e:
            logger.error(f"❌ Failed to read GPU checker config from config.yaml: {e}")
            raise



    def ply_processor_config(self) -> PLYProcessorConfig:
        """
        Returns a PLYProcessorConfig object based on config.yaml
        """
        try:
            # base_dir from artifacts_roots
            base_dir = Path(self.config.get("artifacts_roots", "artifacts"))

            # output_dir and save_normals from ply_to_npy
            ply_cfg = self.config.get("ply_to_npy", {})
            output_dir = Path(ply_cfg.get("output_dir", "dataset"))
            save_normals = ply_cfg.get("save_normals", True)

            return PLYProcessorConfig(base_dir=base_dir, output_dir=output_dir, save_normals=save_normals)
        except Exception as e:
            logger.error(f"❌ Failed to load PLYProcessor config: {e}")
            raise



    def rotation_config(self) -> RotationConfig:
        """
        Returns a RotationConfig object based on config.yaml
        """
        try:
            rotation_cfg = self.params.get("preprocessing", {}).get("rotation", {})
            enabled = rotation_cfg.get("enabled", False)
            direction = rotation_cfg.get("direction", ["z"])
            angle = rotation_cfg.get("angle", 0)
            target_files = rotation_cfg.get("target_files", ["all"])

            base_dir = Path(self.params.get("artifacts_roots", "artifacts"))
            output_dir = base_dir / "rotated"

            return RotationConfig(
                enabled=enabled,
                direction=direction,
                angle=angle,
                target_files=target_files if isinstance(target_files, list) else [target_files],
                base_dir=base_dir,
                output_dir=output_dir
            )
        except Exception as e:
            logger.error(f"❌ Failed to load Rotation config: {e}")
            raise

    def normalization_config(self) -> NormalizationConfig:
        norm_cfg = self.params.get("preprocessing", {}).get("normalization", {})
        enabled = norm_cfg.get("enabled", False)
        target_files = norm_cfg.get("target_files", ["all"])
        targets = norm_cfg.get("targets", ["coord"])
        coord_method = norm_cfg.get("coord_method", "custom")
        color_method = norm_cfg.get("color_method", "minmax")
        intensity_method = norm_cfg.get("intensity_method", "minmax")
        base_dir = Path(self.params.get("dataset_dir", "dataset"))
        vis_dir = Path("artifacts/visualizations")
        create_directories([vis_dir])

        return NormalizationConfig(
            enabled=enabled,
            target_files=target_files if isinstance(target_files, list) else [target_files],
            base_dir=base_dir,
            vis_dir=vis_dir,
            targets=targets,
            coord_method=coord_method,
            color_method=color_method,
            intensity_method=intensity_method
        )

    def label_mapping_config(self) -> LabelMappingConfig:
        cfg = self.params.get("preprocessing", {}).get("label_mapping", {})
        enabled = cfg.get("enabled", True)
        target_files = cfg.get("target_files", ["all"])
        base_dir = Path(self.params.get("dataset_root", self.params.get("preprocessing", {}).get("dataset_dir", "dataset")))
        splits = cfg.get("splits", ["train", "test"])
        output_dir = Path(cfg.get("output_dir", "artifacts/label_maps"))
        create_directories([output_dir])
        return LabelMappingConfig(
            enabled=enabled,
            base_dir=base_dir,
            target_files=target_files if isinstance(target_files, list) else [target_files],
            splits=splits,
            output_dir=output_dir
        )



    def chunking_config(self) -> ChunkingConfig:
        try:

            params = self.params.get("preprocessing", {}).get("chunking", {})

            enabled = params.get("enabled", True)
            target_files = params.get("target_files", ["all"])
            base_dir = Path(params.get("base_dir", "dataset"))
            splits = params.get("splits", ["train", "test"])

            chunk_range = tuple(params.get("chunk_range", (6, 6)))
            chunk_stride = tuple(params.get("chunk_stride", (3, 3)))
            chunk_minimum_size = params.get("chunk_minimum_size", 10000)
            grid_size = params.get("grid_size", None)
            num_workers = params.get("num_workers", None)

            run_on = params.get("run_on", "both")  # new option

            # Determine splits based on run_on
            if run_on == "train":
                splits = ["train"]
            elif run_on == "test":
                splits = ["test"]
            # else keep both (default)

            return ChunkingConfig(
                enabled=enabled,
                base_dir=base_dir,
                splits=splits,
                target_files=target_files if isinstance(target_files, list) else [target_files],
                chunk_range=chunk_range,
                chunk_stride=chunk_stride,
                chunk_minimum_size=chunk_minimum_size,
                grid_size=grid_size,
                num_workers=num_workers,
                run_on=run_on
            )
        except Exception as e:
            logger.error(f"❌ Failed to load Chunking config: {e}")
            raise


    def pointnet2_input_config(self) -> Pointnet2InputConfig:
        params = self.params.get("preprocessing", {}).get("pointnet2_input", {})
        return Pointnet2InputConfig(
            enabled=params.get("enabled", True),
            base_dir=Path(params.get("base_dir", "dataset")),
            chunk_size=params.get("chunk_size", 4096),
            augment=params.get("augment", False),
        )

    def randlanet_input_config(self) -> RandLANetInputConfig:
        params = self.params.get("preprocessing", {}).get("randlanet_input", {})
        return RandLANetInputConfig(
            base_dir=Path(params.get("base_dir", "dataset")),
            use_norm=params.get("use_norm", True),
            max_points=params.get("max_points", 20000)
        )

    def get_randlanet_config(self) -> RandlaNetConfig:
        randla_cfg = self.params["model"]["randlanet_arc"]
        return RandlaNetConfig(
            d_out=randla_cfg["d_out"],
            n_layers=randla_cfg["n_layers"],
            n_classes=randla_cfg["n_classes"],
            k=randla_cfg["k"],
            ratios=randla_cfg["ratios"],
            pool_size=randla_cfg.get("pool_size", 16),
            dropout=randla_cfg.get("dropout", 0.5),
            seed=randla_cfg.get("seed", 42),
            device=randla_cfg.get("device", "cuda")
        )

    def get_pointnet2_model_config(self) -> PointNet2ModelConfig:
        params = self.params.get("model", {}).get("pointnet2_model", {})
        config = self.config.get("models", {}).get("pointnet2", {})

        return PointNet2ModelConfig(
            npoint_sa1=params["npoint_sa1"],
            radius_sa1=params["radius_sa1"],
            nsample_sa1=params["nsample_sa1"],
            in_channel_sa1=params["in_channel_sa1"],
            mlp_sa1=params["mlp_sa1"],

            npoint_sa2=params["npoint_sa2"],
            radius_sa2=params["radius_sa2"],
            nsample_sa2=params["nsample_sa2"],
            in_channel_sa2=params["in_channel_sa2"],
            mlp_sa2=params["mlp_sa2"],

            npoint_sa3=params["npoint_sa3"],
            radius_sa3=params["radius_sa3"],
            nsample_sa3=params["nsample_sa3"],
            in_channel_sa3=params["in_channel_sa3"],
            mlp_sa3=params["mlp_sa3"],

            fp3_channels=params["fp3_channels"],
            fp2_channels=params["fp2_channels"],
            fp1_channels=params["fp1_channels"],
            num_classes=params["num_classes"],
        )


    def get_train_config(self) -> RandlaNetTrainConfig:
        # Architecture parameters
        arc_cfg = self.params.get("model", {}).get("randlanet_arc", {})

        # Training parameters
        train_cfg = self.params.get("model", {}).get("training", {})
        return RandlaNetTrainConfig(
            train_dir=Path(train_cfg.get("train_dir", "./dataset/train/chunks")),
            test_dir=Path(train_cfg.get("test_dir", "./dataset/test/chunks")),
            n_classes=int(arc_cfg.get("n_classes", 16)),
            n_layers=int(arc_cfg.get("n_layers", 4)),
            d_out=[int(x) for x in arc_cfg.get("d_out", [16, 64, 128, 256])],
            k_neighbors=int(arc_cfg.get("k", 16)),
            ratios=[int(x) for x in arc_cfg.get("ratios", [4, 4, 4, 4])],
            pool_size=int(arc_cfg.get("pool_size", 16)),
            batch_size=int(train_cfg.get("batch_size", 2)),
            epochs=int(train_cfg.get("epochs", 200)),
            lr=float(train_cfg.get("lr", 1e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
            save_model_dir=Path(train_cfg.get("save_model_dir", "./saved_models_randlanet")),
            seed=int(arc_cfg.get("seed", 0)),
            device=str(arc_cfg.get("device", "cuda"))
        )

    def get_training_config(self) -> PointNet2TrainingConfig:
        training_params = self.params.get("model", {}).get("pointnet2_training", {})
        dirs = training_params.get("training", {})

        return PointNet2TrainingConfig(
            base_dir=dirs.get("base_dir", "dataset"),
            train_dir=dirs.get("train_dir", "dataset/train/same_size_chunks"),
            test_dir=dirs.get("test_dir", "dataset/test/same_size_chunks"),
            save_model_dir=dirs.get("save_model_dir", "artifacts/saved_models_pointnet2"),

            batch_size=training_params.get("batch_size", 3),
            epochs=training_params.get("epochs", 120),
            lr=training_params.get("lr", 0.001),
            step_size=training_params.get("step_size", 20),
            gamma=training_params.get("gamma", 0.5),
            seed=training_params.get("seed", 42),
            device=training_params.get("device", "cuda")
        )

    def get_randlanet_test_config(self) -> RandLANetTestConfig:
        """Fetch randlanet_test config from params.yaml"""
        cfg = self.params.get("model", {}).get("randlanet_test", {})
        if not cfg:
            raise ValueError("Test config for 'randlanet_test' not found in params.yaml")

        return RandLANetTestConfig(
            model_type=cfg.get("model_type"),
            model_path=Path(cfg.get("model_path")),
            test_dir=Path(cfg.get("test_dir")),
            save_pred_dir=Path(cfg.get("save_pred_dir")),
            num_classes=cfg.get("num_classes"),
            batch_size=cfg.get("batch_size", 1)
        )

    def get_class_names(self):
        """Return class names from config.yaml"""
        class_names = self.config.get("class_names", [])
        if not class_names:
            raise ValueError("No 'class_names' found in config.yaml")
        return class_names

    def get_test_pointnet2_config(self) -> PointNet2TestConfig:
        test_cfg = self.params.get("model", {}).get("pointnet2_test", {})
        class_names = self.params.get("class_names", [])

        if not test_cfg:
            raise ValueError("❌ Test config for 'pointnet2' not found in params.yaml")

        return PointNet2TestConfig(
            model_type=test_cfg.get("model_type", "pointnet2"),
            model_path=Path(test_cfg["model_path"]),
            test_dir=Path(test_cfg["test_dir"]),
            save_pred_dir=Path(test_cfg["save_pred_dir"]),
            batch_size=test_cfg.get("batch_size", 2),
            num_classes=test_cfg.get("num_classes", len(class_names)),
            class_names=class_names if class_names else [f"Class {i}" for i in range(test_cfg.get("num_classes", 14))]
        )  # <- Closing parenthesis added here
