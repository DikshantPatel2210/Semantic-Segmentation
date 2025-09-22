from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


"""
======================================================
Configuration Entities for Semantic Segmentation
======================================================

This module defines all the configuration entities (dataclasses) used throughout
the Semantic Segmentation project. These entities serve as structured representations
of all configurable parameters required at different stages of the pipeline, including
data preprocessing, dataset organization, model architecture, training, and testing. 

By using these dataclasses, the project ensures that all settings are type-safe, 
clearly organized, and easily accessible. Each entity encapsulates related parameters,
such as rotation or normalization settings for preprocessing, architecture details 
for specific models (PointNet++ or RandLANet), and hyperparameters for training or testing. 

Overall, this module provides a centralized and standardized way to manage 
all configurations, reduces the chance of errors from manual parameter handling, 
and improves maintainability and readability across the entire codebase.
"""


@dataclass(frozen=True)
class DirectoryStructureConfig:
    root_dir: Path
    train_dir: Path
    test_dir: Path


@dataclass(frozen=True)
class GPUCheckerConfig:
    fail_on_no_gpu: bool = False
    required_min_cuda_version: Optional[str] = None
    required_min_pytorch_version: Optional[str] = None

@dataclass(frozen=True)
class PLYProcessorConfig:
    base_dir: Path = Path("artifacts")
    output_dir: Path = Path("dataset")
    save_normals: bool = True

@dataclass(frozen=True)
class RotationConfig:
    enabled: bool
    direction: list  # ["x", "y", "z"]
    angle: float     # degrees
    target_files: list  # list of files or ["all"]
    base_dir: Path
    output_dir: Path

@dataclass(frozen=True)
class NormalizationConfig:
    enabled: bool
    target_files: list  # ["all"] or list of files
    base_dir: Path      # Root of dataset (dataset/)
    vis_dir: Path       # Directory to save visualizations
    targets: list       # ["coords", "color", "intensity"]
    coord_method: str   # "minmax", "zscore", "custom"
    color_method: str   # "minmax", "zscore"
    intensity_method: str  # "minmax", "zscore"


@dataclass(frozen=True)
class LabelMappingConfig:
    enabled: bool
    base_dir: Path
    target_files: list        # ["all"] or specific rooms
    splits: list              # ["train", "test"]
    output_dir: Path           # Where to save JSON mapping files

@dataclass(frozen=True)
class ChunkingConfig:
    enabled: bool
    base_dir: Path
    splits: list                  # ["train", "test"]
    target_files: list            # ["all"] or list of specific rooms
    chunk_range: tuple            # e.g., (6,6)
    chunk_stride: tuple           # e.g., (3,3)
    chunk_minimum_size: int       # minimum points per chunk
    grid_size: float = None       # optional grid subsampling
    num_workers: int = None       # optional multiprocessing workers
    run_on: str = "both"          # "train", "test", or "both"

# =========================
# Config
# =========================
@dataclass(frozen=True)
class Pointnet2InputConfig:
    enabled: bool
    base_dir: Path
    chunk_size: int
    augment: bool

@dataclass(frozen=True)
class RandLANetInputConfig:
    base_dir: Path
    use_norm: bool = True
    max_points: int = 20000

from dataclasses import dataclass
from typing import List

@dataclass
class RandlaNetConfig:
    d_out: List[int]
    n_layers: int
    n_classes: int
    k: int
    ratios: List[int]
    pool_size: int = 16
    dropout: float = 0.5
    seed: int = 42
    device: str = "cuda"  # or "cpu"


@dataclass
class PointNet2ModelConfig:
    npoint_sa1: int
    radius_sa1: float
    nsample_sa1: int
    in_channel_sa1: int
    mlp_sa1: list

    npoint_sa2: int
    radius_sa2: float
    nsample_sa2: int
    in_channel_sa2: int
    mlp_sa2: list

    npoint_sa3: int
    radius_sa3: float
    nsample_sa3: int
    in_channel_sa3: int
    mlp_sa3: list

    fp3_channels: list
    fp2_channels: list
    fp1_channels: list
    num_classes: int

@dataclass(frozen=True)
class RandlaNetTrainConfig:
    train_dir: Path
    test_dir: Path
    n_classes: int
    n_layers: int
    d_out: List[int]
    k_neighbors: int
    ratios: List[int]
    pool_size: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    save_model_dir: Path
    seed: int = 0
    device: str = "cuda"

@dataclass
class PointNet2TrainingConfig:
    # Directory info
    base_dir: str
    train_dir: str
    test_dir: str
    save_model_dir: str
    # Training hyperparameters
    batch_size: int
    epochs: int
    lr: float
    step_size: int
    gamma: float
    seed: int
    device: str

@dataclass(frozen=True)
class RandLANetTestConfig:
    model_type: str
    model_path: Path
    test_dir: Path
    save_pred_dir: Path
    num_classes: int
    batch_size: int = 1

@dataclass
class PointNet2TestConfig:
    model_type: str
    model_path: Path
    test_dir: Path
    save_pred_dir: Path
    batch_size: int = 2
    num_classes: int = 14
    class_names: Optional[List[str]] = None