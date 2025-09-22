import os
import json
import joblib
import yaml
import numpy as np
import torch
import open3d as o3d
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations
from pathlib import Path
from typing import Any
import pandas as pd
from src.Segmentation import logger

"""
======================================================
Common Utility Functions for Semantic Segmentation
======================================================

This module provides a collection of reusable utility functions for the 
Semantic Segmentation project. It centralizes file I/O, directory management, 
and data serialization/deserialization tasks to improve code readability, 
reusability, and maintainability across the pipeline. 

For example, instead of writing code to read a YAML file every time, 
these utilities provide a simple and consistent way to perform such tasks 
anytime they are needed, reducing redundancy and potential errors.
"""


# ---------------- YAML ----------------
@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Read a YAML file and return as a ConfigBox."""
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file loaded successfully from: {path_to_yaml}")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("YAML file is empty")
    except Exception as e:
        raise e


# ---------------- CSV ----------------
@ensure_annotations
def save_dataframe_to_csv(df: pd.DataFrame, filepath: Path, index: bool = False, verbose: bool = True):
    """Save a DataFrame to CSV."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=index)
    if verbose:
        logger.info(f"CSV file saved at: {filepath}")



# ---------------- Directories ----------------
@ensure_annotations
def create_directories(paths, verbose: bool = True):

    for path in paths:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Directory created at: {path}")


# ---------------- JSON ----------------
@ensure_annotations
def save_json(path: Path, data: dict):
    """Save dictionary as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON file saved at: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Load JSON file and return as ConfigBox."""
    with open(path) as f:
        content = json.load(f)
    logger.info(f"JSON file loaded from: {path}")
    return ConfigBox(content)


# ---------------- Binary ----------------
@ensure_annotations
def save_bin(data: Any, path: Path):
    """Save Python object as binary using joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """Load binary data saved by joblib."""
    data = joblib.load(path)
    logger.info(f"Binary file loaded from: {path}")
    return data


# ---------------- File Info ----------------
@ensure_annotations
def get_size(path: Path) -> str:
    """Get file size in KB."""
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"


# ---------------- PLY ----------------
@ensure_annotations
def read_ply(path: Path) -> o3d.geometry.PointCloud:
    """Read a .ply file as Open3D PointCloud."""
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    pcd = o3d.io.read_point_cloud(str(path))
    logger.info(f".ply file loaded successfully from: {path}")
    return pcd


@ensure_annotations
def save_ply(pcd: o3d.geometry.PointCloud, path: Path, ascii: bool = True):
    """Save Open3D PointCloud as .ply file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=ascii)
    logger.info(f".ply file saved at: {path}")


# ---------------- NumPy ----------------
@ensure_annotations
def save_numpy(array: np.ndarray, path: Path):
    """Save numpy array as .npy file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), array)
    logger.info(f".npy file saved at: {path}")


@ensure_annotations
def load_numpy(path: Path) -> np.ndarray:
    """Load numpy array from .npy file."""
    array = np.load(str(path))
    logger.info(f".npy file loaded from: {path}")
    return array


@ensure_annotations
def save_npz(path: Path, **arrays):
    """Save multiple numpy arrays to .npz file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), **arrays)
    logger.info(f".npz file saved at: {path}")


@ensure_annotations
def load_npz(path: Path) -> dict:
    """Load arrays from a .npz file."""
    loaded = np.load(str(path))
    arrays = {key: loaded[key] for key in loaded.files}
    logger.info(f".npz file loaded from: {path}")
    return arrays


# ---------------- PyTorch ----------------
@ensure_annotations
def save_pth(obj: Any, path: Path):
    """Save PyTorch object (model/state_dict/tensor) as .pth file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, str(path))
    logger.info(f".pth file saved at: {path}")


@ensure_annotations
def load_pth(path: Path, map_location: str = "cpu") -> Any:
    """Load PyTorch object from .pth file."""
    obj = torch.load(str(path), map_location=map_location)
    logger.info(f".pth file loaded from: {path}")
    return obj
