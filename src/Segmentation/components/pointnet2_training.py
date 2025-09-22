import os
import glob
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score

from src.Segmentation.config.configuration import ConfigurationManager
from src.Segmentation.components.pointnet2 import PointNet2SemSeg5
from src.Segmentation import logger
from src.Segmentation.components.pointnet2_input import PointCloudChunkNPZDataset


# ===================== Helper: Compute Class Weights =====================
def compute_class_weights(root_dir, n_classes):
    """
    Compute inverse-frequency class weights from NPZ dataset.
    This is used for CrossEntropyLoss to handle class imbalance.

    Args:
        root_dir (str): Path containing .npz chunk files
        n_classes (int): Number of semantic classes

    Returns:
        torch.Tensor: Class weights (size: n_classes)
    """
    npz_files = glob.glob(os.path.join(root_dir, "*.npz"))
    all_labels = []

    # Collect all labels from dataset
    for f in npz_files:
        data = np.load(f)
        all_labels.append(data['labels'].flatten())

    all_labels = np.concatenate(all_labels)
    counts = np.bincount(all_labels, minlength=n_classes)

    # Inverse frequency
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * n_classes  # normalize sum to n_classes
    return torch.tensor(weights, dtype=torch.float32)


# ===================== Training Pipeline =====================
class PointNet2TrainingStage:
    """
    Training class for PointNet++ (PointNet2SemSeg5) semantic segmentation.
    Includes:
        - Dataset loading
        - Model initialization
        - Class weighting for loss
        -  LR scheduler
        - Epoch-wise training with metrics logging

    """
    def __init__(self, train_dir):
        self.config_manager = ConfigurationManager()
        self.train_dir = train_dir

    def main(self):
        # --------------------------
        # Load training parameters
        # --------------------------
        params = self.config_manager.get_training_config()
        device = torch.device(params.device)
        batch_size = params.batch_size
        epochs = params.epochs
        lr = params.lr
        step_size = params.step_size
        gamma = params.gamma
        seed = params.seed
        save_model_dir = params.save_model_dir

        torch.manual_seed(seed)  # reproducibility

        # --------------------------
        # Initialize PointNet2 model
        # --------------------------
        pointnet_cfg = self.config_manager.get_pointnet2_model_config()
        num_classes = pointnet_cfg.num_classes
        model = PointNet2SemSeg5(pointnet_cfg).to(device)

        # --------------------------
        # Compute class weights for loss
        # --------------------------
        class_weights = compute_class_weights(self.train_dir, num_classes).to(device)
        max_weight = 8.0
        normalized_weights = class_weights / class_weights.max() * max_weight

        logger.info(f"Device: {device}")
        logger.info(f"Class weights: {np.round(normalized_weights.cpu().numpy(), 6)}")

        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # --------------------------
        # DataLoader
        # --------------------------
        train_dataset = PointCloudChunkNPZDataset(self.train_dir, return_raw_points=False)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        # --------------------------
        # Optimizer & LR Scheduler
        # --------------------------
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        # --------------------------
        # Create checkpoint directory
        # --------------------------
        os.makedirs(save_model_dir, exist_ok=True)

        # --------------------------
        # Training loop
        # --------------------------
        for epoch in range(epochs):
            start_time = time.time()
            model.train()
            running_loss = 0.0
            all_preds, all_labels = [], []

            for batch_points, batch_labels in train_loader:
                batch_points = batch_points.to(device)
                batch_labels = batch_labels.to(device)

                # Split points into xyz and extra features
                xyz = batch_points[:, :, :3].permute(0, 2, 1)
                features = batch_points[:, :, 3:].permute(0, 2, 1)

                optimizer.zero_grad()
                logits = model(xyz, features)

                # Ensure labels are in valid range [0, num_classes-1]
                batch_labels = torch.clamp(batch_labels, min=0, max=num_classes - 1)

                # Compute loss & backpropagate
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Track predictions for metrics
                preds = logits.argmax(dim=1).detach().cpu().numpy().reshape(-1)
                labels = batch_labels.detach().cpu().numpy().reshape(-1)
                all_preds.append(preds)
                all_labels.append(labels)

            scheduler.step()  # update LR

            # Compute epoch metrics
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            train_acc = accuracy_score(all_labels, all_preds)
            train_f1 = f1_score(all_labels, all_preds, average='weighted')

            # Save model checkpoint
            ckpt_path = os.path.join(save_model_dir, f"epoch_{epoch + 1:03d}_pointnet2.pth")
            torch.save(model.state_dict(), ckpt_path)

            # Logging
            logger.info(f"Epoch {epoch+1}/{epochs}, "
                        f"Loss: {running_loss/len(train_loader):.4f}, "
                        f"Acc: {train_acc:.4f}, F1: {train_f1:.4f}, "
                        f"Saved: {ckpt_path}, Time: {time.time() - start_time:.2f}s")

        logger.info("Training complete!")
