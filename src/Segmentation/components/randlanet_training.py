import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from src.Segmentation import logger
from src.Segmentation.config.configuration import ConfigurationManager
from src.Segmentation.components.randlanet_input import RandLANetPatchDataset
from src.Segmentation.components.randlanet import RandlaNet, build_pyramid_gpu_batch
import glob
import numpy as np

# ===============================================
# RandLA-Net Trainer
# ===============================================
class RandlaNetTrainer:
    """
    Trainer class for RandLA-Net:
    - Loads training dataset
    - Computes class weights for imbalanced data
    - Handles training loop with loss, optimizer, and metrics
    - Saves model checkpoints every epoch
    """
    def __init__(self):
        # Load training configuration
        self.cfg = ConfigurationManager().get_train_config()
        torch.manual_seed(self.cfg.seed)  # Ensure reproducibility
        self.device = torch.device(self.cfg.device if torch.cuda.is_available() else "cpu")

        # Create directory to save models
        os.makedirs(self.cfg.save_model_dir, exist_ok=True)
        logger.info(f"[INIT] RandlaNetTrainer initialized on device: {self.device}")

    # -------------------------
    # Compute class weights for imbalanced segmentation
    # -------------------------
    def compute_class_weights(self, root_dir):
        """
        Computes class weights using inverse frequency of labels across all chunks
        Returns a torch tensor of shape [num_classes] on the correct device
        """
        # Find all area folders containing segment.npy
        area_folders = sorted([d for d in glob.glob(os.path.join(root_dir, "*")) if os.path.isdir(d)])
        all_labels = []

        for area in area_folders:
            seg_file = os.path.join(area, "segment.npy")
            if os.path.exists(seg_file):
                all_labels.append(np.load(seg_file).flatten())  # flatten to 1D

        if len(all_labels) == 0:
            logger.error(f"No segment.npy files found in {root_dir}")
            raise ValueError(f"No segment.npy files found in {root_dir}")

        all_labels = np.concatenate(all_labels)
        counts = np.bincount(all_labels, minlength=self.cfg.n_classes)

        # Inverse frequency weighting to handle class imbalance
        class_weights = 1.0 / (counts + 1e-6)
        max_weight = 8.0
        class_weights = class_weights / np.max(class_weights) * max_weight

        # Convert to torch tensor on correct device
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        logger.info(f"[INFO] Device: {self.device}")
        logger.info(f"[INFO] Class weights: {np.round(class_weights, 6)}")  # rounded for readability

        return class_weights_tensor

    # -------------------------
    # Training loop
    # -------------------------
    def train(self):
        # Prepare dataset and DataLoader
        logger.info(f"[INFO] Preparing dataset from: {self.cfg.train_dir}")
        train_ds = RandLANetPatchDataset(self.cfg.train_dir, max_points=20000)
        train_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True, num_workers=0)
        logger.info(f"[INFO] Dataset loaded with {len(train_ds)} patches, batch size: {self.cfg.batch_size}")

        # Initialize model, optimizer, and loss function
        model = RandlaNet(d_out=self.cfg.d_out, n_layers=self.cfg.n_layers, n_classes=self.cfg.n_classes).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        # Compute class weights for CrossEntropyLoss
        class_weights = self.compute_class_weights(self.cfg.train_dir)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # -------------------------
        # Training epochs
        # -------------------------
        for epoch in range(1, self.cfg.epochs + 1):
            model.train()
            running_loss, n_points = 0.0, 0
            all_preds, all_labels = [], []

            start_time = time.time()

            # Adjust learning rate every 20 epochs
            if epoch % 20 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.9
                logger.info(f"[LR] Learning rate reduced to {optimizer.param_groups[0]['lr']:.6f} at epoch {epoch}")

            # -------------------------
            # Batch-wise training
            # -------------------------
            for i, batch in enumerate(train_loader, 1):
                xyz = batch["xyz"].to(self.device)
                feats = batch["feats"].to(self.device)
                labels = batch["labels"].to(self.device).view(-1)  # flatten to 1D

                # Build multi-scale pyramid for RandLA-Net
                xyz_list, neigh_list, sub_list, interp_list = build_pyramid_gpu_batch(
                    xyz_b=xyz,
                    k=self.cfg.k_neighbors,
                    n_layers=self.cfg.n_layers,
                    ratios=self.cfg.ratios,
                    pool_size=self.cfg.pool_size
                )

                inputs = {
                    "features": feats,
                    "xyz": xyz_list,
                    "neigh_idx": neigh_list,
                    "sub_idx": sub_list,
                    "interp_idx": interp_list
                }

                # Forward + backward pass
                optimizer.zero_grad()
                logits = model(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                # Accumulate metrics
                running_loss += loss.item() * labels.numel()
                n_points += labels.numel()
                all_preds.append(logits.argmax(dim=1).cpu())
                all_labels.append(labels.cpu())

            # -------------------------
            # Epoch-level metrics
            # -------------------------
            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()
            epoch_loss = running_loss / max(1, n_points)
            epoch_acc = (all_preds == all_labels).sum() / len(all_labels)
            epoch_f1 = f1_score(all_labels, all_preds, average="weighted")
            elapsed = time.time() - start_time

            logger.info(f"[EPOCH {epoch}/{self.cfg.epochs}] Loss: {epoch_loss:.4f}, "
                        f"Acc: {epoch_acc:.4f}, F1: {epoch_f1:.4f}, "
                        f"LR: {optimizer.param_groups[0]['lr']:.6f}, Time: {elapsed:.2f}s")

            # Save checkpoint
            save_path = os.path.join(self.cfg.save_model_dir, f"epoch_{epoch:03d}_randlanet.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"[MODEL SAVED] {save_path}")
