import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.Segmentation.components.randlanet import RandlaNet, build_pyramid_gpu_batch
from src.Segmentation.components.randlanet_input import RandLANetPatchDataset
from src.Segmentation.config.configuration import RandLANetTestConfig
from src.Segmentation import logger
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


# ===============================================
# Compute per-class IoU for segmentation results
# ===============================================
def compute_iou_per_class(y_true, y_pred, num_classes):
    """
    Compute Intersection over Union (IoU) per class.
    Returns per-class IoU list and mean IoU (mIoU)
    """
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    ious = []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        denom = tp + fp + fn
        ious.append(tp / denom if denom != 0 else float("nan"))
    return ious, np.nanmean(ious)


# ===============================================
# RandlaNet Tester Class
# ===============================================
class RandlaNetTester:
    """
    Handles testing of a trained RandLA-Net model:
    - Loads test dataset
    - Runs inference
    - Computes metrics (Accuracy, F1, IoU)
    - Saves merged numeric results, metrics table, and confusion matrix
    """
    def __init__(self, config: RandLANetTestConfig, class_names):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names

    def test(self):
        # Ensure output directory exists
        os.makedirs(self.config.save_pred_dir, exist_ok=True)

        # Extract epoch number from model filename
        epoch_match = re.search(r'epoch_(\d+)_', os.path.basename(self.config.model_path))
        epoch_number = int(epoch_match.group(1)) if epoch_match else -1
        logger.info(f"Model Epoch: {epoch_number}")

        # =========================
        # Prepare test dataset & loader
        # =========================
        test_ds = RandLANetPatchDataset(self.config.test_dir, max_points=20000)
        test_loader = DataLoader(test_ds, batch_size=self.config.batch_size, shuffle=False)

        # =========================
        # Load RandLA-Net model
        # =========================
        d_out = [16, 64, 128, 256]
        model = RandlaNet(d_out=d_out, n_layers=4, n_classes=self.config.num_classes).to(self.device)
        model.load_state_dict(torch.load(self.config.model_path, map_location=self.device))
        model.eval()  # Set model to evaluation mode

        # Containers for merging outputs
        all_preds, all_labels = [], []
        merged_numeric, merged_gt_names, merged_pred_names = [], [], []

        # =========================
        # Inference loop
        # =========================
        with torch.no_grad():
            for batch in test_loader:
                xyz = batch["xyz"].to(self.device)
                feats = batch["feats"].to(self.device)
                labels = batch["labels"].to(self.device)
                labels_flat = labels.view(-1)

                # Build multi-scale pyramid of points
                xyz_list, neigh_list, sub_list, interp_list = build_pyramid_gpu_batch(
                    xyz_b=xyz, k=16, n_layers=4, ratios=[4, 4, 4, 4], pool_size=16
                )

                # Prepare model inputs
                inputs = {"features": feats, "xyz": xyz_list,
                          "neigh_idx": neigh_list, "sub_idx": sub_list, "interp_idx": interp_list}

                # Forward pass
                logits = model(inputs)
                preds = logits.argmax(dim=1).cpu().numpy()
                gt = labels_flat.cpu().numpy()

                # Save batch-wise numeric and string outputs
                for i in range(xyz.shape[0]):
                    coords_np = batch["xyz"][i].cpu().numpy().astype(np.float32)
                    feats_np = batch["feats"][i].cpu().numpy().astype(np.float32)
                    gt_np = batch["labels"][i].cpu().numpy().astype(np.int32)
                    pred_np = preds.reshape(labels.shape)[i].astype(np.int32)

                    # Extract features
                    feats_np = feats_np[:, 3:]  # remove coordinates
                    rgb = (feats_np[:, 0:3] * 255).clip(0, 255).astype(np.int32)
                    normals = feats_np[:, 3:6].astype(np.float32)
                    strength = feats_np[:, 6:7].astype(np.float32)

                    # Numeric output: coords + features + GT + Pred
                    numeric_output = np.hstack([
                        coords_np, rgb, normals, strength,
                        gt_np.reshape(-1, 1), pred_np.reshape(-1, 1)
                    ])
                    merged_numeric.append(numeric_output)

                    # String labels
                    merged_gt_names.append(np.array([self.class_names[x] for x in gt_np]))
                    merged_pred_names.append(np.array([self.class_names[x] for x in pred_np]))

                all_preds.append(preds)
                all_labels.append(gt)

        # =========================
        # Merge all batch outputs
        # =========================
        merged_numeric = np.vstack(merged_numeric)
        merged_gt_names = np.concatenate(merged_gt_names).reshape(-1, 1)
        merged_pred_names = np.concatenate(merged_pred_names).reshape(-1, 1)

        df_merged = pd.DataFrame(
            merged_numeric,
            columns=["X", "Y", "Z", "R", "G", "B", "Nx", "Ny", "Nz", "Strength", "GT_Label", "Pred_Label"]
        )
        df_merged["GT_Class"] = merged_gt_names
        df_merged["Pred_Class"] = merged_pred_names

        # Save merged numeric + string results CSV
        merged_file_path = os.path.join(self.config.save_pred_dir, f"merged_test_results_epoch_{epoch_number}.csv")
        df_merged.to_csv(merged_file_path, index=False)
        logger.info(f"✅ Merged test predictions saved: '{merged_file_path}'")

        # =========================
        # Metrics computation
        # =========================
        all_preds_flat = np.concatenate(all_preds)
        all_labels_flat = np.concatenate(all_labels)

        # Compute Precision, Recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels_flat, all_preds_flat, labels=np.arange(self.config.num_classes), zero_division=0
        )

        # Per-class accuracy
        per_class_acc = []
        for i in range(self.config.num_classes):
            idx = all_labels_flat == i
            per_class_acc.append(np.sum(all_preds_flat[idx] == i) / np.sum(idx) if np.sum(idx) > 0 else np.nan)

        # IoU
        ious, miou = compute_iou_per_class(all_labels_flat, all_preds_flat, self.config.num_classes)

        # Create metrics table
        table_data = []
        for i in range(self.config.num_classes):
            acc = per_class_acc[i] * 100 if not np.isnan(per_class_acc[i]) else 0.0
            f1_score_val = f1[i] * 100 if not np.isnan(f1[i]) else 0.0
            iou_score = ious[i] * 100 if not np.isnan(ious[i]) else 0.0
            sup = support[i]
            table_data.append([self.class_names[i], acc, f1_score_val, iou_score, sup])

        df_metrics = pd.DataFrame(table_data, columns=["Class", "Accuracy (%)", "F1-score (%)", "IoU (%)", "Support"])
        df_metrics.loc["Mean"] = ["--", np.nanmean(per_class_acc) * 100, np.nanmean(f1) * 100, miou * 100,
                                  np.sum(support)]

        # Save metrics CSV
        metrics_file = os.path.join(self.config.save_pred_dir, f"metrics_epoch_{epoch_number}.csv")
        df_metrics.to_csv(metrics_file, index=False)
        logger.info(f"✅ Metrics table saved: {metrics_file}")

        # =========================
        # Confusion matrix plot
        # =========================
        cm = confusion_matrix(all_labels_flat, all_preds_flat, labels=np.arange(self.config.num_classes))
        cm_percent = cm.astype(float) / cm.sum(axis=1)[:, None] * 100

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm_percent, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"RandLA-Net - Confusion Matrix (%)")
        cm_file = os.path.join(self.config.save_pred_dir, f"confusion_matrix_epoch_{epoch_number}.png")
        plt.savefig(cm_file, bbox_inches="tight")
        plt.close()
        logger.info(f"✅ Confusion matrix saved: {cm_file}")

        return merged_file_path, metrics_file, cm_file
