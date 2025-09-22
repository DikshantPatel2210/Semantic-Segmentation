import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.Segmentation.components.pointnet2 import PointNet2SemSeg5
from src.Segmentation import logger
from src.Segmentation.components.pointnet2_input import PointCloudChunkNPZDataset
from src.Segmentation.entity.config_entity import PointNet2TestConfig
from src.Segmentation.config.configuration import ConfigurationManager


class PointNetTester:
    """
    Tester class for running semantic segmentation inference using PointNet++ (PointNet2SemSeg5).
    Handles:
        - Loading pre-trained model
        - Running inference on chunked NPZ datasets
        - Saving merged CSV with predictions
        - Computing per-class metrics (Accuracy, F1, IoU)
        - Saving confusion matrix plot
    """
    def __init__(self, config: PointNet2TestConfig):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(self.cfg.save_pred_dir, exist_ok=True)  # ensure save directory exists

    def run(self):
        """
        Main method to run testing on the dataset.
        Returns:
            Path to the merged CSV with predictions.
        """
        # Load NPZ dataset for testing (with raw points for merging later)
        dataset = PointCloudChunkNPZDataset(self.cfg.test_dir, return_raw_points=True)
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=4)

        # Load PointNet2 model configuration and override number of classes
        cfg_manager = ConfigurationManager()
        model_cfg = cfg_manager.get_pointnet2_model_config()
        model_cfg.num_classes = self.cfg.num_classes

        # Initialize model and load pre-trained weights
        model = PointNet2SemSeg5(model_cfg).to(self.device)
        model.load_state_dict(torch.load(self.cfg.model_path, map_location=self.device))
        model.eval()  # set model to evaluation mode

        # Containers to accumulate predictions and labels
        all_preds, all_labels, merged_points = [], [], []

        # Iterate over test batches
        for batch_points, batch_labels, original_points_list in loader:
            batch_points, batch_labels = batch_points.to(self.device), batch_labels.to(self.device)

            # Split batch into coordinates and additional features
            xyz = batch_points[:, :, :3].permute(0, 2, 1)  # (B, 3, N)
            feats = batch_points[:, :, 3:].permute(0, 2, 1)  # (B, F, N)

            with torch.no_grad():
                logits = model(xyz, feats)  # forward pass

            preds = logits.argmax(dim=1).cpu().numpy()  # predicted class per point
            labels = batch_labels.cpu().numpy()         # ground truth labels

            # Merge predictions with original point cloud for saving
            for i, orig_points in enumerate(original_points_list):
                points_np = orig_points.copy() if isinstance(orig_points, np.ndarray) else orig_points.numpy()
                pred_labels = preds[i].reshape(-1, 1)
                gt_labels = labels[i].reshape(-1, 1)
                merged_points.append(np.hstack([points_np, gt_labels, pred_labels]))

            # Accumulate for metrics calculation
            all_preds.append(preds.reshape(-1))
            all_labels.append(labels.reshape(-1))

        # Convert merged points to single numpy array
        merged_points = np.vstack(merged_points)

        # Save CSV of predictions merged with original point cloud
        merged_csv_path = self.save_merged_csv(merged_points)

        # Compute per-class metrics and confusion matrix
        self.compute_metrics(all_preds, all_labels, merged_csv_path)

        return merged_csv_path

    def save_merged_csv(self, merged_points):
        """
        Save a CSV file with original point coordinates, features, GT and predicted labels.
        Denormalizes RGB & Intensity values from [0,1] to [0,255].
        """
        def denormalize(val):
            try:
                return int(float(val) * 255)
            except:
                return 0

        # Append predicted class names for readability
        df = pd.DataFrame(
            np.hstack([
                merged_points,
                np.array([self.cfg.class_names[int(x)] for x in merged_points[:, -1]]).reshape(-1, 1)
            ]),
            columns=["X", "Y", "Z", "R", "G", "B", "Nx", "Ny", "Nz", "Intensity",
                     "GT_Label", "Pred_Label", "Pred_Class"]
        )

        # Denormalize color/intensity columns
        for col in ["R", "G", "B", "Intensity"]:
            df[col] = df[col].apply(denormalize)

        merged_csv_path = os.path.join(self.cfg.save_pred_dir, "merged_results.csv")
        df.to_csv(merged_csv_path, index=False)
        logger.info(f"✅ Merged CSV with denormalized RGB & Intensity saved at: {merged_csv_path}")
        return merged_csv_path

    def compute_metrics(self, all_preds, all_labels, merged_csv_path):
        """
        Compute per-class metrics and save CSV + confusion matrix.
        Metrics:
            - Accuracy per class
            - F1-score per class
            - Intersection over Union (IoU) per class
            - Confusion matrix (percentage)
        """
        all_preds_flat = np.concatenate(all_preds)
        all_labels_flat = np.concatenate(all_labels)

        # Precision, recall, F1-score
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels_flat, all_preds_flat, labels=np.arange(self.cfg.num_classes), zero_division=0
        )

        # Confusion matrix
        cm = confusion_matrix(all_labels_flat, all_preds_flat, labels=np.arange(self.cfg.num_classes))

        # IoU calculation per class
        ious = [
            (cm[i, i] / (cm[i, :].sum() + cm[:, i].sum() - cm[i, i]))
            if (cm[i, :].sum() + cm[:, i].sum() - cm[i, i]) > 0 else np.nan
            for i in range(self.cfg.num_classes)
        ]
        miou = np.nanmean(ious)

        # Prepare metrics DataFrame
        df_metrics = pd.DataFrame({
            "Class": self.cfg.class_names,
            "Accuracy (%)": [
                np.sum(all_preds_flat[all_labels_flat == i] == i) / np.sum(all_labels_flat == i) * 100
                if np.sum(all_labels_flat == i) > 0 else np.nan
                for i in range(self.cfg.num_classes)
            ],
            "F1-score (%)": [x * 100 for x in f1],
            "IoU (%)": [x * 100 for x in ious],
            "Support": support
        })

        # Save metrics CSV
        metrics_file = os.path.join(self.cfg.save_pred_dir, "metrics.csv")
        df_metrics.to_csv(metrics_file, index=False)
        logger.info(f"✅ Metrics CSV saved at: {metrics_file}")
        logger.info(f"\n{df_metrics.to_string(index=False)}")

        # Plot normalized confusion matrix
        cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues",
                    xticklabels=self.cfg.class_names, yticklabels=self.cfg.class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("PointNet++ Confusion Matrix (%)")
        plt.tight_layout()

        cm_png_path = os.path.join(self.cfg.save_pred_dir, "confusion_matrix.png")
        plt.savefig(cm_png_path, dpi=300)
        logger.info(f"✅ Confusion matrix PNG saved at: {cm_png_path}")
        plt.close()
