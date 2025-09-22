import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================================
# Helper Functions
# ======================================================

def square_distance(src, dst):
    """
    Compute squared Euclidean distance between each pair of points.

    Args:
        src: (B, N, 3) source points
        dst: (B, M, 3) target points

    Returns:
        dist: (B, N, M) squared distances
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # Cross term
    dist += torch.sum(src ** 2, -1).view(B, N, 1)        # Source squared norms
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)        # Target squared norms
    return dist

def farthest_point_sample(xyz, npoint):
    """
    Farthest Point Sampling (FPS) to downsample points for Set Abstraction.

    Args:
        xyz: (B, N, 3) input points
        npoint: number of points to sample

    Returns:
        centroids: (B, npoint) indices of sampled points
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10  # initialize distances
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid_xyz = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid_xyz) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]  # choose next farthest point

    return centroids

def index_points(points, idx):
    """
    Index points along batch dimension for FPS or grouping.

    Args:
        points: (B, N, C) input points/features
        idx: (B, S) or (B, S, K) indices

    Returns:
        new_points: selected points/features
    """
    device = points.device
    B = points.shape[0]

    if idx.dim() == 2:  # (B, S)
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(B, 1).repeat(1, idx.shape[1])
        new_points = points[batch_indices, idx, :]
    elif idx.dim() == 3:  # (B, S, K)
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(B, 1, 1).repeat(1, idx.shape[1], idx.shape[2])
        new_points = points[batch_indices, idx, :]
    else:
        raise ValueError(f"Unsupported idx shape: {idx.shape}")

    return new_points

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Find all points within a ball of radius for each centroid (used in SA layer).

    Args:
        radius: search radius
        nsample: max points per local region
        xyz: (B, N, 3) all points
        new_xyz: (B, S, 3) centroids

    Returns:
        group_idx: (B, S, nsample) grouped point indices
    """
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    device = xyz.device
    sqrdists = square_distance(new_xyz, xyz)  # compute squared distances

    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    mask = sqrdists > radius ** 2
    group_idx[mask] = N  # mark points outside radius
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]  # keep closest nsample points

    # replace invalid points with the first valid index
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx

# ======================================================
# Set Abstraction Layer
# ======================================================
class PointNetSetAbstraction5(nn.Module):
    """
    PointNet++ Set Abstraction (SA) layer.
    Performs sampling, grouping, local feature extraction using MLPs.
    """
    def __init__(self, npoint, radius, nsample, in_channel, mlp_channels):
        """
        Args:
            npoint: number of points to sample
            radius: local region radius
            nsample: max points per local region
            in_channel: input feature channels
            mlp_channels: list of MLP output channels
        """
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel + 3  # concatenate xyz coords

        for out_channel in mlp_channels:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        Forward pass.

        Args:
            xyz: (B, 3, N) input points
            points: (B, C, N) point features or None

        Returns:
            new_xyz: sampled points
            new_points: aggregated features
        """
        B, _, N = xyz.shape
        xyz = xyz.permute(0, 2, 1)  # (B, N, 3)

        # Sampling
        fps_idx = farthest_point_sample(xyz, self.npoint) if self.npoint else None
        new_xyz = index_points(xyz, fps_idx) if fps_idx is not None else xyz

        # Grouping
        group_idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, group_idx)
        grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)  # normalize to centroid

        # Concatenate features if available
        if points is not None:
            points = points.permute(0, 2, 1)
            grouped_points = index_points(points, group_idx)
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz_norm

        # Apply MLP on each point
        new_points = new_points.permute(0, 3, 1, 2)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))

        # Pooling (max) over nsample points
        new_points = torch.max(new_points, -1)[0]

        return new_xyz.permute(0, 2, 1), new_points

# ======================================================
# Feature Propagation Layer
# ======================================================
class PointNetFeaturePropagation5(nn.Module):
    """
    PointNet++ Feature Propagation (FP) layer.
    Interpolates coarse features to dense points and applies MLP.
    """
    def __init__(self, in_channel, mlp_channels):
        """
        Args:
            in_channel: input feature channels (concat of previous and skip)
            mlp_channels: output MLP channels
        """
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp_channels:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Args:
            xyz1: (B, C, N) target points
            xyz2: (B, C, S) source points (coarse)
            points1: (B, C1, N) target features (skip)
            points2: (B, C2, S) source features

        Returns:
            new_points: (B, C_new, N) propagated features
        """
        B, C, N = xyz1.shape
        _, _, S = xyz2.shape

        if S == 1:
            # Single point case: replicate features
            interpolated_points = points2.repeat(1, 1, N)
        else:
            # kNN interpolation (3 nearest neighbors)
            xyz1_t, xyz2_t = xyz1.permute(0, 2, 1), xyz2.permute(0, 2, 1)
            dist, idx = square_distance(xyz1_t, xyz2_t).sort(dim=-1)
            dist, idx = dist[:, :, :3], idx[:, :, :3]
            dist_recip = 1.0 / (dist + 1e-8)
            weight = dist_recip / torch.sum(dist_recip, dim=2, keepdim=True)

            points2 = points2.permute(0, 2, 1)
            neighbor_feats = torch.gather(
                points2.unsqueeze(1).expand(-1, N, -1, -1),
                2,
                idx.unsqueeze(-1).expand(-1, -1, -1, points2.shape[-1])
            )
            interpolated_points = torch.sum(neighbor_feats * weight.unsqueeze(-1), dim=2).permute(0, 2, 1)

        # Concatenate skip connection features
        new_points = torch.cat([points1, interpolated_points], dim=1) if points1 is not None else interpolated_points

        # Apply MLP
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))

        return new_points

# ======================================================
# PointNet2 Semantic Segmentation Model
# ======================================================
class PointNet2SemSeg5(nn.Module):
    """
    PointNet2-based Semantic Segmentation model.
    Uses Set Abstraction layers for downsampling and local features,
    Feature Propagation layers for upsampling, and a final classifier.
    """
    def __init__(self, cfg):
        """
        Args:
            cfg: configuration object containing layer/channel info
        """
        super().__init__()

        # --------------------------
        # Set Abstraction Layers (sampling + local feature extraction)
        # --------------------------
        self.sa1 = PointNetSetAbstraction5(cfg.npoint_sa1, cfg.radius_sa1, cfg.nsample_sa1,
                                           cfg.in_channel_sa1, cfg.mlp_sa1)
        self.sa2 = PointNetSetAbstraction5(cfg.npoint_sa2, cfg.radius_sa2, cfg.nsample_sa2,
                                           cfg.mlp_sa1[-1], cfg.mlp_sa2)
        self.sa3 = PointNetSetAbstraction5(cfg.npoint_sa3, cfg.radius_sa3, cfg.nsample_sa3,
                                           cfg.mlp_sa2[-1], cfg.mlp_sa3)

        # --------------------------
        # Feature Propagation Layers (upsample + propagate features)
        # --------------------------
        fp3_in = cfg.mlp_sa2[-1] + cfg.mlp_sa3[-1]
        self.fp3 = PointNetFeaturePropagation5(fp3_in, cfg.fp3_channels)

        fp2_in = cfg.mlp_sa1[-1] + cfg.fp3_channels[-1]
        self.fp2 = PointNetFeaturePropagation5(fp2_in, cfg.fp2_channels)

        fp1_in = cfg.in_channel_sa1 + cfg.fp2_channels[-1]
        self.fp1 = PointNetFeaturePropagation5(fp1_in, cfg.fp1_channels)

        # --------------------------
        # Final Classifier
        # --------------------------
        self.classifier = nn.Sequential(
            nn.Conv1d(cfg.fp1_channels[-1], 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(128, cfg.num_classes, 1)
        )

    def forward(self, xyz, features):
        """
        Forward pass of PointNet2 semantic segmentation.

        Args:
            xyz: (B, 3, N) input points
            features: (B, C, N) input point features (can be None)

        Returns:
            logits: (B, num_classes, N) per-point class scores
        """
        # Downsampling + local feature extraction
        l1_xyz, l1_points = self.sa1(xyz, features)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Upsampling + feature propagation
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, features, l1_points)

        # Classifier
        logits = self.classifier(l0_points)
        return logits
