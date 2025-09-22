import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# ===========================
# RandLA-Net Utilities
# ===========================

def farthest_point_sample_batch(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Perform farthest point sampling (FPS) on a batch of point clouds.
    Args:
        xyz: (B, N, 3) input point clouds
        npoint: number of points to sample
    Returns:
        centroids: (B, npoint) sampled point indices
    """
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), device=device)
    batch_indices = torch.arange(B, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid_xyz = xyz[batch_indices, farthest, :].unsqueeze(1)  # (B,1,3)
        dist = torch.sum((xyz - centroid_xyz) ** 2, dim=2)
        distance = torch.minimum(distance, dist)
        farthest = torch.argmax(distance, dim=1)
    return centroids


def knn_gather(xyz_src: torch.Tensor, xyz_query: torch.Tensor, k: int) -> torch.Tensor:
    """
    Find k-nearest neighbors indices for query points from source points.
    """
    dists = torch.cdist(xyz_query, xyz_src, p=2)
    return torch.topk(dists, k, largest=False).indices


@torch.no_grad()
def build_pyramid_gpu_batch(
        xyz_b: torch.Tensor,
        k: int,
        n_layers: int,
        ratios: List[int],
        pool_size: int = 16
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Build multi-layer pyramid of point clouds with neighbors, subsampling and interpolation indices.
    Args:
        xyz_b: (B,N,3) batch of points
        k: K-nearest neighbors
        n_layers: number of pyramid layers
        ratios: downsampling ratios per layer
        pool_size: pooling size for max-pooling
    Returns:
        xyz_list, neigh_list, sub_list, interp_list
    """
    B, N, _ = xyz_b.shape
    device = xyz_b.device
    xyz_list, neigh_list, sub_list, interp_list = [], [], [], []
    xyz_i = xyz_b.clone()

    for level in range(n_layers):
        Ni = xyz_i.shape[1]
        neigh_idx = knn_gather(xyz_i, xyz_i, k + 1)[:, :, 1:]  # exclude self
        neigh_list.append(neigh_idx)
        xyz_list.append(xyz_i)

        ratio = ratios[level]
        Ni_next = max(1, Ni // ratio)
        if Ni_next >= Ni:
            J = torch.arange(Ni, device=device).unsqueeze(0).repeat(B, 1)
        else:
            J = farthest_point_sample_batch(xyz_i, Ni_next)
        batch_indices = torch.arange(B, device=device).unsqueeze(1)
        xyz_next = xyz_i[batch_indices, J, :]

        sub_idx = knn_gather(xyz_i, xyz_next, pool_size)
        interp_idx = knn_gather(xyz_next, xyz_i, 1)
        sub_list.append(sub_idx)
        interp_list.append(interp_idx)

        xyz_i = xyz_next

    return xyz_list, neigh_list, sub_list, interp_list


# -----------------------
# Batch gather utilities
# -----------------------
def batch_gather(tensor, indices):
    """
    Gather values from tensor according to batch indices.
    """
    shape = list(tensor.shape)
    device = tensor.device
    flat_first = tensor.reshape([shape[0] * shape[1]] + shape[2:])
    offset = (torch.arange(shape[0], device=device) * shape[1]).reshape([shape[0]] + [1] * (indices.ndim - 1))
    return flat_first[indices.long() + offset]


def gather_neighbour(point_features, neighbor_idx):
    """
    Gather neighbor features for each point.
    """
    B, N, C = point_features.shape
    K = neighbor_idx.shape[-1]
    index_input = neighbor_idx.reshape(B, -1)
    features = batch_gather(point_features, index_input)
    features = features.reshape(B, N, K, C)
    return features.permute(0, 3, 1, 2).contiguous()


def random_sample(feature, pool_idx):
    """
    Max-pooling along sampled indices.
    """
    feature = torch.squeeze(feature, dim=3)
    B, d, N = feature.shape
    M = pool_idx.shape[-1]
    feature = feature.permute(0, 2, 1).contiguous()
    pool_idx = pool_idx.reshape(B, -1)
    pool_features = batch_gather(feature, pool_idx)
    pool_features = pool_features.reshape(B, -1, M, d)
    pool_features = torch.max(pool_features, dim=2, keepdim=True)[0]
    return pool_features.permute(0, 3, 1, 2).contiguous()


def nearest_interpolation(feature, interp_idx):
    """
    Interpolate feature from subsampled points back to original points.
    """
    feature = torch.squeeze(feature, dim=3)
    B, d, Nc = feature.shape
    Nf = interp_idx.shape[1]
    idx = interp_idx.reshape(B, Nf)
    feature = feature.permute(0, 2, 1).contiguous()
    interp_features = batch_gather(feature, idx)
    return interp_features.permute(0, 2, 1)[:, :, :, None].contiguous()


def relative_pos_encoding(xyz, neighbor_idx):
    """
    Compute relative positional encoding for points.
    """
    neighbor_xyz = gather_neighbour(xyz, neighbor_idx)
    xyz0 = xyz[:, :, None, :].permute(0, 3, 1, 2).contiguous()
    repeated_xyz = xyz0.repeat(1, 1, 1, neighbor_idx.shape[-1])
    relative_xyz = repeated_xyz - neighbor_xyz
    relative_dist = torch.sqrt(torch.sum(relative_xyz ** 2, dim=1, keepdim=True) + 1e-9)
    relative_feature = torch.cat([relative_dist, relative_xyz, repeated_xyz, neighbor_xyz], dim=1)
    return relative_feature


# ============================
# RandLA-Net Components
# ============================

class AttentivePooling(nn.Module):
    """
    Attentive Pooling module for aggregating neighbor features
    """

    def __init__(self, n_feature, d_out):
        super().__init__()
        self.fc1 = nn.Linear(n_feature, n_feature, bias=False)
        self.conv1 = nn.Conv2d(n_feature, d_out, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(d_out, eps=1e-6, momentum=0.01)

    def forward(self, x):
        B, C, N, K = x.shape
        x_perm = x.permute(0, 2, 3, 1).contiguous()  # (B,N,K,C)
        att_activation = self.fc1(x_perm)
        att_score = F.softmax(att_activation, dim=2)
        x_weighted = x_perm * att_score
        x_agg = torch.sum(x_weighted, dim=2)  # sum over neighbors
        x_agg = x_agg.permute(0, 2, 1).unsqueeze(-1).contiguous()
        x = F.leaky_relu(self.bn1(self.conv1(x_agg)), negative_slope=0.2)
        return x


class BuildingBlock(nn.Module):
    """
    Basic building block for DilatedResidualBlock
    """

    def __init__(self, d_out):
        super().__init__()
        self.conv1 = nn.Conv2d(10, d_out // 2, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(d_out // 2, eps=1e-6, momentum=0.01)
        self.attpool1 = AttentivePooling(2 * (d_out // 2), d_out // 2)
        self.conv2 = nn.Conv2d(d_out // 2, d_out // 2, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(d_out // 2, eps=1e-6, momentum=0.01)
        self.attpool2 = AttentivePooling(2 * (d_out // 2), d_out)

    def forward(self, xyz, feature, neigh_idx):
        f_xyz = relative_pos_encoding(xyz, neigh_idx)
        f_xyz = F.leaky_relu(self.bn1(self.conv1(f_xyz)), negative_slope=0.2)
        feature = torch.squeeze(feature, dim=-1).permute(0, 2, 1).contiguous()
        f_neighbours = gather_neighbour(feature, neigh_idx)
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.attpool1(f_concat)

        f_xyz = F.leaky_relu(self.bn2(self.conv2(f_xyz)), negative_slope=0.2)
        f_pc_agg = torch.squeeze(f_pc_agg, dim=-1).permute(0, 2, 1).contiguous()
        f_neighbours = gather_neighbour(f_pc_agg, neigh_idx)
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.attpool2(f_concat)
        return f_pc_agg


class DilatedResidualBlock(nn.Module):
    """
    Dilated residual block with two attentive building blocks
    """

    def __init__(self, f_in, d_out):
        super().__init__()
        self.conv1 = nn.Conv2d(f_in, d_out // 2, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(d_out // 2)
        self.bb = BuildingBlock(d_out)
        self.conv2 = nn.Conv2d(d_out, d_out * 2, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(d_out * 2)
        self.shortcut = nn.Conv2d(f_in, d_out * 2, kernel_size=1)
        self.bn_shortcut = nn.BatchNorm2d(d_out * 2)

    def forward(self, feature, xyz, neigh_idx):
        f_pc = F.leaky_relu(self.bn1(self.conv1(feature)), negative_slope=0.2)
        f_pc = self.bb(xyz, f_pc, neigh_idx)
        f_pc = self.bn2(self.conv2(f_pc))
        shortcut = self.bn_shortcut(self.shortcut(feature))
        return F.leaky_relu(f_pc + shortcut)


class FeatureDecoder(nn.Module):
    """
    Decoder block with interpolation from subsampled features
    """

    def __init__(self, f_in, f_out):
        super().__init__()
        self.trconv1 = nn.ConvTranspose2d(f_in, f_out, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(f_out)

    def forward(self, feature, encoded_feature, interp_idx):
        f_interp_i = nearest_interpolation(feature, interp_idx)
        f_decoded = self.trconv1(torch.cat([encoded_feature, f_interp_i], dim=1))
        f_decoded = self.bn1(f_decoded)
        return f_decoded


# ===========================
# RandLA-Net Main Model
# ===========================
class RandlaNet(nn.Module):
    def __init__(self, d_out, n_layers, n_classes):
        super().__init__()
        self.n_classes = n_classes
        dilate_block_in = 8
        self.fc1 = nn.Linear(10, dilate_block_in)
        self.bn1 = nn.BatchNorm1d(dilate_block_in)
        self.f_encoders = nn.ModuleList()
        decoder_in_list = [d_out[0] * 2]

        for i in range(n_layers):
            self.f_encoders.append(DilatedResidualBlock(dilate_block_in, d_out[i]))
            dilate_block_in = d_out[i] * 2
            decoder_in_list.append(dilate_block_in)

        self.conv2 = nn.Conv2d(dilate_block_in, dilate_block_in, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(dilate_block_in)

        self.f_decoders = nn.ModuleList()
        for i in range(n_layers):
            self.f_decoders.append(
                FeatureDecoder(decoder_in_list[-i - 1] + decoder_in_list[-i - 2],
                               decoder_in_list[-i - 2])
            )

        self.conv3 = nn.Conv2d(decoder_in_list[0], 64, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.drop4 = nn.Dropout2d(0.5)
        self.conv5 = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, inputs):
        x = inputs['features']
        x = self.fc1(x)
        x = x.permute(0, 2, 1).contiguous()[:, :, :, None]
        x = F.leaky_relu(self.bn1(x.squeeze(-1)))
        x = x[:, :, :, None]

        encoded_list = []
        for i, encoder in enumerate(self.f_encoders):
            x = encoder(x, inputs['xyz'][i], inputs['neigh_idx'][i])
            if i == 0: encoded_list.append(x.clone())
            x = random_sample(x, inputs['sub_idx'][i])
            encoded_list.append(x.clone())

        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        for i, decoder in enumerate(self.f_decoders):
            x = decoder(x, encoded_list[-i - 2], inputs['interp_idx'][-i - 1])

        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.2)
        x = self.drop4(x)
        x = self.conv5(x)
        x = x.squeeze(-1).permute(0, 2, 1).reshape([-1, self.n_classes])
        return x
