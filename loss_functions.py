import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def parse_pdb(filename, index_file='atom_indices.npz'):
    # 尝试从文件加载索引
    if os.path.exists(index_file):
        indices = np.load(index_file)
        n_indices = indices['n_indices']
        ca_indices = indices['ca_indices']
        c_indices = indices['c_indices']
        return None, n_indices, ca_indices, c_indices

    atom_data = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith("ATOM"):
                atom_data.append([
                    line[12:16].strip(),   # atom_name
                    line[17:20].strip(),   # residue_name
                    line[21],              # chain
                    int(line[22:26]),      # residue_number
                    float(line[30:38]),    # x
                    float(line[38:46]),    # y
                    float(line[46:54])     # z
                ])

    atom_array = np.array(atom_data)

    # 使用NumPy进行索引
    n_indices = np.where(atom_array[:, 0] == 'N')[0]
    ca_indices = np.where(atom_array[:, 0] == 'CA')[0]
    c_indices = np.where(atom_array[:, 0] == 'C')[0]

    # 保存索引
    np.savez(index_file, n_indices=n_indices, ca_indices=ca_indices, c_indices=c_indices)

    return atom_array, n_indices, ca_indices, c_indices








# def calculate_dihedral_angle(p0, p1, p2, p3):
#     """
#     计算由四点定义的二面角，使用numpy进行向量运算。
#
#     :param p0, p1, p2, p3: 四个连续点的坐标，每个点的形状为(batch_size, 3)
#     :return: 四点定义的二面角，形状为(batch_size,)
#     """
#     # 向量
#     p0=p0.detach().cpu().numpy()
#     p1=p1.detach().cpu().numpy()
#     p2=p2.detach().cpu().numpy()
#     p3=p3.detach().cpu().numpy()
#
#     b0 = p0 - p1
#     b1 = p2 - p1
#     b2 = p3 - p2
#
#     # 归一化b1
#     b1_normalized = b1 / np.linalg.norm(b1, axis=1)[:, np.newaxis]
#
#     # 正交向量
#     v = np.cross(b1_normalized, b0)
#     w = np.cross(b1_normalized, b2)
#
#     # 计算角度
#     x = np.einsum('ij,ij->i', v, w)
#     y = np.einsum('ij,ij->i', np.cross(v, b1_normalized), w)
#
#     return np.arctan2(y, x)  # 返回的是弧度

def calculate_dihedral_angle(p0, p1, p2, p3):
    # Calculate vectors
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    # Normalize b1
    b1_normalized = b1 / torch.norm(b1, dim=1, keepdim=True)

    # Calculate orthogonal vectors
    v = torch.cross(b1_normalized, b0, dim=1)
    w = torch.cross(b1_normalized, b2, dim=1)

    # Compute angles
    x = torch.sum(v * w, dim=1)
    y = torch.sum(torch.cross(v, b1_normalized, dim=1) * w, dim=1)

    return torch.atan2(y, x)  # Return angles in radians




# def compute_phi_psi_angles_from_coordinates(coordinates, n_indices, ca_indices, c_indices):
#     batch_size = coordinates.shape[0]
#     num_residues = len(n_indices)
#     angles = np.zeros((batch_size, num_residues, 2))  # 存储φ和ψ角度
#
#     for i in range(num_residues):
#         if i > 0 and i < num_residues - 1:
#             n_coords = coordinates[:, 3 * n_indices[i]:3 * n_indices[i] + 3]
#             ca_coords = coordinates[:, 3 * ca_indices[i]:3 * ca_indices[i] + 3]
#             c_coords = coordinates[:, 3 * c_indices[i]:3 * c_indices[i] + 3]
#
#             prev_c_coords = coordinates[:, 3 * c_indices[i - 1]:3 * c_indices[i - 1] + 3]
#             next_n_coords = coordinates[:, 3 * n_indices[i + 1]:3 * n_indices[i + 1] + 3]
#
#             phi = calculate_dihedral_angle(prev_c_coords, n_coords, ca_coords, c_coords)
#             psi = calculate_dihedral_angle(n_coords, ca_coords, c_coords, next_n_coords)
#
#             angles[:, i, 0] = phi
#             angles[:, i, 1] = psi
#
#     return angles

import torch

def compute_phi_psi_angles_from_coordinates(coordinates, n_indices, ca_indices, c_indices):
    batch_size = coordinates.shape[0]
    num_residues = len(n_indices)
    angles = torch.zeros((batch_size, num_residues, 2), device=coordinates.device)  # Store phi and psi angles in a PyTorch tensor

    for i in range(num_residues):
        if i > 0 and i < num_residues - 1:
            n_coords = coordinates[:, 3 * n_indices[i]:3 * n_indices[i] + 3]
            ca_coords = coordinates[:, 3 * ca_indices[i]:3 * ca_indices[i] + 3]
            c_coords = coordinates[:, 3 * c_indices[i]:3 * c_indices[i] + 3]

            prev_c_coords = coordinates[:, 3 * c_indices[i - 1]:3 * c_indices[i - 1] + 3]
            next_n_coords = coordinates[:, 3 * n_indices[i + 1]:3 * n_indices[i + 1] + 3]

            phi = calculate_dihedral_angle(prev_c_coords, n_coords, ca_coords, c_coords)
            psi = calculate_dihedral_angle(n_coords, ca_coords, c_coords, next_n_coords)

            angles[:, i, 0] = phi
            angles[:, i, 1] = psi

    return angles



def angle_difference(angle1, angle2):
    """
    Computes the minimum difference between two angles, considering the periodicity of angles.

    :param angle1: The first array of angles in radians (PyTorch tensor)
    :param angle2: The second array of angles in radians (PyTorch tensor)
    :return: The minimum difference between the two angles in radians
    """
    diff = angle1 - angle2
    diff = (diff + torch.pi) % (2 * torch.pi) - torch.pi  # Adjust the difference to be within [-π, π]
    return diff




def compute_ramachandran_loss(input_coordinates, output_coordinates, n_indices, ca_indices, c_indices):
    # Assuming coordinates are PyTorch tensors and operations inside are converted as above
    actual_angles = compute_phi_psi_angles_from_coordinates(input_coordinates, n_indices, ca_indices, c_indices)
    predicted_angles = compute_phi_psi_angles_from_coordinates(output_coordinates, n_indices, ca_indices, c_indices)

    # Calculate angle differences, considering periodicity
    phi_diff = angle_difference(predicted_angles[:, :, 0], actual_angles[:, :, 0])
    psi_diff = angle_difference(predicted_angles[:, :, 1], actual_angles[:, :, 1])

    # Compute total loss using PyTorch operations
    loss = torch.mean(phi_diff ** 2 + psi_diff ** 2)  # Or use other appropriate loss function

    return loss

# 解析PDB文件
pdb_filename = 'pdb_temp.pdb'
atom_data, n_indices, ca_indices, c_indices = parse_pdb(pdb_filename)


def weight_mse_loss(input, target, batch_size):
    # 计算标准 MSE 损失
    mse_loss = F.mse_loss(input, target, reduction='none')
    c_indices_tensor = torch.tensor(c_indices[:8], device=input.device, dtype=torch.long)
    ca_indices_tensor = torch.tensor(ca_indices[:8], device=input.device, dtype=torch.long)
    n_indices_tensor = torch.tensor(n_indices[:8], device=input.device, dtype=torch.long)
    # 创建权重（在 PyTorch 中执行所有操作）
    initial_weights = torch.ones(1183, device=input.device)
    loop_backbone_index = torch.cat((c_indices_tensor[:8], ca_indices_tensor[:8], n_indices_tensor[:8]))
    initial_weights[loop_backbone_index] = 0.1  # 假设您需要调整这里的权重
    expanded_weights = initial_weights.repeat(3)
    batch_weights = expanded_weights.repeat(batch_size, 1)

    # 应用权重
    weighted_mse = (mse_loss * batch_weights).mean()

    return weighted_mse



def pairwise_distance_matrix(x):
    # x is a tensor of shape [batch_size, n_atoms, 3]
    batch_size, n_atoms, _ = x.shape
    triu_indices = torch.triu_indices(row=n_atoms, col=n_atoms, offset=1)  # Get upper triangle indices, offset=1 excludes diagonal
    dists = torch.norm(x[:, triu_indices[0], :] - x[:, triu_indices[1], :], dim=2)
    return dists


def reconstruction_loss(flat_original_coords, flat_reconstructed_coords):
    # Reshape flattened coordinates back to [batch_size, n_atoms, 3]
    original_coords = flat_original_coords.view(-1, 1183, 3)
    reconstructed_coords = flat_reconstructed_coords.view(-1, 1183, 3)

    # Calculate distance matrices
    original_dists = pairwise_distance_matrix(original_coords)
    reconstructed_dists = pairwise_distance_matrix(reconstructed_coords)

    # Calculate MSE of the distance matrices
    loss = torch.mean((original_dists - reconstructed_dists) ** 2)
    return loss


# Example usage
# batch_size = 10
# n_atoms = 1183
# flat_coords = torch.randn(batch_size, n_atoms * 3)  # Random original coordinates, flattened
# flat_reconstructed_coords = torch.randn(batch_size, n_atoms * 3)  # Random reconstructed coordinates, flattened
# weight_mse_loss(flat_coords, flat_reconstructed_coords,batch_size)
# loss = reconstruction_loss(flat_coords, flat_reconstructed_coords)
# print(f"Loss: {loss.item()}")





class FAPEloss(nn.Module):
    """Frame aligned point error loss

    Args:
        Z (int, optional): [description]. Defaults to 10.
        clamp (int, optional): [description]. Defaults to 10.
        epsion (float, optional): [description]. Defaults to -1e4.
    """
    def __init__(self, Z=10.0, clamp=10.0, epsion=-1e4):

        super().__init__()
        self.z = Z
        self.epsion = epsion
        self.clamp = clamp

    def forward(self, predict_T, transformation, pdb_mask=None, padding_mask=None, device='cpu'):
        """
        Args:
            predict_T (`tensor`, `tensor`): ([batch, N_seq, 3, 3], [batch, N_seq, 3])
            transformation (`tensor`, `tensor`): ([batch, N_seq, 3, 3], [batch, N_seq, 3])
            pdb_mask (`tensor`, optional): pdb mask. size: [batch, N_seq, N_seq]. Defaults to None.
            padding_mask (`tensor`, optional): padding mask. size: [batch, N_seq, N_seq]. Defaults to None.
        """
        predict_R, predict_Trans = predict_T
        RotaionMatrix, translation = transformation
        delta_predict_Trans = rearrange(predict_Trans, 'b j t -> b j () t') - rearrange(predict_Trans, 'b i t -> b () i t')
        delta_Trans = rearrange(translation, 'b j t -> b j () t') - rearrange(translation, 'b i t -> b () i t')

        X_hat = torch.einsum('bikq, bjik->bijq', predict_R, delta_predict_Trans)
        X = torch.einsum('bikq, bjik->bijq', RotaionMatrix, delta_Trans)

        distance = torch.norm(X_hat-X, dim=-1)
        distance = torch.clamp(distance, max=self.clamp) * (1/self.z)

        if pdb_mask is not None:
            distance = distance * pdb_mask
        if padding_mask is not None:
            distance = distance * padding_mask

        FAPE_loss = torch.mean(distance)

        return FAPE_loss
def rigid_transform_3D(A, B):
    centroid_A = torch.mean(A, dim=0)
    centroid_B = torch.mean(B, dim=0)

    A_centered = A - centroid_A
    B_centered = B - centroid_B

    H = A_centered.T @ B_centered
    U, S, Vt = torch.svd(H)

    R = Vt.T @ U.T
    if torch.det(R) < 0:  # Proper rotation
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A
    return R, t


# # Sample data generation
# torch.manual_seed(0)
# batch_size = 2
# N_atoms = 1183
# original_coords = torch.randn(batch_size, N_atoms, 3)
# reconstructed_coords = original_coords + 12.3 * torch.randn(batch_size, N_atoms, 3)
#
# # Container for optimal rotations and translations
# rotations = torch.empty(batch_size, 3, 3)
# translations = torch.empty(batch_size, 3)
#
# # Compute optimal rotation and translation for each sample in the batch
# for i in range(batch_size):
#     R, t = rigid_transform_3D(reconstructed_coords[i], original_coords[i])
#     rotations[i] = R
#     translations[i] = t
#
# # Calculate FAPE loss
# fape_loss = FAPEloss()
# loss = fape_loss(
#     predict_T=(rotations, translations),
#     transformation=(torch.eye(3).expand(batch_size, -1, -1), torch.zeros(batch_size, 3))
#     # Identity and zero translation
# )
#
# print("FAPE Loss:", loss)


fape_loss =FAPEloss()