from typing import Optional

import torch
import torch.nn as nn

from .trajvad_layers import TrajVADPoseCoupling


def build_dilation_pattern(layer_index: int, wavenet_dilation: bool, max_dilation: int) -> tuple[int, int]:
    if wavenet_dilation:
        dilation_value = min(2 ** layer_index, max_dilation)
        return (dilation_value, dilation_value)
    return (1, 2)


def build_pose_layers(
    k_pose: int,
    pose_dim: int,
    cond_dim: int,
    hidden_dim: int,
    wavenet_dilation: bool,
    max_dilation: int,
) -> nn.ModuleList:
    layers = nn.ModuleList()
    for layer_index in range(k_pose):
        layers.append(
            TrajVADPoseCoupling(
                target_dim=pose_dim,
                cond_dim=cond_dim,
                hidden_dim=hidden_dim,
                dilation_pattern=build_dilation_pattern(layer_index, wavenet_dilation, max_dilation),
            )
        )
    return layers


def compute_person_gate(x_cat: Optional[torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
    if x_cat is None:
        return torch.ones(batch_size, device=device)
    seg_cls = torch.mode(x_cat.squeeze(-1), dim=1).values
    return (seg_cls == 0).float()


def compute_pose_gate(pose_mask: Optional[torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
    if pose_mask is None:
        return torch.ones(batch_size, device=device)
    return pose_mask.float().view(batch_size)


def initialize_pose_actnorm(
    x_pose: torch.Tensor,
    pose_actnorm_bias: nn.Parameter,
    pose_actnorm_scale: nn.Parameter,
) -> None:
    with torch.no_grad():
        mean = x_pose.mean(dim=[0, 2], keepdim=True)
        std = x_pose.std(dim=[0, 2], keepdim=True)
        pose_actnorm_bias.data.copy_(-mean)
        pose_actnorm_scale.data.copy_(1.0 / (std + 1e-6))


def gaussian_log_prob_sum(z_pose: torch.Tensor, prior_mean_tensor: torch.Tensor) -> torch.Tensor:
    log_prob_pose = -0.5 * ((z_pose - prior_mean_tensor) ** 2)
    log_prob_pose -= 0.5 * torch.log(torch.tensor(2 * torch.pi, device=z_pose.device))
    return torch.sum(log_prob_pose, dim=[1, 2])
