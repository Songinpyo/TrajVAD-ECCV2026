from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .trajvad_t import TrajVADTModel
from .trajvad_pose_utils import (
    build_pose_layers,
    compute_person_gate,
    compute_pose_gate,
    gaussian_log_prob_sum,
    initialize_pose_actnorm,
)


class TrajVADPModel(nn.Module):
    def __init__(
        self,
        cont_dim: int = 27,
        emb_dim: int = 3,
        num_classes: int = 80,
        hidden_channels: int = 64,
        K: int = 18,
        K_pose: Optional[int] = None,
        use_emb_in_input: bool = True,
        prior_mean: float = 3.0,
        seg_len: int = 12,
        device: str = "cuda",
        pose_dim: int = 51,
        lambda_pose: float = 1.0,
        wavenet_dilation: bool = True,
        max_dilation: int = 12,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        self.device = device
        self.pose_dim = pose_dim
        self.lambda_pose = float(lambda_pose)
        self.K = K
        self.K_pose = K if K_pose is None else K_pose
        self.wavenet_dilation = wavenet_dilation
        self.max_dilation = max_dilation
        self.base_model = TrajVADTModel(
            cont_dim=cont_dim,
            emb_dim=emb_dim,
            num_classes=num_classes,
            hidden_channels=hidden_channels,
            K=K,
            use_emb_in_input=use_emb_in_input,
            prior_mean=prior_mean,
            seg_len=seg_len,
            device=device,
            wavenet_dilation=wavenet_dilation,
            max_dilation=max_dilation,
            feature_names=feature_names,
        )

        self.class_emb = self.base_model.class_emb
        self.base_latent_dim = self.base_model.input_dim
        self.input_dim = self.base_latent_dim + pose_dim
        self.pose_shape = (self.input_dim, seg_len, 1)

        self.pose_actnorm_scale = nn.Parameter(torch.ones(1, pose_dim, 1))
        self.pose_actnorm_bias = nn.Parameter(torch.zeros(1, pose_dim, 1))
        self.pose_actnorm_initialized = False

        self.pose_layers = build_pose_layers(
            k_pose=self.K_pose,
            pose_dim=pose_dim,
            cond_dim=self.base_latent_dim,
            hidden_dim=hidden_channels,
            wavenet_dilation=wavenet_dilation,
            max_dilation=max_dilation,
        )

        self.register_buffer("prior_mean_tensor", torch.full((1,), prior_mean))

    def set_actnorm_init(self) -> None:
        self.base_model.set_actnorm_init()
        self.pose_actnorm_initialized = True

    def _initialize_pose_actnorm(self, x_pose: torch.Tensor) -> None:
        initialize_pose_actnorm(
            x_pose=x_pose,
            pose_actnorm_bias=self.pose_actnorm_bias,
            pose_actnorm_scale=self.pose_actnorm_scale,
        )
        self.pose_actnorm_initialized = True

    def _run_pose_branch(
        self,
        x_pose: torch.Tensor,
        z_base: torch.Tensor,
        reverse: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.pose_actnorm_initialized and not reverse:
            self._initialize_pose_actnorm(x_pose)

        batch_size, _, time_steps = x_pose.shape
        logdet_pose_actnorm = torch.sum(torch.log(torch.abs(self.pose_actnorm_scale))) * time_steps
        z_pose = (x_pose + self.pose_actnorm_bias) * self.pose_actnorm_scale
        total_logdet_pose = logdet_pose_actnorm.repeat(batch_size)

        for layer in self.pose_layers:
            z_pose, logdet = layer(z_pose, condition=z_base)
            total_logdet_pose += logdet

        log_prob_pose_sum = gaussian_log_prob_sum(z_pose, self.prior_mean_tensor)
        return z_pose, total_logdet_pose, log_prob_pose_sum

    def forward(
        self,
        x_cont: torch.Tensor,
        x_cat: Optional[torch.Tensor] = None,
        x_pose: Optional[torch.Tensor] = None,
        pose_mask: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if reverse:
            raise NotImplementedError("Reverse pass is not implemented for TrajVADPModel")
        if x_pose is None:
            raise ValueError("TrajVADPModel requires x_pose")

        batch_size, time_steps, _ = x_cont.shape
        z_base, nll_base = self.base_model(x_cont, x_cat, reverse=False)

        g_person = compute_person_gate(x_cat, batch_size, x_cont.device)
        g_pose = compute_pose_gate(pose_mask, batch_size, x_cont.device)
        pose_conf = x_pose.reshape(batch_size, time_steps, 17, 3)[..., 2]
        g_quality = pose_conf.mean(dim=(1, 2)).clamp(0, 1)
        g = g_person * g_pose * g_quality

        x_pose_branch = x_pose.permute(0, 2, 1)
        z_pose, total_logdet_pose, log_prob_pose_sum = self._run_pose_branch(
            x_pose_branch,
            z_base,
            reverse=False,
        )

        pose_term = (log_prob_pose_sum + total_logdet_pose) * g
        raw_nll_base = nll_base * (time_steps * self.base_latent_dim)
        raw_log_prob_base = -raw_nll_base
        raw_log_prob_total = raw_log_prob_base + self.lambda_pose * pose_term
        d_eff = self.base_latent_dim + self.pose_dim * g
        nll_total = -raw_log_prob_total / (time_steps * d_eff)

        return torch.cat([z_base, z_pose], dim=1), nll_total

    def get_anomaly_score(
        self,
        x_cont: torch.Tensor,
        x_cat: Optional[torch.Tensor] = None,
        x_pose: Optional[torch.Tensor] = None,
        pose_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, nll = self.forward(
            x_cont,
            x_cat=x_cat,
            x_pose=x_pose,
            pose_mask=pose_mask,
            reverse=False,
        )
        return nll


def get_trajvad_p_model(
    cont_dim: int = 27,
    emb_dim: int = 3,
    num_classes: int = 80,
    hidden_channels: int = 64,
    K: int = 18,
    K_pose: Optional[int] = None,
    use_emb_in_input: bool = True,
    prior_mean: float = 3.0,
    seg_len: int = 12,
    device: str = "cuda",
    pose_dim: int = 51,
    lambda_pose: float = 1.0,
    wavenet_dilation: bool = True,
    max_dilation: int = 12,
    feature_names: Optional[List[str]] = None,
    **kwargs,
) -> TrajVADPModel:
    del kwargs
    model = TrajVADPModel(
        cont_dim=cont_dim,
        emb_dim=emb_dim,
        num_classes=num_classes,
        hidden_channels=hidden_channels,
        K=K,
        K_pose=K_pose,
        use_emb_in_input=use_emb_in_input,
        prior_mean=prior_mean,
        seg_len=seg_len,
        device=device,
        pose_dim=pose_dim,
        lambda_pose=lambda_pose,
        wavenet_dilation=wavenet_dilation,
        max_dilation=max_dilation,
        feature_names=feature_names,
    )
    return model.to(device)
