"""
TrajVAD-T trajectory normalizing flow.

The model splits the input feature
vector into two arbitrary halves and applying alternating coupling transforms.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .trajvad_layers import TrajVADChannelCoupling


def compute_trajvad_t_split_indices(total_dim: int) -> Tuple[List[int], List[int]]:
    """Compute arbitrary half/half split indices for TrajVAD-T.

    Args:
        total_dim: Total feature dimension after optional embedding concatenation.

    Returns:
        Tuple of index lists `(split_a_indices, split_b_indices)`.

    Raises:
        ValueError: If `total_dim` is smaller than 2.
    """
    if total_dim < 2:
        raise ValueError(f"TrajVAD-T requires total_dim >= 2, got {total_dim}")

    split_point = total_dim // 2
    split_a_indices = list(range(0, split_point))
    split_b_indices = list(range(split_point, total_dim))
    return split_a_indices, split_b_indices


class TrajVADTModel(nn.Module):
    """TrajVAD-T normalizing flow with alternating channel splits.

    Args:
        cont_dim: Continuous input feature dimension.
        emb_dim: Class embedding dimension (if concatenated).
        num_classes: Number of classes.
        hidden_channels: Hidden channels in coupling subnets.
        K: Number of coupling layers.
        use_emb_in_input: Whether to concatenate class embedding to inputs.
        prior_mean: Target mean for latent Gaussian prior.
        seg_len: Segment length.
        device: Target device string.
        wavenet_dilation: Use WaveNet-style exponential dilation if True.
        max_dilation: Maximum dilation when `wavenet_dilation=True`.
        feature_names: Selected feature names; used to infer effective cont_dim.
    """

    def __init__(
        self,
        cont_dim: int = 27,
        emb_dim: int = 3,
        num_classes: int = 80,
        hidden_channels: int = 64,
        K: int = 6,
        use_emb_in_input: bool = True,
        prior_mean: float = 3.0,
        seg_len: int = 12,
        device: str = "cuda",
        wavenet_dilation: bool = True,
        max_dilation: int = 12,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        self.cont_dim = len(feature_names) if feature_names is not None else cont_dim
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.use_emb_in_input = use_emb_in_input
        self.prior_mean = prior_mean
        self.seg_len = seg_len
        self.device = device
        self.wavenet_dilation = wavenet_dilation
        self.max_dilation = max_dilation
        self.feature_names = feature_names
        self.input_dim = self.cont_dim + (emb_dim if use_emb_in_input else 0)
        self.split_a_indices, self.split_b_indices = compute_trajvad_t_split_indices(self.input_dim)
        self.split_a_dim = len(self.split_a_indices)
        self.split_b_dim = len(self.split_b_indices)

        self.class_emb = nn.Embedding(num_classes, emb_dim)

        self.pose_shape = (self.input_dim, seg_len, 1)

        self.layers = nn.ModuleList()
        for layer_index in range(K):
            direction = "left_to_right" if layer_index % 2 == 0 else "right_to_left"
            if wavenet_dilation:
                dilation_value = min(2 ** layer_index, max_dilation)
                dilation_pattern = (dilation_value, dilation_value)
            else:
                dilation_pattern = (1, 2)
            self.layers.append(
                TrajVADChannelCoupling(
                    self.split_a_dim,
                    self.split_b_dim,
                    hidden_channels,
                    direction,
                    dilation_pattern,
                )
            )

        self.actnorm_scale = nn.Parameter(torch.ones(1, self.input_dim, 1))
        self.actnorm_bias = nn.Parameter(torch.zeros(1, self.input_dim, 1))
        self.actnorm_initialized = False

        self.register_buffer("prior_mean_tensor", torch.full((1,), prior_mean))
        self.register_buffer("prior_log_var", torch.zeros(1))

    def set_actnorm_init(self) -> None:
        """Mark ActNorm as initialized (used on checkpoint restore)."""
        self.actnorm_initialized = True

    def _initialize_actnorm(self, x: torch.Tensor) -> None:
        """Initialize ActNorm parameters using batch statistics."""
        with torch.no_grad():
            mean = x.mean(dim=[0, 2], keepdim=True)
            std = x.std(dim=[0, 2], keepdim=True)
            self.actnorm_bias.data.copy_(-mean)
            self.actnorm_scale.data.copy_(1.0 / (std + 1e-6))
            self.actnorm_initialized = True

    def forward(
        self,
        x_cont: torch.Tensor,
        x_cat: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run TrajVAD-T forward pass and compute NLL."""
        batch_size, time_steps, _ = x_cont.shape

        if self.use_emb_in_input and x_cat is not None:
            x_cat = torch.mode(x_cat, dim=1, keepdim=True).values.expand_as(x_cat)
            embedding = self.class_emb(x_cat.squeeze(-1))  # (B, T, emb_dim)
            input_tensor = torch.cat([x_cont, embedding], dim=2)
        else:
            input_tensor = x_cont

        input_tensor = input_tensor.permute(0, 2, 1)  # (B, C, T)

        if not self.actnorm_initialized and not reverse:
            self._initialize_actnorm(input_tensor)

        if reverse:
            raise NotImplementedError("Reverse pass is not implemented for TrajVAD-T")

        logdet_actnorm = torch.sum(torch.log(torch.abs(self.actnorm_scale))) * time_steps
        latent = (input_tensor + self.actnorm_bias) * self.actnorm_scale
        total_logdet = logdet_actnorm.repeat(batch_size)

        split_a = latent[:, self.split_a_indices, :]
        split_b = latent[:, self.split_b_indices, :]
        for layer in self.layers:
            split_a, split_b, logdet = layer(split_a, split_b)
            total_logdet += logdet

        z = torch.cat([split_a, split_b], dim=1)
        gaussian_log_prob = -0.5 * ((z - self.prior_mean_tensor) ** 2)
        gaussian_log_prob -= 0.5 * torch.log(torch.tensor(2 * torch.pi, device=z.device, dtype=z.dtype))
        gaussian_log_prob = torch.sum(gaussian_log_prob, dim=[1, 2])

        nll = -(gaussian_log_prob + total_logdet) / (time_steps * self.input_dim)
        return z, nll

    def get_anomaly_score(
        self,
        x_cont: torch.Tensor,
        x_cat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return per-sample NLL anomaly score."""
        _, nll = self.forward(x_cont, x_cat, reverse=False)
        return nll


def get_trajvad_t_model(
    cont_dim: int = 27,
    emb_dim: int = 3,
    num_classes: int = 80,
    hidden_channels: int = 64,
    K: int = 6,
    use_emb_in_input: bool = True,
    prior_mean: float = 3.0,
    seg_len: int = 12,
    device: str = "cuda",
    wavenet_dilation: bool = True,
    max_dilation: int = 12,
    feature_names: Optional[List[str]] = None,
) -> TrajVADTModel:
    """Factory helper for TrajVAD-T model construction."""
    model = TrajVADTModel(
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
    return model.to(device)
