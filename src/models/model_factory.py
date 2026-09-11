from typing import List, Optional

import torch.nn as nn

from .trajvad_p import get_trajvad_p_model
from .trajvad_t import get_trajvad_t_model


SUPPORTED_MODEL_TYPES = ["TrajVAD-T", "TrajVAD-P"]
_MODEL_TYPE_ALIASES = {
    "TrajVAD-T": "TrajVAD-T",
    "trajvad-t": "TrajVAD-T",
    "trajvad_t": "TrajVAD-T",
    "TrajVAD-P": "TrajVAD-P",
    "trajvad-p": "TrajVAD-P",
    "trajvad_p": "TrajVAD-P",
}
POSE_MODEL_TYPES = ["TrajVAD-P"]


def normalize_model_type(model_type: str) -> str:
    key = str(model_type).strip()
    normalized = _MODEL_TYPE_ALIASES.get(key)
    if normalized is None:
        normalized = _MODEL_TYPE_ALIASES.get(key.lower())
    if normalized is None:
        raise ValueError(
            f"Unknown model_type: {model_type}. Supported: {SUPPORTED_MODEL_TYPES}"
        )
    return normalized




def create_model(
    model_type: str,
    cont_dim: int,
    emb_dim: int = 3,
    num_classes: int = 80,
    hidden_channels: int = 64,
    K: int = 6,
    K_pose: Optional[int] = None,
    use_emb_in_input: bool = True,
    prior_mean: float = 3.0,
    seg_len: int = 12,
    pose_dim: int = 51,
    lambda_pose: float = 1.0,
    wavenet_dilation: bool = True,
    max_dilation: int = 12,
    device: str = "cuda",
    feature_names: Optional[List[str]] = None,
    logger=None,
) -> nn.Module:
    normalized_model_type = normalize_model_type(model_type)

    if normalized_model_type == "TrajVAD-T":
        model = get_trajvad_t_model(
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
    elif normalized_model_type == "TrajVAD-P":
        model = get_trajvad_p_model(
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
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. Supported: {SUPPORTED_MODEL_TYPES}"
        )

    if logger:
        logger.info(f"Model Type: {normalized_model_type}")
        logger.info(f"Input dim: {cont_dim}, K={K}, K_pose={K_pose}")
    return model


def requires_pose(model_type: str) -> bool:
    return normalize_model_type(model_type) in POSE_MODEL_TYPES
