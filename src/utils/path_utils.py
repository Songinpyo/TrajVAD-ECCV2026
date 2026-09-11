"""
Path utilities for Trajectory-VAD project.

Provides consistent path generation for:
- Feature data directories
- Ground truth paths
"""
from pathlib import Path
from typing import Union


def get_feature_path(
    dataset: str,
    feature_variant: str,
    project_root: Union[str, Path]
) -> Path:
    """
    Get feature data root path with variant support.

    Args:
        dataset: Dataset name (e.g., 'shanghaitech', 'ubnormal')
        feature_variant: Feature variant name (e.g., 'seg12_std_nolog')
        project_root: Project root path

    Returns:
        Path to variant feature directory

    Example:
        >>> get_feature_path('shanghaitech', 'seg12_std_nolog', '/path/to/project')
        Path('/path/to/project/data/shanghaitech_tracks/shanghaitech_pose_features/variant_seg12_std_nolog')
    """
    project_root = Path(project_root)
    return (
        project_root / "data" / f"{dataset}_tracks" /
        f"{dataset}_pose_features" / f"variant_{feature_variant}"
    )


def get_gt_path(dataset: str, dataset_config, project_root: Union[str, Path]) -> str:
    """
    Get ground truth path based on dataset.

    Args:
        dataset: Dataset name
        dataset_config: DatasetConfig instance with gt_path attribute
        project_root: Project root path

    Returns:
        Path to ground truth file/directory as string

    Raises:
        ValueError: If dataset is unknown
    """
    project_root = Path(project_root)

    if dataset == 'shanghaitech':
        return str(project_root / "data" / dataset / "testing" / "test_frame_mask")
    elif dataset == 'ubnormal':
        local_gt = project_root / "data" / "ubnormal" / "ubnormal_frame_gt"
        if local_gt.exists():
            return str(local_gt)
        return dataset_config.gt_path
    elif dataset == 'msad':
        return str(project_root / 'data' / 'msad' / 'anomaly_annotation.csv')
    else:
        raise ValueError(f'Unknown dataset: {dataset}')
