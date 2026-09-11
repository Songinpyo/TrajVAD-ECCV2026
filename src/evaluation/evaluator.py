"""
Evaluation utilities for Traj-VAD models.

Provides model evaluation with STG-NF scoring method.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, Any

from utils.scoring import (  # pyright: ignore[reportImplicitRelativeImport]
    compute_micro_auc_stgnf,
    load_ubnormal_hr_masks,
)


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader[Any],
    test_dataset,
    gt_dict: Dict[str, Any],
    device: str,
    seg_len: int | None = None,
    dataset_name: str = 'shanghaitech',
    epoch: int | None = None,
    project_root: Optional[Path] = None,
    model_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate model on test set using STG-NF scoring method.

    This function computes anomaly scores for all test samples and
    calculates frame-level AUC/AP metrics following the STG-NF protocol.

    Args:
        model: Traj-VAD model
        test_loader: Test data loader
        test_dataset: Test dataset instance (must have get_metadata method)
        gt_dict: Ground truth dictionary {video_name: frame_labels}
        device: Device to run evaluation on
        seg_len: Segment length (if None, extracted from model)
        dataset_name: Dataset name for scoring protocol
        epoch: Epoch number for progress bar description
        project_root: Project root path for loading HR masks.

    Returns:
        Dictionary with evaluation results:
            - micro_auc: Micro-averaged AUC
            - micro_ap: Micro-averaged AP
            - micro_auc_hr: AUC for human-related anomalies
            - micro_ap_hr: AP for human-related anomalies
            - micro_auc_non_hr: AUC for non-human-related anomalies
            - micro_ap_non_hr: AP for non-human-related anomalies
            - frame_scores: Per-frame anomaly scores
            - frame_labels: Per-frame ground truth labels
            - scores: Raw per-segment scores
            - metadata: Per-segment metadata
            - clip_results: Per-clip evaluation results
            - num_clips: Number of clips evaluated
            - num_frames: Number of frames evaluated
    """
    model.eval()

    # Extract seg_len from model if not provided
    if seg_len is None:
        if hasattr(model, 'pose_shape') and model.pose_shape is not None:
            seg_len = model.pose_shape[1]
        elif hasattr(model, 'seg_len'):
            seg_len = model.seg_len
        else:
            seg_len = 12

    assert seg_len is not None
    seg_len_val: int = int(seg_len)

    # Infer project_root if not provided
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent

    all_scores = []
    all_metadata = []

    # Progress bar description
    if epoch is not None:
        desc = f"Epoch {epoch} [Test]"
    else:
        desc = "Evaluating"

    pbar = tqdm(test_loader, desc=desc, ncols=100, leave=False)

    def _requires_pose(model_name: Optional[str]) -> bool:
        if model_name is None:
            return False
        try:
            from models.model_factory import requires_pose as _requires_pose_impl  # pyright: ignore[reportImplicitRelativeImport]
        except Exception:
            return False
        return bool(_requires_pose_impl(model_name))

    expects_pose = _requires_pose(model_type)

    with torch.no_grad():
        batch_start_idx = 0
        for batch in pbar:
            batch = [value.to(device) for value in batch]
            x_cont, x_cat = batch[:2]
            kwargs = dict(x_pose=batch[2], pose_mask=batch[3]) if expects_pose else {}
            nll = model.get_anomaly_score(x_cont, x_cat, **kwargs)

            # Convert to scores (higher = more normal)
            scores = (-1 * nll).cpu().numpy()

            # Collect metadata
            batch_size = x_cont.shape[0]
            for i in range(batch_size):
                all_metadata.append(test_dataset.get_metadata(batch_start_idx + i))
            batch_start_idx += batch_size
            all_scores.append(scores)

    all_scores = np.concatenate(all_scores, axis=0)

    # Determine dataset name for scoring
    scoring_dataset = 'ShanghaiTech' if dataset_name == 'shanghaitech' else dataset_name.capitalize()

    # Main evaluation
    frame_results = compute_micro_auc_stgnf(
        all_scores,
        all_metadata,
        gt_dict,
        seg_len_val,
        dataset=scoring_dataset
    )

    # HR and Non-HR protocols
    frame_results_hr = None
    frame_results_non_hr = None

    if dataset_name == 'shanghaitech':
        frame_results_hr = compute_micro_auc_stgnf(
            all_scores, all_metadata, gt_dict, seg_len_val,
            dataset='ShanghaiTech-HR'
        )
        frame_results_non_hr = compute_micro_auc_stgnf(
            all_scores, all_metadata, gt_dict, seg_len_val,
            dataset='ShanghaiTech-Non-HR'
        )

    elif dataset_name == 'ubnormal':
        hr_mask_path = project_root / "data" / "ubnormal" / "MoCoDAD" / "hr_bool_masks" / "testing" / "test_frame_mask"
        if hr_mask_path.exists():
            hr_masks = load_ubnormal_hr_masks(str(hr_mask_path))
            if hr_masks:
                frame_results_hr = compute_micro_auc_stgnf(
                    all_scores, all_metadata, gt_dict, seg_len_val,
                    dataset='UBnormal-HR', hr_mask_dict=hr_masks
                )
                frame_results_non_hr = compute_micro_auc_stgnf(
                    all_scores, all_metadata, gt_dict, seg_len_val,
                    dataset='UBnormal-Non-HR', hr_mask_dict=hr_masks
                )

    elif dataset_name == 'msad':
        frame_results_hr = compute_micro_auc_stgnf(
            all_scores, all_metadata, gt_dict, seg_len_val,
            dataset='MSAD-HR'
        )
        frame_results_non_hr = compute_micro_auc_stgnf(
            all_scores, all_metadata, gt_dict, seg_len_val,
            dataset='MSAD-Non-HR'
        )

    # Build result dictionary
    result = {
        'micro_auc': frame_results['micro_auc'],
        'micro_ap': frame_results.get('micro_ap', 0.0),
        'frame_auc': frame_results['micro_auc'],  # Backward compatibility
        'scores': all_scores,
        'metadata': all_metadata,
        'frame_scores': frame_results['frame_scores'],
        'frame_labels': frame_results['frame_labels'],
        'clip_results': frame_results['clip_results'],
        'num_clips': frame_results['num_clips'],
        'num_frames': frame_results['num_frames'],
        'num_gt_clips': frame_results.get('num_gt_clips', 0),
        'num_missing_segment_clips': frame_results.get('num_missing_segment_clips', 0),
        'num_missing_segment_frames': frame_results.get('num_missing_segment_frames', 0),
    }

    # Add HR results
    if frame_results_hr is not None:
        result['micro_auc_hr'] = frame_results_hr['micro_auc']
        result['micro_ap_hr'] = frame_results_hr.get('micro_ap', 0.0)
    else:
        result['micro_auc_hr'] = 0.0
        result['micro_ap_hr'] = 0.0

    # Add Non-HR results
    if frame_results_non_hr is not None:
        result['micro_auc_non_hr'] = frame_results_non_hr['micro_auc']
        result['micro_ap_non_hr'] = frame_results_non_hr.get('micro_ap', 0.0)
    else:
        result['micro_auc_non_hr'] = 0.0
        result['micro_ap_non_hr'] = 0.0

    return result
