"""
Scoring utilities for anomaly detection evaluation.

This module provides unified AUC computation following STG-NF conventions:
- Micro AUC: All frames concatenated → single AUC (STG-NF standard)
- Segment scores are treated as normality scores (-NLL; higher=normal)

Supports both old and new metadata formats:
- Old: List [scene_id, clip_id, person_id, start_frame]
- New: Dict {'video_name': '...', 'track_id': ..., 'start_frame': ...}
"""
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score, average_precision_score


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


# Scene/clip pairs to skip for ShanghaiTech-HR evaluation
SHANGHAITECH_HR_SKIP: List[Tuple[int, int]] = [
    (1, 130),
    (1, 135),
    (1, 136),
    (6, 144),
    (6, 145),
    (12, 152),
]

# MSAD anomaly type classification
# HR (Human-Related): 7 categories
MSAD_HR_TYPES = [
    'Assault',
    'Fighting',
    'People_falling',
    'Robbery',
    'Shooting',
    'Traffic_accident',
    'Vandalism'
]

# Non-HR: 4 categories
MSAD_NON_HR_TYPES = [
    'Explosion',
    'Fire',
    'Object_falling',
    'Water_incident'
]


def get_msad_video_category(video_name: str) -> str:
    """
    Extract anomaly category from MSAD video name.
    
    Args:
        video_name: Video name (e.g., 'Assault_1', 'Fire_10', 'normal_001')
    
    Returns:
        Category type: 'hr', 'non_hr', or 'normal'
    """
    # Check if video name matches any HR type
    for hr_type in MSAD_HR_TYPES:
        if video_name.startswith(hr_type):
            return 'hr'
    
    # Check if video name matches any Non-HR type
    for non_hr_type in MSAD_NON_HR_TYPES:
        if video_name.startswith(non_hr_type):
            return 'non_hr'
    
    # Default to normal (training videos or unrecognized)
    return 'normal'

def load_ubnormal_hr_masks(hr_mask_dir: str) -> Dict[str, BoolArray]:
    """
    Load UBnormal HR boolean masks for each clip.
    
    Args:
        hr_mask_dir: Path to test_frame_mask directory containing per-clip .npy files
    
    Returns:
        Dictionary mapping video_name -> boolean mask array
        video_name is in GT format: e.g., 'abnormal_scene_10_scenario_1'
        mask[i] = True means frame i should be kept for HR evaluation
    """
    import glob
    import os
    from pathlib import Path
    
    def convert_hr_filename_to_gt_name(filename: str) -> str:
        """
        Convert HR mask filename to GT video_name format.
        
        HR mask format: SSS_NNNN (e.g., '001_0100', '101_0110')
        GT format: {abnormal|normal}_scene_{scene}_scenario_{scenario}[_suffix]
        
        Rules:
        - Scene < 100: abnormal, scene = padded 3 digits
        - Scene >= 100: normal, scene = scene - 100
        - Scenario: first digit is scenario number, remaining are sub-variant/suffix
        """
        parts = filename.split('_')
        if len(parts) != 2:
            return filename  # Can't parse, return as-is
        
        scene_str, scenario_str = parts
        try:
            scene_num = int(scene_str)
            scenario_num = int(scenario_str)
        except ValueError:
            return filename
        
        # Determine prefix (abnormal/normal)
        if scene_num < 100:
            prefix = 'abnormal'
        else:
            prefix = 'normal'
            scene_num = scene_num - 100
        
        # Extract scenario number (first 1-2 digits based on pattern)
        # Pattern observation: 0100 -> scenario 1, 0200 -> scenario 2, etc.
        # The second two digits seem to be sub-variant (00, 51, 52, 53, etc.)
        scenario_main = scenario_num // 100  # e.g., 0100 -> 1, 0200 -> 2
        
        return f"{prefix}_scene_{scene_num}_scenario_{scenario_main}"
    
    hr_masks = {}
    mask_pattern = os.path.join(hr_mask_dir, '*.npy')
    for mask_path in glob.glob(mask_pattern):
        hr_filename = Path(mask_path).stem  # e.g., '001_0100'
        gt_video_name = convert_hr_filename_to_gt_name(hr_filename)
        
        # Load mask
        mask = np.load(mask_path)
        
        # If multiple HR mask files map to the same GT name (e.g., scenario variants),
        # we need to handle this. For now, prefer the one with more HR frames.
        if gt_video_name in hr_masks:
            # Keep the one with more True values (more HR frames)
            if mask.sum() > hr_masks[gt_video_name].sum():
                hr_masks[gt_video_name] = mask
        else:
            hr_masks[gt_video_name] = mask
    
    return hr_masks


def parse_metadata(metadata: List[Any]) -> Tuple[Dict[str, List[Tuple[int, int, int]]], bool]:
    """
    Parse metadata into a unified format.
    
    Args:
        metadata: List of metadata entries (old or new format)
    
    Returns:
        Tuple of (parsed_dict, is_new_format)
        parsed_dict: {video_name: [(track_id, start_frame, score_idx), ...]}
    """
    if len(metadata) == 0:
        return {}, True
    
    # Detect format
    sample = metadata[0]
    is_new_format = isinstance(sample, dict)
    
    video_segments = defaultdict(list)
    
    for idx, m in enumerate(metadata):
        if is_new_format:
            # New format: dict
            video_name = m['video_name']
            track_id = m['track_id']
            start_frame = m['start_frame']
        else:
            # Old format: list/array [scene_id, clip_id, person_id, start_frame]
            if isinstance(m, np.ndarray):
                m = m.tolist()
            scene_id = int(m[0])
            clip_id = int(m[1])
            track_id = int(m[2])
            start_frame = int(m[3])
            video_name = f"{scene_id:02d}_{clip_id:04d}"
        
        video_segments[video_name].append((track_id, start_frame, idx))
    
    return dict(video_segments), is_new_format


def compute_micro_auc(
    scores: FloatArray,
    metadata: List[Any],
    gt_dict: Dict[str, FloatArray],
    seg_len: int = 12,
    smoothing: bool = True,
    smoothing_sigma_range: Tuple[int, int] = (1, 7),
    flip_gt: bool = True,
    verbose: bool = False,
    dataset: str = 'ShanghaiTech',
    hr_mask_dict: Optional[Dict[str, BoolArray]] = None
) -> Dict[str, Any]:
    """
    Compute frame-level micro AUC/AP with the public TrajVAD protocol.

    The public release uses the canonical STG-NF aggregation: segment scores
    are assigned to the center frame, min-pooled within each track, min-pooled
    across tracks, then smoothed with Gaussian sigma 1-6.
    
    Args:
        scores: Segment-level scores (N,)
        metadata: List of metadata entries (old or new format)
        gt_dict: Dictionary mapping video_name -> frame-level GT
        seg_len: Segment length (default: 12)
        smoothing: Apply Gaussian smoothing
        smoothing_sigma_range: Range of sigma values for smoothing
        flip_gt: Flip GT (1=normal, 0=abnormal for STG-NF)
        verbose: Print debug info
    
    Returns:
        Dictionary with:
            - micro_auc: Main metric (ROC AUC over all frames)
            - micro_ap: Average precision over all frames
            - frame_scores: Concatenated scores
            - frame_labels: Concatenated GT labels
            - clip_results: Per-clip stats
            - num_clips, num_frames
    """
    # Parse metadata to unified format
    video_segments, is_new_format = parse_metadata(metadata)
    
    if verbose:
        print(f"Metadata format: {'new (dict)' if is_new_format else 'old (list)'}")
        print(f"Videos with segments: {len(video_segments)}")
    
    clip_list = sorted(gt_dict.keys())
    
    dataset_gt_arr: list[FloatArray] = []
    dataset_scores_arr: list[FloatArray] = []
    clip_results: dict[str, dict[str, Any]] = {}

    num_gt_clips = 0
    num_missing_segment_clips = 0
    num_missing_segment_frames = 0

    clip_entries: list[tuple[str, FloatArray, int, int]] = []
    
    for video_name in clip_list:
        # Dataset-specific video filtering
        if dataset == 'ShanghaiTech-HR':
            # Skip non-HR videos for ShanghaiTech-HR
            try:
                scene_str, clip_str = video_name.split('_')[:2]
                scene_id = int(scene_str)
                clip_id = int(clip_str)
            except (ValueError, IndexError):
                scene_id, clip_id = None, None
            if scene_id is not None and (scene_id, clip_id) in SHANGHAITECH_HR_SKIP:
                continue
        
        elif dataset == 'ShanghaiTech-Non-HR':
            # Only keep non-HR videos for ShanghaiTech-Non-HR (inverse of HR skip list)
            try:
                scene_str, clip_str = video_name.split('_')[:2]
                scene_id = int(scene_str)
                clip_id = int(clip_str)
            except (ValueError, IndexError):
                scene_id, clip_id = None, None
            # Skip if NOT in the skip list (i.e., keep only non-HR)
            if scene_id is not None and (scene_id, clip_id) not in SHANGHAITECH_HR_SKIP:
                continue
        
        elif dataset == 'MSAD-HR':
            # Only keep HR videos for MSAD-HR
            category = get_msad_video_category(video_name)
            if category != 'hr':
                continue
        
        elif dataset == 'MSAD-Non-HR':
            # Only keep Non-HR videos for MSAD-Non-HR
            category = get_msad_video_category(video_name)
            if category != 'non_hr':
                continue
        
        # Load GT
        clip_gt = gt_dict[video_name].copy()
        if flip_gt:
            clip_gt = 1 - clip_gt  # 1=normal, 0=abnormal

        num_gt_clips += 1
        
        init_val = np.inf
        
        segments = video_segments.get(video_name, [])
        is_missing_segments = (video_name not in video_segments) or (len(segments) == 0)

        if is_missing_segments:
            clip_score = init_val * np.ones(len(clip_gt))
        else:
            track_ids = set(seg[0] for seg in segments)
            track_scores = {tid: np.ones(len(clip_gt)) * init_val for tid in track_ids}

            for track_id, start_frame, score_idx in segments:
                frame_idx = start_frame + seg_len // 2
                if frame_idx >= len(clip_gt):
                    continue
                track_scores[track_id][frame_idx] = min(
                    track_scores[track_id][frame_idx],
                    scores[score_idx],
                )

            track_scores_arr = np.stack(list(track_scores.values()))
            clip_score = np.amin(track_scores_arr, axis=0)
        
        # UBnormal-HR: Apply frame-level HR mask filtering
        if dataset == 'UBnormal-HR' and hr_mask_dict is not None:
            if video_name in hr_mask_dict:
                hr_mask = hr_mask_dict[video_name]
                # Ensure mask length matches clip length
                if len(hr_mask) == len(clip_gt):
                    clip_gt = clip_gt[hr_mask]
                    clip_score = clip_score[hr_mask]
                # If lengths mismatch, skip HR filtering for this clip (use all frames)
        
        # UBnormal-Non-HR: normal frames + Non-HR anomaly frames
        elif dataset == 'UBnormal-Non-HR' and hr_mask_dict is not None:
            if video_name in hr_mask_dict:
                hr_mask = hr_mask_dict[video_name]
                # Ensure mask length matches clip length
                if len(hr_mask) == len(clip_gt):
                    # Non-HR = normal frames (clip_gt=1 after flip) OR Non-HR anomaly frames (~hr_mask)
                    non_hr_mask = (clip_gt == 1) | (~hr_mask)
                    clip_gt = clip_gt[non_hr_mask]
                    clip_score = clip_score[non_hr_mask]
                # If lengths mismatch, skip filtering for this clip (use all frames)
        

        if is_missing_segments:
            num_missing_segment_clips += 1
            num_missing_segment_frames += len(clip_gt)

        dataset_gt_arr.append(clip_gt)
        dataset_scores_arr.append(clip_score)

        num_abnormal = int((clip_gt == 0).sum()) if flip_gt else int(clip_gt.sum())
        num_normal = int((clip_gt == 1).sum()) if flip_gt else int((clip_gt == 0).sum())
        clip_entries.append((video_name, clip_gt, num_abnormal, num_normal))
    
    if len(dataset_scores_arr) == 0:
        return {
            'micro_auc': 0.0,
            'frame_scores': np.array([]),
            'frame_labels': np.array([]),
            'clip_results': {},
            'num_clips': 0,
            'num_frames': 0,
            'num_gt_clips': 0,
            'num_missing_segment_clips': 0,
            'num_missing_segment_frames': 0,
        }

    lengths = [len(arr) for arr in dataset_scores_arr]
    scores_all = np.concatenate(dataset_scores_arr)
    finite_mask = np.isfinite(scores_all)
    has_any_finite = bool(finite_mask.any())
    if has_any_finite:
        max_finite = scores_all[finite_mask].max()
        min_finite = scores_all[finite_mask].min()
        scores_all = scores_all.copy()
        scores_all[np.isposinf(scores_all)] = max_finite
        scores_all[np.isneginf(scores_all)] = min_finite
        scores_all[np.isnan(scores_all)] = max_finite
    else:
        scores_all = np.zeros_like(scores_all, dtype=float)

    dataset_scores_arr = []
    cursor = 0
    for length in lengths:
        dataset_scores_arr.append(scores_all[cursor : cursor + length].copy())
        cursor += length

    for (video_name, clip_gt, num_abnormal, num_normal), clip_score in zip(clip_entries, dataset_scores_arr):
        clip_auc: Optional[float]
        if not has_any_finite:
            clip_auc = None
        elif len(np.unique(clip_gt)) > 1:
            clip_auc_raw = roc_auc_score(clip_gt, clip_score)
            if isinstance(clip_auc_raw, np.ndarray):
                clip_auc = float(clip_auc_raw.item())
            else:
                clip_auc = float(clip_auc_raw)
        else:
            clip_auc = None

        clip_results[video_name] = {
            'auc': clip_auc,
            'num_frames': len(clip_score),
            'num_abnormal': num_abnormal,
            'num_normal': num_normal,
        }
    
    # Apply Gaussian smoothing
    if smoothing:
        for s in range(len(dataset_scores_arr)):
            for sig in range(*smoothing_sigma_range):
                dataset_scores_arr[s] = gaussian_filter1d(dataset_scores_arr[s], sigma=sig).astype(np.float64, copy=False)
    
    # Concatenate all clips → Micro AUC
    gt_np = np.concatenate(dataset_gt_arr)
    scores_np = np.concatenate(dataset_scores_arr)
    
    # Compute global metrics
    if len(np.unique(gt_np)) > 1:
        micro_auc = roc_auc_score(gt_np, scores_np)
        micro_ap = average_precision_score(gt_np, scores_np)
    else:
        micro_auc = 0.0
        micro_ap = 0.0
    
    return {
        'micro_auc': micro_auc,
        'micro_ap': micro_ap,
        'frame_scores': scores_np,
        'frame_labels': gt_np,
        'clip_results': clip_results,
        'num_clips': len(clip_results),
        'num_frames': len(scores_np),
        'num_gt_clips': num_gt_clips,
        'num_missing_segment_clips': num_missing_segment_clips,
        'num_missing_segment_frames': num_missing_segment_frames,
    }


def compute_micro_auc_stgnf(
    scores: FloatArray,
    metadata: List[Any],
    gt_dict: Dict[str, FloatArray],
    seg_len: int = 12,
    dataset: str = 'ShanghaiTech',
    hr_mask_dict: Optional[Dict[str, BoolArray]] = None
) -> Dict[str, Any]:
    """
    Compute Micro AUC using exact STG-NF conventions.
    
    - Score: -NLL (higher = more normal)
    - Pooling: MIN (select most abnormal)
    - Frame: Center only
    - GT: Dataset-dependent flip
    - Smoothing: Gaussian (sigma 1-6)
    
    GT Conventions (raw dataset GT):
    - ShanghaiTech: 0=normal, 1=abnormal → needs flip to 1=normal
    - UBnormal: 0=normal, 1=abnormal → needs flip to 1=normal
    - MSAD: 0=normal, 1=abnormal → needs flip to 1=normal
    """
    # Ensure seg_len is not None
    if seg_len is None:
        seg_len = 12
    
    flip_gt = flip_gt_for_dataset_stgnf(dataset)

    return compute_micro_auc(
        scores=scores,
        metadata=metadata,
        gt_dict=gt_dict,
        seg_len=seg_len,
        smoothing=True,
        smoothing_sigma_range=(1, 7),
        flip_gt=flip_gt,
        dataset=dataset,
        hr_mask_dict=hr_mask_dict
    )


def flip_gt_for_dataset_stgnf(dataset: str) -> bool:
    ds = (dataset or "").lower()

    if "ubnormal" in ds:
        return True
    if "msad" in ds:
        return True
    return True
