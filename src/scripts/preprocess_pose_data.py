#!/usr/bin/env python3
"""
Feature Extraction Pipeline for Multi-class Trajectory VAD (with Raw Pose).

Loads processed parquet tracks AND pose data, generates model-ready .pt files.
Includes 'x_pose' (raw keypoints) in the output.

Pipeline:
1. Load parquet (gap-handled tracks)
2. Load pose parquet (processed poses)
3. Merge track and pose data
4. Optional Kalman filtering (on tracks)
5. Extract trajectory features + class_id
6. Extract raw pose keypoints
7. Log transformation (on selected features)
8. Standard scaling (on features, fit on train only)
9. Save as .pt files

Modes:
    Standard: Saves x_cont (27D scaled features)

Usage:
    python src/scripts/preprocess_pose_data.py --dataset shanghaitech --split all
"""
import os
import sys
import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.bbox_features import BBoxFeatureExtractor, FEATURE_NAMES
from utils.kalman_filter import smooth_trajectory_with_kalman
from utils.dataset_config import get_dataset_config
from utils.feature_config import load_feature_config, save_config_to_output

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# Constants
SEG_LEN = 12
SEG_STRIDE_TRAIN = 4
SEG_STRIDE_TEST = 1




def load_video_data(track_path: Path, pose_path: Path) -> Dict[int, pd.DataFrame]:
    """
    Load tracks and poses, merge them, and group by track_id.

    If pose file doesn't exist, creates zero-filled keypoints for all frames.

    Returns:
        Dict[track_id -> DataFrame with columns: frame_id, x1, y1, x2, y2, conf, class_id, keypoints]
    """
    if not track_path.exists():
        raise FileNotFoundError(f"Track file not found: {track_path}")

    df_track = pd.read_parquet(track_path)

    # Handle missing pose file: create zero-filled keypoints
    if not pose_path.exists():
        logger.warning(f"Pose file not found, using zero keypoints: {pose_path.name}")
        # Create zero keypoints (17 keypoints, each with x, y, conf = 0)
        zero_kp = np.zeros((17, 3), dtype=np.float32).tolist()
        df_track['keypoints'] = [zero_kp] * len(df_track)

        tracks = {}
        for track_id in df_track['track_id'].unique():
            track_df = df_track[df_track['track_id'] == track_id].sort_values('frame_id')
            tracks[track_id] = track_df
        return tracks

    df_pose = pd.read_parquet(pose_path)

    # Merge on frame_id and track_id
    # df_pose has 'keypoints' column

    # Select relevant columns from pose
    # Note: df_pose might also have bbox, but we trust df_track for bbox (processed/interpolated)
    # However, df_pose has the raw keypoints we want.

    if 'keypoints' not in df_pose.columns:
        logger.warning(f"Pose file missing 'keypoints' column, using zero keypoints: {pose_path.name}")
        zero_kp = np.zeros((17, 3), dtype=np.float32).tolist()
        df_track['keypoints'] = [zero_kp] * len(df_track)

        tracks = {}
        for track_id in df_track['track_id'].unique():
            track_df = df_track[df_track['track_id'] == track_id].sort_values('frame_id')
            tracks[track_id] = track_df
        return tracks

    # Use LEFT JOIN to keep all frames from preprocessed tracks (which include interpolated gaps)
    # This introduces NaNs in 'keypoints' for the interpolated frames
    join_key = 'source_track_id' if 'source_track_id' in df_track else 'track_id'
    pose_columns = df_pose[['frame_id', 'track_id', 'keypoints']].rename(columns={'track_id': join_key})
    merged = pd.merge(df_track, pose_columns, on=['frame_id', join_key], how='left', validate='one_to_one')

    tracks = {}
    for track_id in merged['track_id'].unique():
        track_df = merged[merged['track_id'] == track_id].sort_values('frame_id')

        # Check if we need to interpolate poses (if there are gaps filled by preprocess_tracks)
        if track_df['keypoints'].isnull().any():
            track_df = interpolate_pose(track_df)

        tracks[track_id] = track_df

    return tracks


def interpolate_pose(track_df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate missing keypoints for frames that were gap-filled.

    Args:
        track_df: DataFrame with 'frame_id' and 'keypoints' (some None/NaN)

    Returns:
        DataFrame with filled 'keypoints'
    """
    result = track_df.copy()
    valid = np.flatnonzero(result['keypoints'].notnull().to_numpy())
    frames = result['frame_id'].to_numpy()
    for left, right in zip(valid[:-1], valid[1:]):
        gap = frames[right] - frames[left] - 1
        if not 0 < gap <= 10:
            continue
        first = np.asarray(result.iloc[left]['keypoints'].tolist(), dtype=np.float32) if isinstance(result.iloc[left]['keypoints'], np.ndarray) else np.asarray(result.iloc[left]['keypoints'], dtype=np.float32)
        last = np.asarray(result.iloc[right]['keypoints'].tolist(), dtype=np.float32) if isinstance(result.iloc[right]['keypoints'], np.ndarray) else np.asarray(result.iloc[right]['keypoints'], dtype=np.float32)
        for index in range(left + 1, right):
            alpha = (frames[index] - frames[left]) / (frames[right] - frames[left])
            result.at[result.index[index], 'keypoints'] = ((1 - alpha) * first + alpha * last).tolist()
    return result



def apply_kalman_to_tracks(
    tracks: Dict[int, pd.DataFrame],
    process_noise: float = 1.0,
    measurement_noise: float = 10.0
) -> Dict[int, pd.DataFrame]:
    """Apply Kalman filtering to all tracks (bbox only)."""
    smoothed_tracks = {}

    for track_id, track_df in tracks.items():
        # Extract bbox trajectory (convert x1y1x2y2 to cxcywh)
        x1 = track_df['x1'].values
        y1 = track_df['y1'].values
        x2 = track_df['x2'].values
        y2 = track_df['y2'].values

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        trajectory = np.stack([cx, cy, w, h], axis=1)  # (T, 4)

        # Smooth trajectory
        smoothed = smooth_trajectory_with_kalman(
            trajectory, process_noise, measurement_noise
        )

        # Convert back to x1y1x2y2
        cx_s, cy_s, w_s, h_s = smoothed[:, 0], smoothed[:, 1], smoothed[:, 2], smoothed[:, 3]
        x1_s = cx_s - w_s / 2
        y1_s = cy_s - h_s / 2
        x2_s = cx_s + w_s / 2
        y2_s = cy_s + h_s / 2

        # Create new DataFrame
        smoothed_df = track_df.copy()
        smoothed_df['x1'] = x1_s
        smoothed_df['y1'] = y1_s
        smoothed_df['x2'] = x2_s
        smoothed_df['y2'] = y2_s

        smoothed_tracks[track_id] = smoothed_df

    return smoothed_tracks


def process_video(
    track_path: Path,
    pose_path: Path,
    extractor: BBoxFeatureExtractor,
    apply_kalman: bool = True,
    seg_stride: int = 1,
    seg_len: int = 12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
    """
    Process a single video file.

    Returns:
        x_cont: (N, T, 27) continuous features
        x_cat: (N, T, 1) class IDs
        x_pose: (N, T, 17, 3) raw pose keypoints
        metadata: List of dicts with video_name, track_id, start_frame
    """
    video_name = track_path.stem

    try:
        tracks = load_video_data(track_path, pose_path)
    except FileNotFoundError as e:
        logger.warning(f"Skipping {video_name}: {e}")
        return (np.empty((0, seg_len, 27), dtype=np.float32),
                np.empty((0, seg_len, 1), dtype=np.int64),
                np.empty((0, seg_len, 17, 3), dtype=np.float32),
                [])

    if apply_kalman:
        tracks = apply_kalman_to_tracks(tracks)

    all_x_cont = []
    all_x_cat = []
    all_x_pose = []
    all_metadata = []

    for track_id, track_df in tracks.items():
        # Skip tracks shorter than segment length
        min_len = seg_len
        if len(track_df) < min_len:
            continue

        # Extract bbox, conf, class_id
        bboxes = track_df[['x1', 'y1', 'x2', 'y2']].values
        conf = track_df['conf'].values
        class_ids = track_df['class_id'].values.astype(np.int64)
        frame_ids = track_df['frame_id'].values

        # Extract keypoints
        # keypoints column contains lists or arrays. Need to stack them.
        # Handle potential None values (though preprocessing should have filled them)
        raw_kps = track_df['keypoints'].values

        # Convert to numpy array (T, 17, 3)
        # We need to ensure all elements are valid arrays
        valid_kps = []
        for kp in raw_kps:
            if kp is None or (isinstance(kp, float) and np.isnan(kp)):
                 # Should not happen with processed data, but fallback to zeros
                 valid_kps.append(np.zeros((17, 3), dtype=np.float32))
            else:
                 # kp is an array of objects (arrays), need to stack them
                 valid_kps.append(np.stack(kp).astype(np.float32))

        keypoints_arr = np.stack(valid_kps) # (T, 17, 3)

        # Update extractor stride for this call
        extractor.seg_stride = seg_stride

        x_cont, x_cat, start_indices = extractor.extract_from_track(
            bboxes, conf, class_ids
        )

        if len(x_cont) == 0:
            continue

        # Slice keypoints using start_indices
        x_pose_segments = []
        for start_idx in start_indices:
            end_idx = start_idx + seg_len
            segment_kp = keypoints_arr[start_idx:end_idx]
            x_pose_segments.append(segment_kp)

        x_pose = np.stack(x_pose_segments) # (N, seg_len, 17, 3)

        all_x_cont.append(x_cont)
        all_x_cat.append(x_cat)
        all_x_pose.append(x_pose)

        # Store metadata
        for i, start_idx in enumerate(start_indices):
            all_metadata.append({
                'video_name': video_name,
                'track_id': track_id,
                'start_frame': int(frame_ids[start_idx]),
                'segment_idx': i
            })

    if len(all_x_cont) == 0:
        return (np.empty((0, seg_len, 27), dtype=np.float32),
                np.empty((0, seg_len, 1), dtype=np.int64),
                np.empty((0, seg_len, 17, 3), dtype=np.float32),
                [])

    x_cont = np.concatenate(all_x_cont, axis=0)
    x_cat = np.concatenate(all_x_cat, axis=0)
    x_pose = np.concatenate(all_x_pose, axis=0)

    return x_cont, x_cat, x_pose, all_metadata


def process_split(
    dataset_name: str,
    vid_res: Tuple[int, int],
    split: str,
    apply_kalman: bool = True,
    feature_config: Optional[Dict] = None
) -> Dict:
    """
    Process all videos in a split.
    """
    # Generate variant name from config
    if feature_config is not None:
        seg_len = feature_config.get('segment', {}).get('length', 12)
        norm_method = feature_config.get('normalization', {}).get('method', 'standard')
        apply_log = feature_config.get('extractor', {}).get('apply_log', False)

        # Variant naming: seg{length}_{norm}_{log}
        norm_short = {
            'standard': 'std',
            'minmax': 'mm',
            'robust': 'robust',
            'none': 'none'
        }.get(norm_method, norm_method)
        log_suffix = 'log' if apply_log else 'nolog'
        feature_variant = f"seg{seg_len}_{norm_short}_{log_suffix}"
    else:
        # Fallback to default
        feature_variant = "seg12_std_log"

    # Determine output path structure
    input_track_dir = DATA_ROOT / f"{dataset_name}_tracks" / f"{dataset_name}_tracks_processed" / split
    input_pose_dir = DATA_ROOT / f"{dataset_name}_tracks" / f"{dataset_name}_tracks_pose_processed" / split
    output_dir = DATA_ROOT / f"{dataset_name}_tracks" / f"{dataset_name}_pose_features" / f"variant_{feature_variant}" / split
    variant_dir = DATA_ROOT / f"{dataset_name}_tracks" / f"{dataset_name}_pose_features" / f"variant_{feature_variant}"

    output_dir.mkdir(parents=True, exist_ok=True)
    variant_dir.mkdir(parents=True, exist_ok=True)

    if not input_track_dir.exists():
        logger.warning(f"Input track directory does not exist: {input_track_dir}")
        return {'split': split, 'num_segments': 0}

    if not input_pose_dir.exists():
        logger.warning(f"Input pose directory does not exist: {input_pose_dir}")
        return {'split': split, 'num_segments': 0}

    parquet_files = sorted(input_track_dir.glob('*.parquet'))

    if len(parquet_files) == 0:
        logger.warning(f"No parquet files found in {input_track_dir}")
        return {'split': split, 'num_segments': 0}

    # Filter out abnormal videos for training split (semi-supervised datasets)
    if split == 'training':
        original_count = len(parquet_files)

        if dataset_name == 'ubnormal':
            # UBnormal: Filter out videos starting with "abnormal_" (if any)
            parquet_files = [f for f in parquet_files if not f.stem.startswith('abnormal_')]
            filtered_count = original_count - len(parquet_files)
            if filtered_count > 0:
                logger.info(f"  UBnormal: Filtered out {filtered_count} abnormal videos from training set")
                logger.info(f"  Remaining: {len(parquet_files)} normal videos")

        if len(parquet_files) == 0:
            logger.warning(f"No videos remaining after filtering")
            return {'split': split, 'num_segments': 0, 'version': 'kf' if apply_kalman else 'raw'}

    variant_label = f"[variant:{feature_variant}] " if feature_variant else ""
    logger.info(f"Processing {variant_label}{split}: {len(parquet_files)} videos")

    # Initialize extractor from config or defaults
    seg_stride = SEG_STRIDE_TRAIN if split == 'training' else SEG_STRIDE_TEST

    if feature_config is not None:
        # Use config to initialize extractor
        segment_cfg = feature_config.get('segment', {})
        seg_len = segment_cfg.get('length', SEG_LEN)
        extractor = BBoxFeatureExtractor(
            vid_res=vid_res,
            seg_len=seg_len,
            seg_stride=seg_stride,
            config=feature_config
        )
        # Save config to output directory for reproducibility
        if split == 'training':  # Only save once per variant
            save_config_to_output(feature_config, variant_dir)
            logger.info(f"  Saved config to {variant_dir}")
    else:
        # Default initialization (backward compatibility)
        extractor = BBoxFeatureExtractor(
            vid_res=vid_res,
            seg_len=SEG_LEN,
            seg_stride=seg_stride,
            apply_log=False
        )

    # Collect all data
    all_x_cont = []
    all_x_cat = []
    all_x_pose = []
    all_metadata = []

    for pq_file in tqdm(parquet_files, desc=f"[{split}]"):
        # Construct corresponding pose file path
        pose_file = input_pose_dir / pq_file.name

        x_cont, x_cat, x_pose, metadata = process_video(
            pq_file, pose_file, extractor, apply_kalman, seg_stride, extractor.seg_len
        )

        if len(x_cont) > 0:
            all_x_cont.append(x_cont)
            all_x_cat.append(x_cat)
            all_x_pose.append(x_pose)
            all_metadata.extend(metadata)

    if len(all_x_cont) == 0:
        logger.warning(f"No segments extracted for {split}")
        return {'split': split, 'num_segments': 0}

    x_cont = np.concatenate(all_x_cont, axis=0)
    x_cat = np.concatenate(all_x_cat, axis=0)
    x_pose = np.concatenate(all_x_pose, axis=0)

    logger.info(f"  Extracted: {len(x_cont)} segments, shape {x_cont.shape}")
    logger.info(f"  Pose shape: {x_pose.shape}")

    # Handle scaling
    version = "kf" if apply_kalman else "raw"
    scaler_dir = variant_dir
    scaler_dir.mkdir(parents=True, exist_ok=True)
    scaler_file = scaler_dir / f"scaler_{version}.pkl"

    if split == 'training':
        # Fit and transform on training data
        extractor.fit_scaler(x_cont)
        extractor.save_scaler(str(scaler_file))
        x_cont_scaled = extractor.transform(x_cont)
        logger.info(f"  Fitted scaler and saved to {scaler_file}")
    else:
        # Load scaler from training and transform
        if scaler_file.exists():
            extractor.load_scaler(str(scaler_file))
            x_cont_scaled = extractor.transform(x_cont)
            logger.info(f"  Loaded scaler from {scaler_file}")
        else:
            logger.warning(f"  Scaler not found at {scaler_file}, using unscaled features")
            x_cont_scaled = x_cont

    # Save to .pt file
    output_filename = f"{dataset_name}_{split}_{version}.pt"
    output_path = output_dir / output_filename

    data_dict = {
        'x_cont': torch.from_numpy(x_cont_scaled),  # (N, T, 16)
        'x_cat': torch.from_numpy(x_cat),           # (N, T, 1)
        'x_pose': torch.from_numpy(x_pose),         # (N, T, 17, 3)
        'metadata': all_metadata,
        'feature_names': extractor.get_feature_names(),
        'log_indices': extractor.get_log_indices(),
        'config': {
            'dataset': dataset_name,
            'split': split,
            'apply_kalman': apply_kalman,
            'seg_len': extractor.seg_len,
            'seg_stride': seg_stride,
            'vid_res': vid_res,
            'num_segments': len(x_cont),
        }
    }

    torch.save(data_dict, output_path)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"  Saved: {output_path} ({file_size_mb:.2f} MB)")

    return {
        'split': split,
        'version': version,
        'num_segments': len(x_cont),
        'shape': x_cont.shape,
        'pose_shape': x_pose.shape,
        'file_size_mb': file_size_mb
    }


def main():
    parser = argparse.ArgumentParser(description="Feature extraction for trajectory VAD (with Pose)")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['shanghaitech', 'ubnormal', 'msad'],
                        help='Dataset name')
    parser.add_argument('--split', type=str, default='all',
                        choices=['train', 'test', 'training', 'testing', 'all'])
    parser.add_argument('--feature_config', type=str, default=None,
                        help='Path to feature config YAML file (default: dataset segment length). '
                             'Variant name is auto-generated from config. '
                             'Config will be saved to output directory for reproducibility.')
    args = parser.parse_args()
    if args.feature_config is None:
        length = {'ubnormal': 12, 'shanghaitech': 16, 'msad': 24}[args.dataset]
        args.feature_config = f'seg{length}_std_nolog.yaml'

    # Load feature config (required)
    config_path = Path(args.feature_config)
    if not config_path.is_absolute():
        # Try relative to project root or configs directory
        config_path = PROJECT_ROOT / "configs" / "features" / args.feature_config
        if not config_path.exists():
            config_path = Path(args.feature_config)  # Try as absolute or relative to CWD

    if not config_path.exists():
        raise FileNotFoundError(f"Feature config not found: {config_path}")

    feature_config = load_feature_config(str(config_path))

    # Variant name is auto-generated from config parameters
    seg_len = feature_config.get('segment', {}).get('length', 12)
    norm_method = feature_config.get('normalization', {}).get('method', 'standard')
    apply_log = feature_config.get('extractor', {}).get('apply_log', False)

    norm_short = {
        'standard': 'std',
        'minmax': 'mm',
        'robust': 'robust',
        'none': 'none'
    }.get(norm_method, norm_method)
    log_suffix = 'log' if apply_log else 'nolog'
    feature_variant = f"seg{seg_len}_{norm_short}_{log_suffix}"

    logger.info(f"Loaded feature config from {config_path}")
    logger.info(f"  Auto-generated variant: {feature_variant}")
    logger.info(f"  Config name: {feature_config.get('name', 'N/A')}")
    logger.info(f"  Description: {feature_config.get('description', 'N/A')}")
    logger.info(f"  Segment length: {seg_len}, Normalization: {norm_method}, Log: {apply_log}")

    # Get dataset config
    dataset_config = get_dataset_config(args.dataset)
    vid_res = dataset_config.resolution

    # Normalize split names
    if args.split == 'train':
        args.split = 'training'
    elif args.split == 'test':
        args.split = 'testing'


    # Determine splits to process
    if args.split == 'all':
        splits = ['training', 'testing']
    else:
        splits = [args.split]

    print("=" * 70)
    print("Feature Extraction Pipeline (with Pose)")
    print("=" * 70)
    print(f"Dataset: {dataset_config.name}")
    print(f"Resolution: {vid_res[0]} × {vid_res[1]}")
    if feature_variant:
        print(f"Feature variant: {feature_variant}")
    print(f"Split(s): {', '.join(splits)}")
    if feature_config:
        segment_cfg = feature_config.get('segment', {})
        print(f"Segment length: {segment_cfg.get('length', SEG_LEN)}")
    else:
        print(f"Segment length: {SEG_LEN}")
    print(f"Train stride: {SEG_STRIDE_TRAIN}, Test stride: {SEG_STRIDE_TEST}")
    print("=" * 70)

    results = []

    versions = [(True, 'kf')]

    for apply_kf, version in versions:
        print(f"\n{'='*70}")
        print(f"Processing version: {version.upper()}")
        print(f"{'='*70}")

        if 'training' in splits:
            result = process_split(
                dataset_name=dataset_config.name,
                vid_res=vid_res,
                split='training',
                apply_kalman=apply_kf,
                feature_config=feature_config
            )
            results.append(result)

        if 'testing' in splits:
            result = process_split(
                dataset_name=dataset_config.name,
                vid_res=vid_res,
                split='testing',
                apply_kalman=apply_kf,
                feature_config=feature_config
            )
            results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print(f"{'Split':<12} {'Version':<8} {'Segments':<12} {'Shape':<20} {'Size (MB)':<10}")
    print("-" * 70)
    for r in results:
        if r['num_segments'] > 0:
            print(f"{r['split']:<12} {r.get('version', 'N/A'):<8} {r['num_segments']:<12} "
                  f"{str(r['shape']):<20} {r['file_size_mb']:<10.2f}")

    print("=" * 70)
    print("\n✓ Feature extraction complete!")

    output_base = DATA_ROOT / f"{dataset_config.name}_tracks" / f"{dataset_config.name}_pose_features" / f"variant_{feature_variant}"
    print(f"  Output base: {output_base}")


if __name__ == '__main__':
    main()
