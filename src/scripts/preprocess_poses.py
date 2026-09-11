"""
Pose Gap Handling and Interpolation

Apply the same gap handling strategy as bbox tracks:
- Gaps ≤ 10 frames: Linear interpolation
- Gaps > 10 frames: Track split
- Tracks < 12 frames: Remove

Usage:
    conda run -n posevad python src/scripts/preprocess_poses.py --dataset shanghaitech --split all
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset_config import get_dataset_config
from utils import setup_logger


MAX_GAP_FOR_INTERPOLATION = 10
MIN_TRACK_LENGTH = 12


def interpolate_keypoints(kp1: np.ndarray, kp2: np.ndarray, num_frames: int) -> np.ndarray:
    """
    Linear interpolation between two keypoint arrays.

    Args:
        kp1: (17, 3) keypoints at start
        kp2: (17, 3) keypoints at end
        num_frames: Number of intermediate frames to generate

    Returns:
        interpolated: (num_frames, 17, 3) interpolated keypoints
    """
    interpolated = []

    for i in range(1, num_frames + 1):
        alpha = i / (num_frames + 1)
        # Interpolate x, y
        kp_interp = kp1.copy()
        kp_interp[:, :2] = (1 - alpha) * kp1[:, :2] + alpha * kp2[:, :2]
        # Average confidence
        kp_interp[:, 2] = (kp1[:, 2] + kp2[:, 2]) / 2
        interpolated.append(kp_interp)

    return np.array(interpolated)


def process_track_gaps(
    track_df: pd.DataFrame,
    max_gap: int = MAX_GAP_FOR_INTERPOLATION,
    min_length: int = MIN_TRACK_LENGTH
) -> pd.DataFrame:
    """
    Process gaps in a single track.

    Args:
        track_df: DataFrame for single track (sorted by frame_id)
        max_gap: Maximum gap size for interpolation

    Returns:
        processed_df: DataFrame with gaps filled or track split
    """
    track_id = track_df['track_id'].iloc[0]
    class_id = track_df['class_id'].iloc[0] if 'class_id' in track_df.columns else 0
    frames = track_df['frame_id'].values
    keypoints_list = track_df['keypoints'].values
    
    if 'pose_score' in track_df.columns:
        pose_scores = track_df['pose_score'].values
    else:
        # Calculate from keypoint confidences
        pose_scores = np.array([np.mean([kp[2] for kp in kps]) if kps is not None else 0.0 
                                for kps in keypoints_list])
    
    if 'has_pose' in track_df.columns:
        has_poses = track_df['has_pose'].values
    else:
        has_poses = np.array([kps is not None for kps in keypoints_list])
    
    if 'bbox' in track_df.columns:
        bboxes = track_df['bbox'].values
    else:
        # Derive bbox from keypoints (x_min, y_min, x_max, y_max)
        bboxes = []
        for kps in keypoints_list:
            if kps is not None:
                kp_arr = np.stack(kps) if isinstance(kps[0], (list, np.ndarray)) else np.array(kps).reshape(-1, 3)
                x_coords = kp_arr[:, 0]
                y_coords = kp_arr[:, 1]
                bbox = [x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max()]
            else:
                bbox = [0, 0, 0, 0]
            bboxes.append(bbox)
        bboxes = np.array(bboxes)

    # Find gaps
    gaps = np.diff(frames) - 1
    gap_indices = np.where(gaps > 0)[0]

    if len(gap_indices) == 0:
        # No gaps
        return track_df

    # Process each gap
    processed_data = []
    current_segment = []

    for i in range(len(track_df)):
        current_segment.append(track_df.iloc[i].to_dict())

        # Check if there's a gap after this frame
        if i in gap_indices:
            gap_size = gaps[i]

            if gap_size <= max_gap:
                # Interpolate
                if has_poses[i] and has_poses[i+1]:
                    # Both endpoints have valid poses
                    # kp_raw is 1D object array (17,) containing 17 arrays of shape (3,)
                    # np.stack converts it to (17, 3)
                    kp1 = np.stack(keypoints_list[i]).astype(np.float32)
                    kp2 = np.stack(keypoints_list[i+1]).astype(np.float32)
                    bbox1 = np.array(bboxes[i])
                    bbox2 = np.array(bboxes[i+1])

                    # Interpolate keypoints
                    interp_kps = interpolate_keypoints(kp1, kp2, int(gap_size))

                    # Interpolate bboxes
                    for j in range(int(gap_size)):
                        alpha = (j + 1) / (gap_size + 1)
                        interp_bbox = (1 - alpha) * bbox1 + alpha * bbox2
                        interp_frame = frames[i] + j + 1

                        # Average pose score
                        interp_score = (pose_scores[i] + pose_scores[i+1]) / 2

                        current_segment.append({
                            'frame_id': int(interp_frame),
                            'track_id': track_id,
                            'keypoints': interp_kps[j].tolist(),  # Convert to list for Parquet compatibility
                            'pose_score': interp_score,
                            'has_pose': True,
                            'bbox': interp_bbox,
                            'class_id': class_id
                        })
                else:
                    # At least one endpoint has no pose, add None frames
                    for j in range(int(gap_size)):
                        interp_frame = frames[i] + j + 1
                        alpha = (j + 1) / (gap_size + 1)
                        interp_bbox = (1 - alpha) * np.array(bboxes[i]) + alpha * np.array(bboxes[i+1])

                        current_segment.append({
                            'frame_id': int(interp_frame),
                            'track_id': track_id,
                            'keypoints': None,
                            'pose_score': 0.0,
                            'has_pose': False,
                            'bbox': interp_bbox,
                            'class_id': class_id
                        })
            else:
                # Gap too large, split track
                if len(current_segment) >= min_length:
                    processed_data.extend(current_segment)

                # Start new segment
                current_segment = []

    # Add remaining segment
    if len(current_segment) >= min_length:
        processed_data.extend(current_segment)

    if len(processed_data) == 0:
        return pd.DataFrame()

    return pd.DataFrame(processed_data)


def process_video(
    video_name: str,
    pose_parquet: Path,
    logger,
    allowed_classes: list = None,
    max_gap: int = MAX_GAP_FOR_INTERPOLATION,
    min_length: int = MIN_TRACK_LENGTH
) -> pd.DataFrame:
    """
    Process poses for a single video.

    Args:
        video_name: Video name
        pose_parquet: Path to pose parquet file
        logger: Logger instance
        allowed_classes: List of allowed class IDs (optional)

    Returns:
        processed_df: DataFrame with gaps processed
    """
    logger.info(f"Processing video: {video_name}")

    # Load pose data
    pose_df = pd.read_parquet(pose_parquet)
    logger.info(f"  Loaded {len(pose_df)} pose detections")

    # Filter by class if specified
    if allowed_classes is not None:
        if 'class_id' in pose_df.columns:
            original_count = len(pose_df)
            pose_df = pose_df[pose_df['class_id'].isin(allowed_classes)]
            filtered_count = len(pose_df)
            logger.info(f"  Filtered by class {allowed_classes}: {original_count} -> {filtered_count} detections")
        else:
            logger.warning("  --classes specified but 'class_id' column not found in data. Skipping filtering.")

    # Get unique tracks
    track_ids = sorted(pose_df['track_id'].unique())
    logger.info(f"  Found {len(track_ids)} unique tracks")

    # Process each track
    processed_tracks = []
    removed_tracks = 0

    for track_id in track_ids:
        track_df = pose_df[pose_df['track_id'] == track_id].sort_values('frame_id').reset_index(drop=True)

        # Process gaps
        processed_track = process_track_gaps(track_df, max_gap, min_length)

        if len(processed_track) > 0:
            processed_tracks.append(processed_track)
        else:
            removed_tracks += 1

    if len(processed_tracks) == 0:
        logger.warning(f"  No valid tracks after processing!")
        return pd.DataFrame()

    # Concatenate all tracks
    result_df = pd.concat(processed_tracks, ignore_index=True)

    logger.info(f"  Processed {len(processed_tracks)} tracks ({removed_tracks} removed)")
    logger.info(f"  Final detections: {len(result_df)}")
    logger.info(f"  Pose success rate: {result_df['has_pose'].sum() / len(result_df) * 100:.1f}%")

    return result_df


def main():
    parser = argparse.ArgumentParser(description='Process pose gaps with interpolation')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['shanghaitech', 'ubnormal', 'msad'],
                        help='Dataset name')
    parser.add_argument('--split', type=str, default='all',
                        choices=['training', 'testing', 'all'],
                        help='Dataset split to process')

    # Processing parameters
    parser.add_argument('--max_gap', type=int, default=MAX_GAP_FOR_INTERPOLATION,
                        help='Maximum gap size for interpolation')
    parser.add_argument('--min_length', type=int, default=MIN_TRACK_LENGTH,
                        help='Minimum track length to keep')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: auto-generated)')
    parser.add_argument('--sample', type=int, default=None,
                        help='Process only N videos for testing')
    parser.add_argument('--classes', type=int, nargs='+', default=[0],
                        help='List of class IDs to process (e.g., --classes 0 1). If not specified, all classes are processed.')

    args = parser.parse_args()

    # Get project root
    project_root = Path(__file__).parent.parent.parent

    # Get dataset config
    dataset_config = get_dataset_config(args.dataset)

    # Determine splits to process
    if args.split == 'all':
        splits = ['training', 'testing']
    else:
        splits = [args.split]

    # Process each split
    for split in splits:
        print("\n" + "=" * 80)
        print(f"Processing {args.dataset} - {split.upper()} split")
        print("=" * 80)

        # Input directory (raw poses)
        input_dir = project_root / 'data' / f'{args.dataset}_tracks' / f'{args.dataset}_tracks_pose' / split

        # Output directory (processed poses)
        if args.output_dir:
            output_dir = Path(args.output_dir) / split
        else:
            output_dir = project_root / 'data' / f'{args.dataset}_tracks' / f'{args.dataset}_tracks_pose_processed' / split

        output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logger
        log_file = output_dir / f'pose_preprocessing_{split}.log'
        logger = setup_logger(str(log_file))

        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Split: {split}")
        logger.info(f"Input: {input_dir}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Max gap: {args.max_gap}")
        logger.info(f"Min length: {args.min_length}")

        # Check input directory
        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            continue

        # Get all pose files
        pose_files = sorted(input_dir.glob('*.parquet'))
        if args.sample:
            pose_files = pose_files[:args.sample]

        logger.info(f"Found {len(pose_files)} videos to process")

        if len(pose_files) == 0:
            logger.warning("No pose files found!")
            continue

        # Process each video
        success_count = 0
        failed_videos = []

        for pose_file in tqdm(pose_files, desc=f"Processing {split}"):
            video_name = pose_file.stem

            try:
                # Process video
                processed_df = process_video(
                    video_name=video_name,
                    pose_parquet=pose_file,
                    logger=logger,
                    allowed_classes=args.classes,
                    max_gap=args.max_gap, min_length=args.min_length
                )

                if len(processed_df) == 0:
                    logger.warning(f"  No valid data for {video_name}")
                    failed_videos.append(video_name)
                    continue

                # Save results
                output_file = output_dir / f'{video_name}.parquet'
                
                # Normalize keypoints column to list format for Parquet compatibility
                def normalize_keypoints(kp):
                    if kp is None:
                        return None
                    if hasattr(kp, 'tolist'):
                        return kp.tolist()
                    if isinstance(kp, np.ndarray) and kp.dtype == object:
                        return [k.tolist() if hasattr(k, 'tolist') else list(k) for k in kp]
                    return kp
                
                processed_df['keypoints'] = processed_df['keypoints'].apply(normalize_keypoints)
                
                processed_df.to_parquet(output_file, index=False)
                logger.info(f"  Saved to: {output_file}")

                success_count += 1

            except Exception as e:
                logger.error(f"Failed to process {video_name}: {e}", exc_info=True)
                failed_videos.append(video_name)

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Successfully processed: {success_count} / {len(pose_files)} videos")

        if failed_videos:
            logger.warning(f"Failed videos ({len(failed_videos)}):")
            for video_name in failed_videos:
                logger.warning(f"  - {video_name}")

        logger.info(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
