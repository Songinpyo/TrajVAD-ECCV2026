#!/usr/bin/env python3
"""
Track Preprocessing Pipeline for Multi-class Trajectory VAD.

Handles gaps in tracked data through:
1. Linear interpolation for short gaps (<=10 frames)
2. Track splitting for long gaps (>10 frames)
3. Minimum length filtering (>=12 frames)

Usage:
    python preprocess_tracks.py --dataset ubnormal --split all
    python preprocess_tracks.py --dataset msad --split test
    
    python preprocess_tracks.py --dataset shanghaitech --split all
    
    python preprocess_tracks.py --dataset ubnormal --split all
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset_config import get_dataset_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# Constants
GAP_THRESHOLD = 10
MIN_TRACK_LENGTH = 12  # frames (sequence length for model)


def interpolate_gap(
    df_before: pd.DataFrame, 
    df_after: pd.DataFrame, 
    gap_frames: List[int]
) -> pd.DataFrame:
    """
    Linear interpolation for gap frames.
    
    Args:
        df_before: DataFrame with the last row before the gap
        df_after: DataFrame with the first row after the gap
        gap_frames: List of frame IDs to interpolate
        
    Returns:
        DataFrame with interpolated rows
    """
    if len(gap_frames) == 0:
        return pd.DataFrame()
    
    # Get start and end values
    start_row = df_before.iloc[-1]
    end_row = df_after.iloc[0]
    
    # Columns to interpolate
    interp_cols = ['x1', 'y1', 'x2', 'y2', 'conf']
    
    interpolated_rows = []
    n_gaps = len(gap_frames)
    
    for i, frame_id in enumerate(gap_frames):
        # Linear interpolation factor (0 to 1)
        t = (i + 1) / (n_gaps + 1)
        
        row = {
            'frame_id': frame_id,
            'track_id': int(start_row['track_id']),
            'source_track_id': int(start_row.get('source_track_id', start_row['track_id'])),
            'class_id': int(start_row['class_id']),  # Keep class from before gap
        }
        
        # Interpolate bbox coordinates and confidence
        for col in interp_cols:
            row[col] = start_row[col] + t * (end_row[col] - start_row[col])
        
        interpolated_rows.append(row)
    
    return pd.DataFrame(interpolated_rows)


def process_track(
    track_df: pd.DataFrame, 
    next_track_id: int,
    gap_threshold: int = GAP_THRESHOLD
) -> Tuple[List[pd.DataFrame], int, Dict]:
    """
    Process a single track: interpolate short gaps, split on long gaps.
    
    Args:
        track_df: DataFrame for a single track (sorted by frame_id)
        next_track_id: Next available track_id for splits
        gap_threshold: Maximum gap size to interpolate
        
    Returns:
        Tuple of (list of processed track DataFrames, next_track_id, stats dict)
    """
    track_df = track_df.sort_values('frame_id').reset_index(drop=True)
    if 'source_track_id' not in track_df:
        track_df['source_track_id'] = track_df['track_id']
    frames = track_df['frame_id'].values
    
    stats = {
        'interpolated_frames': 0,
        'splits': 0,
        'dropped_short': 0
    }
    
    if len(frames) < 2:
        if len(frames) < MIN_TRACK_LENGTH:
            stats['dropped_short'] = 1
            return [], next_track_id, stats
        return [track_df], next_track_id, stats
    
    # Find gaps
    frame_diffs = np.diff(frames)
    gap_indices = np.where(frame_diffs > 1)[0]
    
    if len(gap_indices) == 0:
        # No gaps - return as is if long enough
        if len(track_df) >= MIN_TRACK_LENGTH:
            return [track_df], next_track_id, stats
        else:
            stats['dropped_short'] = 1
            return [], next_track_id, stats
    
    # Process gaps
    processed_segments = []
    current_segment_start = 0
    all_rows = []
    
    for gap_idx in gap_indices:
        gap_size = frame_diffs[gap_idx] - 1  # Number of missing frames
        
        # Get segment before gap
        segment_before = track_df.iloc[current_segment_start:gap_idx + 1]
        
        if gap_size <= gap_threshold:
            # Case A: Short gap - interpolate
            gap_frames = list(range(int(frames[gap_idx]) + 1, int(frames[gap_idx + 1])))
            interpolated = interpolate_gap(
                segment_before.tail(1), 
                track_df.iloc[gap_idx + 1:gap_idx + 2],
                gap_frames
            )
            
            all_rows.append(segment_before)
            if len(interpolated) > 0:
                all_rows.append(interpolated)
                stats['interpolated_frames'] += len(gap_frames)
        else:
            # Case B: Long gap - split
            # Finalize current segment
            if len(all_rows) > 0:
                all_rows.append(segment_before)
                current_track = pd.concat(all_rows, ignore_index=True)
                if len(current_track) >= MIN_TRACK_LENGTH:
                    processed_segments.append(current_track)
                else:
                    stats['dropped_short'] += 1
            elif len(segment_before) >= MIN_TRACK_LENGTH:
                processed_segments.append(segment_before.copy())
            else:
                stats['dropped_short'] += 1
            
            # Start new segment with new track_id
            all_rows = []
            stats['splits'] += 1
        
        current_segment_start = gap_idx + 1
    
    # Add final segment
    final_segment = track_df.iloc[current_segment_start:]
    all_rows.append(final_segment)
    
    if len(all_rows) > 0:
        current_track = pd.concat(all_rows, ignore_index=True)
        if len(current_track) >= MIN_TRACK_LENGTH:
            processed_segments.append(current_track)
        else:
            stats['dropped_short'] += 1
    
    # Assign new track_ids to split segments
    result_dfs = []
    for i, segment in enumerate(processed_segments):
        segment = segment.copy()
        if i == 0:
            # Keep original track_id for first segment
            pass
        else:
            # Assign new track_id for subsequent segments
            segment['track_id'] = next_track_id
            next_track_id += 1
        result_dfs.append(segment)
    
    return result_dfs, next_track_id, stats


def process_video(parquet_path: Path, gap_threshold: int = GAP_THRESHOLD) -> Tuple[pd.DataFrame, Dict]:
    """
    Process all tracks in a video.
    
    Args:
        parquet_path: Path to input parquet file
        gap_threshold: Maximum gap size to interpolate
        
    Returns:
        Tuple of (processed DataFrame, statistics dict)
    """
    df = pd.read_parquet(parquet_path)
    
    video_stats = {
        'original_tracks': df['track_id'].nunique(),
        'original_detections': len(df),
        'interpolated_frames': 0,
        'splits': 0,
        'dropped_short': 0,
        'final_tracks': 0,
        'final_detections': 0
    }
    
    # Find max track_id for new assignments
    next_track_id = df['track_id'].max() + 1
    
    # Process each track
    processed_tracks = []
    
    for track_id in df['track_id'].unique():
        track_df = df[df['track_id'] == track_id]
        
        track_results, next_track_id, track_stats = process_track(
            track_df, next_track_id, gap_threshold=gap_threshold
        )
        
        processed_tracks.extend(track_results)
        video_stats['interpolated_frames'] += track_stats['interpolated_frames']
        video_stats['splits'] += track_stats['splits']
        video_stats['dropped_short'] += track_stats['dropped_short']
    
    if processed_tracks:
        result_df = pd.concat(processed_tracks, ignore_index=True)
        # Ensure correct column order and types
        result_df = result_df[['frame_id', 'track_id', 'source_track_id', 'x1', 'y1', 'x2', 'y2', 'conf', 'class_id']]
        result_df['frame_id'] = result_df['frame_id'].astype(int)
        result_df['track_id'] = result_df['track_id'].astype(int)
        result_df['class_id'] = result_df['class_id'].astype(int)
        
        video_stats['final_tracks'] = result_df['track_id'].nunique()
        video_stats['final_detections'] = len(result_df)
    else:
        result_df = pd.DataFrame(columns=['frame_id', 'track_id', 'source_track_id', 'x1', 'y1', 'x2', 'y2', 'conf', 'class_id'])
    
    return result_df, video_stats


def process_split(dataset_name: str, split: str, gap_threshold: int = GAP_THRESHOLD):
    """Process all videos in a dataset split.
    
    Args:
        dataset_name: Dataset name (e.g., 'ubnormal', 'shanghaitech')
        split: Split name ('training' or 'testing')
        gap_threshold: Maximum gap size to interpolate
    """
    input_dir = DATA_ROOT / f"{dataset_name}_tracks" / f"{dataset_name}_tracks" / split
    output_dir = DATA_ROOT / f"{dataset_name}_tracks" / f"{dataset_name}_tracks_processed" / split
    
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        logger.warning(f"Input directory does not exist: {input_dir}")
        return defaultdict(int)

    parquet_files = sorted(input_dir.glob('*.parquet'))

    if len(parquet_files) == 0:
        logger.warning(f"No parquet files found in {input_dir}")
        return defaultdict(int)

    logger.info(f"Processing {split} set: {len(parquet_files)} videos")

    total_stats = defaultdict(int)

    for pq_file in tqdm(parquet_files, desc=f"[{split}]"):
        result_df, video_stats = process_video(pq_file, gap_threshold=gap_threshold)

        # Save processed parquet
        output_path = output_dir / pq_file.name
        result_df.to_parquet(output_path, index=False, engine='pyarrow')

        # Accumulate stats
        for key, value in video_stats.items():
            total_stats[key] += value

    return dict(total_stats)


def print_summary(stats: Dict, split: str):
    """Print processing summary."""
    print(f"\n{'='*60}")
    print(f"[{split.upper()}] Processing Summary")
    print(f"{'='*60}")
    print(f"Original tracks:      {stats['original_tracks']:,}")
    print(f"Original detections:  {stats['original_detections']:,}")
    print(f"---")
    print(f"Interpolated frames:  {stats['interpolated_frames']:,}")
    print(f"Track splits:         {stats['splits']:,}")
    print(f"Dropped (too short):  {stats['dropped_short']:,}")
    print(f"---")
    print(f"Final tracks:         {stats['final_tracks']:,}")
    print(f"Final detections:     {stats['final_detections']:,}")
    print(f"Track change:         {stats['final_tracks'] - stats['original_tracks']:+,}")
    print(f"Detection change:     {stats['final_detections'] - stats['original_detections']:+,}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess tracks with gap handling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python preprocess_tracks.py --dataset ubnormal --split all
  python preprocess_tracks.py --dataset msad --split test
  
  python preprocess_tracks.py --dataset shanghaitech --split all
  
  python preprocess_tracks.py --dataset ubnormal --split all
        """
    )
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['shanghaitech', 'ubnormal', 'msad'],
                        help='Dataset name')
    parser.add_argument('--split', type=str, default='all',
                        choices=['train', 'test', 'training', 'testing', 'all'],
                        help='Dataset split to process')
    parser.add_argument('--max_gap', type=int, default=GAP_THRESHOLD,
                        help=f'Maximum gap size for interpolation (default: {GAP_THRESHOLD})')
    args = parser.parse_args()

    # Get dataset config
    dataset_config = get_dataset_config(args.dataset)

    # Normalize split names
    if args.split == 'test':
        args.split = 'testing'
    elif args.split == 'train':
        args.split = 'training'


    # Determine splits to process
    if args.split == 'all':
        splits = ['testing', 'training']
    else:
        splits = [args.split]

    print("=" * 60)
    print("Track Preprocessing Pipeline")
    print("=" * 60)
    print(f"Dataset:           {dataset_config.name}")
    print(f"Split(s):           {', '.join(splits)}")
    print(f"Gap threshold:     {args.max_gap} frames")
    print(f"Min track length:  {MIN_TRACK_LENGTH} frames")
    print("=" * 60)

    for split in splits:
        stats = process_split(dataset_config.name, split, gap_threshold=args.max_gap)
        if stats.get('original_tracks', 0) > 0:  # Only print if files were processed
            print_summary(stats, split)

    print("\n✓ Preprocessing complete!")
    output_base = DATA_ROOT / f"{dataset_config.name}_tracks" / f"{dataset_config.name}_tracks_processed"
    print(f"  Output base: {output_base}")


if __name__ == '__main__':
    main()

