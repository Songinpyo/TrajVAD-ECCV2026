"""Ground-truth loaders: 0 is normal, 1 is anomalous."""

from pathlib import Path

from typing import Dict, Optional, List

import numpy as np

def load_shanghaitech_gt_files(
    gt_dir: str,
    flip: bool = False,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Load ShanghaiTech ground truth frame masks from directory.
    
    Args:
        gt_dir: Directory containing .npy GT files
        flip: If True, flip GT (1-gt). Use for STG-NF convention (1=normal)
        verbose: Print loading info
    
    Returns:
        Dictionary mapping video_name (e.g., "01_0014") to frame-level GT array
        
    GT Convention:
        - Original: 0=normal, 1=abnormal
        - After flip: 1=normal, 0=abnormal (STG-NF convention)
    """
    gt_path = Path(gt_dir)
    
    if not gt_path.exists():
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")
    
    gt_dict = {}
    
    for npy_file in sorted(gt_path.glob("*.npy")):
        # Handle different naming conventions
        video_name = npy_file.stem
        
        # Some files may have 'label' prefix, normalize to "XX_XXXX" format
        if 'label' in video_name.lower():
            video_name = video_name.replace('label', '01')
        
        gt_array = np.load(npy_file)
        
        if flip:
            gt_array = 1 - gt_array
        
        gt_dict[video_name] = gt_array
    
    if verbose:
        print(f"✓ Loaded {len(gt_dict)} GT files from {gt_dir}")
        if flip:
            print(f"  GT convention: 1=normal, 0=abnormal (flipped)")
        else:
            print(f"  GT convention: 0=normal, 1=abnormal (original)")
    
    return gt_dict

def get_gt_statistics(gt_dict: Dict[str, np.ndarray]) -> dict:
    """
    Compute statistics from GT dictionary.

    Returns:
        Dictionary with total_frames, normal_frames, abnormal_frames, etc.
    """
    total_frames = 0
    normal_frames = 0
    abnormal_frames = 0

    for video_name, gt in gt_dict.items():
        total_frames += len(gt)
        normal_frames += (gt == 0).sum()
        abnormal_frames += (gt == 1).sum()

    return {
        'num_videos': len(gt_dict),
        'total_frames': total_frames,
        'normal_frames': int(normal_frames),
        'abnormal_frames': int(abnormal_frames),
        'abnormal_ratio': abnormal_frames / total_frames if total_frames > 0 else 0
    }

def load_ubnormal_gt_txt(
    gt_root: str,
    split: str = 'testing',
    flip: bool = False,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Load UBnormal GT from ubnormal_frame_gt/{split}/{video_name}/ground_truth_frame_level.txt

    Args:
        gt_root: Path to ubnormal_frame_gt directory
        split: 'training', 'testing', or 'validation'
        flip: If True, flip GT (1-gt) to normal-positive labels.
        verbose: Print loading info

    Returns:
        Dictionary mapping video_name to frame-level GT array
        
    GT Convention (UBnormal):
        - Original: 0=normal, 1=abnormal.
        - After flip: 1=normal, 0=abnormal.
    """
    gt_root_path = Path(gt_root)
    split_dir = gt_root_path / split

    if not split_dir.exists():
        raise FileNotFoundError(f"UBnormal GT split not found: {split_dir}")

    gt_dict = {}

    for video_dir in sorted(split_dir.iterdir()):
        if not video_dir.is_dir():
            continue

        video_name = video_dir.name
        gt_file = video_dir / 'ground_truth_frame_level.txt'

        if not gt_file.exists():
            continue

        # Load GT (one line per frame)
        with open(gt_file, 'r') as f:
            gt = np.array([int(line.strip()) for line in f.readlines()], dtype=np.float32)

        if flip:
            gt = 1 - gt

        gt_dict[video_name] = gt

    if verbose:
        print(f"✓ Loaded {len(gt_dict)} UBnormal GT files ({split})")
        if flip:
            print(f"  GT convention: 1=normal, 0=abnormal (flipped)")
        else:
            print(f"  GT convention: 0=normal, 1=abnormal (original)")
        stats = get_gt_statistics(gt_dict)
        print(f"  Total frames: {stats['total_frames']}, Abnormal: {stats['abnormal_frames']} ({stats['abnormal_ratio']:.2%})")

    return gt_dict

def load_msad_gt_csv(
    csv_file: str,
    video_list: Optional[List[str]] = None,
    flip: bool = False,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Load MSAD GT from anomaly_annotation.csv.
    
    CSV format:
        name,scenario,total frames,starting frame of anomaly,ending frame of anomaly
        Assault_1,shop,426,55,426
        ...
    
    Args:
        csv_file: Path to anomaly_annotation.csv
        video_list: Optional list of video names to include (if None, uses all videos from CSV + frame directories)
        flip: If True, flip GT (1-gt). Use for STG-NF convention (1=normal)
        verbose: Print loading info
    
    Returns:
        Dictionary mapping video_name (e.g., "Assault_1") to frame-level GT array
        
    GT Convention:
        - Original: 0=normal, 1=abnormal
        - After flip: 1=normal, 0=abnormal (STG-NF convention)
        
    Note:
        - starting/ending frame in CSV are 1-indexed (inclusive)
        - Normal videos (not in CSV) will have all zeros
    """
    import csv
    from pathlib import Path
    
    csv_path = Path(csv_file)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"MSAD GT CSV file not found: {csv_file}")
    
    # Read CSV and build abnormal frame ranges
    abnormal_ranges = {}  # video_name -> (start_frame, end_frame, total_frames)
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_name = row['name'].strip()
            total_frames = int(row['total frames'])
            start_frame = int(row['starting frame of anomaly'])  # 1-indexed
            end_frame = int(row['ending frame of anomaly'])      # 1-indexed
            
            # Convert to 0-indexed (inclusive)
            start_frame_0idx = start_frame - 1
            end_frame_0idx = end_frame  # end_frame is inclusive in CSV
            
            abnormal_ranges[video_name] = (start_frame_0idx, end_frame_0idx, total_frames)
    
    # Get video list if not provided
    if video_list is None:
        # Use videos from CSV + try to find normal videos from frame directories
        video_list = list(abnormal_ranges.keys())
        # Note: Normal videos will be added below if frame directories exist
    
    gt_dict = {}
    
    # Process each video
    for video_name in video_list:
        if video_name in abnormal_ranges:
            # Abnormal video: create GT array with abnormal frames marked
            start_frame_0idx, end_frame_0idx, total_frames = abnormal_ranges[video_name]
            
            gt = np.zeros(total_frames, dtype=np.float32)
            # Mark abnormal frames (0-indexed, end_frame_0idx is exclusive in numpy)
            gt[start_frame_0idx:end_frame_0idx] = 1.0
            
            if flip:
                gt = 1 - gt
            
            gt_dict[video_name] = gt
        else:
            # Normal video: all zeros (or try to get frame count from file system)
            # Try to get frame count from frame directory
            csv_dir = csv_path.parent
            frame_root_testing = csv_dir.parent / 'msad_frames' / 'testing'
            frame_dir = frame_root_testing / video_name
            
            if frame_dir.exists():
                frame_files = list(frame_dir.glob('*.jpg'))
                if len(frame_files) > 0:
                    total_frames = len(frame_files)
                    gt = np.zeros(total_frames, dtype=np.float32)
                    
                    if flip:
                        gt = 1 - gt
                    
                    gt_dict[video_name] = gt
                else:
                    if verbose:
                        print(f"[WARN] Video {video_name} not in CSV and no frames found, skipping")
            else:
                if verbose:
                    print(f"[WARN] Video {video_name} not in CSV and frame directory not found, skipping")
    
    if verbose:
        print(f"✓ Loaded {len(gt_dict)} MSAD GT entries from {csv_file}")
        if flip:
            print(f"  GT convention: 1=normal, 0=abnormal (flipped)")
        else:
            print(f"  GT convention: 0=normal, 1=abnormal (original)")
        stats = get_gt_statistics(gt_dict)
        print(f"  Total frames: {stats['total_frames']}, Abnormal: {stats['abnormal_frames']} ({stats['abnormal_ratio']:.2%})")

    return gt_dict

def load_gt_for_dataset(dataset, dataset_config, project_root, gt_path_override=None):
    from .path_utils import get_gt_path
    path = gt_path_override or get_gt_path(dataset, dataset_config, project_root)
    if dataset == 'shanghaitech':
        return load_shanghaitech_gt_files(path, flip=False)
    if dataset == 'ubnormal':
        return load_ubnormal_gt_txt(path, split='testing', flip=False)
    if dataset == 'msad':
        return load_msad_gt_csv(path, video_list=dataset_config.get_video_list('testing'), flip=False)
    raise ValueError(f'Unknown dataset: {dataset}')
