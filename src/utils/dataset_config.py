"""Dataset paths for ShanghaiTech, UBnormal, and MSAD."""

from dataclasses import dataclass

from pathlib import Path

from typing import Tuple, Optional, List

import os

@dataclass
class DatasetConfig:
    """Dataset configuration dataclass"""
    name: str
    resolution: Tuple[int, int]  # (width, height)
    frame_root: str
    gt_path: str
    frame_ext: str  # 'jpg', 'tif', etc.

    def get_frame_path(self, video_name: str, frame_idx: int, split: str = 'testing') -> str:
        """
        Get frame file path for given video and frame index.

        Args:
            video_name: Video identifier (e.g., '01_0014', 'Test001', 'abnormal_scene_1_scenario_1')
            frame_idx: Frame index (0-indexed)
            split: 'training' or 'testing'

        Returns:
            Absolute path to frame file
        """
        raise NotImplementedError("Subclass must implement get_frame_path()")

    def get_video_list(self, split: str = 'testing') -> List[str]:
        """
        Get list of video names for given split.

        Args:
            split: 'training' or 'testing'

        Returns:
            List of video names
        """
        raise NotImplementedError("Subclass must implement get_video_list()")

class ShanghaiTechConfig(DatasetConfig):
    """ShanghaiTech dataset configuration"""

    def __init__(self, data_root: str = 'data/shanghaitech'):
        super().__init__(
            name='shanghaitech',
            resolution=(856, 480),
            frame_root=os.path.join(data_root, '{split}/frames'),
            gt_path=os.path.join(data_root, 'testing/test_frame_mask'),
            frame_ext='jpg'
        )

    def get_frame_path(self, video_name: str, frame_idx: int, split: str = 'testing') -> str:
        frame_root = self.frame_root.format(split=split)
        if split == 'training':
            return os.path.join(frame_root, video_name, f"{frame_idx:05d}.jpg")
        else:  # testing
            return os.path.join(frame_root, video_name, f"{frame_idx:03d}.jpg")

    def get_video_list(self, split: str = 'testing') -> List[str]:
        frame_root = self.frame_root.format(split=split)
        if not os.path.exists(frame_root):
            return []
        return sorted([d for d in os.listdir(frame_root) if os.path.isdir(os.path.join(frame_root, d))])

class UBnormalConfig(DatasetConfig):
    """UBnormal dataset configuration"""

    def __init__(self, data_root: str = 'data/ubnormal_frames'):
        super().__init__(
            name='ubnormal',
            resolution=(1080, 720),
            frame_root=os.path.join(data_root, '{split}'),
            gt_path='data/ubnormal/ubnormal_frame_gt',
            frame_ext='jpg'
        )
        self.scripts_dir = 'data/ubnormal/scripts'

    def get_frame_path(self, video_name: str, frame_idx: int, split: str = 'testing') -> str:
        frame_root = self.frame_root.format(split=split)
        return os.path.join(frame_root, video_name, f"{frame_idx:06d}.jpg")

    def get_video_list(self, split: str = 'testing') -> List[str]:
        """Get video list from scripts directory"""
        if split == 'training':
            # Only normal videos for training
            txt_file = os.path.join(self.scripts_dir, 'normal_training_video_names.txt')
        else:  # testing
            # Both normal and abnormal
            normal_txt = os.path.join(self.scripts_dir, 'normal_test_video_names.txt')
            abnormal_txt = os.path.join(self.scripts_dir, 'abnormal_test_video_names.txt')

            videos = []
            for txt_file in [normal_txt, abnormal_txt]:
                if os.path.exists(txt_file):
                    with open(txt_file, 'r') as f:
                        videos.extend([line.strip() for line in f])
            return sorted(videos)

        if os.path.exists(txt_file):
            with open(txt_file, 'r') as f:
                return [line.strip() for line in f]
        return []

class MSADConfig(DatasetConfig):
    """MSAD dataset configuration.

    Frames are pre-extracted to 1280x720 JPG:
    - Training:  data/msad_frames/training/{video_name}/{frame_id:06d}.jpg  (normal only)
    - Testing:   data/msad_frames/testing/{video_name}/{frame_id:06d}.jpg   (abnormal + normal)
    """

    def __init__(self, data_root: str = 'data/msad_frames'):
        super().__init__(
            name='msad',
            resolution=(1280, 720),
            frame_root=os.path.join(data_root, '{split}'),
            gt_path=os.path.join('data/msad/anomaly_annotation.csv'),
            frame_ext='jpg'
        )
        self.data_root = data_root

    def get_frame_path(self, video_name: str, frame_idx: int, split: str = 'testing') -> str:
        frame_root = self.frame_root.format(split=split)
        return os.path.join(frame_root, video_name, f"{frame_idx:06d}.jpg")

    def get_video_list(self, split: str = 'testing') -> List[str]:
        """Get video list from pre-extracted frame directories."""
        frame_root = self.frame_root.format(split=split)
        if not os.path.exists(frame_root):
            return []

        video_names: List[str] = []
        for video_dir in sorted(Path(frame_root).iterdir()):
            if video_dir.is_dir():
                video_names.append(video_dir.name)

        return sorted(video_names)

def get_dataset_config(dataset_name, data_root=None):
    classes = {'shanghaitech': ShanghaiTechConfig, 'ubnormal': UBnormalConfig, 'msad': MSADConfig}
    cls = classes[dataset_name]
    return cls(data_root) if data_root is not None else cls()
