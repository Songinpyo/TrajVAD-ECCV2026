"""Load standardized trajectory features and pelvis-centered COCO poses."""
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from utils.bbox_features import FEATURE_NAMES


def robust_normalize_pose(x_pose):
    shape = x_pose.shape
    pose = x_pose.reshape(shape[0], shape[1], 17, 3).float().clone()
    valid_frames = (pose[..., 2] > 0).any(dim=-1)
    mask = valid_frames.any(dim=1)
    # COCO left/right hips (11, 12) define the pelvis at each frame.
    pelvis = pose[:, :, [11, 12], :2].mean(dim=2, keepdim=True)
    coords = pose[..., :2] - pelvis
    # Retain a single body-scale factor per window for both coordinate axes.
    scale = coords[..., 1].std(dim=(1, 2), keepdim=True).unsqueeze(-1).clamp_min(1e-6)
    pose[..., :2] = coords / scale
    pose[~valid_frames] = 0
    return pose.reshape(shape), mask


class MultiClassBBoxDataset(Dataset):
    def __init__(self, data_root, dataset_name, split='training', mode='kf', load_pose=False):
        if mode != 'kf':
            raise ValueError('The paper configuration uses Kalman-filtered features (kf).')
        split = {'train': 'training', 'test': 'testing'}.get(split, split)
        path = Path(data_root) / split / f'{dataset_name}_{split}_kf.pt'
        obj = torch.load(path, map_location='cpu', weights_only=False)
        names = obj.get('feature_names', FEATURE_NAMES)
        self.x_cont = obj['x_cont'].float()[..., [names.index(n) for n in FEATURE_NAMES]]
        self.x_cat = obj['x_cat'].long()
        self.metadata = obj['metadata']
        self.feature_names = FEATURE_NAMES
        self.cont_dim = 27
        self.load_pose = load_pose
        if load_pose:
            pose, self.pose_mask = robust_normalize_pose(obj['x_pose'])
            self.x_pose = pose.flatten(2)

    def __len__(self):
        return len(self.x_cont)

    def __getitem__(self, index):
        base = self.x_cont[index], self.x_cat[index]
        if self.load_pose:
            return (*base, self.x_pose[index], self.pose_mask[index])
        return base

    def get_metadata(self, index):
        return self.metadata[index]


def get_multiclass_loader(data_root, dataset_name, split='training', mode='kf',
                          batch_size=256, num_workers=4, load_pose=False):
    dataset = MultiClassBBoxDataset(data_root, dataset_name, split, mode, load_pose)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=split in ('train', 'training'),
                        num_workers=num_workers, pin_memory=True)
    return loader, dataset
