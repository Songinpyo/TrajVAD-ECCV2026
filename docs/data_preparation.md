# Data Preparation

Place preprocessed feature tensors and ground-truth files under `data/`.

Expected feature layout:

```text
data/{dataset}_tracks/{dataset}_pose_features/variant_{feature_variant}/
├── training/{dataset}_training_kf.pt
├── testing/{dataset}_testing_kf.pt
├── scaler_kf.pkl
└── ...
```

Ground-truth defaults used by the public code:

```text
data/shanghaitech/testing/test_frame_mask/
data/ubnormal/ubnormal_frame_gt/
data/msad/anomaly_annotation.csv
```

Use `--data_root` or `--gt_dir` to point evaluation at a different location.

The paper uses YOLOX, OSNet, ByteTrack, and AlphaPose. Tracking and pose data will be released soon.

Intermediate parquet files use these directories (one file per video):

```text
data/{dataset}_tracks/{dataset}_tracks/{split}/
data/{dataset}_tracks/{dataset}_tracks_processed/{split}/
data/{dataset}_tracks/{dataset}_tracks_pose/{split}/
data/{dataset}_tracks/{dataset}_tracks_pose_processed/{split}/
```

`split` is `training` or `testing`. `preprocess_tracks.py` and `preprocess_poses.py`
process track and pose gaps. `preprocess_pose_data.py` generates combined trajectory
and pose features in the layout above; `preprocess_data.py` writes trajectory-only
features under `{dataset}_tracks_features/variant_{feature_variant}/` (use
`--data_root` to load that directory).

The paper configurations use all 27 trajectory features and `kf` data, with segment
lengths 12 for UBnormal, 16 for ShanghaiTech, and 24 for MSAD.

Before feature extraction, interpolate gaps of up to 10 missing frames and split
tracks at longer gaps. Bounding boxes are Kalman-filtered, and trajectory features
are standardized using training statistics. Pose coordinates are centered on the
COCO hip midpoint; keypoint confidence is retained for the pose gate.
