# TrajVAD: Bounding-Box Trajectories Matter for Video Anomaly Detection

<p align="center">
  <b>ECCV 2026</b><br>
  Inpyo Song and Jangwon Lee<br>
  Sungkyunkwan University
</p>

<p align="center">
  <a href="https://songinpyo.github.io/TrajVAD-ECCV2026-Project/"><img src="https://img.shields.io/badge/Project-Page-4285F4" alt="Project Page"></a>
  <a href="https://link.springer.com/chapter/10.1007/978-3-032-37035-8_10"><img src="https://img.shields.io/badge/Paper-Springer-003D7C" alt="Paper"></a>
  <a href="https://arxiv.org/abs/2605.21957"><img src="https://img.shields.io/badge/arXiv-2605.21957-B31B1B" alt="arXiv"></a>
</p>

Official implementation of **Bounding-Box Trajectories Matter for Video Anomaly Detection**.

TrajVAD learns normal motion patterns from multi-class bounding-box trajectories using normalizing flows. **TrajVAD-T** uses trajectory features, while **TrajVAD-P** incorporates human pose information through an additional pose branch.

![TrajVAD overview](docs/figure1_teaser.png)

## News

- **Tracking and pose data will be released soon.** Download links and preparation instructions will be added here.
- Training and evaluation code and configurations are available. Pretrained weights will be released soon.
- Our paper has been accepted to **ECCV 2026**.

## Installation

```bash
conda env create -f environment.yml
conda activate posevad
```

Run the commands below from the repository root.

## Data Preparation

We support ShanghaiTech, UBnormal, and MSAD. The paper configurations use all 27 trajectory features and Kalman-filtered (`kf`) bounding boxes. Segment lengths are dataset-specific:

| Dataset | Segment length | Feature variant |
| --- | ---: | --- |
| ShanghaiTech | 16 | `seg16_std_nolog` |
| UBnormal | 12 | `seg12_std_nolog` |
| MSAD | 24 | `seg24_std_nolog` |

Place preprocessed features and ground-truth annotations under `data/` following [the data layout](docs/data_preparation.md).

The paper uses **YOLOX** for detection, **OSNet** for re-identification, **ByteTrack** for tracking, and **AlphaPose** for 17-keypoint poses.

The preprocessing scripts in `src/scripts/` convert track and pose parquet files to model-ready features. See [the data layout](docs/data_preparation.md) for the expected directories.

## Pretrained Weights

Pretrained weights will be released soon.

## Training

```bash
python src/train_flow.py --training_config TrajVAD-T/shanghaitech.yaml
python src/train_flow.py --training_config TrajVAD-P/shanghaitech.yaml
```

Configurations for all three datasets are in `configs/paper/`. Training outputs are
saved under `runs/{dataset}/{configuration}/`. Use `--data_root` and `--gt_dir` to
set feature and annotation paths.

## Evaluation

```bash
python src/evaluate_flow.py \
  --weight_path runs/shanghaitech/TrajVAD-T_shanghaitech/best_micro_auc.pth
```

The checkpoint includes the model and data configuration. Evaluation reports
frame-level micro AUROC and AP following the paper evaluation protocol.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{song2026bounding,
  title     = {Bounding-Box Trajectories Matter for Video Anomaly Detection},
  author    = {Song, Inpyo and Lee, Jangwon},
  booktitle = {Computer Vision -- ECCV 2026},
  pages     = {175--193},
  publisher = {Springer},
  year      = {2026},
  doi       = {10.1007/978-3-032-37035-8_10}
}
```
