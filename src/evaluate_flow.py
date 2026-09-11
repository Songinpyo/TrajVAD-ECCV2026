"""Evaluate TrajVAD with the configuration saved in its checkpoint."""
import argparse
import json
from pathlib import Path
from data.dataset_loader import get_multiclass_loader
from models.model_factory import requires_pose
from evaluation.evaluator import evaluate_model
from utils.checkpoint_utils import load_model_from_checkpoint
from utils.dataset_config import get_dataset_config
from utils.gt_loader import load_gt_for_dataset
from utils.path_utils import get_feature_path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weight_path', required=True)
    parser.add_argument('--data_root')
    parser.add_argument('--gt_dir')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--output', help='Optional result JSON file')
    args = parser.parse_args()
    model, checkpoint = load_model_from_checkpoint(args.weight_path, args.device)
    config = checkpoint['config']
    data = config['data']
    dataset = data['dataset']
    root = args.data_root or get_feature_path(dataset, data['feature_variant'], ROOT)
    loader, test_data = get_multiclass_loader(root, dataset, split='testing',
        batch_size=config['training']['batch_size'], num_workers=config['system']['num_workers'],
        load_pose=requires_pose(config['model']['model_type']))
    gt = load_gt_for_dataset(dataset, get_dataset_config(dataset), ROOT, gt_path_override=args.gt_dir)
    results = evaluate_model(model, loader, test_data, gt, args.device,
        seg_len=data['seg_len'], dataset_name=dataset, model_type=config['model']['model_type'])
    metrics = {k:float(results[k]) for k in ['micro_auc', 'micro_ap']}
    print(json.dumps(metrics, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(metrics, indent=2) + '\n')


if __name__ == '__main__':
    main()
