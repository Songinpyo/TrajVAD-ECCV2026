"""Train TrajVAD using a paper configuration."""
import argparse
import csv
import random
from pathlib import Path
import numpy as np
import torch
from data.dataset_loader import get_multiclass_loader
from models.model_factory import create_model, requires_pose
from training.trainer import create_optimizer, train_epoch
from evaluation.evaluator import evaluate_model
from utils.training_config import load_training_config, save_config_to_output
from utils.dataset_config import get_dataset_config
from utils.gt_loader import load_gt_for_dataset
from utils.path_utils import get_feature_path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training_config', required=True)
    parser.add_argument('--data_root')
    parser.add_argument('--gt_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--device', choices=['cuda', 'cpu'])
    args = parser.parse_args()
    path = Path(args.training_config)
    if not path.exists():
        path = ROOT / 'configs' / 'paper' / path
    config = load_training_config(path)
    system, data, training = config['system'], config['data'], config['training']
    device = args.device or system['device']
    seed = system['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = data['dataset']
    root = args.data_root or get_feature_path(dataset, data['feature_variant'], ROOT)
    pose = requires_pose(config['model']['model_type'])
    common = dict(data_root=root, dataset_name=dataset, mode='kf',
                  batch_size=training['batch_size'], num_workers=system['num_workers'], load_pose=pose)
    train_loader, _ = get_multiclass_loader(split='training', **common)
    test_loader, test_data = get_multiclass_loader(split='testing', **common)
    model = create_model(**config['model'], seg_len=data['seg_len'], device=device)
    optimizer = create_optimizer(model, training['lr'], training['weight_decay'], training['momentum'])
    gt = load_gt_for_dataset(dataset, get_dataset_config(dataset), ROOT, gt_path_override=args.gt_dir)
    output = Path(args.output_dir) if args.output_dir else ROOT / 'runs' / dataset / config['name']
    output.mkdir(parents=True, exist_ok=True)
    save_config_to_output(config, output)
    best = float('-inf')
    with (output / 'training_history.csv').open('w') as stream:
        fields = ['epoch', 'train_nll', 'micro_auc', 'micro_ap']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, training['num_epochs'] + 1):
            loss = train_epoch(model, train_loader, optimizer, device, training['grad_clip'])
            metrics = evaluate_model(model, test_loader, test_data, gt, device,
                                     seg_len=data['seg_len'], dataset_name=dataset,
                                     model_type=config['model']['model_type'])
            writer.writerow(dict(epoch=epoch, train_nll=loss, **{k:metrics[k] for k in fields[2:]}))
            stream.flush()
            print(f"Epoch {epoch}: AUROC={metrics['micro_auc']:.4f}, "
                  f"AP={metrics['micro_ap']:.4f}")
            if metrics['micro_auc'] > best:
                best = metrics['micro_auc']
                torch.save(dict(model_state_dict=model.state_dict(), config=config,
                                epoch=epoch, micro_auc=best), output / 'best_micro_auc.pth')


if __name__ == '__main__':
    main()
