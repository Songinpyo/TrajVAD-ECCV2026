"""Restore a model together with its training configuration."""
import torch
from models.model_factory import create_model


def load_model_from_checkpoint(checkpoint_path, device='cuda', logger=None):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = create_model(**config['model'], seg_len=config['data']['seg_len'], device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.set_actnorm_init()
    model.eval()
    return model, checkpoint
