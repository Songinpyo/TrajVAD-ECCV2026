"""Train the trajectory likelihood and reliability-gated joint likelihood."""
import torch
from tqdm import tqdm


def create_optimizer(model, lr=5e-4, weight_decay=1e-5, momentum=0.9):
    return torch.optim.RMSprop(model.parameters(), lr=float(lr),
                              weight_decay=float(weight_decay), momentum=float(momentum))


def train_epoch(model, loader, optimizer, device, grad_clip=3.0):
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc='Training', leave=False):
        batch = [value.to(device) for value in batch]
        x_cont, x_cat = batch[:2]
        kwargs = dict(x_pose=batch[2], pose_mask=batch[3]) if len(batch) == 4 else {}
        _, nll = model(x_cont, x_cat, **kwargs)
        loss = nll.mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item() * len(x_cont)
    return total / len(loader.dataset)
