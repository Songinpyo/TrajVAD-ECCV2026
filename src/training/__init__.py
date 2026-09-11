"""
Training module for Trajectory-VAD.

Provides:
- Optimizer creation
- Training loop
- Scheduler utilities
"""
from .trainer import create_optimizer, train_epoch

__all__ = ['create_optimizer', 'train_epoch']
