"""
Kalman Filter for BBox trajectory smoothing.
Reduces noise in bounding box coordinates from low-quality surveillance footage.
"""
import numpy as np
from typing import Tuple, Optional

try:
    from filterpy.kalman import KalmanFilter as _FilterPyKalmanFilter
except ModuleNotFoundError:
    _FilterPyKalmanFilter = None


class BBoxKalmanFilter:
    """
    Kalman Filter for smoothing bounding box trajectories.

    State vector: [x, y, w, h, dx, dy, dw, dh]
    - x, y: center coordinates
    - w, h: width and height
    - dx, dy, dw, dh: velocities
    """

    def __init__(self,
                 process_noise: float = 1.0,
                 measurement_noise: float = 10.0):
        """
        Initialize Kalman Filter for BBox tracking.

        Args:
            process_noise: Process noise covariance (Q). Lower = smoother but slower response
            measurement_noise: Measurement noise covariance (R). Higher = trust measurements less
        """
        if _FilterPyKalmanFilter is None:
            raise ImportError(
                "filterpy is required for Kalman preprocessing. "
                "Install the project environment or run without --apply_kalman."
            ) from None

        self.kf = _FilterPyKalmanFilter(dim_x=8, dim_z=4)

        # State transition matrix (constant velocity model)
        dt = 1.0  # Time step
        self.kf.F = np.array([
            [1, 0, 0, 0, dt, 0,  0,  0],   # x += dx * dt
            [0, 1, 0, 0, 0,  dt, 0,  0],   # y += dy * dt
            [0, 0, 1, 0, 0,  0,  dt, 0],   # w += dw * dt
            [0, 0, 0, 1, 0,  0,  0,  dt],  # h += dh * dt
            [0, 0, 0, 0, 1,  0,  0,  0],   # dx (constant)
            [0, 0, 0, 0, 0,  1,  0,  0],   # dy (constant)
            [0, 0, 0, 0, 0,  0,  1,  0],   # dw (constant)
            [0, 0, 0, 0, 0,  0,  0,  1],   # dh (constant)
        ])

        # Measurement function (we only observe x, y, w, h)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])

        # Process noise covariance
        self.kf.Q *= process_noise

        # Measurement noise covariance
        self.kf.R *= measurement_noise

        # Initial state covariance
        self.kf.P *= 1000.0

        self.initialized = False

    def update(self, bbox: np.ndarray) -> np.ndarray:
        """
        Update filter with new bbox measurement and return filtered bbox.

        Args:
            bbox: Array of shape (4,) with [x, y, w, h]

        Returns:
            Filtered bbox of shape (4,)
        """
        if not self.initialized:
            # Initialize state with first measurement
            self.kf.x = np.array([bbox[0], bbox[1], bbox[2], bbox[3], 0, 0, 0, 0])
            self.initialized = True
            return bbox

        # Predict
        self.kf.predict()

        # Update with measurement
        self.kf.update(bbox)

        # Return filtered state (only position and size, not velocity)
        return self.kf.x[:4].copy()

    def reset(self):
        """Reset filter state."""
        self.initialized = False
        self.kf.P *= 1000.0


def smooth_trajectory_with_kalman(
    trajectory: np.ndarray,
    process_noise: float = 1.0,
    measurement_noise: float = 10.0
) -> np.ndarray:
    """
    Smooth a trajectory of bounding boxes using Kalman filter.

    Args:
        trajectory: Array of shape (T, 4) with [x, y, w, h] for each timestep
        process_noise: Process noise parameter
        measurement_noise: Measurement noise parameter

    Returns:
        Smoothed trajectory of shape (T, 4)
    """
    kf = BBoxKalmanFilter(process_noise=process_noise, measurement_noise=measurement_noise)
    smoothed = np.zeros_like(trajectory)

    for t in range(len(trajectory)):
        smoothed[t] = kf.update(trajectory[t])

    return smoothed
