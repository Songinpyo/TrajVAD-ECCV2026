"""
Feature engineering for BBox-based anomaly detection.
Computes physical attributes from bounding box trajectories.

Feature Specification (27D continuous + 1D categorical):
- Basic coords (4): cx, cy, w, h (normalized 0~1)
- Differential (6): vx, vy, speed, ax, ay, acc
- Direction (2): sin(θ), cos(θ)
- Geometry (1): area, ratio
- Normalized speed (3): h_norm_speed, w_norm_speed, area_norm_speed
- Normalized acc (3): h_norm_acc, w_norm_acc, area_norm_acc
- High-order kinematics (2): jerk, curvature
- Geometric dynamics (2): expansion_rate, ar_velocity
- Pseudo-physics (2): kinetic_energy, path_efficiency
- Confidence (1): conf

Preprocessing Steps:
1. Normalize coordinates to 0~1
2. Compute derived features
3. Optional log transformation (disabled in paper configurations)
4. Standard scaling (fit on train only)
"""
import numpy as np
from typing import Tuple, List, Optional, Dict
from sklearn.preprocessing import StandardScaler, RobustScaler
import pickle


# Feature indices for log transformation
# Feature indices for log transformation
# Only features that are strictly non-negative and have huge dynamic range driven by perspective (Depth):
# w (2), h (3), area (10), kinetic_energy (24).
# We DO NOT apply log to speed/acc/jerk because high magnitude IS the anomaly signal.
LOG_TRANSFORM_INDICES = [2, 3, 10, 24]
LOG_EPSILON = 1e-6

# Feature names with indices
FEATURE_NAMES = [
    'cx',              # 0: center x (0~1)
    'cy',              # 1: center y (0~1)
    'w',               # 2: width
    'h',               # 3: height
    'vx',              # 4: velocity x
    'vy',              # 5: velocity y
    'speed',           # 6: speed magnitude
    'ax',              # 7: acceleration x
    'ay',              # 8: acceleration y
    'acc',             # 9: acceleration magnitude
    'area',            # 10: w*h
    'ratio',           # 11: w/h
    'h_norm_speed',    # 12: speed/h (perspective-corrected speed)
    'w_norm_speed',    # 13: speed/w
    'area_norm_speed', # 14: speed/area
    'h_norm_acc',      # 15: acc/h
    'w_norm_acc',      # 16: acc/w
    'area_norm_acc',   # 17: acc/area
    'sin_theta',       # 18: direction sin
    'cos_theta',       # 19: direction cos
    'jerk',            # 20: jerk magnitude (rate of change of acceleration)
    'curvature',       # 21: trajectory curvature
    'expansion_rate',  # 22: looming/expansion rate (d/dt log(Area))
    'ar_velocity',     # 23: aspect ratio velocity (rate of change of w/h)
    'kinetic_energy',  # 24: kinetic energy proxy (Area * Speed²)
    'path_efficiency', # 25: path efficiency / tortuosity (net displacement / total path length)
    'conf',            # 26: detection confidence
]


def compute_velocity(positions: np.ndarray) -> np.ndarray:
    """Compute velocity from position trajectory (T, 2) -> (T, 2)."""
    velocity = np.zeros_like(positions)
    if len(positions) > 1:
        velocity[1:] = positions[1:] - positions[:-1]
        velocity[0] = velocity[1]  # Forward fill
    return velocity


def compute_acceleration(velocity: np.ndarray) -> np.ndarray:
    """Compute acceleration from velocity trajectory (T, 2) -> (T, 2)."""
    acceleration = np.zeros_like(velocity)
    if len(velocity) > 1:
        acceleration[1:] = velocity[1:] - velocity[:-1]
        acceleration[0] = acceleration[1]  # Forward fill
    return acceleration


def compute_orientation(velocity: np.ndarray) -> np.ndarray:
    """Compute orientation (sin, cos) from velocity (T, 2) -> (T, 2)."""
    speed = np.linalg.norm(velocity, axis=1, keepdims=True)
    orientation = np.zeros_like(velocity)
    mask = speed[:, 0] > 1e-6
    orientation[mask] = velocity[mask] / speed[mask]
    return orientation


def compute_jerk(acceleration: np.ndarray) -> np.ndarray:
    """
    Compute jerk (rate of change of acceleration) from acceleration trajectory.
    
    Jerk magnitude: ||j|| = ||(a_t - a_{t-1}) / dt||
    
    Args:
        acceleration: (T, 2) acceleration array
    
    Returns:
        jerk: (T,) jerk magnitude array
    """
    jerk_vec = compute_acceleration(acceleration)  # Reuse acceleration computation
    jerk = np.linalg.norm(jerk_vec, axis=1)  # (T,)
    return jerk


def compute_curvature(velocity: np.ndarray, acceleration: np.ndarray) -> np.ndarray:
    """
    Compute trajectory curvature from velocity and acceleration.
    
    Curvature formula: κ = |v_x * a_y - v_y * a_x| / (v_x² + v_y²)^1.5
    
    Args:
        velocity: (T, 2) velocity array
        acceleration: (T, 2) acceleration array
    
    Returns:
        curvature: (T,) curvature array
    """
    T = len(velocity)
    curvature = np.zeros(T, dtype=np.float32)
    
    vx = velocity[:, 0]
    vy = velocity[:, 1]
    ax = acceleration[:, 0]
    ay = acceleration[:, 1]
    
    # Compute speed squared
    speed_sq = vx**2 + vy**2
    
    # Avoid division by zero: set curvature to 0 when speed is too small
    mask = speed_sq > 1e-6
    numerator = np.abs(vx * ay - vy * ax)
    denominator = speed_sq ** 1.5
    curvature[mask] = numerator[mask] / denominator[mask]
    
    return curvature


def compute_expansion_rate(area: np.ndarray) -> np.ndarray:
    """
    Compute expansion rate (looming indicator) from area trajectory.
    
    Expansion rate: E_dot = d/dt log(Area) ≈ (Area_t - Area_{t-1}) / Area_t
    
    Args:
        area: (T,) area array
    
    Returns:
        expansion_rate: (T,) expansion rate array
    """
    T = len(area)
    expansion_rate = np.zeros(T, dtype=np.float32)
    
    if T > 1:
        # Forward difference: (Area_t - Area_{t-1}) / Area_t
        # Use log difference for numerical stability: log(Area_t) - log(Area_{t-1})
        # Ensure area is positive to avoid log of zero/negative values
        safe_area = np.maximum(area, 1e-6)
        log_area = np.log(safe_area)
        expansion_rate[1:] = log_area[1:] - log_area[:-1]
        expansion_rate[0] = expansion_rate[1]  # Forward fill
    
    return expansion_rate


def compute_ar_velocity(ratio: np.ndarray) -> np.ndarray:
    """
    Compute aspect ratio velocity (rate of change of aspect ratio).
    
    AR velocity: AR_dot = (ratio_t - ratio_{t-1}) / dt
    
    Args:
        ratio: (T,) aspect ratio (w/h) array
    
    Returns:
        ar_velocity: (T,) aspect ratio velocity array
    """
    ar_velocity = np.zeros_like(ratio)
    if len(ratio) > 1:
        ar_velocity[1:] = ratio[1:] - ratio[:-1]
        ar_velocity[0] = ar_velocity[1]  # Forward fill
    return ar_velocity


def compute_kinetic_energy(area: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """
    Compute kinetic energy proxy: E_proxy = Area * Speed²
    
    This approximates kinetic energy using area as a proxy for mass.
    
    Args:
        area: (T,) area array
        speed: (T,) speed magnitude array
    
    Returns:
        kinetic_energy: (T,) kinetic energy proxy array
    """
    kinetic_energy = area * (speed ** 2)
    return kinetic_energy


def compute_path_efficiency(
    positions: np.ndarray,
    window_size: int = 12
) -> np.ndarray:
    """
    Compute path efficiency (tortuosity): η = net_displacement / total_path_length
    
    Path efficiency measures how efficiently an object moves from point A to B.
    - η ≈ 1: straight line movement (efficient)
    - η ≈ 0: loitering/fighting (inefficient, lots of movement but little net displacement)
    
    Args:
        positions: (T, 2) position array
        window_size: Window size for computing efficiency (default: 12 frames)
    
    Returns:
        path_efficiency: (T,) path efficiency array
    """
    T = len(positions)
    path_efficiency = np.zeros(T, dtype=np.float32)
    
    if T < 2:
        return path_efficiency
    
    # Compute net displacement and total path length for each window
    for t in range(T):
        start_idx = max(0, t - window_size + 1)
        end_idx = t + 1
        
        if end_idx - start_idx < 2:
            path_efficiency[t] = 1.0  # Default to efficient for short windows
            continue
        
        window_positions = positions[start_idx:end_idx]
        
        # Net displacement: ||x_end - x_start||
        net_displacement = np.linalg.norm(window_positions[-1] - window_positions[0])
        
        # Total path length: sum of ||x_k - x_{k-1}||
        if len(window_positions) > 1:
            diffs = window_positions[1:] - window_positions[:-1]
            total_path_length = np.sum(np.linalg.norm(diffs, axis=1))
        else:
            total_path_length = 0.0
        
        # Path efficiency: net_displacement / total_path_length
        if total_path_length > 1e-6:
            path_efficiency[t] = net_displacement / total_path_length
        else:
            path_efficiency[t] = 1.0  # Default to efficient if no movement
    
    return path_efficiency


def extract_all_features(
    bboxes: np.ndarray,
    conf: np.ndarray,
    vid_res: Tuple[int, int] = (856, 480)
) -> Dict[str, np.ndarray]:
    """
    Extract ALL available features from bounding box trajectory.
    This function computes all features regardless of which ones will be used,
    solving dependency issues (e.g., area is computed even if only norm_speed is needed).
    
    Args:
        bboxes: (T, 4) with [x1, y1, x2, y2] or [cx, cy, w, h] in pixels
        conf: (T,) detection confidence scores
        vid_res: (width, height) for normalization
    
    Returns:
        Dictionary mapping feature names to feature arrays (T,)
        All features are computed: cx, cy, w, h, vx, vy, speed, ax, ay, acc,
                                   sin_theta, cos_theta, area, ratio,
                                   h_norm_speed, w_norm_speed, area_norm_speed,
                                   h_norm_acc, w_norm_acc, area_norm_acc,
                                   jerk, curvature, expansion_rate, ar_velocity,
                                   kinetic_energy, path_efficiency, conf
    """
    T = len(bboxes)
    vid_w, vid_h = vid_res
    
    # Convert x1y1x2y2 to cxcywh if needed
    if bboxes.shape[1] == 4:
        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
    else:
        cx, cy, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    
    # Step 1: Normalize to 0~1
    cx_norm = cx / vid_w
    cy_norm = cy / vid_h
    w_norm = w / vid_w
    h_norm = h / vid_h
    
    # Step 2: Compute ALL differential features (always compute, regardless of usage)
    positions = np.stack([cx_norm, cy_norm], axis=1)  # (T, 2)
    velocity = compute_velocity(positions)             # (T, 2)
    acceleration = compute_acceleration(velocity)      # (T, 2)
    orientation = compute_orientation(velocity)        # (T, 2)
    
    speed = np.linalg.norm(velocity, axis=1)           # (T,)
    acc = np.linalg.norm(acceleration, axis=1)     # (T,)
    
    # Step 3: Compute ALL geometry features (always compute)
    area = w_norm * h_norm                             # (T,)
    ratio = w_norm / (h_norm + 1e-6)                   # (T,)
    
    # Step 4: Compute normalized speed features
    h_norm_speed = speed / (h_norm + 1e-6)             # speed/h (perspective-corrected)
    w_norm_speed = speed / (w_norm + 1e-6)             # speed/w
    area_norm_speed = speed / (area + 1e-6)            # speed/area
    
    # Step 5: Compute normalized acceleration features
    h_norm_acc = acc / (h_norm + 1e-6)                 # acc/h
    w_norm_acc = acc / (w_norm + 1e-6)                 # acc/w
    area_norm_acc = acc / (area + 1e-6)                # acc/area
    
    # Step 6: Compute high-order kinematics features
    jerk = compute_jerk(acceleration)                  # (T,)
    curvature = compute_curvature(velocity, acceleration)  # (T,)
    
    # Step 7: Compute geometric dynamics features
    expansion_rate = compute_expansion_rate(area)      # (T,)
    ar_velocity = compute_ar_velocity(ratio)            # (T,)
    
    # Step 8: Compute pseudo-physics features
    kinetic_energy = compute_kinetic_energy(area, speed)  # (T,)
    path_efficiency = compute_path_efficiency(positions)   # (T,)
    
    # Step 9: Assemble ALL features into dictionary
    all_features = {
        'cx': cx_norm,
        'cy': cy_norm,
        'w': w_norm,
        'h': h_norm,
        'vx': velocity[:, 0],
        'vy': velocity[:, 1],
        'speed': speed,
        'ax': acceleration[:, 0],
        'ay': acceleration[:, 1],
        'acc': acc,
        'area': area,
        'ratio': ratio,
        'h_norm_speed': h_norm_speed,
        'w_norm_speed': w_norm_speed,
        'area_norm_speed': area_norm_speed,
        'h_norm_acc': h_norm_acc,
        'w_norm_acc': w_norm_acc,
        'area_norm_acc': area_norm_acc,
        'sin_theta': orientation[:, 0],
        'cos_theta': orientation[:, 1],
        'jerk': jerk,
        'curvature': curvature,
        'expansion_rate': expansion_rate,
        'ar_velocity': ar_velocity,
        'kinetic_energy': kinetic_energy,
        'path_efficiency': path_efficiency,
        'conf': conf,
    }
    
    return all_features






def segment_trajectory(
    features: np.ndarray,
    class_ids: np.ndarray,
    seg_len: int = 12,
    seg_stride: int = 1
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Split trajectory into fixed-length segments.
    
    Args:
        features: (T, D) continuous features (D can be 16, 21, or custom)
        class_ids: (T,) class IDs
        seg_len: Segment length
        seg_stride: Stride between segments
    
    Returns:
        x_cont: (N, seg_len, D) continuous features
        x_cat: (N, seg_len, 1) categorical features (class_id)
        start_frames: List of start frame indices
    """
    T = len(features)
    D = features.shape[1] if features.ndim == 2 else 16  # Dynamic dimension
    segments_cont = []
    segments_cat = []
    start_frames = []
    
    for start in range(0, T - seg_len + 1, seg_stride):
        end = start + seg_len
        segments_cont.append(features[start:end])
        segments_cat.append(class_ids[start:end].reshape(-1, 1))
        start_frames.append(start)
    
    if len(segments_cont) == 0:
        return (np.empty((0, seg_len, D), dtype=np.float32),
                np.empty((0, seg_len, 1), dtype=np.int64),
                [])
    
    x_cont = np.array(segments_cont, dtype=np.float32)
    x_cat = np.array(segments_cat, dtype=np.int64)
    
    return x_cont, x_cat, start_frames


class FeatureScaler:
    """
    Scaler wrapper for feature normalization.
    Supports StandardScaler and RobustScaler.
    Fits only on continuous features (excluding class_id).
    """

    def __init__(self, method: str = "standard"):
        """
        Args:
            method: "standard" (StandardScaler) or "robust" (RobustScaler)
        """
        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}. Use 'standard' or 'robust'.")
        self.method = method
        self.fitted = False
    
    def fit(self, features: np.ndarray):
        """
        Fit scaler on training data.
        
        Args:
            features: (N, T, D) or (N*T, D) continuous features (D can be 16, 21, or custom)
        """
        if features.ndim == 3:
            # Reshape (N, T, D) -> (N*T, D)
            N, T, D = features.shape
            features = features.reshape(-1, D)
        
        self.scaler.fit(features)
        self.fitted = True
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted scaler.
        
        Args:
            features: (N, T, D) or (N*T, D) continuous features (D can be 16, 21, or custom)
        
        Returns:
            Scaled features with same shape
        """
        if not self.fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        
        original_shape = features.shape
        if features.ndim == 3:
            N, T, D = features.shape
            features = features.reshape(-1, D)
        
        scaled = self.scaler.transform(features)
        
        return scaled.reshape(original_shape).astype(np.float32)
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(features)
        return self.transform(features)
    
    def save(self, path: str):
        """Save scaler to file."""
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load(self, path: str):
        """Load scaler from file."""
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)
        self.fitted = True


class BBoxFeatureExtractor:
    """
    Complete feature extraction pipeline for bounding box trajectories.
    
    Pipeline:
    1. Extract features (normalized coordinates, default 27D)
    2. Apply log transformation
    3. Segment into fixed-length windows
    4. Apply standard scaling (optional)
    
    Can be initialized from config dictionary for variant support.
    """
    
    def __init__(
        self,
        vid_res: Tuple[int, int] = (856, 480),
        seg_len: int = 12,
        seg_stride: int = 1,
        apply_log: bool = False,
        config: Optional[Dict] = None
    ):
        """
        Initialize feature extractor.

        Args:
            vid_res: Video resolution (width, height)
            seg_len: Segment length
            seg_stride: Stride between segments
            apply_log: Whether to apply log transformation (Default: False)
            config: Optional config dictionary (overrides other parameters if provided)
        """
        if config is not None:
            # Initialize from config
            extractor_cfg = config.get('extractor', {})
            segment_cfg = config.get('segment', {})
            norm_cfg = config.get('normalization', {})

            self.apply_log = extractor_cfg.get('apply_log', False)
            self.seg_len = segment_cfg.get('length', 12)
            self.seg_stride = seg_stride  # Keep stride from argument (can be train/test specific)
            self.config = config

            # Extract normalization method from config
            norm_method = norm_cfg.get('method', 'standard')

            # Extract feature names from config
            feature_list = extractor_cfg.get('features', [])
            if isinstance(feature_list, list) and len(feature_list) > 0:
                # Extract feature names from dict format: [{'name': 'cx'}, ...]
                if isinstance(feature_list[0], dict):
                    self.feature_names = [f['name'] for f in feature_list]
                elif isinstance(feature_list[0], str):
                    self.feature_names = feature_list
                else:
                    raise ValueError(f"Invalid feature format in config: {feature_list[0]}")
            else:
                self.feature_names = None  # Use default (all 27D)
        else:
            # Initialize from arguments (backward compatibility)
            self.apply_log = apply_log
            self.seg_len = seg_len
            self.seg_stride = seg_stride
            self.config = None
            self.feature_names = None  # Use default (all 27D)
            norm_method = 'standard'

        self.vid_res = vid_res
        self.scaler = FeatureScaler(method=norm_method)
    
    def extract_from_track(
        self,
        bboxes: np.ndarray,
        conf: np.ndarray,
        class_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Extract features from a single track.
        
        Args:
            bboxes: (T, 4) bounding boxes [x1, y1, x2, y2]
            conf: (T,) confidence scores
            class_ids: (T,) class IDs
        
        Returns:
            x_cont: (N, seg_len, D) continuous features (D depends on feature_names)
            x_cat: (N, seg_len, 1) class IDs
            start_frames: List of start frame indices
        """
        # Extract all features first (27D)
        # Note: We must apply log transform BEFORE selecting subset because LOG_TRANSFORM_INDICES are fixed for the full set.
        all_features_dict = extract_all_features(bboxes, conf, self.vid_res)
        
        # Apply log transform if enabled
        if self.apply_log:
            for idx in self.get_log_indices(): # Access static method via instance or class
                # Indices in LOG_TRANSFORM_INDICES correspond to FEATURE_NAMES order
                # We need to look up the name for this index
                feat_name = FEATURE_NAMES[idx]
                if feat_name in all_features_dict:
                     all_features_dict[feat_name] = np.log(all_features_dict[feat_name] + LOG_EPSILON)

        # Select requested features
        if self.feature_names is None:
            selected_names = FEATURE_NAMES
        else:
            selected_names = self.feature_names
            
        # Construct feature array (T, D)
        T = len(bboxes)
        D = len(selected_names)
        features = np.zeros((T, D), dtype=np.float32)
        for i, name in enumerate(selected_names):
             features[:, i] = all_features_dict[name]
        
        # Segment trajectory
        x_cont, x_cat, start_frames = segment_trajectory(
            features, class_ids, self.seg_len, self.seg_stride
        )
        
        return x_cont, x_cat, start_frames
    
    def fit_scaler(self, all_features: np.ndarray):
        """Fit scaler on training data."""
        self.scaler.fit(all_features)
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Apply scaling to features."""
        return self.scaler.transform(features)
    
    def save_scaler(self, path: str):
        """Save fitted scaler."""
        self.scaler.save(path)
    
    def load_scaler(self, path: str):
        """Load fitted scaler."""
        self.scaler.load(path)
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """Get feature names."""
        return FEATURE_NAMES.copy()
    
    @staticmethod
    def get_log_indices() -> List[int]:
        """Get indices of log-transformed features."""
        return LOG_TRANSFORM_INDICES.copy()
