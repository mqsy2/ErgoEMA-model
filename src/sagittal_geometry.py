"""
Sagittal (90° Side-Profile) Geometry.
Estimates the craniovertebral angle (CVA) from pose landmarks and the body silhouette.

The CVA is the angle between the horizontal and the line from the C7 vertebra to the
tragus of the ear. MediaPipe has no C7 landmark, so C7 is located on the posterior
(back-of-neck) edge of the body silhouette at the base of the neck.
"""

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

# Height (in pixels) at which the silhouette is analysed; masks are resized to this height
ANALYSIS_HEIGHT: int = 480

# Expected position of C7 relative to the shoulder joint, in units of the ear-to-shoulder distance
C7_HEIGHT_ABOVE_SHOULDER: float = 0.30   # C7 sits roughly a third of the way up from the shoulder joint to the ear
C7_DEPTH_BEHIND_SHOULDER: float = 0.35   # Median depth of the neck-base contour behind the shoulder joint across the
                                         # study's lateral clips
# The silhouette refines that expected position only when its back edge passes within this distance of it.
# Long hair, hoods and headrests push the edge further out; the expected position is then used as is.
C7_SILHOUETTE_TOLERANCE: float = 0.30

# Approximate spine levels as fractions of the C7-to-hip height (drawn on the back contour, not used for classification)
SPINE_LEVELS: Tuple[Tuple[str, float], ...] = (("THORACIC", 0.29), ("LUMBAR", 0.71), ("SACRAL", 0.88))

CONTOUR_SMOOTH_ROWS: int = 5             # Half-width of the running median applied to the back contour
MIN_EAR_VISIBILITY: float = 0.3
MIN_HIP_VISIBILITY: float = 0.5

@dataclass
class LateralGeometry:
    """Side-profile measurements for one frame. All points are in normalized image coordinates."""
    facing: float                              # +1.0 facing right (+X), -1.0 facing left (-X)
    ear: Tuple[float, float]
    shoulder: Tuple[float, float]
    c7: Tuple[float, float]
    craniovertebral_angle_deg: float
    c7_from_silhouette: bool                   # False when C7 was placed from landmark proportions alone
    back_contour: List[Tuple[float, float]] = field(default_factory=list)       # Posterior body edge, C7 downwards
    spine_levels: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # Approximate levels on the contour

def _back_contour(body: np.ndarray, rows: np.ndarray, seed_x: np.ndarray, back_dir: int) -> np.ndarray:
    """
    For each row, walks from a seed inside the body toward the back and returns the x of the
    last body pixel. NaN where the seed is outside the body or the body runs to the frame edge.
    """
    width = body.shape[1]
    xs = np.full(len(rows), np.nan)
    for i, (y, sx) in enumerate(zip(rows, seed_x)):
        x = int(round(sx))
        if not (0 <= x < width) or not body[y, x]:
            continue
        run = body[y, x:] if back_dir > 0 else body[y, x::-1]
        gap = int(np.argmin(run))  # First background pixel along the walk
        if run[gap]:
            continue
        xs[i] = x + back_dir * (gap - 1)
    return xs

def _running_median(values: np.ndarray, half_width: int) -> np.ndarray:
    """NaN-aware running median."""
    padded = np.pad(values, half_width, constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half_width + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # All-NaN windows stay NaN
        return np.nanmedian(windows, axis=1)

def analyze_lateral(pose, near_left: bool) -> Optional[LateralGeometry]:
    """
    Measures the craniovertebral angle for a side-profile pose.

    Args:
        pose: PoseLandmarks; its segmentation_mask (if any) locates C7 on the back of the neck.
        near_left: True if the subject's left side faces the camera.
    """
    shoulder = pose.left_shoulder if near_left else pose.right_shoulder
    hip = pose.left_hip if near_left else pose.right_hip
    near_ear, far_ear = (pose.left_ear, pose.right_ear) if near_left else (pose.right_ear, pose.left_ear)
    ear = near_ear if near_ear.visibility > MIN_EAR_VISIBILITY or near_ear.visibility >= far_ear.visibility else far_ear

    mask = pose.segmentation_mask
    if mask is not None:
        height, width = mask.shape[:2]
    else:
        # Work in pixels of a frame with the right aspect ratio so that angles are not distorted
        aspect = pose.image_width / pose.image_height if pose.image_width and pose.image_height else 1.0
        height, width = ANALYSIS_HEIGHT, ANALYSIS_HEIGHT * aspect
    scale = np.array([width, height], dtype=float)

    def to_px(p) -> np.ndarray:
        return np.array([p.x, p.y]) * scale

    ear_px, sh_px = to_px(ear), to_px(shoulder)
    unit = float(np.linalg.norm(ear_px - sh_px))  # Ear-to-shoulder distance: the body-size reference
    if unit < 1e-6:
        return None

    facing = 1.0 if pose.nose.x >= ear.x else -1.0
    back_dir = -int(facing)
    hip_px = to_px(hip) if hip is not None and hip.visibility > MIN_HIP_VISIBILITY and 0.0 < hip.y < 1.0 else None

    # Expected C7 position from landmark proportions, refined below where the silhouette's back edge agrees with it
    c7_px = np.array([sh_px[0] - facing * C7_DEPTH_BEHIND_SHOULDER * unit, sh_px[1] - C7_HEIGHT_ABOVE_SHOULDER * unit])
    c7_from_silhouette = False
    back_contour: List[Tuple[float, float]] = []
    spine_levels: Dict[str, Tuple[float, float]] = {}

    if mask is not None:
        body = mask > 0.5
        bottom = hip_px[1] if hip_px is not None else height
        rows = np.arange(max(0, int(ear_px[1])), min(height, int(bottom)))
        if len(rows) > 2 * CONTOUR_SMOOTH_ROWS:
            # Seeds follow the body axis: ear -> shoulder -> hip (straight down from the shoulder without a hip)
            axis = [ear_px, sh_px, hip_px if hip_px is not None else np.array([sh_px[0], float(height)])]
            seed_x = np.interp(rows, [p[1] for p in axis], [p[0] for p in axis])
            contour_x = _running_median(_back_contour(body, rows, seed_x, back_dir), CONTOUR_SMOOTH_ROWS)
            on_edge = np.isfinite(contour_x)

            if on_edge.any():
                edge = np.stack([contour_x[on_edge], rows[on_edge].astype(float)], axis=1)
                gaps = np.linalg.norm(edge - c7_px, axis=1)
                nearest = int(np.argmin(gaps))
                if gaps[nearest] <= C7_SILHOUETTE_TOLERANCE * unit:
                    c7_px = edge[nearest]
                    c7_from_silhouette = True

                    below = edge[edge[:, 1] >= c7_px[1]]
                    back_contour = [(float(x / width), float(y / height)) for x, y in below[::4]]
                    if hip_px is not None:
                        for name, fraction in SPINE_LEVELS:
                            i = int(np.argmin(np.abs(rows - (c7_px[1] + fraction * (hip_px[1] - c7_px[1])))))
                            if on_edge[i]:
                                spine_levels[name] = (float(contour_x[i] / width), float(rows[i] / height))

    forward = (ear_px[0] - c7_px[0]) * facing
    cva_deg = float(np.degrees(np.arctan2(c7_px[1] - ear_px[1], forward)))

    return LateralGeometry(
        facing=facing,
        ear=(float(ear.x), float(ear.y)),
        shoulder=(float(shoulder.x), float(shoulder.y)),
        c7=(float(c7_px[0] / width), float(c7_px[1] / height)),
        craniovertebral_angle_deg=cva_deg,
        c7_from_silhouette=c7_from_silhouette,
        back_contour=back_contour,
        spine_levels=spine_levels
    )
