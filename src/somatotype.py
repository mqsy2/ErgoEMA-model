"""
Heath-Carter Anthropometric Somatotype.
Computes the endomorphy-mesomorphy-ectomorphy rating and its somatotype category from body
measurements, using the equations of the Heath-Carter instruction manual (Carter, 2002).

The method is defined on measurements taken on the person (stadiometer, scale, skinfold caliper,
bone caliper and tape). None of them can be read from a webcam, so they are entered by the operator.
"""

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional
import json

# Measurement -> (lowest, highest) plausible adult value. Used to catch unit mix-ups such as a height in metres.
PLAUSIBLE_RANGES: Dict[str, tuple] = {
    "height_cm": (100.0, 230.0),                 # Stature
    "weight_kg": (20.0, 250.0),                  # Body mass
    "triceps_skinfold_mm": (1.0, 80.0),
    "subscapular_skinfold_mm": (1.0, 80.0),
    "supraspinale_skinfold_mm": (1.0, 80.0),
    "medial_calf_skinfold_mm": (1.0, 80.0),
    "humerus_breadth_cm": (3.0, 12.0),           # Biepicondylar breadth of the humerus
    "femur_breadth_cm": (4.0, 16.0),             # Biepicondylar breadth of the femur
    "flexed_arm_girth_cm": (10.0, 70.0),         # Upper arm girth, elbow flexed and tensed
    "calf_girth_cm": (15.0, 80.0),               # Maximum standing calf girth
}

MIN_COMPONENT: float = 0.1  # A component is never rated zero or negative

@dataclass
class Anthropometry:
    """The ten Heath-Carter measurements. Any that were not taken are left as None."""
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    triceps_skinfold_mm: Optional[float] = None
    subscapular_skinfold_mm: Optional[float] = None
    supraspinale_skinfold_mm: Optional[float] = None
    medial_calf_skinfold_mm: Optional[float] = None
    humerus_breadth_cm: Optional[float] = None
    femur_breadth_cm: Optional[float] = None
    flexed_arm_girth_cm: Optional[float] = None
    calf_girth_cm: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Anthropometry':
        """Builds the measurements from a dictionary, rejecting values outside the plausible range."""
        if not isinstance(data, dict):
            raise ValueError("measurements must be a JSON object of name: value pairs")
        values = {}
        for name, (low, high) in PLAUSIBLE_RANGES.items():
            value = data.get(name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number, got {value!r}")
            if not low <= value <= high:
                raise ValueError(f"{name} = {value} is outside the plausible range {low}-{high} (check the unit)")
            values[name] = float(value)
        return cls(**values)

    @classmethod
    def load_json(cls, filepath: str) -> 'Anthropometry':
        """Loads the measurements from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

@dataclass
class Somatotype:
    """Heath-Carter rating. A component is None when the measurements it needs were not supplied."""
    endomorphy: Optional[float]
    mesomorphy: Optional[float]
    ectomorphy: Optional[float]
    category: Optional[str]                              # Set only when all three components are rated
    missing: List[str] = field(default_factory=list)     # Measurements still needed for a full rating

    @property
    def rating(self) -> str:
        """The rating in the conventional endomorphy-mesomorphy-ectomorphy order, e.g. '2.5-5.0-3.0'."""
        return "-".join("n/a" if c is None else f"{c:.1f}" for c in (self.endomorphy, self.mesomorphy, self.ectomorphy))

def endomorphy(height_cm: float, triceps_mm: float, subscapular_mm: float, supraspinale_mm: float) -> float:
    """Relative fatness, from the sum of three skinfolds corrected for height."""
    x = (triceps_mm + subscapular_mm + supraspinale_mm) * (170.18 / height_cm)
    return max(MIN_COMPONENT, -0.7182 + 0.1451 * x - 0.00068 * x ** 2 + 0.0000014 * x ** 3)

def mesomorphy(
    height_cm: float,
    humerus_breadth_cm: float,
    femur_breadth_cm: float,
    flexed_arm_girth_cm: float,
    calf_girth_cm: float,
    triceps_mm: float,
    medial_calf_mm: float
) -> float:
    """Musculoskeletal robustness relative to height, from bone breadths and skinfold-corrected girths."""
    corrected_arm_girth = flexed_arm_girth_cm - triceps_mm / 10.0
    corrected_calf_girth = calf_girth_cm - medial_calf_mm / 10.0
    return max(MIN_COMPONENT, 0.858 * humerus_breadth_cm + 0.601 * femur_breadth_cm + 0.188 * corrected_arm_girth
               + 0.161 * corrected_calf_girth - 0.131 * height_cm + 4.5)

def ectomorphy(height_cm: float, weight_kg: float) -> float:
    """Relative linearity, from the height-weight ratio (height divided by the cube root of mass)."""
    hwr = height_cm / weight_kg ** (1.0 / 3.0)
    if hwr >= 40.75:
        return 0.732 * hwr - 28.58
    if hwr > 38.25:
        return 0.463 * hwr - 17.63
    return MIN_COMPONENT

def classify(endo: float, meso: float, ecto: float) -> str:
    """Names the somatotype category (Carter & Heath, 1990) for a three-component rating."""
    ranked = sorted([("endomorph", endo), ("mesomorph", meso), ("ectomorph", ecto)], key=lambda c: c[1], reverse=True)
    (first, high), (second, mid), (_, low) = ranked

    # Differences are compared at the one-decimal precision the rating is reported to
    if round(high - low, 1) <= 1.0:
        return "Central"
    if round(high - mid, 1) <= 0.5:
        # Two components share dominance; named in the conventional order (mesomorph before the other, endomorph before ectomorph)
        pair = sorted([first, second], key=["mesomorph", "endomorph", "ectomorph"].index)
        return f"{pair[0]}-{pair[1]}".capitalize()
    if round(mid - low, 1) <= 0.5:
        return f"Balanced {first}"
    return f"{second}ic {first}".capitalize()

def rate(measurements: Anthropometry) -> Somatotype:
    """Rates every component the supplied measurements allow, and the category when all three are rated."""
    m = measurements
    missing = [f.name for f in fields(m) if getattr(m, f.name) is None]

    def have(*names: str) -> bool:
        return not any(name in missing for name in names)

    endo = meso = ecto = None
    if have("height_cm", "triceps_skinfold_mm", "subscapular_skinfold_mm", "supraspinale_skinfold_mm"):
        endo = round(endomorphy(m.height_cm, m.triceps_skinfold_mm, m.subscapular_skinfold_mm, m.supraspinale_skinfold_mm), 1)
    if have("height_cm", "humerus_breadth_cm", "femur_breadth_cm", "flexed_arm_girth_cm", "calf_girth_cm", "triceps_skinfold_mm", "medial_calf_skinfold_mm"):
        meso = round(mesomorphy(m.height_cm, m.humerus_breadth_cm, m.femur_breadth_cm, m.flexed_arm_girth_cm, m.calf_girth_cm, m.triceps_skinfold_mm, m.medial_calf_skinfold_mm), 1)
    if have("height_cm", "weight_kg"):
        ecto = round(ectomorphy(m.height_cm, m.weight_kg), 1)

    category = classify(endo, meso, ecto) if None not in (endo, meso, ecto) else None
    return Somatotype(endomorphy=endo, mesomorphy=meso, ectomorphy=ecto, category=category, missing=missing)
