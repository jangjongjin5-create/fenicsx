from __future__ import annotations

from pathlib import Path

import numpy as np


BASE = {"tau_brug": 1.1, "Lx": 0.20e-3}
TRAIN_CONFIGS = {"base": dict(BASE)}
TAU_TRUE = 1.1
TAU_GRADCHECK = 0.9
ETA_MAX = 0.405
ETA_MASTER = np.concatenate(
    [np.linspace(0.10, 0.36, 14), [0.375, 0.385, 0.390, 0.395, 0.400, 0.405]]
)
ETA_QUICK = np.array(
    [0.10, 0.20, 0.30, 0.34, 0.36, 0.38, 0.385, 0.390, 0.395, 0.400, 0.405]
)
ETA_GRADCHECK = np.array([0.390, 0.395, 0.400, 0.405])

DETA_WET = 0.005
DETA_VERY_WET = 0.0025
DETA_MIN = 2.5e-4
BRANCH_JUMP_MAX = 0.10

NOISE_REL = 0.0
NOISE_TEST_REL = 0.01
RNG_SEED = 7
GT_SCHEMA_VERSION = 1
QUADRATURE_DEGREE = 4
QUADRATURE_DIAGNOSTICS = (4, 6, 8)
S_NORM_FLOOR = 0.05

TAU_VALUES = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
TAU_LANDSCAPE = np.linspace(0.6, 1.2, 9)
TAU_LANDSCAPE_QUICK = np.linspace(0.6, 1.2, 5)
RECOVERY_STARTS = np.array([1.00, 1.05, 1.15, 1.20])
GRADCHECK_EPS = (1e-3, 5e-4, 1e-4)
GRADCHECK_REL_TOL = 0.01
GRADCHECK_PREFERRED_TOL = 0.001
SENSITIVITY_MIN_CONTRAST = 1e-5
TRUTH_NUMERICAL_ZERO_TOL = 1e-10
LANDSCAPE_MIN_VARIATION = 1e-6
LANDSCAPE_TRUTH_TOL = 0.08
RECOVERY_TAU_TOL = 0.05

OBSERVABLES = {
    "I-only": (0.0, 0.0, 1.0),
    "s-only": (1.0, 0.0, 0.0),
    "O2-only": (0.0, 1.0, 0.0),
    "s+I": (1.0, 0.0, 1.0),
    "s+O2": (1.0, 1.0, 0.0),
    "all": (1.0, 1.0, 1.0),
}

FUNCTIONAL_HIDDEN = 24
FUNCTIONAL_QUAD = 16
TRAIN_EPOCHS = 180
TRAIN_EPOCHS_QUICK = 30
TRAIN_LR = 2e-3
TRAIN_PATIENCE = 30
TRAIN_PATIENCE_QUICK = 8
TRAIN_GRAD_CLIP = 1.0
OPERATOR_DIRECTIONS = 3
OPERATOR_DIRECTIONS_QUICK = 2
OPERATOR_EPS = 1e-4

MESH_ROBUSTNESS = {
    "coarse": {"nx": 16, "ny": 40},
    "base": {"nx": 24, "ny": 60},
    "fine": {"nx": 32, "ny": 80},
}

OUT_DIR = Path(__file__).resolve().parent / "brug_out"
GT_PATH = OUT_DIR / "gt_noise_free.pkl"
GT_QUICK_PATH = OUT_DIR / "gt_noise_free_quick.pkl"
MODEL_PATH = OUT_DIR / "brug_net_best.pt"
MODEL_QUICK_PATH = OUT_DIR / "brug_net_best_quick.pt"


def quickify(overrides: dict[str, float | int]) -> dict[str, float | int]:
    result = dict(overrides)
    result["nx"] = max(8, int(result.get("nx", 24)) // 2)
    result["ny"] = max(16, int(result.get("ny", 60)) // 2)
    return result   
# =====================================================================
# Multi-condition RH extension
# =====================================================================
MULTICOND_OUT = OUT_DIR / "multicond_gt"

# Additional humidified operating points appended to DEFAULT_PARAMS
# baseline RH. Baseline itself is read from pemfc_lib at runtime.
MULTICOND_RH_EXTRA = (0.50, 0.85)

# Candidate eta envelope; each condition truncates at its own pre-fold limit.
MULTICOND_ETA = ETA_MASTER
MULTICOND_ETA_QUICK = ETA_QUICK

# Previous single-condition constitutive support, for comparison.
SUPPORT_BASELINE_PHI = (0.241, 0.297)

SUPPORT_BINS = 24

# Greedy training-snapshot recommendation target.
SUPPORT_TARGET_PER_BIN = 3

# =====================================================================
# Dual landscape configuration
# =====================================================================

# Broad/common landscape:
# all candidate tau values compared on a common pre-fold feasible eta range.
ETA_LANDSCAPE_COMMON_MAX = 0.360

# Wet near-truth landscape:
# retain the informative high-eta flooding states.
ETA_LANDSCAPE_WET_MAX = 0.405

TAU_LANDSCAPE_COMMON = np.array([
    1.00, 1.025, 1.05, 1.075, 1.10,
    1.125, 1.15, 1.175, 1.20
])

# Initial near-truth window.
# If a candidate cannot reach eta=0.405, narrow this range later.
TAU_LANDSCAPE_WET = np.array([
    1.05, 1.075, 1.10, 1.105, 1.11
])

# =====================================================================
# APPEND-ONLY block for configs_brug.py  (multi-condition RH extension)
# Paste at the end of the existing configs_brug.py. Nothing above changes.
# =====================================================================
MULTICOND_OUT = OUT_DIR / "multicond_gt"
# Additional humidified operating points appended to the DEFAULT_PARAMS
# baseline RH (baseline is read from pemfc_lib at run time, never hardcoded).
MULTICOND_RH_EXTRA = (0.50, 0.85)
# Candidate eta envelope; each condition truncates at its own pre-fold limit.
MULTICOND_ETA = ETA_MASTER
MULTICOND_ETA_QUICK = ETA_QUICK
# Prior single-condition constitutive support (task section 8) for comparison.
SUPPORT_BASELINE_PHI = (0.241, 0.297)
SUPPORT_BINS = 24
# greedy training-snapshot recommender: target fill per phi_g bin
SUPPORT_TARGET_PER_BIN = 3

# === BEGIN DUAL LANDSCAPE CONFIG ===

# Broad apples-to-apples landscape.
# tau=1.2 loses its branch near eta~0.375, so all broad candidates
# are compared only inside the common feasible pre-fold window.
ETA_LANDSCAPE_COMMON_MAX = 0.360

TAU_LANDSCAPE_COMMON = np.array([
    1.000,
    1.025,
    1.050,
    1.075,
    1.100,
    1.125,
    1.150,
    1.175,
    1.200,
], dtype=float)

# Local wet landscape.
# Keep the informative full GT up to eta=0.405, but only in the tiny
# truth-centered tau window already relevant to the differential
# gradcheck. This is NOT the broad parameter landscape.
ETA_LANDSCAPE_WET_MAX = 0.405

TAU_LANDSCAPE_WET = np.array([
    1.099,
    1.100,
    1.101,
], dtype=float)

# === END DUAL LANDSCAPE CONFIG ===

