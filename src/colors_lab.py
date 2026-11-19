"""CIELAB color matching for IRC palette quantization.

The "lab" color algorithm: convert sRGB to CIELAB (D65, 2 deg) and pick
the nearest palette entry by plain euclidean distance in Lab space,
which is exactly the CIE delta-E (1976) color difference. This replaces
perceptual weighting hacks in RGB space and needs no clustering: a
KD-tree over the (at most 99) palette points answers nearest-neighbor
queries for every pixel in one vectorized call.
"""

import numpy as np
from scipy.spatial import cKDTree

# sRGB (linearized) -> XYZ, D65
_XYZ_MATRIX = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)

# D65 reference white
_WHITE_POINT = np.array([0.95047, 1.0, 1.08883])

_DELTA = 6.0 / 29.0
_DELTA_CUBED = _DELTA**3
_DELTA_SCALED = 3.0 * _DELTA**2


def srgb_to_lab(rgb) -> np.ndarray:
    """Convert an (N, 3) array of sRGB values (0-255) to CIELAB.

    Out-of-range inputs are clamped so the function is total for any
    caller (the np.where power branch would otherwise warn on
    sufficiently negative values even though they are discarded).
    """
    rgb = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)
    srgb = np.clip(rgb / 255.0, 0.0, 1.0)
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    )
    xyz = (linear @ _XYZ_MATRIX.T) / _WHITE_POINT
    f = np.where(
        xyz > _DELTA_CUBED,
        np.cbrt(xyz),
        xyz / _DELTA_SCALED + 4.0 / 29.0,
    )
    lab = np.empty_like(f)
    lab[:, 0] = 116.0 * f[:, 1] - 16.0  # L
    lab[:, 1] = 500.0 * (f[:, 0] - f[:, 1])  # a
    lab[:, 2] = 200.0 * (f[:, 1] - f[:, 2])  # b
    return lab


def snap_indices(rgb_img, palette) -> np.ndarray:
    """Map every pixel of an (H, W, 3) sRGB image to a palette index.

    Returns an (H, W) int64 array of palette indices chosen by minimum
    delta-E (1976) against the palette.

    Duplicate palette entries (the extended proposal repeats white,
    black, red, yellow, cyan and magenta) are deduplicated before the
    tree is built and remapped to the FIRST (lowest) original index, so
    the result is canonical by construction rather than by KD-tree
    tie-breaking — matching the legacy palette.index() first-match rule.
    """
    img = np.asarray(rgb_img)
    pal = np.asarray(palette)
    unique_rgb, first_index = np.unique(pal, axis=0, return_index=True)
    tree = cKDTree(srgb_to_lab(unique_rgb))
    _, unique_idx = tree.query(srgb_to_lab(img.reshape(-1, 3)))
    return first_index[unique_idx].reshape(img.shape[:-1])


def majority_vote(indices) -> int:
    """Most frequent palette index in a block; ties pick the lowest index."""
    flat = np.asarray(indices).reshape(-1).astype(np.int64)
    return int(np.bincount(flat).argmax())
