"""CIELAB color matching tests.

Reference Lab values are the standard sRGB -> CIELAB (D65, 2 deg)
conversions for pure primaries.
"""

import unittest

import numpy as np

from src.colors_lab import majority_vote, srgb_to_lab, snap_indices
from src.irc import IRC_COLORS


class SrgbToLab(unittest.TestCase):
    def test_white(self):
        lab = srgb_to_lab(np.array([[255, 255, 255]], dtype=np.uint8))
        self.assertAlmostEqual(float(lab[0, 0]), 100.0, places=3)
        self.assertAlmostEqual(float(lab[0, 1]), 0.0, places=3)
        self.assertAlmostEqual(float(lab[0, 2]), 0.0, places=3)

    def test_black(self):
        lab = srgb_to_lab(np.array([[0, 0, 0]], dtype=np.uint8))
        np.testing.assert_allclose(lab, [[0.0, 0.0, 0.0]], atol=1e-6)

    def test_primary_red(self):
        lab = srgb_to_lab(np.array([[255, 0, 0]], dtype=np.uint8))
        np.testing.assert_allclose(lab, [[53.2408, 80.0925, 67.2032]], atol=0.01)

    def test_mid_gray_is_neutral(self):
        lab = srgb_to_lab(np.array([[128, 128, 128]], dtype=np.uint8))
        np.testing.assert_allclose(lab, [[53.585, 0.0, 0.0]], atol=0.01)

    def test_shape_preserved_by_snap(self):
        img = np.zeros((4, 6, 3), dtype=np.uint8)
        idx = snap_indices(img, IRC_COLORS)
        self.assertEqual(idx.shape, (4, 6))

    def test_out_of_domain_input_is_clamped_not_nan(self):
        """Direct callers may pass out-of-range floats; the discarded
        np.where branch must not even evaluate to a warning."""
        import warnings

        bad = np.array([[-50.0, 300.0, 128.0]])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            lab = srgb_to_lab(bad)
        self.assertTrue(np.isfinite(lab).all())
        np.testing.assert_allclose(
            lab, srgb_to_lab(np.array([[0, 255, 128]], dtype=np.uint8)), atol=1e-9
        )


class SnapIndices(unittest.TestCase):
    def test_palette_colors_snap_to_themselves(self):
        """Every palette entry snaps to a code with an identical RGB."""
        pal = np.array(IRC_COLORS, dtype=np.uint8)
        idx = snap_indices(pal, IRC_COLORS)
        for i, code in enumerate(idx):
            self.assertEqual(
                tuple(pal[code]),
                IRC_COLORS[i],
                f"palette color {i} snapped to code {code} ({tuple(pal[code])})",
            )

    def test_exact_palette_primary(self):
        # (0,255,0) is palette code 56; (0,0,255) is code 60
        img = np.array([[[0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
        idx = snap_indices(img, IRC_COLORS)
        self.assertEqual(idx[0, 0], 56)
        self.assertEqual(idx[0, 1], 60)

    def test_near_red_snaps_to_red_family(self):
        img = np.array([[[250, 3, 1]]], dtype=np.uint8)
        idx = snap_indices(img, IRC_COLORS)
        self.assertEqual(int(idx[0, 0]), 4)  # canonical red, not dup 52

    def test_duplicate_palette_entries_canonicalize_to_lowest_index(self):
        """Red/white/black/etc have two codes each; the snap must return
        the lowest one by construction (legacy palette.index parity)."""
        dups = [
            (255, 0, 0),
            (255, 255, 255),
            (0, 0, 0),
            (255, 255, 0),
            (0, 255, 255),
            (255, 0, 255),
        ]
        expected = [4, 0, 1, 8, 11, 13]
        img = np.array([dups], dtype=np.uint8)
        idx = snap_indices(img, IRC_COLORS)
        self.assertEqual(idx[0].tolist(), expected)

    def test_gray_snaps_to_achromatic_entry(self):
        # any neutral gray must land on an achromatic palette entry
        # (mIRC grey 14 or the extended grayscale ramp 88-98)
        for g in (60, 120, 200):
            img = np.array([[[g, g, g]]], dtype=np.uint8)
            idx = snap_indices(img, IRC_COLORS)
            r, gr, b = IRC_COLORS[int(idx[0, 0])]
            self.assertEqual(r, gr)
            self.assertEqual(gr, b)

    def test_16_color_palette_masks(self):
        img = np.array([[[255, 0, 0]]], dtype=np.uint8)
        idx = snap_indices(img, IRC_COLORS[:16])
        self.assertEqual(int(idx[0, 0]), 4)


class MajorityVote(unittest.TestCase):
    def test_majority_wins(self):
        self.assertEqual(majority_vote(np.array([5, 5, 2, 9, 5])), 5)

    def test_tie_picks_smallest_index(self):
        self.assertEqual(majority_vote(np.array([7, 7, 2, 2])), 2)

    def test_single_entry(self):
        self.assertEqual(majority_vote(np.array([42])), 42)


if __name__ == "__main__":
    unittest.main()
