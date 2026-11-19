"""IRCArtConverter integration tests (torch-dependent).

Runs the real node function on small synthetic images, exercising both
color algorithms and both block modes. IRC output is verified by
decoding it back to a color grid.
"""

import re
import unittest

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

HALF = "\u2580"
CC = "\x03"


def decode_line(line):
    grid = []
    fg = bg = None
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == CC:
            i += 1
            m = re.match(r"(\d{1,2})(?:,(\d{1,2}))?", line[i:])
            fg = int(m.group(1))
            if m.group(2) is not None:
                bg = int(m.group(2))
            i += m.end()
            continue
        grid.append((bg, bg) if ch != HALF else (fg, bg))
        i += 1
    return grid


def IRC_RGB(code):
    from src.irc import IRC_COLORS

    return IRC_COLORS[code]


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class ConverterLabAlgorithm(unittest.TestCase):
    def setUp(self):
        from src.nodes import IRCArtConverter

        self.converter = IRCArtConverter()

    def run_convert(self, img, **kw):
        params = dict(
            block_width=4,
            block_height=4,
            half_block_mode=False,
            use_16_colors=False,
            distance_method="perceptual_weighted",
            color_algorithm="lab",
        )
        params.update(kw)
        return self.converter.convert_to_irc_art(img, **params)

    def make_image(self, w, h, cells):
        """cells: list of (color, width_fraction) column bands."""
        arr = np.zeros((h, w, 3), dtype=np.float32)
        x = 0
        for color, frac in cells:
            end = x + int(w * frac)
            arr[:, x:end] = np.array(color, dtype=np.float32) / 255.0
            x = end
        return torch.from_numpy(arr).unsqueeze(0)

    def test_input_types_has_color_algorithm_default_lab(self):
        it = self.converter.INPUT_TYPES()
        spec = it["required"]["color_algorithm"]
        self.assertEqual(spec[0], ["lab", "kmeans"])
        self.assertEqual(spec[1]["default"], "lab")

    def test_solid_image_full_block(self):
        # 8x8 solid red image, 4x4 blocks -> 2 cells per row; red is
        # codes 4 and 52 (both (255,0,0) by design)
        img = self.make_image(8, 8, [((255, 0, 0), 1.0)])
        out_img, irc_text, ansi_text = self.run_convert(img)
        self.assertIn("\x1b", ansi_text)  # ANSI path still functional
        lines = irc_text.split("\n")
        self.assertEqual(len(lines), 2)  # 8 // 4 rows
        for line in lines:
            cells = decode_line(line)
            self.assertEqual(len(cells), 2)
            for top, bottom in cells:
                self.assertIn(top, (4, 52))
                self.assertEqual(top, bottom)

    def test_banded_image_half_block(self):
        # rows 0-1 red, rows 2-3 blue -> one text row of ▀ fg red bg blue
        img = self.make_image(8, 4, [((255, 0, 0), 1.0)])
        img[0, 2:, :] = torch.tensor([0.0, 0.0, 1.0])
        out_img, irc_text, ansi_text = self.run_convert(
            img, half_block_mode=True, block_height=4
        )
        lines = irc_text.split("\n")
        self.assertEqual(len(lines), 1)  # 4 // 4 text rows
        for top, bottom in decode_line(lines[0]):
            self.assertEqual(IRC_RGB(top), (255, 0, 0))
            self.assertEqual(IRC_RGB(bottom), (0, 0, 255))

    def test_16_color_mode(self):
        img = self.make_image(4, 4, [((200, 10, 10), 1.0)])
        _, irc_text, _ = self.run_convert(img, use_16_colors=True)
        top, bottom = decode_line(irc_text.split("\n")[0])[0]
        self.assertLessEqual(top, 15)

    def test_processed_image_is_palette_colored(self):
        from src.irc import IRC_COLORS

        img = self.make_image(8, 8, [((30, 200, 90), 1.0)])
        out_img, irc_text, _ = self.run_convert(img)
        # preview must be painted in palette colors only
        flat = (out_img[0].numpy() * 255.0).round().astype(int)
        colors = {tuple(c) for c in flat.reshape(-1, 3)}
        for c in colors:
            self.assertIn(c, IRC_COLORS)

    def test_line_lengths_never_exceed_512(self):
        rng = np.random.default_rng(7)
        arr = rng.random((64, 64, 3)).astype(np.float32)
        img = torch.from_numpy(arr).unsqueeze(0)
        _, irc_text, _ = self.run_convert(
            img, block_width=8, block_height=15, half_block_mode=True
        )
        for line in irc_text.split("\n"):
            self.assertLessEqual(len(line), 512, f"line exceeds 512 bytes: {len(line)}")

    def test_ragged_width_clamps_right_edge_blocks(self):
        # width 10 with 4px blocks -> 2 cells; the second block covers
        # only columns 8-9. The index-map slice must mirror extract_block
        # bounds exactly, not index past the edge.
        img = self.make_image(10, 4, [((0, 255, 0), 1.0)])
        out_img, irc_text, _ = self.run_convert(img, block_width=4, block_height=4)
        lines = irc_text.split("\n")
        self.assertEqual(len(lines), 1)  # 4 // 4 rows
        cells = decode_line(lines[0])
        self.assertEqual(len(cells), 2)  # 10 // 4 cols
        for top, bottom in cells:
            self.assertEqual(top, 56)  # exact palette green
            self.assertEqual(top, bottom)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class ConverterKmeansAlgorithm(unittest.TestCase):
    """The legacy pipeline stays selectable and functional."""

    def test_kmeans_still_works(self):
        from src.nodes import IRCArtConverter

        converter = IRCArtConverter()
        # legacy KMeans needs >= 99 pixels (one cluster per palette entry)
        arr = np.zeros((32, 32, 3), dtype=np.float32)
        arr[:, :] = np.array([255, 0, 0]) / 255.0
        img = torch.from_numpy(arr).unsqueeze(0)
        out = converter.convert_to_irc_art(
            img,
            block_width=8,
            block_height=8,
            half_block_mode=False,
            use_16_colors=False,
            distance_method="weighted_manhattan",
            color_algorithm="kmeans",
        )
        irc_text = out[1]
        top, bottom = decode_line(irc_text.split("\n")[0])[0]
        self.assertEqual(IRC_RGB(top), (255, 0, 0))


if __name__ == "__main__":
    unittest.main()
