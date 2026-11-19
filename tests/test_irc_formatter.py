"""IRC formatter tests: stateful delta color-code emission.

Emission rules (per modern.ircdocs.horse/formatting):
- \x03<fg>,<bg> sets both; \x03<fg> sets fg and KEEPS the current bg;
  there is no bg-only form, so a bg change must resend fg.
- Identical consecutive runs emit characters only (no code).
- Spaces with fg == bg force fg to 0 (shortest fg digit string).
- The first run of a line always uses the full form (start-of-message
  color state is unknown).
"""

import random
import re
import unittest

from src.irc import IRCFormatter

HALF = "\u2580"  # ▀
CC = "\x03"


class GoldenCases(unittest.TestCase):
    def fmt(self, line):
        return IRCFormatter.format_line(line)

    def test_line_start_always_full_form(self):
        self.assertEqual(self.fmt([(1, 2)]), f"{CC}1,2{HALF}")

    def test_two_digit_colors(self):
        self.assertEqual(self.fmt([(52, 88)]), f"{CC}52,88{HALF}")

    def test_run_of_identical_entries_is_one_code(self):
        self.assertEqual(self.fmt([(1, 2)] * 3), f"{CC}1,2{HALF}{HALF}{HALF}")

    def test_same_state_after_run_emits_no_code(self):
        # (3,3) spaces then a bare int 3: identical effective color
        self.assertEqual(self.fmt([(3, 3), 3, 3]), f"{CC}0,3   ")

    def test_space_run_uses_fg_zero_trick(self):
        self.assertEqual(self.fmt([(4, 4), (4, 4)]), f"{CC}0,4  ")

    def test_bg_change_resends_fg(self):
        self.assertEqual(self.fmt([(1, 2), (1, 3)]), f"{CC}1,2{HALF}{CC}1,3{HALF}")

    def test_fg_change_with_same_bg_is_fg_only(self):
        self.assertEqual(self.fmt([(1, 2), (5, 2)]), f"{CC}1,2{HALF}{CC}5{HALF}")

    def test_two_digit_fg_only_form(self):
        self.assertEqual(self.fmt([(52, 88), (60, 88)]), f"{CC}52,88{HALF}{CC}60{HALF}")

    def test_space_to_block_same_bg_is_fg_only(self):
        # the common half-block art transition: background row color
        # stays, only the top (fg) color changes
        self.assertEqual(self.fmt([(3, 3), (7, 3)]), f"{CC}0,3 {CC}7{HALF}")

    def test_block_to_space_same_bg_is_fg_only(self):
        # space run keeps bg=3, only fg resets to 0
        self.assertEqual(self.fmt([(7, 3), (3, 3)]), f"{CC}7,3{HALF}{CC}0 ")

    def test_block_to_space_bg_change_full_form(self):
        self.assertEqual(self.fmt([(7, 3), (5, 5)]), f"{CC}7,3{HALF}{CC}0,5 ")

    def test_full_block_mode_ints(self):
        self.assertEqual(self.fmt([9, 9, 12]), f"{CC}0,9  {CC}0,12 ")

    def test_empty_line(self):
        self.assertEqual(self.fmt([]), "")


def decode_line(line):
    """Decode formatter output back into a (fg, bg) grid.

    Inverse of IRCFormatter.format_line: spaces carry their bg color as
    both cells, half-blocks carry fg as top and bg as bottom.
    """
    grid = []
    fg = bg = None
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == CC:
            i += 1
            m = re.match(r"(\d{1,2})(?:,(\d{1,2}))?", line[i:])
            if not m:
                # bare \x03 resets both — never emitted by the formatter
                raise AssertionError(f"unexpected bare color code at {i}")
            fg = int(m.group(1))
            if m.group(2) is not None:
                bg = int(m.group(2))
            i += m.end()
            continue
        if ch == HALF:
            grid.append((fg, bg))
        else:
            grid.append((bg, bg))
        i += 1
    return grid


def normalize(entries):
    """Normalize input entries to the decoder's (fg, bg) tuple form."""
    out = []
    for e in entries:
        if isinstance(e, tuple):
            out.append(tuple(e))
        else:
            out.append((e, e))
    return out


class RoundtripProperty(unittest.TestCase):
    def test_random_grids_roundtrip(self):
        rng = random.Random(20261119)
        for trial in range(200):
            n = rng.randint(1, 120)
            grid = []
            for _ in range(n):
                r = rng.random()
                if r < 0.3:
                    c = rng.randint(0, 98)
                    grid.append((c, c))  # space cell
                elif r < 0.6:
                    grid.append(rng.randint(0, 98))  # full-block int
                else:
                    grid.append((rng.randint(0, 98), rng.randint(0, 98)))
            line = IRCFormatter.format_line(grid)
            with self.subTest(trial=trial, grid=grid, line=repr(line)):
                self.assertEqual(decode_line(line), normalize(grid))

    def test_fg_only_never_longer_than_full_form(self):
        """Delta emission can never exceed one full code per run."""
        rng = random.Random(42)
        grid = [(rng.randint(0, 98), rng.randint(0, 98)) for _ in range(80)]
        line = IRCFormatter.format_line(grid)
        # worst case: one code per cell; full form is at most 6 bytes of code
        codes = line.count(CC)
        self.assertLessEqual(codes, 80)
        self.assertLessEqual(len(line), 80 * (1 + 6))


if __name__ == "__main__":
    unittest.main()
