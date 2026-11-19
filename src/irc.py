"""IRC-side color palette and text formatting.

Pure Python (no torch/sklearn) so it stays unit-testable and importable
anywhere. Palette values follow the extended mIRC colour proposal as
documented at modern.ircdocs.horse/formatting: codes 0-15 are the
traditional mIRC client defaults, codes 16-98 the extended table.

Emission rules (IRCFormatter.format_line):
- \\x03<fg>,<bg> sets both colors; \\x03<fg> sets fg and keeps the
  current background (modern.ircdocs.horse: "If only the foreground
  color is set, the background color stays the same"). There is no
  bg-only form, so a background change must resend the foreground.
- Runs whose effective (fg, bg) equals the current state emit characters
  only; a run that changes only the foreground emits the short fg-only
  form; anything else emits the full form.
- Spaces drawn with fg == bg force fg to 0: \\x030,<bg> is the shortest
  unambiguous encoding of a background-colored space.
- The first run of each line always uses the full form: each line is its
  own PRIVMSG and clients reset formatting per message, so no state may
  be assumed at line start and none needs to be reset at line end.
"""

SPACE_CHAR = " "
HALF_BLOCK_CHAR = "\u2580"
COLOR_CODE = "\x03"

# 99-color IRC palette.
# Codes 0-15: mIRC client defaults. Codes 16-98: extended mIRC colour
# proposal (modern.ircdocs.horse/formatting, "Extended Colors").
# Code 38 is #740047 = (116, 0, 71); an earlier revision of this list
# carried the transcription error (116, 0, 116) = #740074.
# Duplicate RGBs (0/98, 1/88, 4/52, 8/54, 11/58, 13/62) exist in the
# proposal by design.
IRC_COLORS = [
    (255, 255, 255),
    (0, 0, 0),
    (0, 0, 127),
    (0, 147, 0),
    (255, 0, 0),
    (127, 0, 0),
    (156, 0, 156),
    (252, 127, 0),
    (255, 255, 0),
    (0, 252, 0),
    (0, 147, 147),
    (0, 255, 255),
    (0, 0, 252),
    (255, 0, 255),
    (127, 127, 127),
    (210, 210, 210),
    (71, 0, 0),
    (71, 33, 0),
    (71, 71, 0),
    (50, 71, 0),
    (0, 71, 0),
    (0, 71, 44),
    (0, 71, 71),
    (0, 39, 71),
    (0, 0, 71),
    (46, 0, 71),
    (71, 0, 71),
    (71, 0, 42),
    (116, 0, 0),
    (116, 58, 0),
    (116, 116, 0),
    (81, 116, 0),
    (0, 116, 0),
    (0, 116, 73),
    (0, 116, 116),
    (0, 64, 116),
    (0, 0, 116),
    (75, 0, 116),
    (116, 0, 71),
    (116, 0, 69),
    (181, 0, 0),
    (181, 99, 0),
    (181, 181, 0),
    (125, 181, 0),
    (0, 181, 0),
    (0, 181, 113),
    (0, 181, 181),
    (0, 99, 181),
    (0, 0, 181),
    (117, 0, 181),
    (181, 0, 181),
    (181, 0, 107),
    (255, 0, 0),
    (255, 140, 0),
    (255, 255, 0),
    (178, 255, 0),
    (0, 255, 0),
    (0, 255, 160),
    (0, 255, 255),
    (0, 140, 255),
    (0, 0, 255),
    (165, 0, 255),
    (255, 0, 255),
    (255, 0, 152),
    (255, 89, 89),
    (255, 180, 89),
    (255, 255, 113),
    (207, 255, 96),
    (111, 255, 111),
    (101, 255, 201),
    (109, 255, 255),
    (89, 180, 255),
    (89, 89, 255),
    (196, 89, 255),
    (255, 102, 255),
    (255, 89, 188),
    (255, 156, 156),
    (255, 211, 156),
    (255, 255, 156),
    (226, 255, 156),
    (156, 255, 156),
    (156, 255, 219),
    (156, 255, 255),
    (156, 211, 255),
    (156, 156, 255),
    (220, 156, 255),
    (255, 156, 255),
    (255, 148, 211),
    (0, 0, 0),
    (19, 19, 19),
    (40, 40, 40),
    (54, 54, 54),
    (77, 77, 77),
    (101, 101, 101),
    (129, 129, 129),
    (159, 159, 159),
    (188, 188, 188),
    (226, 226, 226),
    (255, 255, 255),
]


class TextFormatter:
    """Base class for text formatting operations."""

    @staticmethod
    def count_consecutive(items, index: int) -> int:
        """Count consecutive identical entries starting from index."""
        if index >= len(items):
            return 0

        count = 1
        current = items[index]

        while index + count < len(items) and items[index + count] == current:
            count += 1

        return count


class IRCFormatter(TextFormatter):
    """IRC text formatting with stateful delta color-code emission."""

    @staticmethod
    def format_digits(n: int) -> str:
        """Minimal 1-2 digit form (a color is 1 or 2 ASCII digits).

        Single digits are safe here: the characters following a code are
        only ever spaces or the half-block char, never digits, so a
        one-digit color can never be extended by the text that follows
        (the "mistaken eating of text" hazard in the spec).
        """
        return f"{n:02d}" if n >= 10 else str(n)

    @staticmethod
    def format_line(line_colors) -> str:
        """Format a line of block colors, emitting minimal color codes.

        line_colors entries are ints (full-block: one color per cell,
        rendered as background-colored spaces) or (top, bottom) tuples
        (half-block: fg over bg; top == bottom renders as spaces).
        """
        if not line_colors:
            return ""

        result = []
        cur_fg = None  # unknown start-of-line state (fresh PRIVMSG)
        cur_bg = None
        i = 0
        while i < len(line_colors):
            entry = line_colors[i]
            if isinstance(entry, tuple):
                top, bottom = entry
                if top == bottom:
                    fg, bg, ch = 0, top, SPACE_CHAR
                else:
                    fg, bg, ch = top, bottom, HALF_BLOCK_CHAR
            else:
                fg, bg, ch = 0, entry, SPACE_CHAR

            count = IRCFormatter.count_consecutive(line_colors, i)

            if fg == cur_fg and bg == cur_bg:
                prefix = ""
            elif bg == cur_bg and cur_bg is not None:
                # background unchanged: the fg-only form keeps it
                prefix = f"{COLOR_CODE}{IRCFormatter.format_digits(fg)}"
            else:
                prefix = (
                    f"{COLOR_CODE}{IRCFormatter.format_digits(fg)},"
                    f"{IRCFormatter.format_digits(bg)}"
                )

            result.append(prefix + ch * count)
            cur_fg, cur_bg = fg, bg
            i += count

        return "".join(result)
