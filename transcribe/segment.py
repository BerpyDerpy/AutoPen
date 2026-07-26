"""Find the handwriting area on a scanned page and split it into line images.

Nothing here is tied to a fixed page layout. The content box is found by
scanning inward from each edge, so scans with different borders, different
margins, or a border that runs off the page all still work.
"""

import numpy as np
from PIL import Image


# --- Tunable settings
# These are the only numbers to adjust if a batch of scans behaves oddly.

# A pixel counts as "dark" when its average brightness (0 = black, 255 = white)
# is at or below this.
DARK_MAX = 170

# A pixel counts as "gray" when the gap between its brightest and dimmest color
# channel is at or below this. Black pen ink sits close to gray. The printed
# teal border, teal section labels, and the teal watermark are far from gray,
# so they fail this test and never become ink.
COLOR_GAP_MAX = 40

# A row or column is treated as a printed border line when at least this
# fraction of its pixels are dark. A border line runs nearly edge to edge, so
# it scores far higher than any row of handwriting.
BORDER_FILL_MIN = 0.45

# How far in from an edge we look for a border, as a fraction of the page's
# width or height. Keeps the search in the margin area.
BORDER_SEARCH_FRACTION = 0.25

# Two border lines with a gap smaller than this (in pixels) are treated as one
# double-line border, so we crop inside the inner line rather than between them.
BORDER_LINE_GAP_MAX = 40

# Extra pixels trimmed off just inside the border, so no sliver of the printed
# line survives into the crop.
BORDER_INSET = 8

# Settings for the fallback used when a side has no printed border at all.
# We instead look for where real content starts: a stretch of rows or columns
# that each hold at least a little ink.
CONTENT_FILL_MIN = 0.002
CONTENT_RUN_MIN = 5
CONTENT_PAD = 10


def load_page(path):
    """Load a scan as a plain RGB pixel array of shape (height, width, 3)."""
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def build_masks(rgb):
    """Return two true/false maps for the page: dark pixels, and ink pixels.

    dark_mask is every dark pixel, printed teal included. It is what makes the
    printed border easy to spot.

    ink_mask is only dark pixels that are also close to gray, which is the
    user's black pen. Printed teal and the watermark are left out.
    """
    brightness = rgb.mean(axis=2)
    brightest_channel = rgb.max(axis=2).astype(np.int16)
    dimmest_channel = rgb.min(axis=2).astype(np.int16)
    color_gap = brightest_channel - dimmest_channel

    dark_mask = brightness <= DARK_MAX
    ink_mask = dark_mask & (color_gap <= COLOR_GAP_MAX)
    return dark_mask, ink_mask


def _find_filled_bands(values, limit, threshold):
    """Group consecutive entries of `values` that are at or above `threshold`.

    Only looks at entries before `limit`. Returns a list of (start, end) pairs,
    where end is one past the last entry in the band.
    """
    bands = []
    index = 0
    while index < limit:
        if values[index] >= threshold:
            start = index
            while index < limit and values[index] >= threshold:
                index += 1
            bands.append((start, index))
        else:
            index += 1
    return bands


def _content_start(ink_fill, limit):
    """Fallback edge finder for a side with no printed border.

    Walks inward and returns the index where a sustained stretch of ink begins,
    backed off a little so the content itself is not clipped.
    """
    run_length = 0
    for index in range(limit):
        if ink_fill[index] >= CONTENT_FILL_MIN:
            run_length += 1
            if run_length >= CONTENT_RUN_MIN:
                run_start = index - run_length + 1
                return max(0, run_start - CONTENT_PAD)
        else:
            run_length = 0
    return 0


def _inner_edge(dark_fill, ink_fill, limit):
    """Find where the content area begins, scanning inward from index 0.

    `dark_fill` and `ink_fill` are one number per row (or per column): the
    fraction of that row that is dark, and the fraction that is ink.
    """
    border_bands = _find_filled_bands(dark_fill, limit, BORDER_FILL_MIN)
    if not border_bands:
        return _content_start(ink_fill, limit)

    # Start at the end of the first border line, then absorb any further lines
    # that sit right beside it. This is what handles a double-line border.
    edge = border_bands[0][1]
    for start, end in border_bands[1:]:
        if start - edge <= BORDER_LINE_GAP_MAX:
            edge = end
        else:
            break
    return edge + BORDER_INSET


def find_content_box(dark_mask, ink_mask):
    """Find the handwriting area of the page.

    Returns (top, bottom, left, right) as pixel indices into the full page.
    """
    height, width = dark_mask.shape

    # One number per row: what fraction of that row is dark, and is ink.
    dark_by_row = dark_mask.mean(axis=1)
    ink_by_row = ink_mask.mean(axis=1)
    # Same thing per column.
    dark_by_column = dark_mask.mean(axis=0)
    ink_by_column = ink_mask.mean(axis=0)

    row_limit = int(height * BORDER_SEARCH_FRACTION)
    column_limit = int(width * BORDER_SEARCH_FRACTION)

    top = _inner_edge(dark_by_row, ink_by_row, row_limit)
    left = _inner_edge(dark_by_column, ink_by_column, column_limit)

    # For the bottom and right edges, reverse the numbers so the same
    # scan-inward code applies, then convert the result back.
    bottom = height - _inner_edge(dark_by_row[::-1], ink_by_row[::-1], row_limit)
    right = width - _inner_edge(dark_by_column[::-1], ink_by_column[::-1], column_limit)

    return top, bottom, left, right
