"""Write preview images so the segmentation can be checked by eye.

Run from the project root:

    .venv/bin/python transcribe/debug_preview.py

For every scan in Scans/ this writes two files into debug/:
  <name>_box.png   the page shrunk down, with the detected content box in red
  <name>_ink.png   the content box's ink mask, black ink on white

Nothing here is used by the real tool. It exists only to check the settings
at the top of segment.py.
"""

import os

import numpy as np
from PIL import Image, ImageDraw

import segment


SCANS_DIR = "Scans"
DEBUG_DIR = "debug"

# The scans are big, so previews are shrunk by this much before saving.
PREVIEW_SHRINK = 3


def preview_one_scan(scan_name):
    path = os.path.join(SCANS_DIR, scan_name)
    rgb = segment.load_page(path)
    dark_mask, ink_mask = segment.build_masks(rgb)
    top, bottom, left, right = segment.find_content_box(dark_mask, ink_mask)

    height, width = dark_mask.shape
    print(
        "%s  page %dx%d  box top=%d bottom=%d left=%d right=%d"
        % (scan_name, width, height, top, bottom, left, right)
    )

    base_name = os.path.splitext(scan_name)[0]

    # Preview 1: the whole page with the detected box drawn on it.
    page_image = Image.fromarray(rgb)
    drawing = ImageDraw.Draw(page_image)
    drawing.rectangle([left, top, right, bottom], outline=(255, 0, 0), width=8)
    shrunk = page_image.resize((width // PREVIEW_SHRINK, height // PREVIEW_SHRINK))
    shrunk.save(os.path.join(DEBUG_DIR, base_name + "_box.png"))

    # Preview 2: what survived the ink test inside the box.
    box_ink = ink_mask[top:bottom, left:right]
    # True means ink, and we want ink drawn black, so flip it before saving.
    ink_picture = np.where(box_ink, 0, 255).astype(np.uint8)
    ink_image = Image.fromarray(ink_picture)
    ink_image = ink_image.resize(
        (ink_image.width // PREVIEW_SHRINK, ink_image.height // PREVIEW_SHRINK)
    )
    ink_image.save(os.path.join(DEBUG_DIR, base_name + "_ink.png"))


def main():
    os.makedirs(DEBUG_DIR, exist_ok=True)
    for scan_name in sorted(os.listdir(SCANS_DIR)):
        if scan_name.lower().endswith(".png"):
            preview_one_scan(scan_name)


if __name__ == "__main__":
    main()
