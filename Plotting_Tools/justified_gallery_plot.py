import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageOps

"""
No-crop, variable-size photo mosaic (justified layout) into a fixed 3:2 canvas,
with explicit outer margins on ALL sides.

Key properties:
- NO crop, NO letterboxing: each tile has the same aspect ratio as its image.
- Variable-sized tiles (justified gallery layout).
- Packs into an inner rectangle (canvas minus margins), then offsets/centers.
- Pillow version compatible: works with older Pillow that lacks Image.Resampling.

Example (600 dpi, 0.2 inch margin, 18x12 inches):
  18*600 = 10800 px, 12*600 = 7200 px, margin = 0.2*600 = 120 px

  python mosaic_justified_margin.py /path/to/photos \
      --out poster_18x12_600dpi.tif \
      --width 10800 --height 7200 --dpi 600 --margin-in 0.2 --gap 30 --seed 7 --inner-align center
"""

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_LANCZOS = Image.LANCZOS


@dataclass
class Item:
    path: Path
    r: float  # aspect ratio w/h


@dataclass
class Placed:
    path: Path
    x: int
    y: int
    w: int
    h: int


def list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def read_aspect_ratio(path: Path) -> float:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        w, h = im.size
    return w / max(1, h)


def layout_justified(
    items: List[Item],
    W: int,
    gap: int,
    target_row_h: float,
    justify_last_row: bool = True,
) -> Tuple[List[Placed], int]:
    """
    Build a justified layout inside width W.

    target_row_h controls row breaking; each justified row height is computed so
    the row fills W exactly (up to a rounding correction on the last tile).

    Returns:
      placed rects (inner coords)
      used_h: total packed height excluding trailing gap
    """
    placed: List[Placed] = []
    y = 0
    row: List[Item] = []
    row_sum_r = 0.0

    def flush_row(is_last: bool) -> None:
        nonlocal y, row, row_sum_r, placed
        if not row:
            return

        gaps_total = gap * (len(row) - 1)
        avail_w = W - gaps_total
        if avail_w <= 1:
            return

        if is_last and not justify_last_row:
            # Ragged last row: keep target height
            h = max(1, int(round(target_row_h)))
            widths = [max(1, int(round(it.r * h))) for it in row]
            # If it doesn't fit, shrink to fit
            total_w = sum(widths)
            if total_w > avail_w:
                h = max(1, int(round(avail_w / max(1e-9, row_sum_r))))
                widths = [max(1, int(round(it.r * h))) for it in row]
        else:
            # Justified: choose h so widths fill avail_w
            h = max(1, int(round(avail_w / max(1e-9, row_sum_r))))
            widths = [max(1, int(round(it.r * h))) for it in row]

            # Rounding correction: force exact width fill
            total_w = sum(widths)
            delta = avail_w - total_w
            widths[-1] = max(1, widths[-1] + delta)

        x = 0
        for it, w in zip(row, widths):
            placed.append(Placed(it.path, x, y, w, h))
            x += w + gap

        y += h + gap
        row.clear()
        row_sum_r = 0.0

    # Build rows
    for it in items:
        row.append(it)
        row_sum_r += it.r

        gaps_total = gap * (len(row) - 1)
        predicted_w = row_sum_r * target_row_h + gaps_total
        if predicted_w >= W:
            flush_row(is_last=False)

    flush_row(is_last=True)

    used_h = max(0, y - gap)
    return placed, used_h


def solve_layout_to_fit(
    items: List[Item],
    W: int,
    H: int,
    gap: int,
    justify_last_row: bool,
    samples: int = 140,
) -> Tuple[List[Placed], int, float]:
    """
    Robust solver: try many target row heights and pick the best layout that FITS.

    Priority:
      1) Choose layout with used_h <= H and minimal slack (H-used_h).
      2) If none fit, choose layout with minimal overflow (used_h-H).

    Returns: (placed, used_h, chosen_target_row_h)
    """
    if not items:
        return [], 0, 0.0

    # Reasonable target row height range:
    # lower bound: small but not absurd (affects row breaks)
    # upper bound: at most H (one very tall row)
    min_h = max(20.0, H / 40.0)          # heuristic
    max_h = float(max(40, H))            # allow big
    if min_h >= max_h:
        min_h = 20.0
        max_h = float(max(40, H))

    # Use geometric spacing to cover range well
    def geom_space(a: float, b: float, n: int) -> List[float]:
        if n <= 1:
            return [a]
        ratio = (b / a) ** (1.0 / (n - 1))
        vals = [a * (ratio ** i) for i in range(n)]
        return vals

    candidates = geom_space(min_h, max_h, samples)

    best_fit: Optional[Tuple[List[Placed], int, float, int]] = None   # slack
    best_over: Optional[Tuple[List[Placed], int, float, int]] = None  # overflow

    for t in candidates:
        placed, used_h = layout_justified(items, W, gap, t, justify_last_row=justify_last_row)

        if used_h <= H:
            slack = H - used_h
            if best_fit is None or slack < best_fit[3]:
                best_fit = (placed, used_h, t, slack)
        else:
            overflow = used_h - H
            if best_over is None or overflow < best_over[3]:
                best_over = (placed, used_h, t, overflow)

    if best_fit is not None:
        placed, used_h, t, _ = best_fit
        return placed, used_h, t

    # Fallback: nothing fit (rare unless N is huge / gap too big / constraints too tight)
    placed, used_h, t, _ = best_over
    return placed, used_h, t


def render(
    placed: List[Placed],
    out_path: Path,
    W: int,
    H: int,
    bg: Tuple[int, int, int],
    dpi: int,
) -> None:
    canvas = Image.new("RGB", (W, H), color=bg)

    for p in placed:
        with Image.open(p.path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            tile = im.resize((p.w, p.h), resample=RESAMPLE_LANCZOS)  # exact resize, no crop
            canvas.paste(tile, (p.x, p.y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        canvas.save(out_path, quality=95, subsampling=0, optimize=True, dpi=(dpi, dpi))
    elif ext in {".tif", ".tiff"}:
        canvas.save(out_path, compression="tiff_lzw", dpi=(dpi, dpi))
    else:
        canvas.save(out_path, dpi=(dpi, dpi))


def parse_args():
    ap = argparse.ArgumentParser(description="No-crop justified-gallery mosaic with margins + alignment.")
    ap.add_argument("folder", type=str, help="Folder containing images")
    ap.add_argument("--out", type=str, default="poster.tif", help="Output file (tif/png/jpg)")
    ap.add_argument("--width", type=int, default=6000, help="Canvas width in px")
    ap.add_argument("--height", type=int, default=9000, help="Canvas height in px")
    ap.add_argument("--gap", type=int, default=30, help="Gap between images in px")
    ap.add_argument("--dpi", type=int, default=600, help="DPI metadata for printing")
    ap.add_argument("--bg", type=str, default="255,255,255", help="Background RGB, e.g. 255,255,255")
    ap.add_argument("--seed", type=int, default=0, help="Shuffle seed (0 keeps sorted order)")
    ap.add_argument("--limit", type=int, default=None, help="Use only first N images (after sorting)")

    ap.add_argument("--margin-in", type=float, default=0.2, help="Outer margin in inches (default 0.2)")
    ap.add_argument("--margin-px", type=int, default=None, help="Override outer margin in pixels")

    ap.add_argument(
        "--inner-align",
        choices=["top", "center", "bottom"],
        default="center",
        help="How to place the collage inside the inner rectangle (default: center).",
    )

    ap.add_argument(
        "--no-justify-last-row",
        action="store_true",
        help="Do not force last row to fill width (usually leave OFF for strict filling).",
    )

    ap.add_argument(
        "--samples",
        type=int,
        default=140,
        help="How many candidate target row heights to try (higher = more robust, slower).",
    )

    return ap.parse_args()


def main():
    args = parse_args()
    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    bg = tuple(int(x) for x in args.bg.split(","))
    if len(bg) != 3:
        raise ValueError("--bg must be like 255,255,255")

    margin_px = args.margin_px if args.margin_px is not None else int(round(args.margin_in * args.dpi))
    margin_px = max(0, margin_px)

    inner_W = args.width - 2 * margin_px
    inner_H = args.height - 2 * margin_px
    if inner_W <= 0 or inner_H <= 0:
        raise ValueError("Margin too large for the chosen canvas size.")

    paths = list_images(folder)
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise ValueError("No images found in folder.")

    items = [Item(p, read_aspect_ratio(p)) for p in paths]
    if args.seed != 0:
        rng = random.Random(args.seed)
        rng.shuffle(items)

    placed_inner, used_h, chosen_t = solve_layout_to_fit(
        items,
        W=inner_W,
        H=inner_H,
        gap=args.gap,
        justify_last_row=not args.no_justify_last_row,
        samples=args.samples,
    )

    # Slack is guaranteed >= 0 if we found a fitting layout
    slack = inner_H - used_h
    if slack < 0:
        # This can only happen if no candidate fit; we'll pin to top to avoid violating margins.
        slack = 0

    if args.inner_align == "top":
        y_in = 0
    elif args.inner_align == "bottom":
        y_in = slack
    else:  # center
        y_in = slack // 2

    x_offset = margin_px
    y_offset = margin_px + y_in

    placed: List[Placed] = []
    for p in placed_inner:
        placed.append(Placed(p.path, p.x + x_offset, p.y + y_offset, p.w, p.h))

    render(
        placed=placed,
        out_path=Path(args.out),
        W=args.width,
        H=args.height,
        bg=bg,
        dpi=args.dpi,
    )

    print(f"Saved: {args.out}")
    print(f"- Canvas: {args.width} x {args.height} px")
    print(f"- DPI: {args.dpi}")
    print(f"- Margin: {margin_px}px ({margin_px / args.dpi:.4f} in)")
    print(f"- Inner: {inner_W} x {inner_H} px")
    print(f"- Used inner height: ~{used_h}px (slack {inner_H - used_h}px)")
    print(f"- Gap: {args.gap}px")
    print(f"- Images (input): {len(items)}")
    print(f"- Inner align: {args.inner_align}")
    print(f"- Justify last row: {not args.no_justify_last_row}")
    print(f"- Chosen target_row_h (heuristic): {chosen_t:.2f}")
    print(f"- Samples tried: {args.samples}")


if __name__ == "__main__":
    main()