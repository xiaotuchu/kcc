from PIL import Image, ImageOps
import numpy as np
from math import ceil, radians, tan

'''
Gutter split parameters
A pixel is white when its gray value is >= 'white_threshold'.
A straight line across the whole page is a gutter line when at least 'gutter_white_ratio' of its pixels are white.
Gutters may be slanted up to 'max_gutter_angle' degrees; they are searched on a copy scaled down to 'detection_width'.
Consecutive gutter lines form a gutter only when at least max(3px, 'min_gutter_height' * page height) thick.
Leading/trailing regions shorter than 'noise_band_height' and darker than 'noise_band_dark_ratio' are dropped (page numbers).
A side column is a narrow column of text along the left or right edge (at most 'side_column_width' of the page),
set apart from the panels by a full-height white gap and having at least 'side_column_gap_rows' of empty rows.
Inside it, a cut moves to the nearest empty row between characters, at most 'side_column_shift' of the page height away.
'''
white_threshold = 235
gutter_white_ratio = 0.98
min_gutter_height = 0.006 # percent
max_gutter_angle = 15 # degrees
detection_width = 400 # px
noise_band_height = 0.05 # percent
noise_band_dark_ratio = 0.08
side_column_width = 0.1 # percent
side_column_gap_rows = 0.15 # percent
side_column_shift = 0.03 # percent
merge_tolerance = 1.3
strip_padding = 4 # px


'''
Splits a page into strips along full-width empty gutters (horizontal or slanted), so each strip can be read
on a device held in landscape. Cuts never go through panels or speech balloons; across a slanted gutter the
part of the neighbouring panel inside the strip's bounding box is filled with the background color.
Content that is too tall and has no gutter is never forced apart: it stays in one strip.

    Parameters:
            img (PIL image): A PIL image.
            screen_size (2-tuple, width|height): Device screen resolution.
            background_color (string): 'white' for white background, anything else for black.
    Returns:
            strips (list of PIL images): Strips from top to bottom. Empty when the page can't be split
            (no gutter to cut along, or a blank page) and should be kept as a regular page.
'''
def get_gutter_split_strips(img, screen_size, background_color='white'):
    fill = 'white' if background_color == 'white' else 'black'
    return [cut_strip(img, top_y, bot_y, top, bot, fill)
            for top_y, bot_y, top, bot in plan_gutter_split(img, screen_size, background_color)]


'''
Plans the strips of a page.

    Returns:
            plans (list of 4-tuple, top boundary|bottom boundary|top|bot): Each strip spans rows [top, bot) and keeps
            only the pixels between its two boundaries. A boundary is the y of the cut for every column.
            Empty when the page should be kept as a regular page.
'''
def plan_gutter_split(img, screen_size, background_color='white'):
    if img.mode != 'L':
        img = ImageOps.grayscale(img)

    if background_color != 'white':
        img = ImageOps.invert(img)

    w, h = img.size
    white = np.array(img) >= white_threshold
    # side columns would block every gutter: look for gutters across the panels only
    x0, x1 = panels_area(white)
    lines = find_gutter_lines(img.crop((x0, 0, x1, h)))
    boundaries = [gutter_boundary(white, line, x0, x1) for line in lines]
    regions = content_regions(white, boundaries)
    if not regions:
        return []
    regions = drop_noise_regions(regions, h)

    # strip aspect ratio must not exceed the landscape screen aspect ratio
    limit = w * min(screen_size) / max(screen_size) * merge_tolerance
    groups = merge_regions(regions, limit)
    plans = []
    for group in groups:
        top, bot = min(r[2] for r in group), max(r[3] for r in group)
        plans.append((group[0][0], group[-1][1], max(0, top - strip_padding), min(h, bot + strip_padding)))
    # a single strip too tall for landscape means there is nothing to cut along
    if len(plans) == 1 and plans[0][3] - plans[0][2] > limit:
        return []
    return plans


'''
Finds the columns holding the panels, leaving out a side column of text on either edge.

    Parameters:
            white (NumPy array): Boolean mask of white pixels.
    Returns:
            area (2-tuple, left|right): Columns [left, right) to search gutters in.
'''
def panels_area(white):
    h, w = white.shape
    gaps = [(s, e) for s, e in true_runs(white.mean(axis=0) >= gutter_white_ratio) if e - s >= 3]
    x0, x1 = 0, w
    inner = [gap for gap in gaps if gap[0] > 0 and gap[1] < w]
    if inner and inner[0][0] <= w * side_column_width and is_side_column(white[:, :inner[0][0]]):
        x0 = inner[0][0]
    if inner and w - inner[-1][1] <= w * side_column_width and is_side_column(white[:, inner[-1][1]:]):
        x1 = inner[-1][1]
    return x0, x1


def is_side_column(white):
    empty_rows = white.all(axis=1)
    return not empty_rows.all() and empty_rows.mean() >= side_column_gap_rows


'''
Turns a gutter line found across the panels into a cut for every column of the page: inside a side column
the cut moves to the nearest empty row, so it doesn't go through a character.
'''
def gutter_boundary(white, line, x0, x1):
    h, w = white.shape
    ys = np.empty(w)
    ys[x0:x1] = line_y(line, x1 - x0)
    if x0 > 0:
        ys[:x0] = nearest_empty_row(white[:, :x0], ys[x0])
    if x1 < w:
        ys[x1:] = nearest_empty_row(white[:, x1:], ys[x1 - 1])
    return ys


def nearest_empty_row(white, y):
    reach = white.shape[0] * side_column_shift
    best = None
    for start, end in true_runs(white.all(axis=1)):
        if start <= y < end:
            return y
        center = (start + end - 1) / 2
        if abs(center - y) <= reach and (best is None or abs(center - y) < abs(best - y)):
            best = center
    # no empty row close enough: cut where the gutter is
    return y if best is None else best


'''
Finds straight full-width gutter lines, horizontal or slanted.

    Parameters:
            img (PIL image): A grayscale PIL image with white background.
    Returns:
            lines (list of 2-tuple, left y|right y): Gutter center lines from top to bottom, never crossing each other.
'''
def find_gutter_lines(img):
    w, h = img.size
    scale = min(1.0, detection_width / w)
    if scale < 1:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.BOX)
    white = np.array(img) >= white_threshold
    hs, ws = white.shape
    min_gutter = max(2, round(max(3, h * min_gutter_height) * scale))
    xs = np.arange(ws)

    candidates = []
    max_dy = int(ws * tan(radians(max_gutter_angle)))
    for dy in range(-max_dy, max_dy + 1):
        shift = np.round(xs * dy / max(1, ws - 1)).astype(int)
        # only lines fully inside the page
        lo, hi = -shift.min(), hs - shift.max()
        if hi - lo < min_gutter:
            continue
        ys = np.arange(lo, hi)[:, None] + shift
        is_gutter = white[ys, xs].mean(axis=1) >= gutter_white_ratio
        for start, end in true_runs(is_gutter):
            if end - start >= min_gutter:
                y = lo + (start + end - 1) / 2
                if not crosses_panel(white, xs, shift, y):
                    candidates.append((end - start, -abs(dy), y, y + dy))

    # the true direction of a gutter is the one where it is thickest
    candidates.sort(reverse=True)
    lines = []
    for thickness, _, y0, y1 in candidates:
        if all(lines_apart((y0, y1, thickness), line) for line in lines):
            lines.append((y0, y1, thickness))
    lines.sort(key=lambda line: line[0] + line[1])
    return [(y0 / scale, y1 / scale) for y0, y1, _ in lines]


'''
Tells whether a nearly white line runs through the inside of a panel: the few dark pixels it has belong to
borders that continue above and below it, and there are at least two of them (both sides of the panel).
A single one is tolerated, like a small mark printed in a real gutter.
'''
def crosses_panel(white, xs, shift, y, reach=3):
    hs = white.shape[0]

    def dark(offset):
        return ~white[np.clip(round(y + offset) + shift, 0, hs - 1), xs]

    crossings = dark(0) & dark(-reach) & dark(reach)
    return sum(1 for _ in true_runs(crossings)) >= 2


def true_runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return zip(np.where(edges == 1)[0], np.where(edges == -1)[0])


def lines_apart(a, b):
    d0, d1 = a[0] - b[0], a[1] - b[1]
    return d0 * d1 > 0 and min(abs(d0), abs(d1)) >= (a[2] + b[2]) / 2


def line_y(line, w):
    return line[0] + (line[1] - line[0]) * np.arange(w) / max(1, w - 1)


'''
Finds content between consecutive gutter boundaries.

    Parameters:
            white (NumPy array): Boolean mask of white pixels.
            boundaries (list of NumPy arrays): Cuts from top to bottom, the y of the cut for every column.
    Returns:
            regions (list of 5-tuple, top boundary|bottom boundary|top|bot|dark ratio): Regions with content,
            rows [top, bot) hold all their content. Empty for a blank page.
'''
def content_regions(white, boundaries):
    h, w = white.shape
    bounds = [np.zeros(w)] + boundaries + [np.full(w, h)]
    regions = []
    for top_y, bot_y in zip(bounds, bounds[1:]):
        r0, r1 = max(0, int(top_y.min())), min(h, ceil(bot_y.max()))
        rows = np.arange(r0, r1)[:, None]
        inside = (rows >= top_y) & (rows < bot_y)
        dark = inside & ~white[r0:r1]
        content_rows = np.where(dark.any(axis=1))[0]
        if len(content_rows) == 0:
            continue
        top, bot = content_rows[0], content_rows[-1] + 1
        dark_ratio = dark[top:bot].sum() / max(1, inside[top:bot].sum())
        regions.append((top_y, bot_y, r0 + top, r0 + bot, dark_ratio))
    return regions


def drop_noise_regions(regions, h):
    def is_noise(region):
        return region[3] - region[2] < h * noise_band_height and region[4] < noise_band_dark_ratio

    while len(regions) > 1 and is_noise(regions[0]):
        regions = regions[1:]
    while len(regions) > 1 and is_noise(regions[-1]):
        regions = regions[:-1]
    return regions


'''
Greedily groups adjacent regions while the strip (bounding box of their content) stays under 'limit'.
'''
def merge_regions(regions, limit):
    groups = [[regions[0]]]
    for region in regions[1:]:
        group = groups[-1]
        members = group + [region]
        if max(r[3] for r in members) - min(r[2] for r in members) <= limit:
            group.append(region)
        else:
            groups.append([region])
    return groups


'''
Crops rows [top, bot) and fills everything outside the two boundaries with the background color.
'''
def cut_strip(img, top_y, bot_y, top, bot, fill):
    w, _ = img.size
    strip = img.crop((0, top, w, bot))
    top_y, bot_y = top_y - top, bot_y - top
    if top_y.max() <= 0 and bot_y.min() >= strip.height:
        return strip
    if strip.mode not in ('L', 'RGB'):
        strip = strip.convert('RGB')
    rows = np.arange(strip.height)[:, None]
    mask = Image.fromarray(((rows >= top_y) & (rows < bot_y)).astype(np.uint8) * 255)
    return Image.composite(strip, Image.new(strip.mode, strip.size, fill), mask)
