from PIL import Image, ImageDraw, ImageOps
import numpy as np
from math import ceil, radians, tan

'''
Gutter split parameters
A pixel is white when its gray value is >= 'white_threshold'.
A straight line across the whole page is a gutter line when at least 'gutter_white_ratio' of its pixels are white.
Gutters may be slanted up to 'max_gutter_angle' degrees; they are searched on a copy scaled down to 'detection_width'.
Consecutive gutter lines form a gutter only when at least max(3px, 'min_gutter_height' * page height) thick.
Leading/trailing regions shorter than 'noise_band_height' and darker than 'noise_band_dark_ratio' are dropped (page numbers).
'''
white_threshold = 235
gutter_white_ratio = 0.98
min_gutter_height = 0.006 # percent
max_gutter_angle = 15 # degrees
detection_width = 400 # px
noise_band_height = 0.05 # percent
noise_band_dark_ratio = 0.08
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
    return [cut_strip(img, *plan, fill) for plan in plan_gutter_split(img, screen_size, background_color)]


'''
Plans the strips of a page.

    Returns:
            plans (list of 4-tuple, top line|bottom line|top|bot): Each strip spans rows [top, bot) and keeps only the
            pixels between its two boundary lines. A line is (y at the left edge, y at the right edge).
            Empty when the page should be kept as a regular page.
'''
def plan_gutter_split(img, screen_size, background_color='white'):
    if img.mode != 'L':
        img = ImageOps.grayscale(img)

    if background_color != 'white':
        img = ImageOps.invert(img)

    w, h = img.size
    white = np.array(img) >= white_threshold
    regions = content_regions(white, find_gutter_lines(img))
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
Finds content between consecutive gutter lines.

    Parameters:
            white (NumPy array): Boolean mask of white pixels.
            lines (list of 2-tuple): Gutter lines from top to bottom.
    Returns:
            regions (list of 5-tuple, top line|bottom line|top|bot|dark ratio): Regions with content, rows [top, bot)
            hold all their content. Empty for a blank page.
'''
def content_regions(white, lines):
    h, w = white.shape
    bounds = [(0, 0)] + lines + [(h, h)]
    regions = []
    for top_line, bot_line in zip(bounds, bounds[1:]):
        top_y, bot_y = line_y(top_line, w), line_y(bot_line, w)
        r0, r1 = max(0, int(top_y.min())), min(h, ceil(bot_y.max()))
        rows = np.arange(r0, r1)[:, None]
        inside = (rows >= top_y) & (rows < bot_y)
        dark = inside & ~white[r0:r1]
        content_rows = np.where(dark.any(axis=1))[0]
        if len(content_rows) == 0:
            continue
        top, bot = content_rows[0], content_rows[-1] + 1
        dark_ratio = dark[top:bot].sum() / max(1, inside[top:bot].sum())
        regions.append((top_line, bot_line, r0 + top, r0 + bot, dark_ratio))
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
Crops rows [top, bot) and fills everything outside the two boundary lines with the background color.
'''
def cut_strip(img, top_line, bot_line, top, bot, fill):
    w, _ = img.size
    strip = img.crop((0, top, w, bot))
    top_y, bot_y = line_y(top_line, w) - top, line_y(bot_line, w) - top
    if top_y.max() <= 0 and bot_y.min() >= strip.height:
        return strip
    if strip.mode not in ('L', 'RGB'):
        strip = strip.convert('RGB')
    mask = Image.new('L', strip.size, 0)
    ImageDraw.Draw(mask).polygon([(0, top_y[0]), (w - 1, top_y[-1]), (w - 1, bot_y[-1]), (0, bot_y[0])], fill=255)
    return Image.composite(strip, Image.new(strip.mode, strip.size, fill), mask)
