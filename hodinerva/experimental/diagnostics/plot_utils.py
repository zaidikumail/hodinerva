import colorsys

import matplotlib.colors as mcolors


def scale_luminance(color, scale):
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0, min(1, l * scale))
    return colorsys.hls_to_rgb(h, l, s)
