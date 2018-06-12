import re

import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from shapely.wkt import loads as load_wkt
from shapely import affinity

import numpy as np
from simdna.simulations import loaded_motifs
from pkg_resources import resource_filename

##########################################################################
# copied from descartes
# https://pypi.python.org/pypi/descartes


class Polygon(object):
    # Adapt Shapely or GeoJSON/geo_interface polygons to a common interface

    def __init__(self, context):
        if hasattr(context, 'interiors'):
            self.context = context
        else:
            self.context = getattr(context, '__geo_interface__', context)

    @property
    def geom_type(self):
        return (
            getattr(self.context, 'geom_type', None) or self.context['type'])

    @property
    def exterior(self):
        return (
            getattr(self.context, 'exterior', None) or self.context['coordinates'][0])

    @property
    def interiors(self):
        value = getattr(self.context, 'interiors', None)
        if value is None:
            value = self.context['coordinates'][1:]
        return value


def PolygonPath(polygon):
    """Constructs a compound matplotlib path from a Shapely or GeoJSON-like
    geometric object"""
    this = Polygon(polygon)
    assert this.geom_type == 'Polygon'

    def coding(ob):
        # The codes will be all "LINETO" commands, except for "MOVETO"s at the
        # beginning of each subpath
        n = len(getattr(ob, 'coords', None) or ob)
        vals = np.ones(n, dtype=Path.code_type) * Path.LINETO
        vals[0] = Path.MOVETO
        return vals
    vertices = np.concatenate(
        [np.asarray(this.exterior)] + [np.asarray(r)
                                       for r in this.interiors])
    codes = np.concatenate(
        [coding(this.exterior)] + [coding(r)
                                   for r in this.interiors])
    return Path(vertices, codes)


def PolygonPatch(polygon, **kwargs):
    """Constructs a matplotlib patch from a geometric object

    The `polygon` may be a Shapely
    or GeoJSON-like object with or without holes.
    The `kwargs` are those supported by the matplotlib.patches.Polygon class
    constructor. Returns an instance of matplotlib.patches.PathPatch.

    Example (using Shapely Point and a matplotlib axes):

      >>> b = Point(0, 0).buffer(1.0)
      >>> patch = PolygonPatch(b, fc='blue', ec='blue', alpha=0.5)
      >>> axis.add_patch(patch)

    """
    return PathPatch(PolygonPath(polygon), **kwargs)

#
# END copied from descartes
#
##########################################################################

##########################################################################
# Initialize the polygon paths for A,C,G,T
#
# Geometry taken from JTS TestBuilder Monospace font with fixed precision model
# of 1000.0
#

A_data = """
MULTIPOLYGON (
((24.7631 57.3346, 34.3963 57.3346, 52.391 -1.422, 44.1555 -1.422, 39.8363
  13.8905, 19.2476 13.8905, 15.0039 -1.422, 6.781 -1.422, 24.7631 57.3346)),
((29.5608 50.3205, 21.1742 20.2623, 37.9474 20.2623, 29.5608 50.3205))
)
"""

C_data = """POLYGON((
52.391 2.5937, 48.5882 0.8417, 44.68 -0.4142, 40.5998 -1.17, 36.2814 -1.422,
32.8755 -1.2671, 29.6656 -0.8024, 26.6518 -0.0278, 23.834 1.0565,
21.2122 2.4507, 18.7865 4.1547, 16.5569 6.1686, 14.5233 8.4922,
12.7087 11.0966, 11.136 13.9527, 9.8053 17.0606, 8.7166 20.4201,
7.8698 24.0314, 7.2649 27.8943, 6.902 32.009, 6.781 36.3754, 6.9027 40.7209,
7.2678 44.8198, 7.8764 48.6722, 8.7283 52.278, 9.8236 55.6371,
11.1624 58.7497, 12.7446 61.6157, 14.5702 64.2351, 16.6133 66.5753,
18.8481 68.6034, 21.2745 70.3195, 23.8926 71.7235, 26.7023 72.8156,
29.7037 73.5956, 32.8967 74.0637, 36.2814 74.2197, 40.5998 73.9697,
44.68 73.2196, 48.5882 71.9696, 52.391 70.2196, 52.391 60.1101,
48.6468 62.739, 44.6331 64.657, 40.4709 65.8289, 36.2814 66.2196,
31.7716 65.7557, 29.7437 65.1758, 27.8672 64.3641, 26.1421 63.3203,
24.5684 62.0447, 23.146 60.5371, 21.875 58.7976, 19.7831 54.6129,
18.289 49.481, 17.3925 43.4019, 17.0936 36.3754, 17.3925 29.3763,
18.289 23.3166, 19.7831 18.1964, 21.875 14.0157, 23.146 12.2762,
24.5684 10.7686, 26.1421 9.4929, 27.8672 8.4492, 29.7437 7.6375,
31.7716 7.0576, 36.2814 6.5937, 40.5354 6.9844, 44.7034 8.1563,
48.6878 10.0743, 52.391 12.7032, 52.391 2.5937))"""

G_data = """POLYGON((
52.391 5.4974, 50.49 3.8964, 48.4724 2.502, 46.3383 1.3144, 44.0877 0.3334,
41.7314 -0.4346, 39.2805 -0.9832, 34.0946 -1.422, 30.9504 -1.2772,
27.9859 -0.843, 25.2009 -0.1191, 22.5956 0.8942, 20.1698 2.197,
17.9236 3.7894, 15.857 5.6713, 13.9699 7.8428, 12.285 10.2753,
10.8248 12.9404, 9.5892 15.8381, 8.5782 18.9685, 7.7919 22.3315,
7.2303 25.9271, 6.8933 29.7553, 6.781 33.8161, 6.8948 37.8674,
7.2362 41.6888, 7.8053 45.2803, 8.6019 48.6419, 9.6262 51.7737, 10.878 54.6755,
12.3575 57.3474, 14.0646 59.7895, 15.9743 61.9712, 18.0615 63.862,
20.3262 65.4618, 22.7685 66.7708, 25.3884 67.789, 28.1857 68.5162,
31.1606 68.9525, 34.3131 69.098, 38.5048 68.7957, 42.5144 67.8889,
46.3638 66.3703, 50.0748 64.2325, 50.0748 54.8075, 46.342 57.8466,
42.5144 59.9716, 38.5266 61.2226, 34.3131 61.6395, 30.1132 61.2053,
28.2228 60.6624, 26.4723 59.9024, 24.8614 58.9253, 23.3904 57.731,
22.0591 56.3195, 20.8675 54.691, 18.9046 50.7806, 17.5025 45.998,
16.6612 40.3432, 16.3808 33.8161, 16.6526 27.1962, 17.4679 21.4959,
18.8267 16.7151, 20.7291 12.8539, 21.8892 11.2595, 23.1951 9.8776,
24.6469 8.7084, 26.2446 7.7517, 27.9883 7.0076, 29.8778 6.4762, 34.0946 6.051,
36.9534 6.2276, 39.4407 6.7575, 41.6331 7.6625, 43.607 8.9644, 43.607 27.2172,
33.7304 27.2172, 33.7304 34.7776, 52.391 34.7776, 52.391 5.4974
))"""

T_data = """POLYGON((
6.781 58.3746, 52.391 58.3746, 52.391 51.5569, 33.6933 51.5569, 33.6933 -1.422,
25.5684 -1.422, 25.5684 51.5569, 6.781 51.5569, 6.781 58.3746
))"""


def standardize_polygons_str(data_str):
    """Given a POLYGON string, standardize the coordinates to a 1x1 grid.

    Input : data_str (taken from above)
    Output: tuple of polygon objects
    """
    # find all of the polygons in the letter (for instance an A
    # needs to be constructed from 2 polygons)
    path_strs = re.findall("\(\(([^\)]+?)\)\)", data_str.strip())

    # convert the data into a numpy array
    polygons_data = []
    for path_str in path_strs:
        data = np.array([
            tuple(map(float, x.split())) for x in path_str.strip().split(",")])
        polygons_data.append(data)

    # standardize the coordinates
    min_coords = np.vstack(data.min(0) for data in polygons_data).min(0)
    max_coords = np.vstack(data.max(0) for data in polygons_data).max(0)
    for data in polygons_data:
        data[:, ] -= min_coords
        data[:, ] /= (max_coords - min_coords)

    polygons = []
    for data in polygons_data:
        polygons.append(load_wkt(
            "POLYGON((%s))" % ",".join(" ".join(map(str, x)) for x in data)))

    return tuple(polygons)


letters_polygons = {}
letters_polygons['A'] = standardize_polygons_str(A_data)
letters_polygons['C'] = standardize_polygons_str(C_data)
letters_polygons['G'] = standardize_polygons_str(G_data)
letters_polygons['T'] = standardize_polygons_str(T_data)


colors = dict(zip(
    'ACGT', (('green', 'white'), ('blue',), ('orange',), ('red',))
))


def add_letter_to_axis(ax, let, x, y, height):
    """Add 'let' with position x,y and height height to matplotlib axis 'ax'.

    """
    for polygon, color in zip(letters_polygons[let], colors[let]):
        new_polygon = affinity.scale(
            polygon, yfact=height, origin=(0, 0, 0))
        new_polygon = affinity.translate(
            new_polygon, xoff=x, yoff=y)
        patch = PolygonPatch(
            new_polygon, edgecolor=color, facecolor=color)
        ax.add_patch(patch)
    return


def plot_bases_on_ax(letter_heights, ax):
    """
    Plot the N letters with heights taken from the Nx4 matrix letter_heights.

    Parameters
    ----------
    letter_heights: Nx4 array
    ax: axis to plot on
    """

    assert letter_heights.shape[1] == 4, letter_heights.shape
    for x_pos, heights in enumerate(letter_heights):
        letters_and_heights = sorted(zip(heights, 'ACGT'))
        y_pos_pos = 0.0
        y_neg_pos = 0.0
        for height, letter in letters_and_heights:
            if height > 0:
                add_letter_to_axis(ax, letter, 0.5 + x_pos, y_pos_pos, height)
                y_pos_pos += height
            elif height < 0:
                add_letter_to_axis(ax, letter, 0.5 + x_pos, y_neg_pos, height)
                y_neg_pos += height
    ax.set_xlim(0, letter_heights.shape[0] + 1)
    ax.set_xticks(np.arange(1, letter_heights.shape[0] + 1))
    ax.set_aspect(aspect='auto', adjustable='box')
    ax.autoscale_view()


def plot_bases(letter_heights, figsize=(12, 6), ylab='bits'):
    """
    Plot the N letters with heights taken from the Nx4 matrix letter_heights.

    Parameters
    ----------
    letter_heights: Nx4 array
    ylab: y axis label

    Returns
    -------
    pyplot figure
    """
    assert letter_heights.shape[1] == 4, letter_heights.shape

    fig = pyplot.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_xlabel('pos')
    ax.set_ylabel(ylab)
    plot_bases_on_ax(letter_heights, ax)

    return fig


def plot_pwm(letter_heights,
             figsize=(12, 6), ylab='bits', information_content=True):
    """
    Plots pwm. Displays information content by default.
    """
    if information_content:
        letter_heights = letter_heights * (
            2 + (letter_heights *
                 np.log2(letter_heights)).sum(axis=1))[:, np.newaxis]
    return plot_bases(letter_heights, figsize, ylab=ylab)


def plot_motif(motif_name, figsize, ylab='bits', information_content=True):
    """
    Plot motifs from encode motifs file
    """
    motif_letter_heights = loaded_motifs.getPwm(motif_name).getRows()
    return plot_pwm(motif_letter_heights, figsize,
                    ylab=ylab, information_content=information_content)


def add_letters_to_axis(ax, letter_heights):
    """
    Plots letter on user-specified axis.

    Parameters
    ----------
    ax : axis
    letter_heights: Nx4 array
    """
    assert letter_heights.shape[1] == 4

    x_range = [1, letter_heights.shape[0]]
    pos_heights = np.copy(letter_heights)
    pos_heights[letter_heights < 0] = 0
    neg_heights = np.copy(letter_heights)
    neg_heights[letter_heights > 0] = 0

    for x_pos, heights in enumerate(letter_heights):
        letters_and_heights = sorted(zip(heights, 'ACGT'))
        y_pos_pos = 0.0
        y_neg_pos = 0.0
        for height, letter in letters_and_heights:
            if height > 0:
                add_letter_to_axis(ax, letter, 0.5 + x_pos, y_pos_pos, height)
                y_pos_pos += height
            else:
                add_letter_to_axis(ax, letter, 0.5 + x_pos, y_neg_pos, height)
                y_neg_pos += height

    ax.set_xlim(x_range[0] - 1, x_range[1] + 1)
    ax.set_xticks(list(range(*x_range)) + [x_range[-1]])
    ax.set_aspect(aspect='auto', adjustable='box')
    ax.autoscale_view()
