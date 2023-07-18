import logomaker
import pandas as pd
import re

import matplotlib
from matplotlib import pyplot

import numpy as np
from simdna.simulations import loaded_motifs


def plot_bases_on_ax(letter_heights, ax, show_ticks=True):
    """
    Plot the N letters with heights taken from the Nx4 matrix letter_heights.

    Parameters
    ----------
    letter_heights: Nx4 array
    ax: axis to plot on
    """

    logomaker.Logo(pd.DataFrame(letter_heights, columns=['A','C','G','T']), ax=ax)

    return ax


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
    ax.set_xlabel('base pair position')
    ax.set_ylabel(ylab)
    plot_bases_on_ax(letter_heights, ax)

    return fig,ax


