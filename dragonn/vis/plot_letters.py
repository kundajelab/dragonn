import logomaker
import pandas as pd

def plot_bases_on_ax(letter_heights, ax, show_ticks=True):
    """
    Plot the N letters with heights taken from the Nx4 matrix letter_heights.

    Parameters
    ----------
    letter_heights: Nx4 array
    ax: axis to plot on
    """

    logomaker.Logo(pd.DataFrame(letter_heights, columns=['A','C','G','T']), ax=ax)
