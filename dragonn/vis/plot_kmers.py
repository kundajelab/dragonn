#helper functions for plotting model filters
from dragonn.vis.plot_letters import *
from matplotlib import pyplot as plt

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



def plot_motifs(simulation_data):
    for motif_name in simulation_data.motif_names:
        plot_motif(motif_name, figsize=(10, 4), ylab=motif_name)

