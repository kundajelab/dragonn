#helper functions for plotting model filters

def plot_motifs(simulation_data):
    for motif_name in simulation_data.motif_names:
        plot_motif(motif_name, figsize=(10, 4), ylab=motif_name)

def plot_sequence_filters(model):
    fig = plt.figure(figsize=(15, 8))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    conv_filters=model.layers[0].get_weights()[0]
    #transpose for plotting
    conv_filters=np.transpose(conv_filters,(3,1,2,0)).squeeze(axis=-1)
    num_plots_per_axis = int(len(conv_filters)**0.5) + 1
    for i, conv_filter in enumerate(conv_filters):
        ax = fig.add_subplot(num_plots_per_axis, num_plots_per_axis, i+1)
        add_letters_to_axis(ax, conv_filter)
        ax.axis("off")
        ax.set_title("Filter %s" % (str(i+1)))


def plot_filters(model,simulation_data):    
    print("Plotting simulation motifs...")
    plot_motifs(simulation_data)
    plt.show()
    print("Visualizing convolutional sequence filters in SequenceDNN...")
    plot_sequence_filters(model)
    plt.show()

