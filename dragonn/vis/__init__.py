import matplotlib
from matplotlib import pyplot as plt
import  numpy as np 
from dragonn.vis.plot_letters import * 

def plot_motif_scores(motif_scores,title="",figsize=(20,3),ymin=0,ymax=20):
    plt.figure(figsize=figsize)
    plt.plot(pos_motif_scores, "-o")
    plt.xlabel("Sequence base")
    plt.ylabel("Motif scan score")
    #threshold motif scores at 0; any negative scores are noise that we do not need to visualize
    plt.ylim(ymin, ymax)
    plt.title(title)
    plt.show()

def plot_model_weights(model,layer_idx=-2):
    W_dense, b_dense = model.layers[layer_idx].get_weights()
    plt.plot(W_dense,'-o')
    plt.xlabel('Filter index')
    plt.ylabel('Weight value')
    plt.show()
    
def plot_ism(ism_mat,title="",figsize=(20,5)):
    """ Plot the 4xL heatmap and also the identity and score of the highest scoring (mean subtracted) allele at each position 
    
    Args:
      ism_math: (n_positions x 4) 
      title: optional string specifying plot title 
      figsize: optional tuple indicating output figure dimensions in x, y 
    Returns: 
      generates a heatmap and letter plot of the ISM matrix 
    """
    highest_scoring_pos=np.argmax(np.abs(ism_mat),axis=1)
    zero_map=np.zeros(ism_mat.shape)
    zero_map[:,highest_scoring_pos]=1
    product=zero_map*ism_mat
    
    fig,axes=plt.subplots(2, 1,sharex='row',figsize=figsize)
    axes[0]=plot_bases_on_ax(product,axes[0],show_ticks=False)
    extent = [0, ism_mat.shape[0], 0, 100*ism_mat.shape[1]]
    ymin=np.amin(ism_mat)
    ymax=np.amax(ism_mat)
    axes[1].imshow(ism_mat.T,extent=extent,vmin=ymin, vmax=ymax, interpolation='nearest',aspect='auto')
    axes[1].set_xlabel("Sequence base")
    axes[1].set_ylabel("ISM Score")
    axes[1].set_title(title)
    axes[1].set_yticks(np.arange(50,100*ism_mat.shape[1],100),("A","C","G","T"))
    plt.set_cmap('RdBu')
    plt.tight_layout()
    plt.colorbar()
    plt.show()


def plot_seq_importance(grads, x, xlim=None, ylim=None, figsize=(25, 3),title="",snp_pos=0):
    """Plot  sequence importance score
    
    Args:
      grads: either deeplift or gradientxinput score matrix 
      x: one-hot encoded DNA sequence
      xlim: restrict the plotted xrange
      figsize: matplotlib figure size
    """
    grads=grads.squeeze()
    x=x.squeeze()
    
    seq_len = x.shape[0]
    vals_to_plot=grads*x
    if xlim is None:
        xlim = (0, seq_len)
    if ylim is None:
        ylim= (np.amin(vals_to_plot),np.amax(vals_to_plot))
    seqlogo_fig(vals_to_plot, figsize=figsize)
    plt.xticks(list(range(xlim[0], xlim[1], 5)))
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(title)
    plt.axvline(x=snp_pos, color='k', linestyle='--')

def plot_learning_curve(history):
    train_losses=history.history['loss']
    valid_losses=history.history['val_loss']
    min_loss_indx = min(enumerate(valid_losses), key=lambda x: x[1])[0]
    f = plt.figure(figsize=(10, 4))
    ax = f.add_subplot(1, 1, 1)
    ax.plot(range(len(train_losses)), train_losses, 'b', label='Training',lw=4)
    ax.plot(range(len(train_losses)), valid_losses, 'r', label='Validation', lw=4)
    ax.plot([min_loss_indx, min_loss_indx], [0, 1.0], 'k--', label='Early Stop')
    ax.legend(loc="upper right")
    ax.set_ylabel("Loss")
    ax.set_ylim((min(train_losses+valid_losses),max(train_losses+valid_losses)))
    ax.set_xlabel("Epoch")
    plt.show()
