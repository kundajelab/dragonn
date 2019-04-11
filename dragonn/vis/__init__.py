import matplotlib
from matplotlib import pyplot as plt


def plot_model_weights(model,layer_idx=-2):
    W_dense, b_dense = model.layers[layer_idx].get_weights()
    plt.plot(W_dense,'-o')
    plt.xlabel('Filter index')
    plt.ylabel('Weight value')
    plt.show()
    
def plot_ism(ism_mat,title,vmin=None,vmax=None):
    # create discrete colormap of ISM scores
    extent = [0, ism_mat.shape[0], 0, 100*ism_mat.shape[1]]
    plt.figure(figsize=(20,3))
    if vmin==None:
        vmin=np.amin(ism_mat)
    if vmax==None:
        vmax=np.amax(ism_mat)
    plt.imshow(ism_mat.T,extent=extent,vmin=vmin, vmax=vmax)
    plt.xlabel("Sequence base")
    plt.ylabel("ISM Score")
    plt.title(title)
    plt.yticks(np.arange(50,100*ism_mat.shape[1],100),("A","C","G","T"))
    plt.set_cmap('RdBu')
    plt.colorbar()
    plt.show()


def plot_seq_importance(grads, x, xlim=None, ylim=None, layer_idx=-2, figsize=(25, 3),title="",snp_pos=0):
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
