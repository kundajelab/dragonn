import matplotlib
from matplotlib import pyplot as plt
import  numpy as np 
from dragonn.vis.plot_letters import * 
from dragonn.vis.plot_kmers import * 

def plot_all_interpretations(interp_dict,X,xlim=None,figsize=(20,3)):
    if xlim==None:
        xlim=(0,X.squeeze().shape[0]) 
    f_scan,axes_scan=plot_motif_scores(interp_dict['motif_scan'],title="Motif Scan Scores",figsize=figsize,xlim=xlim,show=False)
    f_ism,axes_sim=plot_ism(interp_dict['ism'],'ISM',xlim=xlim,figsize=figsize,show=False)
    f_input_grad,axes_input_grad=plot_seq_importance(interp_dict['input_grad'],X,title="GradXInput",xlim=xlim,figsize=figsize,show=False)
    f_deeplift,axes_deeplift=plot_seq_importance(interp_dict['deeplift'],X,title="DeepLIFT",xlim=xlim,figsize=figsize,show=False)

def plot_sequence_filters(model,show=True):
    if show==False:
        plt.ioff()
    else:
        plt.ion() 
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
    if show==True:
        plt.show() 

def plot_filters(model,simulation_datam,show=True):
    if show==False:
        plt.ioff()
    else:
        plt.ion() 
    print("Plotting simulation motifs...")
    plot_motifs(simulation_data)
    plt.show()
    print("Visualizing convolutional sequence filters in SequenceDNN...")
    plot_sequence_filters(model)
    if show==True:
        plt.show()
    return f,f.get_axes()

def plot_motif_scores(motif_scores,title="",figsize=(20,3),ylim=(0,20),xlim=None,show=True):
    if show==False:
        plt.ioff()
    else:
        plt.ion() 
    #remove any redundant axes
    motif_scores=motif_scores.squeeze()
    f=plt.figure(figsize=figsize)
    plt.plot(motif_scores, "-o")
    plt.xlabel("Sequence base")
    #threshold motif scores at 0; any negative scores are noise that we do not need to visualize
    if plt.ylim!=None:
        plt.ylim(ylim)
    if xlim!=None:
        plt.xlim(xlim)
    plt.title(title)
    if show==True:
        plt.show()
    return f,f.axes

def plot_model_weights(model,layer_idx=-2,show=True):
    if show==False:
        plt.ioff()
    else:
        plt.ion() 

    W_dense, b_dense = model.layers[layer_idx].get_weights()
    f=plt.figure()
    plt.plot(W_dense,'-o')
    plt.xlabel('Filter index')
    plt.ylabel('Weight value')
    if show==True:
        plt.show()
    return f,f.get_axes() 

def plot_ism(ism_mat,title="", xlim=None, ylim=None, figsize=(20,5),show=True):
    """ Plot the 4xL heatmap and also the identity and score of the highest scoring (mean subtracted) allele at each position 
    
    Args:
      ism_math: (n_positions x 4) 
      title: optional string specifying plot title 
      figsize: optional tuple indicating output figure dimensions in x, y 
    Returns: 
      generates a heatmap and letter plot of the ISM matrix 
    """
    if show==False:
        plt.ioff()
    else:
        plt.ion() 
    if ism_mat.shape!=2:
        ism_mat=np.squeeze(ism_mat)
    assert len(ism_mat.shape)==2
    assert ism_mat.shape[1]==4
    
    highest_scoring_pos=np.argmax(np.abs(ism_mat),axis=1)
    zero_map=np.zeros(ism_mat.shape)
    for i in range(zero_map.shape[0]):
        zero_map[i][highest_scoring_pos[i]]=1
    product=zero_map*ism_mat    
    f,axes=plt.subplots(2, 1,sharex='row',figsize=(figsize[0],2*figsize[1]))
    axes[0]=plot_bases_on_ax(product,axes[0],show_ticks=False)
    axes[0].set_title(title)
    extent = [0, ism_mat.shape[0], 0, 100*ism_mat.shape[1]]
    ymin=np.amin(ism_mat)
    ymax=np.amax(ism_mat)
    hmap=axes[1].imshow(ism_mat.T,extent=extent,vmin=ymin, vmax=ymax, interpolation='nearest',aspect='auto')
    axes[1].set_yticks(np.arange(50,100*ism_mat.shape[1],100),("A","C","G","T"))
    axes[1].set_xlabel("Sequence base")
    if xlim!=None:
        axes[0].set_xlim(xlim)
        axes[1].set_xlim(xlim) 
    if ylim!=None:
        axes[0].set_ylim(ylim)
        axes[1].set_ylim(ylim)
        
    plt.set_cmap('RdBu')
    plt.tight_layout()
    plt.colorbar(hmap,ax=axes[1],orientation='horizontal')
    if show==True:
        plt.show()
    return f,axes

def plot_seq_importance(grads, x, xlim=None, ylim=None, figsize=(25, 3),title="",snp_pos=0,show=True):
    """Plot  sequence importance score
    
    Args:
      grads: either deeplift or gradientxinput score matrix 
      x: one-hot encoded DNA sequence
      xlim: restrict the plotted xrange
      figsize: matplotlib figure size
    """
    if show==False:
        plt.ioff()
    else:
        plt.ion() 
    grads=grads.squeeze()
    x=x.squeeze()
    
    seq_len = x.shape[0]
    vals_to_plot=grads*x
    if xlim is None:
        xlim = (0, seq_len)
    if ylim is None:
        ylim= (np.amin(vals_to_plot),np.amax(vals_to_plot))
    f,ax=plot_bases(vals_to_plot, figsize=figsize,ylab="")
    plt.xticks(list(range(xlim[0], xlim[1], 5)))
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(title)
    plt.axvline(x=snp_pos, color='k', linestyle='--')
    if show==True:
        plt.show() 
    return f,ax

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


