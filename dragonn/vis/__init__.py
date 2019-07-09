import matplotlib
from matplotlib import pyplot as plt
import  numpy as np 
from dragonn.vis.plot_letters import * 
from dragonn.vis.plot_kmers import * 
import warnings
warnings.filterwarnings('ignore')

def extract_index_interp_dict(interp_dict_list,index):
    '''
    Extract interpretation metrics for a single example 
    '''
    new_interp_dict_list=[]
    for i in range(len(interp_dict_list)):
        new_interp_dict_list.append({}) 
        for key in interp_dict_list[i]:
            new_interp_dict_list[i][key]=interp_dict_list[i][key][index] 
    return new_interp_dict_list

def plot_all_interpretations(interp_dict_list,X,xlim=None,figsize=(20,3),title=None,snp_pos=0,out_fname_svg=None,index=None):
    '''
    interp_dict_list -- list of dictionaries of interpretation metrics for inputs 
    X -- input 
    title -- list of titles, or None 
    out_fname_svg -- filename to save svg figure (or None if it shouldn't be saved to an output file)
    index -- index of sample whose interpretations should be plotted 
    '''
    if index!=None:
        interp_dict_list=extract_index_interp_dict(interp_dict_list,index)
        X=X[index] 
    num_samples=len(interp_dict_list)
    if title==None:
        title=["" for i in range(num_samples)]
    if xlim==None:
        xlim=(0,X.squeeze().shape[0])
    if interp_dict_list[0]['motif_scan'] is not None:
        f,axes=plt.subplots(nrows=5,ncols=num_samples, dpi=80,figsize=(figsize[0],figsize[1]*5))
        axes = np.array(axes)
        if num_samples==1:
            axes=np.expand_dims(axes,axis=1)
        ism_axes=axes[3:5,:]
        input_grad_axes=axes[2,:]
        deeplift_axes=axes[1,:]
        scan_axes=axes[0,:]
    else:
        f,axes=plt.subplots(4,num_samples, dpi=80,figsize=(figsize[0],figsize[1]*4))
        if num_samples==1:
            axes=np.expand_dims(axes,axis=1)
        axes = np.array(axes)
        ism_axes=axes[2:4,:]
        input_grad_axes=axes[1,:]
        deeplift_axes=axes[0,:]

    for sample_index in range(num_samples):
        ism_axes[:,sample_index]=plot_ism(interp_dict_list[sample_index]['ism'],
                                          X,
                                          title=':'.join(["ISM",title[sample_index]]),
                                          figsize=figsize,
                                          xlim=xlim,
                                          axes=ism_axes[:,sample_index])
        input_grad_axes[sample_index]=plot_seq_importance(interp_dict_list[sample_index]['input_grad'],
                                                          X,
                                                          title=":".join(["GradXInput",title[sample_index]]),
                                                          figsize=figsize,
                                                          xlim=xlim,
                                                          snp_pos=snp_pos,
                                                          axes=input_grad_axes[sample_index])
        
        deeplift_axes[sample_index]=plot_seq_importance(interp_dict_list[sample_index]['deeplift'],
                                                        X,
                                                        title=":".join(["DeepLIFT",title[sample_index]]),
                                                        figsize=figsize,
                                                        xlim=xlim,
                                                        snp_pos=snp_pos,
                                                        axes=deeplift_axes[sample_index])
        if interp_dict_list[sample_index]['motif_scan'] is not None:
            scan_axes[sample_index]=plot_motif_scores(interp_dict_list[sample_index]['motif_scan'],
                                                      title=":".join(["Motif Scan Scores",title[sample_index]]),
                                                      figsize=figsize,
                                                      xlim=xlim,
                                                      axes=scan_axes[sample_index])

    if out_fname_svg is not None:
        plt.savefig(out_fname_svg,dpi=80,format="svg")
    f.show()

def plot_snp_interpretation(ref_interp_dict_list,
                    alt_interp_dict_list,
                    ref_X,
                    alt_X,
                    xlim=None,
                    figsize=(20,3),
                    title=None,
                    out_fname_svg=None,
                    snp_pos=0):
    '''
    ref_interp_dict -- list of interpretation metrics for reference allele tasks
    alt_interp_dict -- list of interpretation mterics for alternate allele tasks
    ref_X -- reference sequence 
    alt_X -- alternate sequence 
    title -- list of titles, or None 
    out_fname_svg -- filename to save svg figure (or None if it shouldn't be saved to an output file)
    '''
    num_samples=len(ref_interp_dict_list)
    if title==None:
        title=["" for i in range(num_samples)]
    if xlim==None:
        xlim=(0,ref_X.squeeze().shape[0])
    f,axes=plt.subplots(6,num_samples, dpi=80,figsize=(figsize[0],figsize[1]*6))
    if num_samples==1:
        axes=np.expand_dims(axes,axis=1)
    axes = np.array(axes)
    ism_axes=axes[0:2,:]
    input_grad_axes=axes[2,:]
    deeplift_ref_axes=axes[3,:]
    deeplift_alt_axes=axes[4,:]
    deeplift_delta_axes=axes[5,:]
    
    for sample_index in range(num_samples):
        ism_axes[:,sample_index]=plot_ism(ref_interp_dict_list[sample_index]['ism'],
                                          title=':'.join(["ISM",title[sample_index]]),
                                          figsize=figsize,
                                          xlim=xlim,
                                          axes=ism_axes[:,sample_index])
        input_grad_axes[sample_index]=plot_seq_importance(ref_interp_dict_list[sample_index]['input_grad'],
                                                          ref_X,
                                                          title=":".join(["GradXInput",title[sample_index]]),
                                                          figsize=figsize,
                                                          xlim=xlim,
                                                          axes=input_grad_axes[sample_index])
        deeplift_ref_axes[sample_index]=plot_seq_importance(ref_interp_dict_list[sample_index]['deeplift'],
                                                            ref_X,
                                                            title=":".join(["Ref DeepLIFT",title[sample_index]]),
                                                            figsize=figsize,
                                                            xlim=xlim,
                                                            snp_pos=snp_pos,
                                                            axes=deeplift_ref_axes[sample_index])
        deeplift_alt_axes[sample_index]=plot_seq_importance(alt_interp_dict_list[sample_index]['deeplift'],
                                                            alt_X,
                                                            title=":".join(["Alt DeepLIFT",title[sample_index]]),
                                                            figsize=figsize,
                                                            xlim=xlim,
                                                            snp_pos=snp_pos,
                                                            axes=deeplift_alt_axes[sample_index])
        deeplift_delta_axes[sample_index]=plot_seq_importance(alt_interp_dict_list[sample_index]['deeplift'] - ref_interp_dict_list[sample_index]['deeplift'],
                                                              alt_X,
                                                              title=":".join(["Alt - Ref DeepLIFT",title[sample_index]]),
                                                              figsize=figsize,
                                                              xlim=xlim,
                                                              snp_pos=snp_pos,
                                                              axes=deeplift_delta_axes[sample_index])
        
    if out_fname_svg is not None:
        plt.savefig(out_fname_svg,dpi=80,format="svg")
    f.show()

    


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

def plot_filters(model,simulation_data,show=True):
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


def plot_motif_scores(motif_scores,title="",figsize=(20,3),ylim=(0,20),xlim=None,axes=None):
    #remove any redundant axes
    motif_scores=motif_scores.squeeze()
    if axes is None:
        f,axes=plt.subplots(1,dpi=80,figsize=figsize)
        show=True
    else:
        show=False
    axes.plot(motif_scores, "-o")
    axes.set_xlabel("Sequence base")
    #threshold motif scores at 0; any negative scores are noise that we do not need to visualize
    if ylim!=None:
        axes.set_ylim(ylim)
    if xlim!=None:
        axes.set_xlim(xlim)
    axes.set_title(title)
    if show==True:
        plt.show()
    else:
        return axes

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

def plot_ism(ism_mat,x,title="", xlim=None, ylim=None, figsize=(20,5),axes=None):
    """ Plot the 4xL heatmap and also the identity and score of the highest scoring (mean subtracted) allele at each position 
    
    Args:
      ism_mat: (n_positions x 4) 
      title: optional string specifying plot title 
      figsize: optional tuple indicating output figure dimensions in x, y 
    Returns: 
      generates a heatmap and letter plot of the ISM matrix 
    """
    if axes is None:
        f,axes=plt.subplots(2, 1,sharex='row',figsize=(figsize[0],2*figsize[1]))
        show=True
    else:
        show=False

    if ism_mat.shape!=2:
        ism_mat=np.squeeze(ism_mat)
    assert len(ism_mat.shape)==2
    assert ism_mat.shape[1]==4

    if x.shape!=2: 
        x=x.squeeze()
    seq_len = x.shape[0]
    product=ism_mat*x
    plt.set_cmap('RdBu')
    axes[0]=plot_bases_on_ax(product,axes[0],show_ticks=False)
    axes[0].set_title(title)
    extent = [0, ism_mat.shape[0], 0, 100*ism_mat.shape[1]]
    ymin=np.amin(ism_mat)
    ymax=np.amax(ism_mat)
    abs_highest=max([abs(ymin),abs(ymax)])
    hmap=axes[1].imshow(ism_mat.T,extent=extent,vmin=-1*abs_highest, vmax=abs_highest, interpolation='nearest',aspect='auto')
    axes[1].set_yticks(np.array([100,200,300,400]))
    axes[1].set_yticklabels(["T","G","C","A"])
    axes[1].set_xlabel("Sequence base")
    if xlim!=None:
        axes[0].set_xlim(xlim)
        axes[1].set_xlim(xlim) 
        

    plt.tight_layout()
    plt.colorbar(hmap,ax=axes[1],orientation='horizontal')
    if show==True:
        plt.show()
    else:
        return axes

def plot_seq_importance(grads, x, xlim=None, ylim=None, figsize=(25, 3),title="",snp_pos=0,axes=None):
    """Plot  sequence importance score
    
    Args:
      grads: either deeplift or gradientxinput score matrix 
      x: one-hot encoded DNA sequence
      xlim: restrict the plotted xrange
      figsize: matplotlib figure size
    """
    if axes is None:
        f,axes=plt.subplots(1,dpi=80,figsize=figsize)
        show=True
    else:
        show=False
    grads=grads.squeeze()
    x=x.squeeze()
    
    seq_len = x.shape[0]
    vals_to_plot=grads*x
    if xlim is None:
        xlim = (0, seq_len)
    if ylim is None:
        ylim= (np.amin(vals_to_plot),np.amax(vals_to_plot))
    axes=plot_bases_on_ax(vals_to_plot,axes,show_ticks=True)
    plt.xticks(list(range(xlim[0], xlim[1], 5)))
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    axes.set_title(title)
    axes.axvline(x=snp_pos, color='k', linestyle='--')
    if show==True:
        plt.show()
    else:
        return axes


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


def plot_positionalPRC(positionalPRC_output):
    '''
    accepts output dictionary from the positionalPRC function of the form: motif_name --> [precision,recall,auPRC] 
    generates PRC curves for each motif on same coordinates 
    '''
    #from sklearn.utils.fixes import signature
    for motif_name,values in positionalPRC_output.items():
        recall=values[0]
        precision=values[1]
        auPRC=str(round(values[2],3))
        #step_kwargs = ({'step': 'post'}
        #                              if 'step' in signature(plt.fill_between).parameters
        #                              else {})
        plt.plot(recall, precision, '-',label=motif_name+":"+auPRC)
        #uncomment to fill the area below the curve, generally not desirable if multiple curves plotted on same axes.
        #plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend()
    plt.show()
        
