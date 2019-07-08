from dragonn.interpret.ism import *
from dragonn.interpret.deeplift import *
from dragonn.interpret.input_grad import *
from dragonn.utils import get_motif_scores
from dragonn.vis import * 
from dragonn.models import load_dragonn_model
import warnings
warnings.filterwarnings('ignore') 
def multi_method_interpret(model,
                           X,
                           task_idx,
                           deeplift_score_func,
                           motif_names=None,
                           batch_size=200,
                           target_layer_idx=-2,
                           num_refs_per_seq=10,
                           reference="shuffled_ref",
                           one_hot_func=None,
                           pfm=None,
                           GC_fraction=0.4,
                           generate_plots=True):
    """
    Arguments:
        model -- keras model object 
        X -- numpy array with shape (1, 1, n_bases_in_sample,4) or list of FASTA sequences 
        task_idx -- numerical index (starting with 0)  of task to interpet. For a single-tasked model, you should set this to 0 
        deeplift_score_fun -- scoring function to use with DeepLIFT algorithm. 
        motif_names -- a list of motif name strings to scan for in the input sequence; if this is unknown, keep the default value of None  
        batch_size -- number of samples to interpret at once 
        target_layer_idx -- should be -2 for classification; -1 for regression 
        reference -- one of 'shuffled_ref','gc_ref','zero_ref'
        num_refs_per_seq -- integer indicating number of references to use for each input sequence \
                            if the reference is set to 'shuffled_ref';if 'zero_ref' or 'gc_ref' is \
                            used, this argument is ignored.
        one_hot_func -- one hot function to use for encoding FASTA string inputs; if the inputs \
                            are already one-hot-encoded, use the default of None 
        generate_plots -- default True. Flag to indicate whether or not interpretation plots \
                            should be generated 
    Returns:
        dictionary with keys 'motif_scan','ism','gradxinput','deeplift' 
    """
    outputs=dict()
    #1) motif scan (if motif_names !=None)
    if motif_names is not None:
        print("getting 'motif_scan' value")
        outputs['motif_scan']=get_motif_scores(X,motif_names,pfm=pfm,GC_fraction=GC_fraction,return_positions=True)
    else:
        outputs['motif_scan']=None 
    #2) ISM
    print("getting 'ism' value") 
    outputs['ism']=in_silico_mutagenesis(model,X,task_idx,target_layer_idx=target_layer_idx)
    #3) Input_Grad
    print("getting 'input_grad' value")
    outputs['input_grad']=input_grad(model,X,target_layer_idx=target_layer_idx)
    #4) DeepLIFT
    print("getting 'deeplift' value") 
    outputs['deeplift']=deeplift(deeplift_score_func,X,batch_size=batch_size,task_idx=task_idx,num_refs_per_seq=num_refs_per_seq,reference=reference,one_hot_func=one_hot_func) 

    
    
    #generate plots
    if generate_plots==True:
        plot_all_interpretations([outputs],X)    
    return outputs 

