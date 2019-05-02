from dragonn.interpret.ism import *
from dragonn.interpret.deeplift import *
from dragonn.interpret.input_grad import *
from dragonn.utils import get_motif_scores
from dragonn.vis import * 
from keras.models import load_model

def multi_method_interpret(model_string, X, motif_names,batch_size=200,target_layer_idx=-2,task_idx=0, num_refs_per_seq=10,reference="shuffled_ref",one_hot_func=None,generate_plots=True):
    """
    Arguments:
        model_string -- a string containing the path to the hdf5 exported model 
        X -- numpy array with shape (1, 1, n_bases_in_sample,4) or list of FASTA sequences 
        motif_names -- a list of motif name strings to scan for in the input sequence 
        batch_size -- number of samples to interpret at once 
        target_layer_idx -- should be -2 for classification; -1 for regression 
        task_idx -- index indicating which task to perform interpretation on 
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
    #load the model for use with ism, gradxinput
    model=load_model(model_string) 
    outputs=dict()
    #1) motif scan
    outputs['motif_scan']=get_motif_scores(X,motif_names,return_positions=True)
    #2) ISM 
    outputs['ism']=in_silico_mutagenesis(model,X,target_layer_idx=target_layer_idx)
    #3) Input_Grad
    outputs['input_grad']=input_grad(model,X,target_layer_idx=target_layer_idx)
    #4) DeepLIFT 
    outputs['deeplift']=deeplift(model_string,X,batch_size=batch_size,target_layer_idx=target_layer_idx,task_idx=task_idx,num_refs_per_seq=num_refs_per_seq,reference=reference,one_hot_func=one_hot_func) 

    #generate plots
    if generate_plots==True:
        plot_all_interpretations(outputs,X)    
    return outputs 

