from dragonn.utils import rolling_window
import numpy as np
from sklearn.metrics import auc, precision_recall_curve

def positionalPRC(embeddings, scores,window_stride=1, coverage_thresh_for_positive=0.8):
    '''
    window_stride: number of bases to shift adjacent sequence windows by; default=1
    coverage_thresh_for_positive: sequence window must overlap the motif by this fraction (0 - 1) for the window to be labeled positive. 
    embeddings: the list of motif embeddings from simulation data (most likely from  simulation_data.valid_embeddings)
    scores: a list of scores for each position along a sequence, for each sequence in the dataset (2d-list). Generally one of: motif scan scores, ISM scores, gradient x input scores, deepLIFT scores for each sequence in the dataset 

    returns: dictionary of motif_name-->[precision, recall, auPRC]
    '''
    #we concatenate across all sequences in the input dataset
    assert len(scores)==len(embeddings)
    assert len(scores)>0

    #get the length of input sequences in the dataset 
    seq_length=len(scores[0])

    #keep lists of labels and predictions for each embedded entity
    all_prc_inputs={} 

    #iterate through all input sequences 
    for i in range(len(embeddings)):
        seq_embeddings=embeddings[i]
        seq_scores=np.asarray(scores[i])

        seq_prc_inputs=dict() 
        
        #sequence may have multiple embeddings
        for embedding in seq_embeddings:
            motif_length=len(embedding.what.string)
            motif_name=embedding.what.stringDescription
            embedding_start_pos=embedding.startPos 
            
            if motif_name not in all_prc_inputs:
                all_prc_inputs[motif_name]=dict()
                all_prc_inputs[motif_name]['labels']=[]
                all_prc_inputs[motif_name]['scores']=[]

            if motif_name not in seq_prc_inputs:
                seq_prc_inputs[motif_name]=dict()
                seq_prc_inputs[motif_name]['labels']=np.zeros((seq_length,1))
                seq_prc_inputs[motif_name]['scores']=np.sum(rolling_window(seq_scores,motif_length))

            #label the window that starts at the embedding start position with 1.
            tmp_label_array=np.zeros((seq_length,1))
            tmp_label_array[embedding_start_pos:embedding_start_pos+motif_length]=1
            tmp_label_windows=np.sum(rolling_window(tmp_label_array,motif_length))
            min_window_sum=coverage_thresh_for_positive*motif_length
            tmp_label_windows[tmp_label_windows>=min_window_sum]=1 #positive
            
            #ambiguous windows are designated with 0.5 to allow for use of np.maximum below 
            tmp_label_windows[(tmp_label_windows>0) & (tmp_label_windows<min_window_sum)]=0.5 
            seq_prc_iputs[motif_name]['labels']=np.maximum(seq_prc_iputs[motif_name]['labels'],tmp_label_windows)

        #update the dictionary of PRC inputs concatenated across sequences
        for motif_name in seq_prc_inputs.keys():
            #drop any ambiguous indices
            non_ambiguous_indices=np.where(seq_prc_inputs[motif_name]['labels']!=0.5)
            all_prc_inputs[motif_name]['labels']+=list(seq_prc_inputs[motif_name]['labels'][non_ambiguous_indices])
            all_prc_inputs[motif_name]['scores']+=list(seq_prc_inputs[motif_name]['scores'][non_ambiguous_indices])

        #calculate the PRC values and auPRC
        prc_values=dict()
        for motif_name in all_prc_inputs:
            labels=all_prc_inputs[motif_name]['labels']
            scores=all_prc_inputs[motif_name]['scores']
            precision, recall = precision_recall_curve(labels, scores)[:2]
            auPRC=auc(recall,precision)
            prc_values[motif_name]=[precision,recall,auPRC]
            
        return prc_values
    

    
