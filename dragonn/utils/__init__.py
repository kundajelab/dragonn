from __future__ import absolute_import, division, print_function
import numpy as np
import sys
from scipy.signal import correlate2d
from simdna.simulations import loaded_motifs
import pysam
import random 
ltrdict = {'a':[1,0,0,0],
           'c':[0,1,0,0],
           'g':[0,0,1,0],
           't':[0,0,0,1],
           'n':[0,0,0,0],
           'A':[1,0,0,0],
           'C':[0,1,0,0],
           'G':[0,0,1,0],
           'T':[0,0,0,1],
           'N':[0,0,0,0]}


def unpack_params(params_dict):
    import argparse
    params=argparse.Namespace()
    for key in params_dict:
        vars(params)[key]=params_dict[key]
    return params
                                                                                        
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def get_motif_scores(encoded_sequences, motif_names,
                     max_scores=None, return_positions=False, GC_fraction=0.4, pfm=None,log_pfm=None,include_rc=True):
    """
    Computes pfm log odds.

    Parameters
    ----------
    encoded_sequences : 4darray
    motif_names : list of strings
    max_scores : int, optional
    return_positions : boolean, optional
    GC_fraction : float, optional
    pfm: position weight matrix for the motif, optional
    log_pfm: log(pfm), optional, this is the format that  HOCOMOCO Provides in their PFM download links 
    include_rc: boolean indicating whether both the forward strand and the reverse complement of the motif should be used (default True) 
    Returns
    -------
    (num_samples, num_motifs, seq_length) complete score array by default.
    If max_scores, (num_samples, num_motifs*max_scores) max score array.
    If max_scores and return_positions, (num_samples, 2*num_motifs*max_scores)
    array with max scores and their positions.
    """
    encoded_sequences=np.transpose(encoded_sequences,(0,1,3,2))
    num_samples, _, _, seq_length = encoded_sequences.shape
    scores = np.ones((num_samples, len(motif_names), seq_length))
    for j, motif_name in enumerate(motif_names):
        if (pfm is None) and (log_pfm is None):
            pfm = loaded_motifs.getPwm(motif_name).getRows().T
            log_pfm = np.log(pfm)
        elif log_pfm is None:
            log_pfm = np.log(pfm)
        #get the background pfm either based on GC fraction or on shuffling the input sequence
        background_pfm = 0.5 * np.array([[1 - GC_fraction, GC_fraction,
                                          GC_fraction, 1 - GC_fraction]] * len(log_pfm[0])).T
        background_log_pfm = np.log(background_pfm)
        scores[:, j, :] =get_pssm_scores(encoded_sequences, log_pfm,include_rc=include_rc) - get_pssm_scores(encoded_sequences, background_log_pfm,include_rc=include_rc)
    if max_scores is not None:
        sorted_scores = np.sort(scores)[:, :, ::-1][:, :, :max_scores]
        if return_positions:
            sorted_positions = scores.argsort()[:, :, ::-1][:, :, :max_scores]
            return np.concatenate((sorted_scores.reshape((num_samples, len(motif_names) * max_scores)),
                                   sorted_positions.reshape((num_samples, len(motif_names) * max_scores))),
                                  axis=1)
        else:
            return sorted_scores.reshape((num_samples, len(motif_names) * max_scores))
    else:
        return scores

    
def get_pssm_scores(encoded_sequences, pssm,include_rc=True):
    """
    Convolves pssm and its reverse complement with encoded sequences
    and returns the maximum score at each position of each sequence.

    Parameters
    ----------
    encoded_sequences: 3darray
         (num_examples, 1, 4, seq_length) array
    pssm: 2darray
        (4, pssm_length) array
    rc

    Returns
    -------
    scores: 2darray
        (num_examples, seq_length) array
    """
    encoded_sequences = encoded_sequences.squeeze(axis=1)
    # initialize fwd and reverse scores to -infinity
    fwd_scores = np.full_like(encoded_sequences, -np.inf, float)
    rc_scores = np.full_like(encoded_sequences, -np.inf, float)
    # cross-correlate separately for each base,
    # for both the PSSM and its reverse complement
    for base_indx in range(encoded_sequences.shape[1]):
        base_pssm = pssm[base_indx][None]
        base_pssm_rc = base_pssm[:, ::-1]
        fwd_scores[:, base_indx, :] = correlate2d(
            encoded_sequences[:, base_indx, :], base_pssm, mode='same')
        if include_rc==True:
            rc_scores[:, base_indx, :] = correlate2d(
                encoded_sequences[:, -(base_indx + 1), :], base_pssm_rc, mode='same')
    # sum over the bases
    fwd_scores = fwd_scores.sum(axis=1)
    if include_rc==True:
        rc_scores = rc_scores.sum(axis=1)
    if include_rc==True:
    # take max of fwd and reverse scores at each position
        scores = np.maximum(fwd_scores, rc_scores)
    else:
        scores=fwd_scores
    return scores

def one_hot_from_bed(bed_entries,ref_fasta):
    ref=pysam.FastaFile(ref_fasta)
    seqs=[ref.fetch(i[0],i[1],i[2]) for i in bed_entries]
    seqs=one_hot_encode(seqs)
    return seqs

def allele_freqs_from_bed(bed_file,ref_fasta):
    import pandas as pd
    bed_entries=pd.read_csv(bed_file,header=None,sep='\t',usecols=[0,1,2])
    print("read in bed file") 
    ref=pysam.FastaFile(ref_fasta)
    print("extracted fasta") 
    seqs=''.join([ref.fetch(row[0],row[1],row[2]) for index,row in bed_entries.iterrows()])
    seqs=seqs.lower()
    a_count=seqs.count('a') 
    c_count=seqs.count('c') 
    g_count=seqs.count('g') 
    t_count=seqs.count('t')
    all_counts=a_count+c_count+g_count+t_count 
    a_freq=a_count/all_counts
    c_freq=c_count/all_counts
    g_freq=g_count/all_counts
    t_freq=t_count/all_counts
    return [a_freq,c_freq,g_freq,t_freq]



def one_hot_encode(seqs):
    encoded_seqs=np.array([[ltrdict.get(x,[0,0,0,0]) for x in seq] for seq in seqs])
    encoded_seqs=np.expand_dims(encoded_seqs,1)
    return encoded_seqs

def reverse_complement(encoded_seqs):
    return encoded_seqs[..., ::-1, ::-1]

def get_sequence_strings(encoded_sequences):
    """
    Converts encoded sequences into an array with sequence strings
    """
    num_samples, _, seq_length,_ = np.shape(encoded_sequences)
    sequence_characters = np.chararray((num_samples, seq_length))
    sequence_characters[:] = 'N'
    for i, letter in enumerate(['A', 'C', 'G', 'T']):
        try:
            letter_indxs = encoded_sequences[:, :, :,i] == 1
            sequence_characters[letter_indxs] = letter
        except:
            letter_indxs = (encoded_sequences[:, :, :,i] == 1).squeeze()
            sequence_characters[letter_indxs] = letter
    # return 1D view of sequence characters
    return [seq.decode('utf-8') for seq in sequence_characters.view('S%s' % (seq_length)).ravel()]

def fasta_from_onehot(onehot_mat,outf):
    strings=get_sequence_strings(onehot_mat)
    outf=open(outf,'w')
    for i in range(len(strings)):
        outf.write(">"+str(i)+'\n')
        outf.write(strings[i]+'\n')
        

def encode_fasta_sequences(fname):
    """
    One hot encodes sequences in fasta file
    """
    name, seq_chars = None, []
    sequences = []
    with open(fname) as fp:
        for line in fp:
            line = line.rstrip()
            if line.startswith(">"):
                if name:
                    sequences.append(''.join(seq_chars).upper())
                name, seq_chars = line, []
            else:
                seq_chars.append(line)
    if name is not None:
        sequences.append(''.join(seq_chars).upper())
    return one_hot_encode(np.array(sequences))


    
