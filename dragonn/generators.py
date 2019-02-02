from argparse import Namespace
import numpy as np
import pandas as pd
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

def dinuc_shuffle(seq):
    #get list of dinucleotides
    nucs=[]
    for i in range(0,len(seq),2):
        nucs.append(seq[i:i+2])
    #generate a random permutation
    random.shuffle(nucs)
    return ''.join(nucs) 


def revcomp(seq):
    seq=seq[::-1].upper()
    comp_dict=dict()
    comp_dict['A']='T'
    comp_dict['T']='A'
    comp_dict['C']='G'
    comp_dict['G']='C'
    rc=[]
    for base in seq:
        if base in comp_dict:
            rc.append(comp_dict[base])
        else:
            rc.append(base)
    return ''.join(rc)

    
def shuffled_ref_generator(data_path,ref_fasta,batch_size=128,add_revcomp=True,tasks=None):
    '''
    Generates tuple([x_batch, y_batch]). x_batch is a numpy ndarray (batch_size, 1, seq_length,4). y_batch is an ndarray (batch_size,ntasks). regions from data_path are assumed to be positives. These are dinucleotide-shuffled to generate corresponding negatives. 

    Inputs: 
    data_path: string containing path to the bed file (with task names in header) of the positive regions to use for training 
    ref_fasta: .fa.gz or .fa file containing the reference genome. 
    batch_size: batch_size for input to keras model training. Default=128 
    add_revcomp: Boolean indicating whether or not reverse-complement sequences should be extracted for entries in each batch. Default=True 
    tasks: a list of task names to extract from the input bed file (i.e. useful if you only want to train on a subset of the tasks). Default=None means that all tasks will be extracted from data_path. 
    '''
    #open the reference file
    ref=pysam.FastaFile(ref_fasta)

    #read in the label bed file 
    data=pd.read_csv(data_path,header=0,sep='\t',index_col=[0,1,2])
    if tasks!=None:
        data=data[tasks]
    #iterate through batches and one-hot-encode on the fly
    start_index=0
    num_generated=0
    total_entries=data.shape[0]-batch_size
    
    #decide if reverse complement should be used
    if add_revcomp==True:
        batch_size=batch_size/4
    else:
        batch_size=batch_size
        
    while True:
        if (num_generated >=total_entries):
            start_index=0
        end_index=start_index+int(batch_size)
        #get seq positions
        bed_entries=data.index[start_index:end_index]
        #bed_entries=[data.index[i] for i in range(start_index,end_index)]
        #get sequences
        seqs=[ref.fetch(i[0],i[1],i[2]) for i in bed_entries]
        if add_revcomp==True:
            #add in the reverse-complemented sequences for training.
            seqs_rc=[revcomp(s) for s in seqs]
            seqs=seqs+seqs_rc
            
        #generate the corresponding negative set by dinucleotide-shuffling the sequences
        seqs_shuffled=[dinuc_shuffle(s) for s in seqs]
        seqs=seqs+seqs_shuffled

        #one-hot-encode the fasta sequences 
        seqs=np.array([[ltrdict.get(x,[0,0,0,0]) for x in seq] for seq in seqs])
        
        x_batch=np.expand_dims(seqs,1)
        y_batch=np.asarray(data[start_index:end_index])
        if add_revcomp==True:
            y_batch=np.concatenate((y_batch,y_batch),axis=0)
        y_shape=y_batch.shape 
        y_batch=np.concatenate((y_batch,np.zeros(y_shape)))
        num_generated+=batch_size
        start_index=end_index
        yield tuple([x_batch,y_batch])


def data_generator(data_path,ref_fasta,batch_size=128,add_revcomp=True,tasks=None,upsample=True,upsample_ratio=0.1):
    if upsample==False:
        return data_generator_bed(data_path, ref_fasta,batch_size,add_revcomp,tasks)
    else:
        return data_generator_bed_upsample(data_path, ref_fasta, batch_size, add_revcomp, tasks, upsample_ratio)

def data_generator_bed_upsample(data_path,ref_fasta,batch_size=128,add_revcomp=True,tasks=None,upsample_ratio=0.1):
    #open the reference file
    ref=pysam.FastaFile(ref_fasta)
    #load the train data as a pandas dataframe, skip the header
    data=pd.read_csv(bed_source,header=0,sep='\t',index_col=[0,1,2])
    if tasks!=None:
        data=data[tasks]
    ones = data.loc[(data > 0).any(axis=1)]
    zeros = data.loc[(data < 1).all(axis=1)]
    #decide if reverse complement should be used
    if add_revcomp==True:
        batch_size=batch_size/2
    else:
        batch_size=batch_size
    pos_batch_size = int(batch_size * upsample_ratio)
    neg_batch_size = batch_size - pos_batch_size
    #iterate through batches and one-hot-encode on the fly
    pos_start_index = 0
    pos_num_generated = 0
    pos_total_entries = ones.shape[0] - pos_batch_size
    neg_start_index = 0
    neg_num_generated = 0
    neg_total_entries = zeros.shape[0] - neg_batch_size
    while True:
        if (pos_num_generated >= pos_total_entries):
            pos_start_index=0
            ones = pd.concat([ones[pos_num_generated:], ones[:pos_num_generated]])
            pos_num_generated = 0
        if (neg_num_generated >= neg_total_entries):
            neg_start_index = 0
            zeros = pd.concat([zeros[neg_num_generated:], zeros[:neg_num_generated]])
            neg_num_generated = 0
        pos_end_index = pos_start_index + int(pos_batch_size)
        neg_end_index = neg_start_index + int(neg_batch_size)
        #get seq positions
        pos_bed_entries=[(ones.index[i]) for i in range(pos_start_index,pos_end_index)]
        neg_bed_entries=[(zeros.index[i]) for i in range(neg_start_index, neg_end_index)]
        bed_entries = pos_bed_entries + neg_bed_entries
        #get sequences
        seqs=[ref.fetch(i[0],i[1],i[2]) for i in bed_entries]
        if add_revcomp==True:
            #add in the reverse-complemented sequences for training.
            seqs_rc=[revcomp(s) for s in seqs]
            seqs=seqs+seqs_rc
        seqs=np.array([[ltrdict.get(x,[0,0,0,0]) for x in seq] for seq in seqs])
        x_batch=np.expand_dims(seqs,1)
        y_labels_ones = ones[pos_start_index:pos_end_index]
        y_labels_zeros = zeros[neg_start_index:neg_end_index]
        y_batch_ones = np.asarray(y_labels_ones)
        y_batch_zeros = np.asarray(y_labels_zeros)
        y_batch = np.concatenate([y_batch_ones, y_batch_zeros])
        if add_revcomp==True:
            y_batch=np.concatenate((y_batch,y_batch),axis=0)
        pos_num_generated += pos_batch_size
        neg_num_generated += neg_batch_size
        pos_start_index = pos_end_index
        neg_start_index = neg_end_index
        yield tuple([x_batch,y_batch])

def data_generator_bed(data_path,ref_fasta,batch_size=128,add_revcomp=True,tasks=None):
    #open the reference file
    ref=pysam.FastaFile(ref_fasta)
    data=pd.read_csv(bed_source,header=0,sep='\t',index_col=[0,1,2])
    if tasks!=None:
        data=data[tasks]
    #iterate through batches and one-hot-encode on the fly
    start_index=0
    num_generated=0
    total_entries=data.shape[0]-batch_size
    #decide if reverse complement should be used
    if add_revcomp==True:
        batch_size=batch_size/2
    else:
        batch_size=batch_size
    while True:
        if (num_generated >=total_entries):
            start_index=0
        end_index=start_index+int(batch_size)
        #get seq positions
        bed_entries=[(data.index[i]) for i in range(start_index,end_index)]
        #get sequences
        seqs=[ref.fetch(i[0],i[1],i[2]) for i in bed_entries]
        if add_revcomp==True:
            #add in the reverse-complemented sequences for training.
            seqs_rc=[revcomp(s) for s in seqs]
            seqs=seqs+seqs_rc
        seqs=np.array([[ltrdict.get(x,[0,0,0,0]) for x in seq] for seq in seqs])
        x_batch=np.expand_dims(seqs,1)
        y_batch=np.asarray(data[start_index:end_index])
        if add_revcomp==True:
            y_batch=np.concatenate((y_batch,y_batch),axis=0)
        num_generated+=batch_size
        start_index=end_index
        yield tuple([x_batch,y_batch])
