from keras.utils import Sequence
import pandas as pd
import numpy as np
import random
import math 
import pysam
from dragonn.utils import ltrdict
import threading 

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


def open_data_file(data_path,tasks,num_to_read=None):
    if data_path.endswith('.hdf5'):
        if (tasks is None) and (num_to_read is not None):
            data=pd.read_hdf(data_path,start=0,stop=num_to_read)
        elif (tasks is not None) and (num_to_read is None):
            data=pd.read_hdf(data_path,columns=tasks)
        elif (tasks is None) and (num_to_read is None):
            data=pd.read_hdf(data_path)
        else: 
            data=pd.read_hdf(data_path,columns=tasks,start=0,stop=num_to_read)
    else:
        #treat as bed file 
        if (tasks is None) and (num_to_read is not None):
            data=pd.read_csv(data_path,header=0,sep='\t',index_col=[0,1,2],start=0,stop=num_to_read)
        elif (tasks is None) and (num_to_read is None):
            data=pd.read_csv(data_path,header=0,sep='\t',index_col=[0,1,2])
        else:
            data=pd.read_csv(data_path,header=0,sep='\t',nrows=1)
            chrom_col=data.columns[0]
            start_col=data.columns[1]
            end_col=data.columns[2]
            if num_to_read is None:
                data=pd.read_csv(data_path,header=0,sep='\t',usecols=[chrom_col,start_col,end_col]+tasks,index_col=[0,1,2])
            else:
                data=pd.read_csv(data_path,header=0,sep='\t',usecols=[chrom_col,start_col,end_col]+tasks,index_col=[0,1,2],start=0,stop=num_to_read)
    return data 

#use wrappers for keras Sequence generator class to allow batch shuffling upon epoch end
class DataGenerator(Sequence):
    def __init__(self,data_path,ref_fasta,batch_size=128,add_revcomp=True,tasks=None,shuffled_ref_negatives=False,upsample=True,upsample_ratio=0.1,upsample_thresh=0.5,num_to_read=None):
        self.lock = threading.Lock()        
        self.batch_size=batch_size
        #decide if reverse complement should be used
        self.add_revcomp=add_revcomp
        if add_revcomp==True:
            self.batch_size=int(batch_size/2)

        #determine whether negative set should consist of the shuffled refs.
        # If so, split batch size in 2, as each batch will be augmented with shuffled ref negatives
        # in ratio equal to positives 
        self.shuffled_ref_negatives=shuffled_ref_negatives
        if self.shuffled_ref_negatives==True:
            self.batch_size=int(self.batch_size/2)
            
        #open the reference file
        self.ref_fasta=ref_fasta
        self.data=open_data_file(data_path,tasks,num_to_read)
        
        self.indices=np.arange(self.data.shape[0])
        num_indices=self.indices.shape[0]
        self.add_revcomp=add_revcomp
        
        #set variables needed for upsampling the positives
        self.upsample=upsample
        if self.upsample==True:
            self.upsample_ratio=upsample_ratio
            self.upsample_thresh=upsample_thresh
            self.ones = self.data.loc[(self.data >= self.upsample_thresh).any(axis=1)]
            self.zeros = self.data.loc[(self.data < self.upsample_thresh).all(axis=1)]
            self.pos_batch_size = int(self.batch_size * self.upsample_ratio)
            self.neg_batch_size = self.batch_size - self.pos_batch_size
            self.pos_indices=np.arange(self.ones.shape[0])
            self.neg_indices=np.arange(self.zeros.shape[0])
            
            #wrap the positive and negative indices to reach size of self.indices
            num_pos_wraps=math.ceil(num_indices/self.pos_indices.shape[0])
            num_neg_wraps=math.ceil(num_indices/self.neg_indices.shape[0])
            self.pos_indices=np.repeat(self.pos_indices,num_pos_wraps)[0:num_indices]
            np.random.shuffle(self.pos_indices)
            self.neg_indices=np.repeat(self.neg_indices,num_neg_wraps)[0:num_indices]
            np.random.shuffle(self.neg_indices)
            
    def __len__(self):
        return math.ceil(self.data.shape[0]/self.batch_size)

    def __getitem__(self,idx):
        with self.lock:
            self.ref=pysam.FastaFile(self.ref_fasta)
            if self.shuffled_ref_negatives==True:
                return self.get_shuffled_ref_negatives_batch(idx)
            elif self.upsample==True:
                return self.get_upsampled_positives_batch(idx)
            else:
                return self.get_basic_batch(idx) 
        
    def get_shuffled_ref_negatives_batch(self,idx): 
        #get seq positions
        inds=self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        bed_entries=self.data.index[inds]
        #get sequences
        seqs=[self.ref.fetch(i[0],i[1],i[2]) for i in bed_entries]
        if self.add_revcomp==True:
            #add in the reverse-complemented sequences for training.
            seqs_rc=[revcomp(s) for s in seqs]
            seqs=seqs+seqs_rc
            
        #generate the corresponding negative set by dinucleotide-shuffling the sequences
        seqs_shuffled=[dinuc_shuffle(s) for s in seqs]
        seqs=seqs+seqs_shuffled
        #one-hot-encode the fasta sequences 
        seqs=np.array([[ltrdict.get(x,[0,0,0,0]) for x in seq] for seq in seqs])
        x_batch=np.expand_dims(seqs,1)
        y_batch=np.asarray(self.data.iloc[inds])
        if self.add_revcomp==True:
            y_batch=np.concatenate((y_batch,y_batch),axis=0)
        y_shape=y_batch.shape 
        y_batch=np.concatenate((y_batch,np.zeros(y_shape)))
        return (x_batch,y_batch)

    def get_upsampled_positives_batch(self,idx):
        #get seq positions
        pos_inds=self.pos_indices[idx*self.pos_batch_size:(idx+1)*self.pos_batch_size]
        pos_bed_entries=self.ones.index[pos_inds]
        neg_inds=self.neg_indices[idx*self.neg_batch_size:(idx+1)*self.neg_batch_size]
        neg_bed_entries=self.zeros.index[neg_inds]
    
        #print(neg_inds[0:10])
        #bed_entries=pos_bed_entries+neg_bed_entries

        #get sequences
        pos_seqs=[self.ref.fetch(i[0],i[1],i[2]) for i in pos_bed_entries]
        neg_seqs=[self.ref.fetch(i[0],i[1],i[2]) for i in neg_bed_entries]
        seqs=pos_seqs+neg_seqs 
        if self.add_revcomp==True:
            #add in the reverse-complemented sequences for training.
            seqs_rc=[revcomp(s) for s in seqs]
            seqs=seqs+seqs_rc
            
        #one-hot-encode the fasta sequences 
        seqs=np.array([[ltrdict.get(x,[0,0,0,0]) for x in seq] for seq in seqs])
        x_batch=np.expand_dims(seqs,1)
        
        #extract the positive and negative labels at the current batch of indices
        y_batch_pos=self.ones.iloc[pos_inds]
        y_batch_neg=self.zeros.iloc[neg_inds]
        y_batch=np.concatenate((y_batch_pos,y_batch_neg),axis=0)
        #add in the labels for the reverse complement sequences, if used 
        if self.add_revcomp==True:
            y_batch=np.concatenate((y_batch,y_batch),axis=0)
        return (x_batch,y_batch)            
    
    def get_basic_batch(self,idx):
        #get seq positions
        inds=self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        bed_entries=self.data.index[inds]
        #get sequences
        seqs=[self.ref.fetch(i[0],i[1],i[2]) for i in bed_entries]
        if self.add_revcomp==True:
            #add in the reverse-complemented sequences for training.
            seqs_rc=[revcomp(s) for s in seqs]
            seqs=seqs+seqs_rc
        #one-hot-encode the fasta sequences 
        seqs=np.array([[ltrdict.get(x,[0,0,0,0]) for x in seq] for seq in seqs])
        x_batch=np.expand_dims(seqs,1)
        #extract the labels at the current batch of indices 
        y_batch=np.asarray(self.data.iloc[inds])
        #add in the labels for the reverse complement sequences, if used 
        if self.add_revcomp==True:
            y_batch=np.concatenate((y_batch,y_batch),axis=0)
        return (x_batch,y_batch)    
    
    def on_epoch_end(self):
        #if upsampling is being used, shuffle the positive and negative indices 
        if self.upsample==True:
            np.random.shuffle(self.pos_indices)
            np.random.shuffle(self.neg_indices)
        else:
            np.random.shuffle(self.indices)

#generate batches of bQTL data with specified allele column name and flank size 
class BQTLGenerator(DataGenerator):
    def __init__(self,data_path,ref_fasta,allele_col,flank_size=500,batch_size=128):
        DataGenerator.__init__(self,data_path,ref_fasta,batch_size,add_revcomp=False,upsample=False)
        self.allele_col=allele_col
        self.flank_size=flank_size 

    #override the get_basic_batch definition to extract flanks around the bQTL
    # and insert the specified allele 
    def get_basic_batch(self,idx): 
        #get seq positions
        inds=self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        
        entries=self.data.iloc[inds]
        seqs=[]
        for index,row in entries.iterrows():
            allele=row[self.allele_col]
            chrom=index[0]
            pos=index[1]
            left_flank_start=(pos-1)-self.flank_size
            left_flank_end=(pos-1)
            right_flank_start=pos
            right_flank_end=(pos+self.flank_size-1)
            left_seq=self.ref.fetch(chrom,left_flank_start,left_flank_end)
            right_seq=self.ref.fetch(chrom,right_flank_start,right_flank_end)
            seq=left_seq+allele+right_seq
            seqs.append(seq)
            
        #one-hot-encode the fasta sequences 
        seqs=np.array([[ltrdict.get(x,[0,0,0,0]) for x in seq] for seq in seqs])
        x_batch=np.expand_dims(seqs,1)
        return x_batch

#Returns batch index in addition to x,y labels. Useful for prediction   
class IndexedBatchGenerator(DataGenerator):
    def __init__(self,data_path,ref_fasta,batch_size=128,add_revcomp=True,tasks=None,shuffled_ref_negatives=False,upsample=True,upsample_ratio=0.1,upsample_thresh=0.5):
        DataGenerator.__init__(self,data_path,ref_fasta,batch_size,add_revcomp=False,upsample=False)

    def __getitem__(self,idx):
        with self.lock:
            self.ref=pysam.FastaFile(self.ref_fasta)
            if self.shuffled_ref_negatives==True:
                (x_batch,y_batch)=self.get_shuffled_ref_negatives_batch(idx)
                return (x_batch,y_batch,idx)
            elif self.upsample==True:
                (x_batch,y_batch)=self.get_upsampled_positives_batch(idx)
                return (x_batch,y_batch,idx) 
            else:
                (x_batch,y_batch)=self.get_basic_batch(idx)
                return (x_batch,y_batch,idx)
            
    def on_epoch_end(self):
        return
    
