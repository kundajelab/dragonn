import gzip
from collections import namedtuple
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

class BQTL(object):
    def __init__(self, chrom, pos,
                 depth, altdepth, refdepth,
                 altallele, postallele,
                 postfreq,prefreq,pvalue):
        self.chrom=chrom
        self.start=pos-1
        self.end=pos
        self.depth=depth
        self.altdepth=altdepth
        self.refdepth=refdepth
        self.altallele=altallele
        self.postallele=postallele
        self.postfreq=postfreq
        self.prefreq=prefreq
        self.logratio=np.log((self.postfreq+0.01)/(self.prefreq+0.01))
        self.pvalue=pvalue



def sample_matched_bqtls(bqtls_to_match, bqtls_to_sample, attrfunc):
    #sort bqtls_to_sample by attr_name
    sorted_bqtls_to_sample = sorted([x for x in bqtls_to_sample
                                     if np.isnan(attrfunc(x))==False],
                                    key=lambda x: attrfunc(x))
    sorted_bqtls_to_sample_vals = [attrfunc(x) for x in sorted_bqtls_to_sample]
    bqtls_to_match_vals = [attrfunc(x) for x in bqtls_to_match]
    searchsorted_indices = np.searchsorted(a=sorted_bqtls_to_sample_vals, v=bqtls_to_match_vals)
    
    matched_sampled_bqtls_indices = set()
    
    for idx in searchsorted_indices:
        #shift the index until you find one that isn't taken
        shift = 1
        while (idx in matched_sampled_bqtls_indices or idx==len(sorted_bqtls_to_sample)):
            if idx == len(sorted_bqtls_to_sample):
                shift = -1
            idx += shift
        if (idx < 0 or idx > len(sorted_bqtls_to_sample)):
            print(idx)
        matched_sampled_bqtls_indices.add(idx)        
    matched_sampled_bqtls = [sorted_bqtls_to_sample[idx] for idx in sorted(matched_sampled_bqtls_indices)]
    sns.distplot([attrfunc(x) for x in bqtls_to_match])
    sns.distplot([attrfunc(x) for x in matched_sampled_bqtls])
    plt.show()    
    return matched_sampled_bqtls
