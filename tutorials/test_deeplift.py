from keras.models import load_model
from dragonn.tutorial_utils import deeplift
from dragonn.utils import get_sequence_strings, one_hot_encode
from deeplift import dinuc_shuffle
import numpy as np


import wget
url_data="http://mitra.stanford.edu/kundaje/projects/dragonn/deep_lift_input_classification_spi1.npy"
url_model="http://mitra.stanford.edu/kundaje/projects/dragonn/SPI1.classification.model.hdf5"
wget.download(url_data)
wget.download(url_model) 


deep_lift_input_classification_spi1=np.load("deep_lift_input_classification_spi1.npy")
deep_lift_input_classification_spi1_strings=get_sequence_strings(deep_lift_input_classification_spi1)

#get scores with GC reference 
deep_lift_scores_spi1_gc_ref=deeplift("SPI1.classification.model.hdf5",deep_lift_input_classification_spi1,reference="gc_ref")
print(deep_lift_scores_spi1_gc_ref.shape)
print(np.max(deep_lift_scores_spi1_gc_ref))
print(np.min(deep_lift_scores_spi1_gc_ref))


#Get scores with shuffled reference (starting with strings ) 
deep_lift_scores_spi1_shuffled_ref_strings=deeplift("SPI1.classification.model.hdf5",deep_lift_input_classification_spi1_strings,one_hot_func=one_hot_encode)
print(deep_lift_scores_spi1_shuffled_ref_strings.shape)
print(np.max(deep_lift_scores_spi1_shuffled_ref_strings))
print(np.min(deep_lift_scores_spi1_shuffled_ref_strings))


#Get scores with shuffled reference (starting with one-hot-encoded ) 
deep_lift_scores_spi1_shuffled_ref=deeplift("SPI1.classification.model.hdf5",deep_lift_input_classification_spi1,one_hot_func=None)
print(deep_lift_scores_spi1_shuffled_ref.shape)
print(np.max(deep_lift_scores_spi1_shuffled_ref))
print(np.min(deep_lift_scores_spi1_shuffled_ref))


assert deep_lift_scores_spi1_shuffled_ref_strings.all()==deep_lift_scores_spi1_shuffled_ref.all()

