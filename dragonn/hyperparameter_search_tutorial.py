#To prepare for model training, we import the necessary functions and submodules from keras
from keras.models import Sequential
from keras.layers.core import Dropout, Reshape, Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adadelta, SGD, RMSprop;
import keras.losses;
from keras.constraints import maxnorm;
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping, History
from keras import backend as K
K.set_image_data_format('channels_last')
from dragonn.callbacks import *

#Import matplotlib utilities for plotting grid search results: 
import matplotlib.pyplot as plt


def hyperparam_get_model(input_shape,
              num_tasks=1,
              num_layers=1,
              num_filters=10,
              kernel_size=15,
              pool_size=35):
    '''
    Performs hyperparameter search in accordance with Figure 6 in the DragoNN manuscript 

    input_shape: tuple with dimensions of input 
    num_tasks: number of prediction tasks, default=1 
    num_layers: number of 2D convolution layers, default=1 
    num_filters: number of filters to use in each 2D convolution layer, default=10 
    kernel_size: filter dimension for each 2D convolution layer, (1,n), default n=15  
    pool_size: 2D MaxPooling pool size, default=35 

    Returns a Sequential model with the specified parameters. 
    '''
    model=Sequential()
    for i in range(num_layers):
        model.add(Conv2D(filters=num_filters,
                         kernel_size=(1,kernel_size),
                         input_shape=input_shape))
        model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(1,pool_size)))
    model.add(Flatten())
    model.add(Dense(num_tasks))
    model.add(Activation("sigmoid"))
    model.compile(optimizer="adam",loss="binary_crossentropy")
    return model

def hyperparam_train_model(data,model,num_training_examples=None,epochs=150,patience=7):
    #train the model
    if num_training_examples==None:
        #use all training examples
        x=data.X_train
        y=data.y_train
    else:
        #use the specified number of training examples 
        x=data.X_train[0:num_training_examples]
        y=data.y_train[0:num_training_examples]
        
    history=model.fit(x=x,
                      y=y,
                      batch_size=128,
                      epochs=epochs,
                      verbose=0,
                      callbacks=[EarlyStopping(patience=patience),
                                 History()],
                      validation_data=(data.X_valid,data.y_valid))
    
    #get auPRC on test set
    test_predictions=model.predict(data.X_test)
    test_performance=ClassificationResult(data.y_test,test_predictions)
    return [i['auROC'] for i in test_performance.results]

def hyperparam_plot_test_auPRC(param_grid,auPRC_dict,xlabel="",ylabel=""):
    #define a list of colors for multi-tasked models 
    colors=["#000000","#1f78b4","#e31a1c","#33a02c","#ff7f00","#a6cee3","#cab2d6","#fdbf6f","#fb9a99","#b2df8a"]
    plt.figure(1,figsize=(20,5))
    datasets=list(auPRC_dict.keys())
    num_datasets=len(datasets)
    
    for i in range(num_datasets):
        cur_dataset=datasets[i]
        cur_auPRC=auPRC_dict[cur_dataset]
        
        #create subplot for current dataset
        plt.subplot(1,num_datasets,i+1)

        #make sure that the input is not an empty list (i.e. some performrance values recorded) 
        assert len(cur_auPRC)>0

        num_tasks=len(cur_auPRC[0])        
        #for purposes of the tutorial, we want to limit analysis to 10 tasks. 
        assert num_tasks < 11

        #add line for param value vs test auPRC for each task 
        for task_index in range(num_tasks):
            x=param_grid
            y=[entry[task_index] for entry in cur_auPRC]
            plt.plot(x,y,colors[task_index])
            plt.title(cur_dataset)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.ylim(0.4,1)
    plt.subplots_adjust(hspace=0) 
    plt.show() 



