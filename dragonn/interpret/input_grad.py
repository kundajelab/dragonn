#Careful! Gradientxinput is summed across tasks, there is no support in tensorflow for calculating the per-task gradient
#(see thread here: https://github.com/tensorflow/tensorflow/issues/4897) 
def input_grad(model,X,target_layer_idx=-2):
    print("WARNING: this function provides aggregated gradients across tasks. Not recommended for multi-tasked models")
    from keras import backend as K 
    fn = K.function([model.input], K.gradients(model.layers[target_layer_idx].output, [model.input]))
    return fn([X])[0]


