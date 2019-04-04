def input_grad(model,X,layer_idx=-2):
    from keras import backend as K 
    fn = K.function([model.input], K.gradients(model.layers[layer_idx].output, [model.input]))
    return fn([X])[0]


