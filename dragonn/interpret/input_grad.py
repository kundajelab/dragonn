import tensorflow as tf

def input_grad(model,X,target_layer_idx=-2):
    print("WARNING: this function provides aggregated gradients across tasks. Not recommended for multi-tasked models")
    X = tf.Variable(X, dtype=tf.float32)

    # if not taking final output
    if target_layer_idx != -1:
        int_model = tf.keras.Model(model.input, model.layers[target_layer_idx].output)
    else:
        int_model = model

    with tf.GradientTape() as tape:
        out = tf.reduce_sum(int_model(X))

    grad = tape.gradient(out, X)
    return grad.numpy()
