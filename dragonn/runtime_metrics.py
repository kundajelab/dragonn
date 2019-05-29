#Note: this is ugly w/ use of tf & K --> needed to avoid custom keras modifications 
import tensorflow as tf
import keras.backend as K

pseudocount=0.01


def contingency_table(y, z):
    """Note:  if y and z are not rounded to 0 or 1, they are ignored
    """
    y = K.cast(K.round(y), K.floatx())
    z = K.cast(K.round(z), K.floatx())
    
    def count_matches(y, z):
        return K.sum(K.cast(y, K.floatx()) * K.cast(z, K.floatx()))
    
    ones = K.ones_like(y)
    zeros = K.zeros_like(y)
    y_ones = K.equal(y, ones)
    y_zeros = K.equal(y, zeros)
    z_ones = K.equal(z, ones)
    z_zeros = K.equal(z, zeros)
    
    tp = count_matches(y_ones, z_ones)
    tn = count_matches(y_zeros, z_zeros)
    fp = count_matches(y_zeros, z_ones)
    fn = count_matches(y_ones, z_zeros)
    return (tp, tn, fp, fn)

def recall(y, z):
    """True positive rate `tp / (tp + fn)`
    """
    tp, tn, fp, fn = contingency_table(y, z)
    return tp / (tp + fn+pseudocount)


def specificity(y, z):
    """True negative rate `tn / (tn + fp)`
    """
    tp, tn, fp, fn = contingency_table(y, z)
    return tn / (tn + fp+pseudocount)


def fpr(y, z):
    """False positive rate `fp / (fp + tn)`
    """
    tp, tn, fp, fn = contingency_table(y, z)
    return fp / (fp + tn+pseudocount)


def fnr(y, z):
    """False negative rate `fn / (fn + tp)`
    """
    tp, tn, fp, fn = contingency_table(y, z)
    return fn / (fn + tp+pseudocount)


def precision(y, z):
    """Precision `tp / (tp + fp)`
    """
    tp, tn, fp, fn = contingency_table(y, z)
    return tp / (tp + fp+pseudocount)


def fdr(y, z):
    """False discovery rate `fp / (tp + fp)`
    """
    tp, tn, fp, fn = contingency_table(y, z)
    return fp / (tp + fp+pseudocount)


def f1(y, z):
    """F1 score: `2 * (p * r) / (p + r)`, where p=precision and r=recall.
    """
    _recall = recall(y, z)
    _prec = precision(y, z)
    return 2 * (_prec * _recall) / (_prec + _recall+pseudocount)


def spearman_corr(y_true,y_pred):
    import K.contribs.metrics.streaming_pearson_correlation
    return K.contribs.metrics.streaming_pearson_correlation(y_pred,y_true)

