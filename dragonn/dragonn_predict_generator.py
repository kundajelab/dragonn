from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import numpy as np

from keras.utils.data_utils import Sequence
from keras.utils.data_utils import OrderedEnqueuer
from keras.utils.generic_utils import Progbar
from keras.utils.generic_utils import to_list
from keras.utils.generic_utils import unpack_singleton
from keras import callbacks as cbks


def dragonn_predict_generator(model, generator,
                              steps=None,
                              max_queue_size=10,
                              workers=1,
                              use_multiprocessing=False,
                              verbose=1):
    """See docstring for `Model.predict_generator`."""
    model._make_predict_function()
    generator_indices=generator.indices
    prediction_indices=None
    batch_size=generator.batch_size 
    steps_done = 0
    wait_time = 0.01
    all_outs = []
    steps=len(generator)
    enqueuer = OrderedEnqueuer(
        generator,
        use_multiprocessing=use_multiprocessing)
    enqueuer.start(workers=workers, max_queue_size=max_queue_size)
    output_generator = enqueuer.get()
    if verbose == 1:
        progbar = Progbar(target=steps)
    try:
        while steps_done < steps:
            generator_output = next(output_generator)
            #print("got batch") 
            if isinstance(generator_output, tuple):
                # Compatibility with the generators
                # used for training.
                if len(generator_output) == 2:
                    x, idx = generator_output
                elif len(generator_output) == 3:
                    x, y, idx = generator_output
                else:
                    raise ValueError('Output of generator should be '
                                     'a tuple `(x, y, idx)` '
                                     'or `(x, idx)`. Found: ' +
                                     str(generator_output))
            else:
                raise ValueError('Output of generator should be '
                                 'a tuple `(x, y, idx)` '
                                 'or `(x, idx)`. Found: ' +
                                 str(generator_output))
            outs = model.predict_on_batch(x)
            cur_inds=generator_indices[idx*batch_size:(idx+1)*batch_size]
            if prediction_indices is None:
                prediction_indices=cur_inds
            else:
                prediction_indices=np.concatenate((prediction_indices,cur_inds),axis=0)
            outs = to_list(outs)
            if not all_outs:
                for out in outs:
                    all_outs.append([])

            for i, out in enumerate(outs):
                all_outs[i].append(out)
            steps_done += 1
            if verbose == 1:
                progbar.update(steps_done)
    except:
        print("Error, stopping enqueuer") 
        enqueuer.stop()
        print("exiting")
    enqueuer.stop()
    
    if len(all_outs) == 1:
        if steps_done == 1:
            return (all_outs[0][0],prediction_indices)
        else:
            return (np.concatenate(all_outs[0]),prediction_indices)
    if steps_done == 1:
        return ([out[0] for out in all_outs],prediction_indices)
    else:
        return ([np.concatenate(out) for out in all_outs],prediction_indices)
