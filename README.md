# [DragoNN](http://kundajelab.github.io/dragonn/)
[![Build Status](https://travis-ci.org/kundajelab/dragonn.svg?branch=master)](https://travis-ci.org/kundajelab/dragonn)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/kundajelab/dragonn/blob/master/LICENSE)

The `dragonn` package implements Deep RegulAtory GenOmic Neural Networks (DragoNNs) for predictive modeling of regulatory genomics, nucleotide-resolution feature discovery, and simulations for systematic development and benchmarking.

![demo](http://i.imgur.com/1fAgrt2.gif)


## 15 seconds to your first DragoNN model
The `dragonn` package provides a simple command line interface to train DragoNN models, test them, and predict on sequence data. Train an example model by running:

```
dragonn train --pos-sequences examples/example_pos_sequences.fa --neg-sequences examples/example_neg_sequences.fa --prefix training_example
```

This will store a model file, training_example.model.json, with the model architecture and a weights file, training_example.weights.hd5, with the parameters of the trained model. Test the model by running:

```
dragonn test --pos-sequences examples/example_pos_sequences.fa --neg-sequences examples/example_neg_sequences.fa --arch-file training_example.model.json --weights-file training_example.weights.hd5
```

This will print the model's test performance metrics. Model predictions on sequence data can be obtained by running:

```
dragonn predict --sequences examples/example_pos_sequences.fa --arch-file training_example.model.json --weights-file training_example.weights.hd5 --output-file example_predictions.txt
```

This will store the model predictions for sequences in example_pos_sequences.fa in the output file example_predictions.txt. Interpret sequence data with a dragonn model by running:

```
dragonn interpret --sequences examples/example_pos_sequences.fa --arch-file training_example.model.json --weights-file training_example.weights.hd5 --prefix example_interpretation
```

This will write the most important subsequence in each input sequence along with its location in the input sequence in the file example_interpretation.task_0.important_sequences.txt.
Note: by default, only examples with predicted positive class probability >0.5 are interpreted. Examples below this thershold yield important subsequence of Ns with location -1. This default can be changed with the flag --pos-thershold.

We encourage DragoNN users to share models in the [Model Zoo](https://github.com/kundajelab/dragonn/wiki/Model-Zoo). Enjoy!

## Upcoming Features

See our [roadmap](https://github.com/kundajelab/dragonn/issues/5) for an outline of upcoming features. Additional feature suggestions are always welcome!

