---
layout: default
title: {{ site.name }}
---
# Tutorial Overview

We begin with instructions to configure software and hardware for DragoNN tutorials. We provide two tutorials, one for beginner users and another for advanced users. The beginner tutorial provides ipython notebooks that simulate regulatory sequence, train DragoNN models on them, and show how to how to interpret them. The advanced tutorial show how to use the `dragonn` command-line interface to train and test DragoNN models on fasta data, and predict and interpret fasta data.    

# Table of Contents
 - [Getting access to software and hardware](#getting-access-to-dragonn-software-and-gpu-hardware])  
 - [IPython Noteboook Tutorial](#ipython-notebook-tutorial) 
 - [Command-line Tutorial](#command-line-tutorial) 

## Getting access to DragoNN software and GPU hardware

The tutorial requires access to the DragoNN software and Graphical Porcessing Units (GPUs) to run the software. If you do not have access to GPUs, see our [cloud resources]({{ site.baseurl }}/cloud_resources.html) to access a public image through Amazon Web Services with the necessary software and hardware.

If you do have access to GPUs, you can install DragoNN locally using [Anaconda](<http://www.continuum.io/downloads>).
Once you have installed Anaconda for your computing platform, the latest released version of DragoNN can be installed with the following command from the terminal:

```
conda install -c kundajelab dragonn
```
This will be sufficient for the IPython notebook tutorial. The command-line tutorial uses features in development that have not been released yet. To access code in development, clone the [DragoNN repository](<https://github.com/kundajelab/dragonn>) and run:

```
python setup.py install
```

## IPython Notebook Tutorial

To explore the ipython notebook tutorial, navigate to the examples directory and run

```
jupyter notebook 
```

This will start a jupter notebook server, allowing you to navigate to the **workshop_tutorial.ipynb** notebook in your browser. 

## Command-line Tutorial

The `dragonn` package provides a simple command line interface to train DragoNN models, test them, and predict on sequence data. Train an example model by running:

```
dragonn train --pos-sequences examples/example_pos_sequences.fa --neg-sequences examples/example_neg_sequences.fa --prefix training_example
```

This will store a model file, training_example.model.json, with the model architecture and a weights file, training_example.weights.hd5, with the parameters of the trained model. Test the model by running:

```
dragonn test --pos-sequences examples/example_pos_sequences.fa --neg-sequences examples/example_neg_sequences.fa --model-file training_example.model.json --weights-file training_example.weights.hd5
```

This will print the model's test performance metrics. Model predictions on sequence data can be obtained by running:

```
dragonn predict --sequences examples/example_pos_sequences.fa --model-file training_example.model.json --weights-file training_example.weights.hd5 --output-file example_predictions.txt
```

This will store the model predictions for sequences in example_pos_sequences.fa in the output file example_predictions.txt.


We encourage DragoNN users to share models in the [Model Zoo](https://github.com/kundajelab/dragonn/wiki/Model-Zoo). Enjoy!