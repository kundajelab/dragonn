---
layout: default
title: {{ site.name }}
---
# DragoNN code design

We provide two ways to interface with the dragonn package. The interface used in the IPython notebook tutorials is in [`./dragonn/tutorial_utils.py`](https://github.com/kundajelab/dragonn/blob/master/dragonn/tutorial_utils.py), which provides simple wrappers for the dragonn code for simulations of regulatory sequence, model training, and data interpretation. The command-line interface for modeling and interpretation of non-simulated data is in [`./dragonn/__main__.py`](https://github.com/kundajelab/dragonn/blob/master/dragonn/__main__.py), which provide console entry points for the `dragonn train`, `dragonn test`, and `dragonn predict` commands. 

For documentation of dragonn package go [here]({{ site.baseurl }}/html/index.html).