---
layout: default
title: {{ site.name }}
---

## Online paper supplement contents
We provide trained models, data, and code in [paper_supplement](https://github.com/kundajelab/dragonn/tree/master/paper_supplement) to reproduce results in the DragoNN manuscript. Data files in [paper_supplement/simulation_data](https://github.com/kundajelab/dragonn/tree/master/paper_supplement/simulation_data) are named based on parameters of [simulations functions](https://github.com/kundajelab/simdna/blob/master/simdna/simulations.py) and contain training, validation, and test data for each simulation in the manuscript. We provide architecture and weights files for 139 models discussed in the manuscript in [paper_supplement/simulation_models](https://github.com/kundajelab/dragonn/tree/master/paper_supplement/simulation_models). Architecture and weights files of models based on simulations are prefixed with `<simulation_function_name>.<model_architecture_parameters>`.

## How to reproduce results in the DragoNN manuscript
To obtain the plots with model performance on simulations for varying data size and model architectures, run:

```
python paper_supplement/simulation_performance_results.py --model-files-dir paper_supplement/simulation_models/ --data-files-dir paper_supplement/simulation_data/ --results-dir paper_supplement/simulation_results
```

This command will write pdf files with the plots in `paper_supplement/simulation_results/`. The pdf files are named with the following format: `<simulation_type>.results.<plot_type>.pdf`.

For example, `simulate_multi_motif_embedding.results.pool_width.pdf` will plot performance on the multiple motif embedding simulation for varying pooling width. For each simulation, there is also a `<simulation_type>.results.RandomForest.pdf` plot showing how a Random Forest model based on motif features performs for varying training data size. This model serves as an "empirical upper bound" for the rest of the results for that simulation as it is based on the motif features in the simulation (see the paper supplement for more details). 
