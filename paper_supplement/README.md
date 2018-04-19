## DragoNN paper supplement contents
We provide trained models, data, and code to reproduce results in the DragoNN manuscript:

- The [primer notebook](primer_tutorial.ipynb) and accompanying [primer models](primer_tutorial_models) are provided to reproduce results in the "understanding architectures with simulations" section of the DragoNN manuscript

- [Simulation data files](simulation_data) and accompanying [simulation models](simulation_models) are provided to reproduce results in the "understanding how neural networks model different properties of regulatory DNA sequences with simulations" section of the DragoNN manuscript

## How to reproduce results in the DragoNN paper
### Paper section: understanding architectures with simulations
Start a jupyter notebook server:

```
jupyter notebook
```

Navigate to the [primer_tutorial.ipynb](primer_tutorial.ipynb) notebook in your browser. Run the notebook to reproduce the architecture exploration in manuscript.

### Paper section: understanding how neural networks model different properties of regulatory DNA sequences with simulations

To obtain plots in this section with model performance on simulations, run:

```
python simulation_performance_results.py --model-files-dir simulation_models --data-files-dir simulation_data --results-dir simulation_results
```

#### --results-dir
The script will write pdf files with the plots in this directory. The pdf files are named with the following format: `<simulation_type>.results.<plot_type>.pdf`. For example, `simulate_multi_motif_embedding.results.pool_width.pdf` will plot performance on the multiple motif embedding simulation for varying pooling width. For each simulation, there is also a `<simulation_type>.results. RandomForest.pdf` plot showing how a Random Forest model based on motif features performs for varying training data size. This model serves as an "empirical upper bound" for the rest of the results for that simulation as it is based on the motif features in the simulation.

#### --data-files-dir
Data files in this directory are named based on parameters of [simulations functions](https://github.com/kundajelab/simdna/blob/master/simdna/simulations.py) and contain training, validation, and test data for each simulation in the manuscript.

#### --model-files-dir
Directory with architecture and weights files for 139 models discussed in the manuscript. Architecture and weights files of models based on simulations are prefixed with `<simulation_function_name>.<model_architecture_parameters>`.