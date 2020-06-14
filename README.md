# README #

This software accompanies the 2017 NIPS [paper](https://papers.nips.cc/paper/7231-protein-interface-prediction-using-graph-convolutional-networks) and [poster](https://zenodo.org/record/1134154), Protein Interface Prediction using Graph Convolutional Networks.
We implemented multiple versions of graph convolution and applied them to the problem of protein interface prediction.
This work was supported by the National Science Foundation under grant no DBI-1564840.

## Setup ##
### Requirements ###

- python 3.6+(ԭ����2.7�����ڸĳ���3.6+��
- PyYAML 3.12
- numpy 1.13.3
- scikit-learn 0.19.1
- tensorflow 2.0.1+(ԭ����1.0.1�����ڸĳ�2.0.+)

### Environment Variables ###
The software assumes the following environment variables are set:

- PL_DATA: full path of data directory (where data files are kept)
- PL_OUT: full path of output directory (where experiment results are placed)
- PL_EXPERIMENTS: full path of experiment library (YAML files)

An alternative to setting these variables is to edit the portions of configuration.py which reference these environment variables.

### CUDA Setup ###
Consider setting the following environment variables for CUDA use:

- LD_LIBRARY_PATH: path to cuda libraries
- CUDA_VISIBLE_DEVICES: Specify (0, 1, etc.) which GPU to use or set to "" to force CPU

### Data ###

To run the provided experiments, you need the pickle files found [here](https://zenodo.org/record/1127774#.WkLewGGnGcY).


## Running Experiments ##

Simply run:
```python experiment_runner.py <experiment>```.
Where ```<experiment>``` is the name of the experiment file (including .yml extension) in the experiments directory.
Alternatively you may run ```run_experiments.sh```, which contains expressions for all provided experiments.


## Contact ##

Please direct any questions to:

* Alex Fout (fout[at]colostate.edu)
* Jonathon Byrd (jonbyrd[at]colostate.edu)
* Basir Shariat (basir[at]cs.colostate.edu
* Asa Ben-Hur (asa[at]cs.colostate.edu)
