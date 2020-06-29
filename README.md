

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<h1 align="center">SVRG</h1>

## Install requirements
`pip install -r requirements.txt` 


## Experiments

Run the experiments for the paper using the commands below:

### Synthetic and Kernels

Use trainval_svrg for SVRG, and trainval for all other optimizers. 

```
python trainval.py -e adaptive_II_syn adaptive_II_kernels -d <datadir> -sb <savedir_base>  -r 1 -c <enable_cuda>
python trainval_svrg.py -e adaptive_II_syn adaptive_II_kernels -d <datadir> -sb <savedir_base>  -r 1 -c <enable_cuda>
```
where `<datadir>` is where the data is saved (example `.tmp/data`),  `<savedir_base>` is where the results will be saved (example `.tmp/results`), and `<enable_cuda>` is either 0 or 1. It is 1 if the user enables cuda.


## Results

View results by running the following command.

```
python trainval.py -e adaptive_III_mnist -v 1 -d <datadir> -sb <savedir_base>
```

where `<datadir>` is where the data is saved, and `<savedir_base>` is where the results will be saved.
