# Stochastic differential equation to describe GRB prompt emission

This repository contains the codes to optimise a stochastic model aiming to generate realistic GRB light curves (LC) from different GRB experiments using a genetic algorithm (GA). 

## Overview

The model is based on a stochastic differential equation (SDE) involving a generalised geometric Brownian motion (gGBM). The GA is implemented in python via the pygad package which is needed in your environment to optimise the model. The GRB LCs can be simply generated. To optimise the light curves, you need to use geneticgrbs_SPEWIE.py, reyling on statistical_tests.py that defines the different metrics to and sde_GA.py where the LC is built using the SDE solution.

## Requirements

The code was tested with Python 3.10. The required packages are listed in the `environment.yml` file.

## Installation

Clone the repository:

```bash
git clone https://github.com/romanomaccary/Stochastic_GRBs_SDE.git
cd Stochastic_GRBs_SDE
```
Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate pygad3
```

If you want to use the environment as a Jupyter kernel, run:

```bash
python -m ipykernel install --user --name pygad3 --display-name "Python (pygad3)"
```
## License

The license will be specified in a future release.

## Contact

For questions, please contact:

Romain Maccary
romain.maccary[at]edu[dot]unife[dot]it
