# CNN estimation of turbulent channel flow using wall measurements 

# Introduction

The code in this repository features a Python/Pytorch implementation of a convolutional neural network to the model to predict the two-dimensional velocity-fluctuation fields at different wall-normal locations in a turbulent open channel flow, using the wall-shear-stress components and the wall pressure as inputs. 
Input data are obtained from ["Resolvent-based estimation of turbulent channel flow using wall measurements", Filipe R. Amaral, André V.G. Cavalieri1, Eduardo Martini, Peter Jordan and Aaron Towne]

Implementation methodology based on ["Convolutional-network models to predict wall-bounded turbulence from wall quantities", L. Guastoni, A. Güemes, A.Ianiro, S Discetti, P. Schlatter, H. Azizpour, R. Vinuesa]

#Pre-requisite

The Python packages required to run the code are listed in `requirements.txt`

# Dataset
The dataset is read and pre-processed in the `utils` folder. 
The dataset is composed of the following input:
- `du_dy0` is the streamwise wall-shear stress
- `du_dy1` is the spanwise wall-shear stress
- `p` is the wall pressure

The outputs are at 15:
- `u15` is the streamwise velocity fluctuations
- `v15` is the wall-normal velocity fluctuations
- `w15` is the spanwise velocity fluctuations

# Usage

The Pytorch model is defined in the `model` folder. With the training parameters ared defined in the `config_file.py`
