import h5py
import os

import torch                    # PyTorch
import torch.nn as nn           # Neural network module
from icecream import ic
import numpy as np
from sklearn.model_selection import train_test_split



# read_h5
def read_h5(filename,  x_z_axis=True):
    """Reads an HDF5 file and returns the data.

    Args:
        filename: The path to the HDF5 file.

    Returns:
        A dictionary containing the data from the HDF5 file.
    """
    with h5py.File(filename, 'r') as f:

        data = {}
        for key in f.keys():

            if isinstance(f[key], h5py.Group):
                for key2 in f[key].keys():
                    if isinstance(f[key][key2], h5py.Dataset):
                        data[key2] = f[key][key2][()]
                    elif isinstance(f[key][key2], h5py.Group):
                        for key3 in f[key][key2].keys():
                            data[key3] = f[key][key2][key3][()]
                    else:
                        data[key2] = {}     
            else:
                data[key] = f[key][()]

        if x_z_axis:
            data['x'] = f['scales']['x']['1.0'][()]
            data['z'] = f['scales']['z']['1.0'][()]

    return data

"""
Example of use

# Import the module
import os
from read_utils import read_h5

# Define the directory and file name
"""
#dir = r'C:\Users\totti\OneDrive\Work\ITA-PUC\Data\Meio_canal_Retau180'
file = 'analysis_s2.h5'
"""
dir_p_file = os.path.join(dir, file)

# Read the data
data = read_h5(dir_p_file) 
"""

def read_data(args,data):
    #INPUT
    du_dy0 = data['dUdy0']
    dw_dy0 = data['dWdy0']
    p0 = data['p0']

    #Reshape
    du_dy0 = np.transpose(du_dy0, axes = (0,2,1,3)) 
    dw_dy0 = np.transpose(dw_dy0, axes = (0,2,1,3))
    p0 = np.transpose(p0, axes = (0,2,1,3))

    #Make the input
    inputs = np.concatenate((du_dy0, dw_dy0, p0), axis=3)
    inputs = np.transpose(inputs, axes = (0,3,1,2))

    #OUTPUT
    u15 = data['u15']
    v15 = data['v15']
    w15 = data['w15']

    #Reshape
    u15 = np.transpose(u15, axes = (0,2,1,3))
    v15 = np.transpose(v15, axes = (0,2,1,3))
    w15 = np.transpose(w15, axes = (0,2,1,3))

    #Make the output
    if args.N_VARS_OUT == 2:
        outputs = np.concatenate((u15, v15), axis=3)
    elif args.N_VARS_OUT == 3:
        outputs = np.concatenate((u15, v15, w15), axis=3)
    else:
        outputs = u15
    outputs = np.transpose(outputs, axes = (0,3,1,2))

    return inputs, outputs

def import_data(args, mean_output = False):
    # Import the data
    dir_p_file = args.file_location
    data = read_h5(dir_p_file)
    inputs, outputs = read_data(args,data)
    
    #Ouptut of the mean of one them
    mean_s = np.mean(outputs[:,0,:,:], axis=0)

    if args.NORMALIZE_INPUT:
        for i in range(args.N_VARS_IN):
            inputs[:,i,:,:] = (inputs[:,i,:,:] - np.mean(inputs[:,i,:,:], axis=0)) / np.std(inputs[:,i,:,:],axis=0)
    
    if args.PRED_FLUCT:
        for i in range(args.N_VARS_OUT):
            outputs[:,i,:,:] = (outputs[:,i,:,:] - np.mean(outputs[:,i,:,:]))

    if args.SCALE_OUTPUT:
        if args.N_VARS_OUT == 2:
            outputs[:,1,:,:] = outputs[:,1,:,:]*(rms(outputs[:,0,:,:], axis=0) / rms(outputs[:,1,:,:], axis=0))

        elif args.N_VARS_OUT == 3:
            outputs[:,1,:,:] = outputs[:,1,:,:]*(rms(outputs[:,0,:,:], axis=0) / rms(outputs[:,1,:,:], axis=0))
            outputs[:,2,:,:] = outputs[:,2,:,:]*(rms(outputs[:,0,:,:], axis=0) / rms(outputs[:,2,:,:], axis=0))
    
    # train is now 75% of the entire data set
    x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size=args.VAL_SPLIT)

    # test is now 10% of the initial data set
    # validation is now 15% of the initial data set
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=args.TEST_SPLIT/(args.TEST_SPLIT + args.VAL_SPLIT)) 

    if mean_output:
        return x_train, x_val, x_test, y_train, y_val, y_test, mean_s

    return x_train, x_val, x_test, y_train, y_val, y_test

def rms(x, axis=None):
    return np.sqrt(np.mean(x**2, axis=axis))