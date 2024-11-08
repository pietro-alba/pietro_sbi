#!/usr/bin/env python3

import glob
import numpy as np
import torch


def combine_files_l1l2(file_paths):
    '''
    This function concatenates .npy files together to
    create one large torch tensor.
    It is used to combine together files of l1l2 summaries

    Inputs:
        - file_paths: list of the paths to the .npy files

    Outputs:
        - data_arr: array of the l1l2 summaries. 
                    It is reshaped and covert it to torch tensor
    '''

    # Load the first .npy file
    # The slicing is done to remove the value of k modes which are saved together with l1l2
    data_arr = np.load(file_paths[0])[:,:-1]

    # For loop to concatenate all the other .npy files to the first one
    for path in file_paths[1:]:
        data = np.load(path)[:,:-1]
        data_arr = np.concatenate((data_arr, data), axis=0)

    return reshape_and_torch_l1l2(data_arr)


def reshape_and_torch_l1l2(array):
    '''
    This function reshapes arrays and converts them to torch tensors
    If array has shape = (1000, 5, 6), it will be reshaped to (1000, 30)
    Te reshaping is needed because the 1000 samples (in this case) cannot
    be 2-dimensional when given as input to the NDE

    Inputs:
        - array: numpy array

    Outputs:
        - arr: reshaped arr and converted to torch.tensor
    '''
    
    arr = array.reshape(array.shape[0], -1)
    arr = torchify(arr)

    return arr


def torchify(array):
    '''
    This function converst a numpy array to torch.tensor
    whose values are numnpy.float32 (instead of numpy.float64)
    The NDE wants torch.tensors whose values are float32 and not float64

    Inputs:
        - array: numpy array

    Outputs:
        - a torch.tensor with float32 values
    '''
    
    return torch.from_numpy(np.float32(array))
    

def import_obs_l1l2(path):
    '''
    This functions import the "observed" l1l2 summaries
    from a specific path.

    Inputs:
        - path: path to the file

    Outputs:
        - obs: torch tensor of the file with l1l2 summaries found in path
    '''

    # Loading the .npy file.
    # The slicing is needed to remove the k modes which 
    # are saved with the l1l2 summaries
    obs = np.load(path)[:-1]

    # Flattening the array and converting it to torch.tensor
    obs = obs.flatten()
    obs = torchify(obs)

    return obs


def import_data_l1l2(path):
    '''
    This functions import the l1l2 summaries
    from a list of paths. 

    Inputs:
        - path: path to the l1l2 summaries.
                Since the paths to these summaries are similar the path string
                contains an *, so the function glob.glob() will be useful to handle this

    Outputs:
        - data: combined data of the l1l2 summaries
    '''

    # Create a list of paths given the path
    # Path contains an *, so with glob.glob(path) I select all the paths which satisfy 
    # this condition and placed them in a sorted list
    # The sorting (based on the file name) is done to have a consistent list of paths between 
    # summaries and parameters so that the data can be easily matched to the corresponding paramters
    data_paths = sorted(glob.glob(path))

    # Combine the files inside the list data_paths
    data = combine_files_l1l2(data_paths)

    return data



