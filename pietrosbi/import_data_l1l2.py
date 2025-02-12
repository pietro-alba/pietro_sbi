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
    data_arr = np.load(file_paths[0])

    # For loop to concatenate all the other .npy files to the first one
    for path in file_paths[1:]:
        data = np.load(path)
        data_arr = np.concatenate((data_arr, data), axis=0)

    return data_arr


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



