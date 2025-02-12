#!/usr/bin/env python3

import glob
import numpy as np
import pywt
import os

def import_data_test(path):
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
    data = combine_files_test(data_paths)

    return data


def combine_files_test(file_paths):
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
    data_arr = np.load(file_paths[0])

    # For loop to concatenate all the other .npy files to the first one
    for path in file_paths[1:]:
        data = np.load(path)
        data_arr = np.concatenate((data_arr, data), axis=0)

    return data_arr


def l1_l2_summary(data, n_pixels, wavelet_type, l1, l2):
    '''
    This function calculates the line of sight decomposition of the data statistics.
    It calculates the l1 and l2 summaries by doing the Wavelet Scattering Transforms
    with the Morlet wavelets.
    l1: sum(|values|)
    l2: sqrt(sum(values^2))

    Inputs:
        - data: array for which we want to calculate the line of sight decomposition
        - n_pixels: number of pixels of the simulation
        - wavelet_type: wavelet to be used for the scattering transform
        - l1: boolean value if you want to calculate l1
        - l2: boolean value if you want to calculate l2

    Outputs:
        - l1_summary: l1 summary of the data if l1 is true 
        - l2_summary: l2 summary of the data if l2 is true 
    '''

    # Number of bins is taken from the data.
    # It was chosen when simulating the data
    bins = len(data[0])

    # j scales for the wavelet scattering transforms
    j_max = int(np.log2(n_pixels))
    scales = [pow(2,j) for j in range(1, j_max+1)]

    # Create array of zeros for l1 and l2 summaries
    if l1:
        l1_summary = np.zeros((len(scales), bins))

    if l2:
        l2_summary = np.zeros((len(scales), bins))

    # For every bin calculate the wavelet scattering transform
    for i in range(bins):
        cwtmatr, _ = pywt.cwt(data[:, i], scales, wavelet_type)

        # Calculating the l1 and l2 summaries
        for l in range(len(cwtmatr)):
            if l1:
                l1_summary[l, i] =  np.sum(np.abs(cwtmatr[l]))
            if l2:
                l2_summary[l, i] =  np.linalg.norm(cwtmatr[l])


    if l1 and l2:    
        return l1_summary, l2_summary
    elif l1:
        return l1_summary
    elif l2:
        return l2_summary
    else:
        return 0


def l1_l2_summary_S2(S2, n_pixels, wavelet_type, l1, l2):
    '''
    This function calculates the line of sight decomposition of the S2 statistic.
    It calculates the l1 and l2 summaries by doing the Wavelet Scattering Transforms
    with the Morlet wavelets.
    l1: sum(|values|)
    l2: sqrt(sum(values^2))

    Inputs:
        - data: array for which we want to calculate the line of sight decomposition 
        - n_pixels: number of pixels of the simulation
        - wavelet_type: wavelet to be used for the scattering transform
        - l1: boolean value if you want to calculate l1
        - l2: boolean value if you want to calculate l2

    Outputs:
        - l1_summary: l1 summary of the data if l1 is true 
        - l2_summary: l2 summary of the data if l2 is true 
    '''

    # Number of bins is taken from the data.
    # It was chosen when simulating the data
    bins = len(S2[0])

    # j scales for the wavelet scattering transforms
    j_max = int(np.log2(n_pixels))
    scales = [pow(2,j) for j in range(1, j_max+1)]

    # Create array of zeros for l1 and l2 summaries
    if l1:
        l1_summary = np.zeros((len(scales), bins, bins))

    if l2:
        l2_summary = np.zeros((len(scales), bins, bins))

    # For every bin calculate the wavelet scattering transform
    for i in range(bins):
        for j in range(bins):
            cwtmatr, _ = pywt.cwt(S2[:, i, j], scales, wavelet_type)

            # Calculating the l1 and l2 summaries
            for l in range(len(cwtmatr)):
                if l1:
                    l1_summary[l, i, j] =  np.sum(np.abs(cwtmatr[l]))
                if l2:
                    l2_summary[l, i, j] =  np.linalg.norm(cwtmatr[l])


    if l1 and l2:    
        return l1_summary, l2_summary
    elif l1:
        return l1_summary
    elif l2:
        return l2_summary
    else:
        return 0

    
def calculate_l1l2(data_PS, data_S1, data_S2, n_pixels, wavelet_type, l1, l2, save_path = ''):
    '''
    This function calculates l1 and l2 summaries and reshape the arrays.
    Reshaping is done in such a way to change the shape 
    from (sample, j_max, bins) to (samples, j_max * bins)

    Inputs:
        - data_PS: array for PS
        - data_S1: array for S1
        - data_S2: array for S2
        - n_pixels: number of pixels of the coeval cube
        - wavelet_type: wavelet to be used for the wavelet scattering transform
        - l1: boolean value if you want to calculate l1
        - l2: boolean value if you want to calculate l2

    Outputs:
        - l1l2_summary_PS: l1 and l2 summaries of PS
        - l1l2_summary_S1: l1 and l2 summaries of S1
        - l1l2_summary_S2: l1 and l2 summaries of S2
    '''

    # Calculating the l1l2 summaries
    l1l2_summary_PS = np.array([list(l1_l2_summary(data_PS[i], n_pixels, wavelet_type, l1, l2)) for i in range(len(data_PS))])
    l1l2_summary_S1 = np.array([list(l1_l2_summary(data_S1[i], n_pixels, wavelet_type, l1, l2)) for i in range(len(data_S1))])
    l1l2_summary_S2 = np.array([list(l1_l2_summary_S2(data_S2[i], n_pixels, wavelet_type, l1, l2)) for i in range(len(data_S2))])   

    # Reshaping the summaries
    l1l2_summary_PS = np.reshape(l1l2_summary_PS, (l1l2_summary_PS.shape[0], -1))
    l1l2_summary_S1 = np.reshape(l1l2_summary_S1, (l1l2_summary_S1.shape[0], -1))
    l1l2_summary_S2 = np.reshape(l1l2_summary_S2, (l1l2_summary_S2.shape[0], -1))

    if save_path:
        np.save(os.path.join(save_path, 'l1l2_summary_PS.npy'), np.float32(l1l2_summary_PS))
        np.save(os.path.join(save_path, 'l1l2_summary_S1.npy'), np.float32(l1l2_summary_S1))
        np.save(os.path.join(save_path, 'l1l2_summary_S2.npy'), np.float32(l1l2_summary_S2))


    return l1l2_summary_PS, l1l2_summary_S1, l1l2_summary_S2


def calculate_obs_l1l2(data_obs_PS, data_obs_S1, data_obs_S2, n_pixels, wavelet_type, l1, l2, save_path = ''):
    '''
    This function calculates l1 and l2 summaries and reshape the arrays for "observed" data.
    Reshaping is done in such a way to change the shape 
    from (sample, j_max, bins) to (samples, j_max * bins)

    Inputs:
        - data_obs_PS: array for "observed" PS
        - data_obs_S1: array for "observed" S1
        - data_obs_S2: array for "observed" S2
        - n_pixels: number of pixels of the coeval cube
        - wavelet_type: wavelet to be used for the wavelet scattering transform
        - l1: boolean value if you want to calculate l1
        - l2: boolean value if you want to calculate l2

    Outputs:
        - l1l2_summary_obs_PS: l1 and l2 summaries of "observed" PS
        - l1l2_summary_obs_S1: l1 and l2 summaries of "observed" S1
        - l1l2_summary_obs_S2: l1 and l2 summaries of "observed" S2
    '''

    # Calculating the l1l2 summaries
    l1l2_summary_obs_PS = np.array([list(l1_l2_summary(data_obs_PS, n_pixels, wavelet_type, l1, l2))])
    l1l2_summary_obs_S1 = np.array([list(l1_l2_summary(data_obs_S1, n_pixels, wavelet_type, l1, l2))])
    l1l2_summary_obs_S2 = np.array([list(l1_l2_summary_S2(data_obs_S2, n_pixels, wavelet_type, l1, l2))])

    # Reshaping the summaries
    l1l2_summary_obs_PS = l1l2_summary_obs_PS.flatten()
    l1l2_summary_obs_S1 = l1l2_summary_obs_S1.flatten()
    l1l2_summary_obs_S2 = l1l2_summary_obs_S2.flatten()

    if save_path:
        np.save(os.path.join(save_path, 'l1l2_summary_obs_PS.npy'), np.float32(l1l2_summary_obs_PS))
        np.save(os.path.join(save_path, 'l1l2_summary_obs_S1.npy'), np.float32(l1l2_summary_obs_S1))
        np.save(os.path.join(save_path, 'l1l2_summary_obs_S2.npy'), np.float32(l1l2_summary_obs_S2))

    return l1l2_summary_obs_PS, l1l2_summary_obs_S1, l1l2_summary_obs_S2





