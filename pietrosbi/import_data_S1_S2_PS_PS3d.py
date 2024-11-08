#!/usr/bin/env python3

import glob
import numpy as np
import torch
import pywt

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


def norm_data(data_PS, data_S1, data_S2):
    '''
    This function normalizes S1 and S2 with PS and S1 respectively

    Inputs:
        - data_PS: array for PS
        - data_S1: array for S1
        - data_S2: array for S2

    Outputs:
        - data_S1_new: normalized arrau for S1
        - data_S2_new: normalized arrau for S2
    '''
    
    # Nprmalization of S1
    data_S1_new = data_S1 / np.sqrt(data_PS)

    # Reversing S1, becasue S2 is define in the opposite way as S1 (look at the class Pietro_Wavelet_Transorms())
    data_S1_rev = data_S1_new[:,:,::-1]

    # Creating a new axis to match the size of the shape of S2
    data_S1_rev = data_S1_rev[:,:,:,np.newaxis] 

    # Normalization of S2
    data_S2_new = data_S2 / data_S1_rev 

    # Setting nan values to 0
    data_S1_new[np.isnan(data_S1_new)] = 0
    data_S2_new[np.isnan(data_S2_new)] = 0

    return data_S1_new, data_S2_new


def norm_obs_data(data_obs_PS, data_obs_S1, data_obs_S2):
    '''
    This function normalizes S1 and S2 with PS and S1 respectively

    Inputs:
        - data_PS: array for PS
        - data_S1: array for S1
        - data_S2: array for S2

    Outputs:
        - data_S1_new: normalized arrau for S1
        - data_S2_new: normalized arrau for S2
    '''

    # Nprmalization of S1
    data_obs_S1_new = data_obs_S1 / np.sqrt(data_obs_PS)

    # Reversing S1, becasue S2 is define in the opposite way as S1 (look at the class Pietro_Wavelet_Transorms())
    data_obs_S1_rev = data_obs_S1_new[:,::-1]

    # Creating a new axis to match the size of the shape of S2
    data_obs_S1_rev = data_obs_S1_rev[:,:,np.newaxis] 

    # Normalization of S2
    data_obs_S2_new = data_obs_S2 / data_obs_S1_rev 

    # Setting nan values to 0
    data_obs_S1_new[np.isnan(data_obs_S1_new)] = 0
    data_obs_S2_new[np.isnan(data_obs_S2_new)] = 0

    return data_obs_S1_new, data_obs_S2_new

    
def calculate_l1l2(data_PS, data_S1, data_S2, n_pixels, wavelet_type, l1, l2):
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

    # Normalization of S1 and S2
    data_S1_new, data_S2_new = norm_data(data_PS, data_S1, data_S2)

    # Calculating the l1l2 summaries
    l1l2_summary_PS = np.array([list(l1_l2_summary(data_PS[i], n_pixels, wavelet_type, l1, l2)) for i in range(len(data_PS))])
    l1l2_summary_S1 = np.array([list(l1_l2_summary(data_S1_new[i], n_pixels, wavelet_type, l1, l2)) for i in range(len(data_S1_new))])
    l1l2_summary_S2 = np.array([list(l1_l2_summary_S2(data_S2_new[i], n_pixels, wavelet_type, l1, l2)) for i in range(len(data_S2_new))])   

    # Reshaping the summaries
    l1l2_summary_PS = np.reshape(l1l2_summary_PS, (l1l2_summary_PS.shape[0], -1))
    l1l2_summary_S1 = np.reshape(l1l2_summary_S1, (l1l2_summary_S1.shape[0], -1))
    l1l2_summary_S2 = np.reshape(l1l2_summary_S2, (l1l2_summary_S2.shape[0], -1))

    return l1l2_summary_PS, l1l2_summary_S1, l1l2_summary_S2


def calculate_obs_l1l2(data_obs_PS, data_obs_S1, data_obs_S2, n_pixels, wavelet_type, l1, l2):
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

    # Normalization of S1 and S2
    data_obs_S1_new, data_obs_S2_new = norm_obs_data(data_obs_PS, data_obs_S1, data_obs_S2)

    # Calculating the l1l2 summaries
    l1l2_summary_obs_PS = np.array([list(l1_l2_summary(data_obs_PS, n_pixels, wavelet_type, l1, l2))])
    l1l2_summary_obs_S1 = np.array([list(l1_l2_summary(data_obs_S1_new, n_pixels, wavelet_type, l1, l2))])
    l1l2_summary_obs_S2 = np.array([list(l1_l2_summary_S2(data_obs_S2_new, n_pixels, wavelet_type, l1, l2))])

    # Reshaping the summaries
    l1l2_summary_obs_PS = l1l2_summary_obs_PS.flatten()
    l1l2_summary_obs_S1 = l1l2_summary_obs_S1.flatten()
    l1l2_summary_obs_S2 = l1l2_summary_obs_S2.flatten()

    return l1l2_summary_obs_PS, l1l2_summary_obs_S1, l1l2_summary_obs_S2


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


def l1l2_and_torchify_data(data_PS, data_S1, data_S2, data_PS3d, param_list, n_pixels, wavelet_type, l1, l2):
    '''
    This function calls the function to calculate the l1 and l2 summaries
    of PS, S1 and S2 and converts them to torch.tensor.
    It also convert PS3D and the paramter list to torch.tensors.

    Inputs:
        - data_PS: array for PS
        - data_S1: array for S1
        - data_S2: array for S2
        - data_PS3d: array for PS3d
        - param_list: list of parameters
        - n_pixels: number of pixels of the coeval cube
        - wavelet_type: wavelet to be used for the wavelet scattering transform
        - l1: boolean value if you want to calculate l1
        - l2: boolean value if you want to calculate l2

    Outputs:
        - l1l2_summary_PS: l1 and l2 summaries of PS
        - l1l2_summary_S1: l1 and l2 summaries of S1
        - l1l2_summary_S2: l1 and l2 summaries of S2
        - data_PS3d: torch.tensor for PS3d
        - param_list: torch.tensor of parameters
    '''

    # Calculating the l1 and l2 summaries for PS, S1 and S2
    l1l2_summary_PS, l1l2_summary_S1, l1l2_summary_S2 = calculate_l1l2(data_PS, 
                                                                       data_S1, 
                                                                       data_S2, 
                                                                       n_pixels, 
                                                                       wavelet_type,
                                                                       l1, 
                                                                       l2
                                                                      )

    # Converting the summaries, PS3d and parameter list to torch.tensors
    l1l2_summary_PS = torchify(l1l2_summary_PS)
    l1l2_summary_S1 = torchify(l1l2_summary_S1)
    l1l2_summary_S2 = torchify(l1l2_summary_S2)
    data_PS3d = torchify(data_PS3d)
    param_list = torchify(param_list)

    return l1l2_summary_PS, l1l2_summary_S1, l1l2_summary_S2, data_PS3d, param_list


def l1l2_and_torchify_obs_data(data_obs_PS, data_obs_S1, data_obs_S2, data_obs_PS3d, param_true, n_pixels, wavelet_type, l1, l2):
    '''
    This function calls the function to calculate the l1 and l2 summaries
    of PS, S1 and S2 and converts them to torch.tensor.
    It also convert PS3D and the paramter list to torch.tensors.

    Inputs:
        - data_obs_PS: array for "observed" PS
        - data_obs_S1: array for "observed" S1
        - data_obs_S2: array for "observed" S2
        - data_obs_PS3d: array for "observed" PS3d
        - param_true: true parameters for the observation
        - n_pixels: number of pixels of the coeval cube
        - wavelet_type: wavelet to be used for the wavelet scattering transform
        - l1: boolean value if you want to calculate l1
        - l2: boolean value if you want to calculate l2

    Outputs:
        - l1l2_summary_obs_PS: l1 and l2 summaries of "observed" PS
        - l1l2_summary_obs_S1: l1 and l2 summaries of "observed" S1
        - l1l2_summary_obs_S2: l1 and l2 summaries of "observed" S2
        - data_obs_PS3d: torch.tensor for PS3d
        - param_true: torch.tensor of the true parameters
    '''

    # Calculating the l1 and l2 summaries for PS, S1 and S2
    l1l2_summary_obs_PS, l1l2_summary_obs_S1, l1l2_summary_obs_S2 = calculate_obs_l1l2(data_obs_PS, 
                                                                                       data_obs_S1, 
                                                                                       data_obs_S2, 
                                                                                       n_pixels, 
                                                                                       wavelet_type,
                                                                                       l1,
                                                                                       l2
                                                                                      )

    # Converting the summaries, PS3d and true parameters to torch.tensors
    l1l2_summary_obs_PS = torchify(l1l2_summary_obs_PS)
    l1l2_summary_obs_S1 = torchify(l1l2_summary_obs_S1)
    l1l2_summary_obs_S2 = torchify(l1l2_summary_obs_S2)
    data_obs_PS3d = torchify(data_obs_PS3d)
    param_true = torchify(param_true)

    return l1l2_summary_obs_PS, l1l2_summary_obs_S1, l1l2_summary_obs_S2, data_obs_PS3d, param_true
