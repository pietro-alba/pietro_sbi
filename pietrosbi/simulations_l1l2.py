#!/usr/bin/env python3

import numpy as np
from pietrosbi.NEW_pietro_wavelet_transforms import Pietro_Wavelet_Transforms
from pietrosbi.coeval_cubes import remove_21cmfast_cache, create_coeval
import itertools as it
import os

def create_data_l1l2(HIIeff_Tvir_arr, 
                     z, 
                     n_pixels, 
                     dim, 
                     bins_list, 
                     l1, 
                     l2, 
                     wavelet_type,
                     start_seed, 
                     ind, 
                     save_files, 
                     path):
    
    """
    It produces coeval cubes and calculates the data summaries to compress the cubes.
    Then, it saves the data summaries in .npy files.
    These data are needed for the training of the Neural Density Estimators.

    Inputs:
        - HIIeff_Tvir_arr: an array with shape (num, 2), where num is the number of cubes to simulate and 2 is the 
                         number of astrophysical paramters to change for each simulation.
                         The two astrophysical parameters are the ionising efficiency and the virial temperature
        - z: redshift of the coeval cubes
        - n_pixels: number of pixel of the cubes
        - dim: size of the cubes in Mpc 
        - bins_list: list with the number of bins for the data summaires. This allows to find the best value for the number of bins
        - l1: True if we want to use the \ell 1 summary, or False otherwise, for the Line of Sight decomposition
        - l2: True if we want to use the \ell 2 summary, or False otherwise, for the Line of Sight decomposition
        - wavelet_type: the type of wavelet to use for the Line of Sight decomposition
        - start_seed: random starting seed. The seed will change for each simulation
        - ind: index of the batch
        - save_files: True if you want to save the .npy files
        - path: path to directory where you want to save the files

    Outputs:
        - seed: the seed of the last simualation, so you can continue changing the seed for the next batch
                Therefore, the starting seed of each batch is the seed of the last simulation of the previous batch
    """

    # Number of simulations in the batch
    num = len(HIIeff_Tvir_arr)
    
    # Array that will save the values of the two atrophysical parameters and the seed of each simulation
    param_list = np.zeros((num, 3))

    # Dictionaries for each summary statistic, where the keys are the number of bins present in the list of bins
    data_summary_S1_dict = {b: np.zeros((num, l1 + l2, 5, b)) for b in bins_list}
    data_summary_S2_dict = {b: np.zeros((num, l1 + l2, 5, b, b)) for b in bins_list}
    data_summary_PS_dict = {b: np.zeros((num, l1 + l2, 5, b)) for b in bins_list}
    data_summary_PS3d_dict = {b: np.zeros((num, b)) for b in bins_list}

    # Dictionary for my wavelet transforms
    My_WT_dict = {b: Pietro_Wavelet_Transforms(box_size=dim, n_pixels=n_pixels, bins=b, l1=l1, l2=l2, wavelet_type = wavelet_type) for b in bins_list}
    
    # Creating the simulations and calculating the data summaries
    seed = start_seed
    for i in range(num):
        seed += np.random.randint(1,10)
        astro_params_dict = {"HII_EFF_FACTOR":HIIeff_Tvir_arr[i,0], "ION_Tvir_MIN": HIIeff_Tvir_arr[i,1]}
        
        coeval = create_coeval(z=z, n_pixels=n_pixels, dim=dim, astro_params_dict=astro_params_dict, seed=seed)
        
        param_list[i,0] = coeval.astro_params.HII_EFF_FACTOR
        param_list[i,1] = coeval.astro_params.ION_Tvir_MIN
        param_list[i,2] = seed

        for b in bins_list:
            My_WT_dict[b].load_sim(coeval.brightness_temp)
            data_summary_S1_dict[b][i,0], data_summary_S1_dict[b][i,1], _ = My_WT_dict[b].l1_l2_summary('S1')
            data_summary_PS_dict[b][i,0], data_summary_PS_dict[b][i,1], _ = My_WT_dict[b].l1_l2_summary('PS')
            data_summary_S2_dict[b][i,0], data_summary_S2_dict[b][i,1], _ = My_WT_dict[b].l1_l2_summary_S2()
            data_summary_PS3d_dict[b][i] = My_WT_dict[b].PS3d
            
        remove_21cmfast_cache()
        
    # Saving the data summaries in .npy files
    if save_files:
        if path is None:
                raise ValueError('path should be a string a not None')
        for b in bins_list:
            full_path = os.path.join(path, f'bins_{b}')
            if not os.path.exists(full_path):
                os.makedirs(full_path)

            l1l2_summary_PS = np.reshape(data_summary_PS_dict[b], (data_summary_PS_dict[b].shape[0], -1))
            l1l2_summary_S1 = np.reshape(data_summary_S1_dict[b], (data_summary_S1_dict[b].shape[0], -1))
            l1l2_summary_S2 = np.reshape(data_summary_S2_dict[b], (data_summary_S2_dict[b].shape[0], -1))
            
            np.save(os.path.join(full_path, f'param_list_{ind}.npy'), np.float32(param_list))
            np.save(os.path.join(full_path, f'data_summary_S1_l1l2_{ind}.npy'), np.float32(l1l2_summary_S1))
            np.save(os.path.join(full_path, f'data_summary_PS_l1l2_{ind}.npy'), np.float32(l1l2_summary_PS))
            np.save(os.path.join(full_path, f'data_summary_S2_l1l2_{ind}.npy'), np.float32(l1l2_summary_S2))
            np.save(os.path.join(full_path, f'data_summary_PS3d_{ind}.npy'), np.float32(data_summary_PS3d_dict[b]))
    
    return seed

def create_data_obs_l1l2(HIIeff_Tvir_true, 
                      z, 
                      n_pixels, 
                      dim, 
                      bins_list, 
                      l1, 
                      l2, 
                      wavelet_type,
                      seed, 
                      save_files, 
                      path):
    
    """
    It produces one coeval cube and calculates its data summaries. This cube will act as an observation.
    Then, it saves the data summaries in .npy files.
    These data are needed after the training of the Neural Density Estimators to test its performance.

    Inputs:
        - HIIeff_Tvir_true: array with the two values of astrophysical paramters used for the observation.            
        - z: redshift of the coeval cube
        - n_pixels: number of pixel of the cube
        - dim: size of the cube in Mpc 
        - bins_list: list with the number of bins for the data summaires. This allows to find the best value for the number of bins
        - l1: True if we want to use the \ell 1 summary, or False otherwise, for the Line of Sight decomposition
        - l2: True if we want to use the \ell 2 summary, or False otherwise, for the Line of Sight decomposition
        - wavelet_type: the type of wavelet to use for the Line of Sight decomposition
        - seed: random seed
        - save_files: True if you want to save the .npy files
        - path: path to directory where you want to save the files

    Outputs:
        - seed: the seed of the simualation
    """

    # Array that will contain the true paramters of the observation
    param_true = np.zeros(3)

    # Dictionaries for each summary statistic, where the keys are the number of bins present in the list of bins
    data_obs_S1_dict = {b: np.zeros((l1 + l2, 5, b)) for b in bins_list}
    data_obs_S2_dict = {b: np.zeros((l1 + l2, 5, b, b)) for b in bins_list}
    data_obs_PS_dict = {b: np.zeros((l1 + l2, 5, b)) for b in bins_list}
    data_obs_PS3d_dict = {b: np.zeros(b) for b in bins_list}

    # Dictionary for my wavelet transforms
    My_WT_dict = {b: Pietro_Wavelet_Transforms(box_size=dim, n_pixels=n_pixels, bins=b, l1=l1, l2=l2, wavelet_type = wavelet_type) for b in bins_list}

    # Coeval cube for the observation
    seed += np.random.randint(1,10)
    astro_params_dict = {"HII_EFF_FACTOR": HIIeff_Tvir_true[0], "ION_Tvir_MIN": HIIeff_Tvir_true[1]}
    coeval = create_coeval(z=z, n_pixels=n_pixels, dim=dim, astro_params_dict=astro_params_dict, seed=seed)
    param_true[0] = coeval.astro_params.HII_EFF_FACTOR
    param_true[1] = coeval.astro_params.ION_Tvir_MIN
    param_true[2] = seed

    
    # Calculation of the data summaries
    for b in bins_list:
        My_WT_dict[b].load_sim(coeval.brightness_temp)
        data_obs_S1_dict[b][0], data_obs_S1_dict[b][1], _ = My_WT_dict[b].l1_l2_summary('S1')
        data_obs_PS_dict[b][0], data_obs_PS_dict[b][1], _ = My_WT_dict[b].l1_l2_summary('PS')
        data_obs_S2_dict[b][0], data_obs_S2_dict[b][1], _ = My_WT_dict[b].l1_l2_summary_S2()
        data_obs_PS3d_dict[b] = My_WT_dict[b].PS3d

    remove_21cmfast_cache()
    
    # Saving the data summaries in .npy files
    if save_files:
        if path is None:
                raise ValueError('path should be a string a not None')
        for b in bins_list:
            full_path = os.path.join(path, f'bins_{b}')
            if not os.path.exists(full_path):
                os.makedirs(full_path)

                
            np.save(os.path.join(full_path, 'param_true.npy'), np.float32(param_true))
            np.save(os.path.join(full_path, 'l1l2_summary_obs_S1.npy'), np.float32(data_obs_S1_dict[b].flatten()))
            np.save(os.path.join(full_path, 'l1l2_summary_obs_PS.npy'), np.float32(data_obs_PS_dict[b].flatten()))
            np.save(os.path.join(full_path, 'l1l2_summary_obs_S2.npy'), np.float32(data_obs_S2_dict[b].flatten()))
            np.save(os.path.join(full_path, 'data_obs_PS3d.npy'), np.float32(data_obs_PS3d_dict[b]))
            
    return seed

def create_batched_data_l1l2(n_batches, 
                          n_per_batch, 
                          z, 
                          n_pixels, 
                          dim, 
                          bins_list, 
                          l1, 
                          l2, 
                          wavelet_type,
                          start_seed, 
                          save_files,
                          path = None):

    """
    It produces the data for the training of the Neural Density Estimator.
    The data are divided in batches.
    It also generates one additional cube which is used as an observation.

    Inputs:
        - n_batches: number of batches
        - n_per_batch: number of simulations per batch
        - z: redshift of the coeval cubes
        - n_pixels: number of pixel of the cubes
        - dim: size of the cubes in Mpc 
        - bins_list: list with the number of bins for the data summaires. This allows to find the best value for the number of bins
        - l1: True if we want to use the \ell 1 summary, or False otherwise, for the Line of Sight decomposition
        - l2: True if we want to use the \ell 2 summary, or False otherwise, for the Line of Sight decomposition
        - wavelet_type: the type of wavelet to use for the Line of Sight decomposition
        - start_seed: random starting seed. The seed will change for each simulation
        - save_files: True if you want to save the .npy files
        - path: path to directory where you want to save the files

    Outputs:
        - seed: the seed of the last simualation, so you can continue changing the seed for other simulations if you need them
    """

    # Lowest and highest values of the uniform distribution which acts as the prior
    low = [20, 4.1]
    high = [40, 5.5]
    
    seed = start_seed
    for i in range(n_batches):
        # Generating random values of the astrophysical paramters and creating the data summaries
        HIIeff_Tvir_arr = np.random.uniform(low = low, high = high, size = (n_per_batch, 2))
        seed = create_data_l1l2(HIIeff_Tvir_arr,
                                z = z, 
                                n_pixels = n_pixels, 
                                dim = dim, 
                                bins_list = bins_list, 
                                l1 = l1, 
                                l2 = l2, 
                                wavelet_type = wavelet_type,
                                start_seed = seed, 
                                ind = i, 
                                save_files = save_files,
                                path = path)
    
    # Generating random values of the astrophysical paramters for the observation and generating the observation
    HIIeff_Tvir_true = np.random.uniform(low = low, high = high, size = (2,))
    seed = create_data_obs_l1l2(HIIeff_Tvir_true, 
                             z = z, 
                             n_pixels = n_pixels, 
                             dim = dim, 
                             bins_list = bins_list, 
                             l1 = l1, 
                             l2 = l2,
                             wavelet_type = wavelet_type,
                             seed = seed, 
                             save_files = save_files, 
                             path = path)
    
    return seed 
    