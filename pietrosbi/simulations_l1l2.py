#!/usr/bin/env python3

import numpy as np
from pietrosbi.pietro_wavelet_transforms import Pietro_Wavelet_Transforms_3
from pietrosbi.coeval_cubes import remove_21cmfast_cache, create_coeval
import itertools as it


def create_param_pairs_2(n=10):
    '''
    This function creates all the possible parameter pairs
    in the parameter space with the corrsponding probability

    Inputs:
        - n: number of parts in which one axis of the parameter space is subdivided

    Outputs:
        - eff_Tvir_pairs: ionization efficiency and virial temperature value pairs
        - eff_Tvir_prob_vals: probability values for all the corresponding pairs of paramters
    '''
    
    # Probabilites for each value of HII_EFF and T_vir. It's a uniform distribution
    prob_EFF = np.ones(n)/n
    EFF_vals = np.linspace(20, 40, n)
    prob_Tvir = np.ones(n)/n
    Tvir_vals = np.linspace(4.1, 5.5, n)

    # Creating an array with all possible pairs of HII_eff and T_vir, and an array with the corresponding probabilities
    eff_Tvir_pairs = np.array(list(it.product(EFF_vals, Tvir_vals)))
    eff_Tvir_prob_pairs = np.array(list(it.product(prob_EFF, prob_Tvir)))
    eff_Tvir_prob_vals = np.array(list(map(lambda x: x[0] * x[1], eff_Tvir_prob_pairs)))

    return eff_Tvir_pairs, eff_Tvir_prob_vals


def create_x_theta_5(HIIeff_Tvir_arr, z = 7, n_pixels = 32, dim = 300, bins = 6, l1 = True, l2 = True, wavelet_type = 'morl', start_seed = 1):
    '''
    This function creates the data summaries with the corresponding parameters used to generate them. 
    l1l2 summaries for PS, S1 and S2; and the PS3d

    Inputs:
        - HIIeff_Tvir_arr: array of pairs of ionization efficency and virial temperature
        - z: redshift
        - n_pixels: number of pixels of the coeval cube
        - dim: dimension of the cooeval cube in Mpc
        - bins: number of bins for the statistics 
        - l1: whether or not to calculate the l1 summary (bool)
        - l2: whether or not to calculate the l2 summary (bool)
        - wavelet_type: wavelet type to be used for the l1 and l2 summary
        - start_seed: first seed for the coeval cube.

    Outputs:
        - param_list: list of paramters used for the simulations
        - data_summary_S1_l1l2: l1 and l2 data summaries for S1
        - data_summary_PS_l1l2: l1 and l2 data summaries for PS
        - data_summary_S2_l1l2: l1 and l2 data summaries for S2
        - data_summary_PS3d: data for PS3d
        - seed: seed used for the last simulation
    '''
    
    # Creation of the class to calculate the l1 and l2 summaries
    My_WT = Pietro_Wavelet_Transforms_3(box_size=dim, n_pixels=n_pixels, bins=bins, l1=l1, l2=l2, wavelet_type = wavelet_type)

    num = len(HIIeff_Tvir_arr)
    
    # Creating coeval cubes and saving the data summaries and parameters with different values of HII_EFF_FACTOR and ION_Tvir_MIN
    param_list = np.zeros((num, 3))
    data_summary_S1_l1l2 = np.zeros((num, l1 + l2 + 1, 5, bins))
    data_summary_PS_l1l2 = np.zeros((num, l1 + l2 + 1, 5, bins))
    data_summary_S2_l1l2 = np.zeros((num, l1 + l2 + 1, 5, bins, bins))
    data_summary_PS3d = np.zeros((num, 2, bins))
    seed = start_seed
    for i in range(num):
        # Changing the seed for each coeval cube
        seed += np.random.randint(1,10)
        astro_params_dict = {"HII_EFF_FACTOR":HIIeff_Tvir_arr[i,0], "ION_Tvir_MIN": HIIeff_Tvir_arr[i,1]}
        coeval = create_coeval(z=z, n_pixels=n_pixels, dim=dim, astro_params_dict=astro_params_dict, seed=seed)
        param_list[i,0] = coeval.astro_params.HII_EFF_FACTOR
        param_list[i,1] = coeval.astro_params.ION_Tvir_MIN
        param_list[i,2] = seed

        # Loading the simulation to My_WT and calculating PS3d and the l1 and l2 summaries for S1, S2 and PS
        My_WT.load_sim(coeval.brightness_temp)
        data_summary_S1_l1l2[i,0], data_summary_S1_l1l2[i,1], data_summary_S1_l1l2[i,2], scales = My_WT.l1_l2_summary('S1')
        data_summary_PS_l1l2[i,0], data_summary_PS_l1l2[i,1], data_summary_PS_l1l2[i,2], scales = My_WT.l1_l2_summary('PS')
        data_summary_S2_l1l2[i,0], data_summary_S2_l1l2[i,1], data_summary_S2_l1l2[i,2], scales = My_WT.l1_l2_summary_S2()
        data_summary_PS3d[i,0] = My_WT.PS3d
        data_summary_PS3d[i,1] = My_WT.ks_3d

        # Removing the files inside the cache
        remove_21cmfast_cache()
    
    return param_list, data_summary_S1_l1l2, data_summary_PS_l1l2, data_summary_S2_l1l2, data_summary_PS3d, seed

def create_data_obs_2(HIIeff_Tvir_true, z = 7, n_pixels = 32, dim = 300, bins = 6, l1 = True, l2 = True, wavelet_type = 'morl', seed = 1):
    '''
    This function creates the data summaries with the corresponding parameters used to generate them. 
    l1l2 summaries for PS, S1 and S2; and the PS3d

    Inputs:
        - HIIeff_Tvir_arr: array of pairs of ionization efficency and virial temperature
        - z: redshift
        - n_pixels: number of pixels of the coeval cube
        - dim: dimension of the cooeval cube in Mpc
        - bins: number of bins for the statistics 
        - l1: whether or not to calculate the l1 summary (bool)
        - l2: whether or not to calculate the l2 summary (bool)
        - wavelet_type: wavelet type to be used for the l1 and l2 summary
        - seed: seed for the coeval cube.

    Outputs:
        - param_true: true paramters used for the simulations
        - data_obs_S1_l1l2: l1 and l2 data summaries for the "observed" S1
        - data_obs_PS_l1l2: l1 and l2 data summaries for the "observed" PS
        - data_obs_S2_l1l2: l1 and l2 data summaries for the "observed" S2
        - data_obs_PS3d: data for the "observed" PS3d

    '''

    # Creation of the class to calculate the l1 and l2 summaries
    My_WT = Pietro_Wavelet_Transforms_3(box_size=dim, n_pixels=n_pixels, bins=bins, l1=l1, l2=l2, wavelet_type = wavelet_type)

    astro_params_dict = {"HII_EFF_FACTOR": HIIeff_Tvir_true[0], "ION_Tvir_MIN": HIIeff_Tvir_true[1]}

    # Observation summary and the true paramters
    param_true = np.zeros(3)
    data_obs_S1_l1l2 = np.zeros((l1 + l2 + 1, 5, bins))
    data_obs_PS_l1l2 = np.zeros((l1 + l2 + 1, 5, bins))
    data_obs_S2_l1l2 = np.zeros((l1 + l2 + 1, 5, bins, bins))
    data_obs_PS3d = np.zeros((2, bins))
    
    # Coeval cube for the observation after changing the seed
    seed += np.random.randint(1,10)
    coeval = create_coeval(z=z, n_pixels=n_pixels, dim=dim, astro_params_dict=astro_params_dict, seed=seed)
    param_true[0] = coeval.astro_params.HII_EFF_FACTOR
    param_true[1] = coeval.astro_params.ION_Tvir_MIN
    param_true[2] = seed

    # Loading the simulation to My_WT and calculating PS3d and the l1 and l2 summaries for S1, S2 and PS
    My_WT.load_sim(coeval.brightness_temp)
    data_obs_S1_l1l2[0], data_obs_S1_l1l2[1], data_obs_S1_l1l2[2], scales = My_WT.l1_l2_summary('S1')
    data_obs_PS_l1l2[0], data_obs_PS_l1l2[1], data_obs_PS_l1l2[2], scales = My_WT.l1_l2_summary('PS')
    data_obs_S2_l1l2[0], data_obs_S2_l1l2[1], data_obs_S2_l1l2[2], scales = My_WT.l1_l2_summary_S2()
    data_obs_PS3d[0] = My_WT.PS3d
    data_obs_PS3d[1] = My_WT.ks_3d

    # Removing the files inside the cache
    remove_21cmfast_cache()

    return param_true, data_obs_S1_l1l2, data_obs_PS_l1l2, data_obs_S2_l1l2, data_obs_PS3d


def create_batched_data_2(n_batches = 10, 
                          n_per_batch = 10, 
                          z = 7, 
                          n_pixels = 32, 
                          dim = 300, 
                          bins = 6, 
                          l1 = True, 
                          l2 = True, 
                          wavelet_type = 'morl', 
                          start_seed = 1, 
                          save_files = False
                         ):
    '''
    This function creates and saves the l1 and l2 summaries of the simulations of coeval cubes.
    The simulations are divided in batches.

    Inputs:
        - n_batches: number of batches
        - n_per_batch: number of simulations per batch
        - z: redshift
        - n_pixels: number of pixels of the coeval cube 
        - dim: dimension of the coeval cube in Mpc 
        - bins: number of bins for the statistics
        - l1: whether or not to calculate the l1 summary (bool)
        - l2: whether or not to calculate the l2 summary (bool) 
        - wavelet_type: type of wavelet to be used for the l1 and l2 summaries
        - start_seed: seed to be used for the first coeval cube 
        - save_files: whether or not to save the .npy files (bool) 

    Outputs: no outputs; it just creates the statistics and summaries and it saves them
    
    '''

    # Total number of simulations
    n = n_batches * n_per_batch

    # Creating the pairs of ionization efficiency and the corresponding probabilities
    eff_Tvir_pairs, eff_Tvir_prob_vals = create_param_pairs_2(n)
    
    seed = start_seed

    # For loop over the number of batches
    for i in range(n_batches):
        # Choosing random indices which correspond to random pairs of HII_eff and T_vir according to the right probabilities
        HIIeff_Tvir_ind_arr = np.random.choice(np.arange(0, n*n), size = n_per_batch, p=eff_Tvir_prob_vals)
        
        # Selecting the right pairs of HII_eff and T_vir based on the randomly chosen indices
        HIIeff_Tvir_arr = np.array([eff_Tvir_pairs[ind] for ind in HIIeff_Tvir_ind_arr])

        # Creating the parameter list and calculating the statistics and summaries
        param_list, data_summary_S1_l1l2, data_summary_PS_l1l2, data_summary_S2_l1l2, data_summary_PS3d, seed = create_x_theta_5(HIIeff_Tvir_arr, 
                                                                                                                                 z = 7, 
                                                                                                                                 n_pixels = 32, 
                                                                                                                                 dim = 300, 
                                                                                                                                 bins = 6, 
                                                                                                                                 l1 = True, 
                                                                                                                                 l2 = True, 
                                                                                                                                 wavelet_type = 'morl', 
                                                                                                                                 start_seed = seed)
    
        # Saving the arrays in .npy files
        if save_files:
            np.save(f'/travail/pguidi/data_for_sbi_l1l2/param_list_{n}_{i}.npy', param_list)
            np.save(f'/travail/pguidi/data_for_sbi_l1l2/data_summary_S1_l1l2_{n}_{i}.npy', data_summary_S1_l1l2)
            np.save(f'/travail/pguidi/data_for_sbi_l1l2/data_summary_PS_l1l2_{n}_{i}.npy', data_summary_PS_l1l2)
            np.save(f'/travail/pguidi/data_for_sbi_l1l2/data_summary_S2_l1l2_{n}_{i}.npy', data_summary_S2_l1l2)
            np.save(f'/travail/pguidi/data_for_sbi_l1l2/data_summary_PS3d_{n}_{i}.npy', data_summary_PS3d)

    # Similar to the body of the for loop but for the "observation"
    HIIeff_Tvir_ind_true = np.random.choice(np.arange(0, n*n), size = 1, p=eff_Tvir_prob_vals)[0]
    HIIeff_Tvir_true = eff_Tvir_pairs[HIIeff_Tvir_ind_true]
    param_true, data_obs_S1_l1l2, data_obs_PS_l1l2, data_obs_S2_l1l2, data_obs_PS3d = create_data_obs_2(HIIeff_Tvir_true, 
                                                                                                        z = 7, 
                                                                                                        n_pixels = 32, 
                                                                                                        dim = 300, 
                                                                                                        bins = 6, 
                                                                                                        l1 = True, 
                                                                                                        l2 = True, 
                                                                                                        wavelet_type = 'morl', 
                                                                                                        seed = seed)

    # Saving the arrays for the "observation" in .npy files
    if save_files:
        np.save(f'/travail/pguidi/data_for_sbi_l1l2/param_true_{n}.npy', param_true)
        np.save(f'/travail/pguidi/data_for_sbi_l1l2/data_obs_S1_l1l2_{n}.npy', data_obs_S1_l1l2)
        np.save(f'/travail/pguidi/data_for_sbi_l1l2/data_obs_PS_l1l2_{n}.npy', data_obs_PS_l1l2)
        np.save(f'/travail/pguidi/data_for_sbi_l1l2/data_obs_S2_l1l2_{n}.npy', data_obs_S2_l1l2)
        np.save(f'/travail/pguidi/data_for_sbi_l1l2/data_obs_PS3d_{n}.npy', data_obs_PS3d)
