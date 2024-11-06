#!/usr/bin/env python3

import glob
import numpy as np
import torch
import pywt

def import_data_test(path):
    data_paths = sorted(glob.glob(path))
    print(data_paths)
    data = combine_files_test(data_paths)

    return data


def combine_files_test(file_paths):
    data_arr = np.load(file_paths[0])
    for path in file_paths[1:]:
        data = np.load(path)
        data_arr = np.concatenate((data_arr, data), axis=0)

    return data_arr


def l1_l2_summary(data, n_pixels, wavelet_type, l1, l2):
    # Calculates the l1 and l2 summaries with the Morlet wavelets
    ps2d_list = data
    #ps2d_list = np.array([self.S_func(box_map[i], plot_fig = False)[0] for i in range(len(box_map))])
    #ks2d = self.PS_window(box_map[0], dimension, bins, plot_fig = False)[1]

    bins = len(data[0])
    
    j_max = int(np.log2(n_pixels))
    scales = [pow(2,j) for j in range(1, j_max+1)]

    if l1:
        l1_summary = np.zeros((len(scales), bins))

    if l2:
        l2_summary = np.zeros((len(scales), bins))

    for i in range(bins):
        cwtmatr, _ = pywt.cwt(ps2d_list[:, i], scales, wavelet_type)
        
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
    # Calculates the l1 and l2 summaries with the Morlet wavelets
    ps2d_list = S2
    #ps2d_list = np.array([self.S_func(box_map[i], plot_fig = False)[0] for i in range(len(box_map))])
    #ks2d = self.PS_window(box_map[0], dimension, bins, plot_fig = False)[1]

    bins = len(S2[0])
    
    j_max = int(np.log2(n_pixels))
    scales = [pow(2,j) for j in range(1, j_max+1)]

    if l1:
        l1_summary = np.zeros((len(scales), bins, bins))

    if l2:
        l2_summary = np.zeros((len(scales), bins, bins))

    for i in range(bins):
        for j in range(bins):
            cwtmatr, _ = pywt.cwt(ps2d_list[:, i, j], scales, wavelet_type)
            
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
    data_S1_rev = data_S1[:,:,::-1]
    data_S1_rev = data_S1_rev[:,:,:,np.newaxis] 
    data_S2_new = data_S2 / data_S1_rev 
    data_S1_new = data_S1 / np.sqrt(data_PS)
    
    data_S1_new[np.isnan(data_S1_new)] = 0
    data_S2_new[np.isnan(data_S2_new)] = 0

    return data_S1_new, data_S2_new


def norm_obs_data(data_obs_PS, data_obs_S1, data_obs_S2):
    data_obs_S1_rev = data_obs_S1[:,::-1]
    data_obs_S1_rev = data_obs_S1_rev[:,:,np.newaxis] 
    data_obs_S2_new = data_obs_S2 / data_obs_S1_rev 
    data_obs_S1_new = data_obs_S1 / np.sqrt(data_obs_PS)
    
    data_obs_S1_new[np.isnan(data_obs_S1_new)] = 0
    data_obs_S2_new[np.isnan(data_obs_S2_new)] = 0

    return data_obs_S1_new, data_obs_S2_new

    
def calculate_l1l2(data_PS, data_S1, data_S2, n_pixels, wavelet_type, l1, l2):
    data_S1_new, data_S2_new = norm_data(data_PS, data_S1, data_S2)
    
    l1l2_summary_PS = np.array([list(l1_l2_summary(data_PS[i], n_pixels, wavelet_type, l1, l2)) for i in range(len(data_PS))])
    l1l2_summary_S1 = np.array([list(l1_l2_summary(data_S1_new[i], n_pixels, wavelet_type, l1, l2)) for i in range(len(data_S1_new))])
    l1l2_summary_S2 = np.array([list(l1_l2_summary_S2(data_S2_new[i], n_pixels, wavelet_type, l1, l2)) for i in range(len(data_S2_new))])   
    
    l1l2_summary_PS = np.reshape(l1l2_summary_PS, (l1l2_summary_PS.shape[0], -1))
    l1l2_summary_S1 = np.reshape(l1l2_summary_S1, (l1l2_summary_S1.shape[0], -1))
    l1l2_summary_S2 = np.reshape(l1l2_summary_S2, (l1l2_summary_S2.shape[0], -1))

    return l1l2_summary_PS, l1l2_summary_S1, l1l2_summary_S2


def calculate_obs_l1l2(data_obs_PS, data_obs_S1, data_obs_S2, n_pixels, wavelet_type, l1, l2):
    data_obs_S1_new, data_obs_S2_new = norm_obs_data(data_obs_PS, data_obs_S1, data_obs_S2)
    
    l1l2_summary_obs_PS = np.array([list(l1_l2_summary(data_obs_PS, n_pixels, wavelet_type, l1, l2))])
    l1l2_summary_obs_S1 = np.array([list(l1_l2_summary(data_obs_S1_new, n_pixels, wavelet_type, l1, l2))])
    l1l2_summary_obs_S2 = np.array([list(l1_l2_summary_S2(data_obs_S2_new, n_pixels, wavelet_type, l1, l2))])
    
    l1l2_summary_obs_PS = l1l2_summary_obs_PS.flatten()
    l1l2_summary_obs_S1 = l1l2_summary_obs_S1.flatten()
    l1l2_summary_obs_S2 = l1l2_summary_obs_S2.flatten()

    return l1l2_summary_obs_PS, l1l2_summary_obs_S1, l1l2_summary_obs_S2


def torchify(array):
    return torch.from_numpy(np.float32(array))


def l1l2_and_torchify_data(data_PS, data_S1, data_S2, data_PS3d, param_list, n_pixels, wavelet_type, l1, l2):
    l1l2_summary_PS, l1l2_summary_S1, l1l2_summary_S2 = calculate_l1l2(data_PS, 
                                                                       data_S1, 
                                                                       data_S2, 
                                                                       n_pixels, 
                                                                       wavelet_type,
                                                                       l1, 
                                                                       l2
                                                                      )
    l1l2_summary_PS = torchify(l1l2_summary_PS)
    l1l2_summary_S1 = torchify(l1l2_summary_S1)
    l1l2_summary_S2 = torchify(l1l2_summary_S2)
    
    data_PS3d = torchify(data_PS3d)
    param_list = torchify(param_list)

    return l1l2_summary_PS, l1l2_summary_S1, l1l2_summary_S2, data_PS3d, param_list


def l1l2_and_torchify_obs_data(data_obs_PS, data_obs_S1, data_obs_S2, data_obs_PS3d, param_true, n_pixels, wavelet_type, l1, l2):
    l1l2_summary_obs_PS, l1l2_summary_obs_S1, l1l2_summary_obs_S2 = calculate_obs_l1l2(data_obs_PS, 
                                                                                       data_obs_S1, 
                                                                                       data_obs_S2, 
                                                                                       n_pixels, 
                                                                                       wavelet_type,
                                                                                       l1,
                                                                                       l2
                                                                                      )
    l1l2_summary_obs_PS = torchify(l1l2_summary_obs_PS)
    l1l2_summary_obs_S1 = torchify(l1l2_summary_obs_S1)
    l1l2_summary_obs_S2 = torchify(l1l2_summary_obs_S2)
    
    data_obs_PS3d = torchify(data_obs_PS3d)
    param_true = torchify(param_true)

    return l1l2_summary_obs_PS, l1l2_summary_obs_S1, l1l2_summary_obs_S2, data_obs_PS3d, param_true
