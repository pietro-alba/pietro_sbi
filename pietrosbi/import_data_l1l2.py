#!/usr/bin/env python3

import glob
import numpy as np
import torch


def combine_files_l1l2(file_paths):
    data_arr = np.load(file_paths[0])[:,:-1]
    for path in file_paths[1:]:
        data = np.load(path)[:,:-1]
        data_arr = np.concatenate((data_arr, data), axis=0)

    return reshape_and_torch_l1l2(data_arr)


def reshape_and_torch_l1l2(array):
    arr = array.reshape(array.shape[0], -1)
    arr = torchify(arr)

    return arr


def import_obs_l1l2(path):
    obs = np.load(path)[:-1]
    obs = obs.flatten()
    obs = torchify(obs)

    return obs


def import_data_l1l2(path):
    data_paths = sorted(glob.glob(path))
    data = combine_files_l1l2(data_paths)

    return data

def torchify(array):
    return torch.from_numpy(np.float32(array))

