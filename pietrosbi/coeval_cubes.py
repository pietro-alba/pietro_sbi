#!/usr/bin/env python3

import os, shutil
import py21cmfast as p21c
import numpy as np

def remove_21cmfast_cache():
    # Removes the cache from the folder '21cmFAST-cache'
    
    folder = '/obs/pguidi/21cmFAST-cache'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
                os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def create_coeval(z, n_pixels, dim, astro_params_dict, seed):
    # Creates the coeval cube with 21cmfast
    coeval = p21c.run_coeval(
        redshift=z,
        user_params={'HII_DIM': n_pixels, "BOX_LEN": dim, "USE_INTERPOLATION_TABLES": False},
        cosmo_params = p21c.CosmoParams(SIGMA_8=0.8),
        astro_params = p21c.AstroParams(astro_params_dict),
        random_seed=seed
    )

    # Reshaping the coeval cube so that the z-axis is the first index
    coeval.brightness_temp = np.transpose(coeval.brightness_temp, (2, 0, 1))

    # Subtracting the mean from each slice in the z-diretion
    for i in range(len(coeval.brightness_temp)):
       coeval.brightness_temp[i] -= np.mean(coeval.brightness_temp[i])
    
    return coeval#.brightness_temp
