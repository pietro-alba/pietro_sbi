#!/usr/bin/env python3

import torch
from torch.optim import Adam

from sbi.analysis import pairplot
from sbi.inference import NPE, NLE
from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)

from sbi.inference.posteriors import MCMCPosterior
from sbi.inference.potentials import likelihood_estimator_based_potential
from sbi.neural_nets.net_builders import build_nsf, build_maf, build_mdn

from sbi.inference.posteriors import MCMCPosterior
from sbi.inference.potentials import likelihood_estimator_based_potential

import matplotlib.pyplot as plt
import numpy as np
import corner
import matplotlib.lines as mlines
import pywt


def my_cornerplot(samples, bins = 20, datapoints = True, smooth=False, limits=None):
    fig = plt.figure(figsize = (8, 8))
    corner.corner(
        np.array(samples), 
        fig = fig, 
        color = "blue", 
        truths = param_true, 
        levels = (0.68, 0.95, 0.98), 
        plot_contour=False,
        plot_density=False,
        plot_datapoints=datapoints,
        fill_contours=True,
        smooth = smooth,
        bins = bins,
        labels = ['$HII_{eff}$','$T_{vir}$'],
        range=limits#[(20, 40),(4.1, 6)]
    )
    print(samples.mean(axis = 0), samples.std(axis = 0))

    plt.show()

def shuffle_two_arr(a, b):
    # Shuffles two arrays in the same way
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def training(density_estimator, params, data, epochs):
    # Training without validation or patience. The code was taken from sbi tutorial
    
    opt = Adam(list(density_estimator.parameters()), lr=5e-4)
    
    for _ in range(epochs):
        opt.zero_grad()
        losses = density_estimator.loss(params, condition=data)
        loss = torch.mean(losses)
        loss.backward()
        opt.step()

    return density_estimator



def training_2(density_estimator, params, data, epochs_max, val_size = 0.1, patience = 10):
    #Trainig with validation and patience. The code is similar to training(...)
    
    opt = Adam(list(density_estimator.parameters()), lr=5e-4)
    params, data = shuffle_two_arr(params, data)
    split = int(len(params) * (1-val_size))
    params_train, params_val = params[:split], params[split:]
    data_train, data_val = data[:split], data[split:]

    p = 0
    e = 0
    loss_val_min = np.inf
    while p < patience and e < epochs_max:
        opt.zero_grad()
        losses = density_estimator.loss(params_train, condition=data_train)
        loss = torch.mean(losses)
        loss.backward()
        opt.step()

        # Calculates the validation loss
        loss_val = torch.mean(density_estimator.loss(params_val, condition=data_val))

        # Updates the min validation loss and the number of times in a row the validation loss didn't improve (p)
        if loss_val < loss_val_min:
            loss_val_min = loss_val
            p = 0
        else:
            p += 1

        e += 1

    print(f'Density estimator was trained for {e} epochs')
    return density_estimator

def posterior_NDE(data_summary, param_arr, data_obs):
    # Defines the prior limits
    lower = torch.tensor([20, 4.1])
    upper = torch.tensor([40, 6])
    
    prior = BoxUniform(low=lower, high=upper)
    # Check prior, return PyTorch prior.
    prior, num_parameters, prior_returns_numpy = process_prior(prior)

    inference = NPE(prior=prior, density_estimator='mdn')
    inference = inference.append_simulations(param_arr, data_summary)
    density_estimator = inference.train()
    posterior = inference.build_posterior(density_estimator)
    
    # Sampling from the posterior and plotting the corner plot
    samples = posterior.sample((100000,), x=data_obs)

    return samples


def posterior_built_NDE(data_summary, param_arr, data_obs):
    ###### NO PRIOR???? #########
    data_obs = data_obs.reshape(1,data_obs.shape[0])
    density_estimator = build_mdn(param_arr, data_summary)

    # Calling one of the two functions at the beginning of the notebook for the training process
    #density_estimator = training(density_estimator, param_arr, data_summary, epochs = 250)
    density_estimator = training_2(density_estimator, param_arr, data_summary, epochs_max = 2000, val_size = 0.1, patience = 20)
    
    # Sampling from the posterior 
    samples = density_estimator.sample((100000,), condition=data_obs).detach()
    samples = samples.squeeze(dim=1)

    return samples


def select_data(summary_type):
    if summary_type == 'S1':
        data_summary = l1l2_summary_S1
        data_obs = l1l2_summary_obs_S1
    elif summary_type == 'PS':
        data_summary = l1l2_summary_PS
        data_obs = l1l2_summary_obs_PS
    elif summary_type == 'S2':
        data_summary = l1l2_summary_S2
        data_obs = l1l2_summary_obs_S2
    elif summary_type == 'PS3d':
        data_summary = data_PS3d
        data_obs = data_obs_PS3d

    return data_summary, data_obs


def concatenate_summaries(summary_list):
    data_summary_list = []
    data_obs_list = []
    for s in summary_list:
        data, data_obs = select_data(s)
        data_summary_list.append(data)
        data_obs_list.append(data_obs)

    if len(summary_list) != 1:
        data_summary = torch.from_numpy(np.concatenate(data_summary_list, axis=1))
        data_obs = torch.from_numpy(np.concatenate(data_obs_list))
    else:
        data_summary = data_summary_list[0]
        data_obs = data_obs_list[0]

    return data_summary, data_obs


def posterior_corner_pair(summary_list1, summary_list2, built_NDE = False):

    assert type(summary_list1) is list, 'summary_type1 must be a list'
    assert type(summary_list2) is list, 'summary_type1 must be a list'

    data_summary_1, data_obs_1 = concatenate_summaries(summary_list1)
    data_summary_2, data_obs_2 = concatenate_summaries(summary_list2)
    print(data_summary_1.shape, data_obs_1.shape)
    print(data_summary_2.shape, data_obs_2.shape)
    if built_NDE:
        samples_1 = posterior_built_NDE(data_summary_1, param_list, data_obs_1)
        samples_2 = posterior_built_NDE(data_summary_2, param_list, data_obs_2)
    else:
        samples_1 = posterior_NDE(data_summary_1, param_list, data_obs_1)
        samples_2 = posterior_NDE(data_summary_2, param_list, data_obs_2)

    fig = plt.figure(figsize = (6, 6))
    c1 = corner.corner(
            np.array(samples_1),
            color = "blue", 
            fig=fig,
            truths = param_true, 
            levels = (0.68, 0.95), 
            plot_contour=False,
            plot_density=False,
            plot_datapoints=False,
            fill_contours=False,
            smooth = False,
            bins = 20,
            labels = ['$HII_{eff}$','$T_{vir}$'],
            range=None #limits#[(20, 40),(4.1, 6)]
        )
    
    c2 = corner.corner(
            np.array(samples_2),
            fig = fig,
            color = "green", 
            truths = param_true, 
            levels = (0.68, 0.95), 
            plot_contour=False,
            plot_density=False,
            plot_datapoints=False,
            fill_contours=False,
            smooth = False,
            bins = 20,
            labels = ['$HII_{eff}$','$T_{vir}$'],
            range=None #limits#[(20, 40),(4.1, 6)]
        )
    
    blue_line = mlines.Line2D([],[],color='blue', label='-'.join(summary_list1))
    green_line = mlines.Line2D([],[],color='green', label='-'.join(summary_list2))
    fig.legend(handles=[blue_line, green_line], loc=(0.75,0.75))
    
    plt.show()


def posterior_corner(summary_list, built_NDE = False):

    assert type(summary_list) is list, 'summary_type1 must be a list'

    data_summary, data_obs = concatenate_summaries(summary_list)
    print(data_summary.shape, data_obs.shape)

    if built_NDE:
        samples = posterior_built_NDE(data_summary, param_list, data_obs)
    else:
        samples = posterior_NDE(data_summary, param_list, data_obs)

    fig = plt.figure(figsize = (6, 6))
    c1 = corner.corner(
            np.array(samples),
            color = "blue", 
            fig=fig,
            truths = param_true, 
            levels = (0.68, 0.95), 
            plot_contour=False,
            plot_density=False,
            plot_datapoints=False,
            fill_contours=False,
            smooth = False,
            bins = 20,
            labels = ['$HII_{eff}$','$T_{vir}$'],
            range=None #limits#[(20, 40),(4.1, 6)]
        )
    
    blue_line = mlines.Line2D([],[],color='blue', label='-'.join(summary_list))
    fig.legend(handles=[blue_line], loc=(0.75,0.75))
    
    plt.show()