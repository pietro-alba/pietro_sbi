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


def my_cornerplot(samples, truths, labels, bins = 20, datapoints = True, smooth=False, limits=None):
    '''
    This function plots a cornerplot of the posterior.

    Inputs:
        - samples: datapoint sampled from the posterior
        - labels: labels of the plots
        - bins: bins for the distributions plotted on the diagonal
        - datapoints: whether or not to show the sample points
        - smooth: smoothin of the posterior
        - limits: limits of the plots

    Outputs: it draws the cornerplot
    '''
    
    fig = plt.figure(figsize = (8, 8))
    corner.corner(
        np.array(samples), 
        fig = fig, 
        color = "blue", 
        truths = truths, 
        levels = (0.68, 0.95), 
        plot_contour=False,
        plot_density=False,
        plot_datapoints=datapoints,
        fill_contours=True,
        smooth = smooth,
        bins = bins,
        labels = labels,
        range=limits
    )

    plt.show()


class My_Posterior():
    '''
    Class that traines the NDE and plots the posterior cornerplots.
    '''
    
    def __init__(self, 
                 data_l1l2_PS, 
                 data_l1l2_S1, 
                 data_l1l2_S2, 
                 data_PS3d, 
                 param_list, 
                 data_obs_l1l2_PS, 
                 data_obs_l1l2_S1, 
                 data_obs_l1l2_S2, 
                 data_obs_PS3d, 
                 param_true
                ):

        '''
        Initialization of the class.
        It saves the l1 and l2 summaries as attributes.

        Inputs:
            - data_l1l2_PS: l1 and l2 summaries of PS
            - data_l1l2_S1: l1 and l2 summaries of S1
            - data_l1l2_S2: l1 and l2 summaries of S2
            - data_PS3d: data of the 3-dim PS
            - param_list: list of parameters
            - data_obs_l1l2_PS: l1 and l2 summaries of the "observed" PS
            - data_obs_l1l2_S1: l1 and l2 summaries of the "observed" S1
            - data_obs_l1l2_S2: l1 and l2 summaries of the "observed" S2
            - data_obs_PS3d: data of the "observed" 3-dim PS
            - param_true: true paramters of the "observation"
        '''

        self.data_l1l2_PS = data_l1l2_PS
        self.data_l1l2_S1 = data_l1l2_S1
        self.data_l1l2_S2 = data_l1l2_S2
        self.data_PS3d = data_PS3d
        self.param_list = param_list 
        self.data_obs_l1l2_PS = data_obs_l1l2_PS 
        self.data_obs_l1l2_S1 = data_obs_l1l2_S1 
        self.data_obs_l1l2_S2 = data_obs_l1l2_S2 
        self.data_obs_PS3d = data_obs_PS3d
        self.param_true = param_true
        

    def shuffle_two_arr(self, a, b):
        '''
        This function shuffles two arrays in the same way.

        Inputs:
            - a: first array
            - b: second array

        Output:
            - first shuffled array
            - second shuffled array
        '''
        
        # Checking that the two arrays have the same length
        assert len(a) == len(b)

        # Shuffling the indices of the array
        p = np.random.permutation(len(a))
        
        return a[p], b[p]

    
    def training(self, density_estimator, params, data, epochs):
        '''
        Training without validation or patience. The code was taken from sbi tutorial.

        Inputs:
            - density_estimator: the density estimator to be trained
            - params: paramters of the simulation
            - data: data of the simulations
            - epochs: number of epochs for the training

        Outputs:
            - density estimator: the trained density estimator
        '''
        
        # Selecting the optimizer for the training. lr: learning rate
        opt = Adam(list(density_estimator.parameters()), lr=5e-4)

        # For loop of the training process.
        # NUmber of cycles = number of epochs
        for _ in range(epochs):
            opt.zero_grad()
            losses = density_estimator.loss(params, condition=data)
            loss = torch.mean(losses)
            loss.backward()
            opt.step()
    
        return density_estimator
    
    
    
    def training_2(self, density_estimator, params, data, epochs_max, val_size = 0.1, patience = 10):
        '''
        Trainig with validation and patience. The code is similar to training(...).

        Inputs:
            - density_estimator: the density estimator to be trained
            - params: paramters of the simulation
            - data: data of the simulations
            - epochs_max: max number of epochs for the training
            - val_size: validation size
            - patience: number of consecutive epochs after which the training stops 
                        in case the loss function of the validation set did not improve 

        Outputs:
            - density estimator: the trained density estimator
        '''
        
        # Selecting the optimizer for the training. lr: learning rate
        opt = Adam(list(density_estimator.parameters()), lr=5e-4)

        # Shuffling paramters and data. I don't if it's needed.
        params, data = self.shuffle_two_arr(params, data)

        # Splitting of the data between training and validation set
        split = int(len(params) * (1-val_size))
        params_train, params_val = params[:split], params[split:]
        data_train, data_val = data[:split], data[split:]

        # Counters for the number of consecutives epochs the loss of the validation set does not improve
        p = 0

        # Counter of the nìtotal number of epochs
        e = 0

        # Initial value of the loss of the validation
        loss_val_min = np.inf

        # Training loop as long as the conditions are true
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
    
    def posterior_NDE(self, data_summary, param_arr, data_obs):
        '''
        This functions takes the default NPE and trains it to 
        learn the posterior.

        Inputs:
            - data_summary: data needed to train the NPE
            - param_arr: parameters of the simulations
            - data_obs: "observed" data. It is needed to sample from the posterior

        Outputs:
            - samples: datapoints samples from the posterior
        '''
        
        # Defines the prior limits and the prior
        lower = torch.tensor([20, 4.1])
        upper = torch.tensor([40, 6])
        prior = BoxUniform(low=lower, high=upper)
        
        # Check prior, return PyTorch prior.
        prior, num_parameters, prior_returns_numpy = process_prior(prior)

        # Creates and train the NPE for inference. A Mixture Density Network is used
        inference = NPE(prior=prior, density_estimator='mdn')
        inference = inference.append_simulations(param_arr, data_summary)
        density_estimator = inference.train()
        posterior = inference.build_posterior(density_estimator)
        
        # Sampling from the posterior and plotting the corner plot
        samples = posterior.sample((100000,), x=data_obs)
    
        return samples
    
    
    def posterior_built_NDE(self, data_summary, param_arr, data_obs):
        '''
        This functions builds the NPE and trains it to 
        learn the posterior.

        Inputs:
            - data_summary: data needed to train the NPE
            - param_arr: parameters of the simulations
            - data_obs: "observed" data. It is needed to sample from the posterior

        Outputs:
            - samples: datapoints samples from the posterior
        '''

        # Reshaping the "observed" data to the accepeted shape for the sampling process
        data_obs = data_obs.reshape(1,data_obs.shape[0])

        # Creating the NPE with Mixture Density Network
        density_estimator = build_mdn(param_arr, data_summary)
    
        # Calling one of the two functions at the beginning of the notebook for the training process
        #density_estimator = training(density_estimator, param_arr, data_summary, epochs = 250)
        density_estimator = self.training_2(density_estimator, param_arr, data_summary, epochs_max = 2000, val_size = 0.15, patience = 20)
        
        # Sampling from the posterior 
        samples = density_estimator.sample((100000,), condition=data_obs).detach()
        samples = samples.squeeze(dim=1)
    
        return samples

    
    def select_data(self, summary_type):
        '''
        This function selects the right dataset based on
        the summary type that you want

        Inputs:
            - summary_type: 'S1', 'PS', 'S2' or 'PS3d'

        Outputs:
            - data_summary: the selected data summary
            - data_obs: the selected "observed" data summary
        '''

        # Selecting the right datasets
        if summary_type == 'S1':
            data_summary = self.data_l1l2_S1
            data_obs = self.data_obs_l1l2_S1
        elif summary_type == 'PS':
            data_summary = self.data_l1l2_PS
            data_obs = self.data_obs_l1l2_PS
        elif summary_type == 'S2':
            data_summary = self.data_l1l2_S2
            data_obs = self.data_obs_l1l2_S2
        elif summary_type == 'PS3d':
            data_summary = self.data_PS3d
            data_obs = self.data_obs_PS3d
    
        return data_summary, data_obs
    
    
    def concatenate_summaries(self, summary_list):
        '''
        This functions concatenates the data summaries if 
        multiple data summaries were chosen

        Inputs:
            - summary_list: list of strings representing the 
                            statistics you want to concatenate

        Outputs:
            - data_summary: the concatenated data summary
            - data_obs: the concatenated "observed" data summary
        '''
        
        data_summary_list = []
        data_obs_list = []

        # For loop to sleect the datasets corresponding to the chosen statistics
        for s in summary_list:
            data, data_obs = self.select_data(s)
            data_summary_list.append(data)
            data_obs_list.append(data_obs)

        # Concatenating the datasets if more than one statistics is chosen
        # Otherwhise it selects the only chosen statistic
        if len(summary_list) != 1:
            data_summary = torch.from_numpy(np.concatenate(data_summary_list, axis=1))
            data_obs = torch.from_numpy(np.concatenate(data_obs_list))
        else:
            data_summary = data_summary_list[0]
            data_obs = data_obs_list[0]
    
        return data_summary, data_obs
    
    
    def posterior_corner_pair(self, summary_list1, summary_list2, built_NDE = False):
        '''
        This functions builds and train the NDE, then it plots the 
        posterior with corner plots.
        It takes two data summaries as inputs to compare their
        constraining power.

        Inputs:
            - summary_list1: first list of data summaries. Elements are strings.
            - summary_list2: second list of data summaries. Elements are strings.
            - built_NDE: false if you want the defaul NDE
                         true if you want my version of the training

        Outputs: no output, it draws the cornerplot
        
        '''

        # Checking that the two data summaries lists are actually lists
        assert type(summary_list1) is list, 'summary_type1 must be a list'
        assert type(summary_list2) is list, 'summary_type1 must be a list'

        # Concatenating the chosen data summaries
        data_summary_1, data_obs_1 = self.concatenate_summaries(summary_list1)
        data_summary_2, data_obs_2 = self.concatenate_summaries(summary_list2)
        print(data_summary_1.shape, data_obs_1.shape)
        print(data_summary_2.shape, data_obs_2.shape)

        # Training the NDE: either my version or the default one
        if built_NDE:
            samples_1 = self.posterior_built_NDE(data_summary_1, self.param_list, data_obs_1)
            samples_2 = self.posterior_built_NDE(data_summary_2, self.param_list, data_obs_2)
        else:
            samples_1 = self.posterior_NDE(data_summary_1, self.param_list, data_obs_1)
            samples_2 = self.posterior_NDE(data_summary_2, self.param_list, data_obs_2)

        # Plotting the cornerplots on the same figure
        fig = plt.figure(figsize = (6, 6))
        c1 = corner.corner(
                np.array(samples_1),
                color = "blue", 
                fig=fig,
                truths = self.param_true, 
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
                truths = self.param_true, 
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

        # For the legend
        blue_line = mlines.Line2D([],[],color='blue', label='-'.join(summary_list1))
        green_line = mlines.Line2D([],[],color='green', label='-'.join(summary_list2))
        fig.legend(handles=[blue_line, green_line], loc=(0.75,0.75))
        
        plt.show()
    
    
    def posterior_corner(self, summary_list, built_NDE = False):
        '''
        This functions builds and train the NDE, then it plots the 
        posterior with corner plots.
        It takes two data summaries as inputs to compare their
        constraining power.

        Inputs:
            - summary_list: list of data summaries. Elements are strings.
            - built_NDE: false if you want the defaul NDE
                         true if you want my version of the training

        Outputs: no output, it draws the cornerplot
        
        '''

        # Checking that the data summary is aactually a list
        assert type(summary_list) is list, 'summary_type1 must be a list'

        # Concatenating the chosen data summaries
        data_summary, data_obs = self.concatenate_summaries(summary_list)
        print(data_summary.shape, data_obs.shape)

        # Training the NDE: either my version or the default one
        if built_NDE:
            samples = self.posterior_built_NDE(data_summary, self.param_list, data_obs)
        else:
            samples = self.posterior_NDE(data_summary, self.param_list, data_obs)

        # Plotting the cornerplots on the same figure
        fig = plt.figure(figsize = (6, 6))
        c1 = corner.corner(
                np.array(samples),
                color = "blue", 
                fig=fig,
                truths = self.param_true, 
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

        # For the legend
        blue_line = mlines.Line2D([],[],color='blue', label='-'.join(summary_list))
        fig.legend(handles=[blue_line], loc=(0.75,0.75))
        
        plt.show()