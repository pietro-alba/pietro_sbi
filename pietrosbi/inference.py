#!/usr/bin/env python3

import torch
from torch.optim import Adam



from sbi.neural_nets.net_builders import build_nsf, build_maf, build_mdn

import matplotlib.pyplot as plt
import numpy as np
import corner
import matplotlib.lines as mlines

from matplotlib import colors, colormaps
from tqdm import tqdm
from sbi import EnsemblePosterior
#from .ensemble import EnsemblePosterior_2
from scipy.optimize import curve_fit
import scipy
from sbi.inference.posteriors import DirectPosterior


class Inference():
    '''
    This class is used to train the NDE.
    It also plots the posterior and the likelihood.
    '''
    
    def __init__(self, 
                 l1l2_summary_PS_dict, 
                 l1l2_summary_S1_dict, 
                 l1l2_summary_S2_dict, 
                 data_PS3d_dict, 
                 param_list, 
                 l1l2_summary_obs_PS_dict, 
                 l1l2_summary_obs_S1_dict, 
                 l1l2_summary_obs_S2_dict, 
                 data_obs_PS3d_dict, 
                 param_true,
                 prior
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

        self.data_l1l2_PS = l1l2_summary_PS_dict
        self.data_l1l2_S1 = l1l2_summary_S1_dict
        self.data_l1l2_S2 = l1l2_summary_S2_dict
        self.data_PS3d = data_PS3d_dict
        self.param_list = param_list 
        self.data_obs_l1l2_PS = l1l2_summary_obs_PS_dict 
        self.data_obs_l1l2_S1 = l1l2_summary_obs_S1_dict 
        self.data_obs_l1l2_S2 = l1l2_summary_obs_S2_dict 
        self.data_obs_PS3d = data_obs_PS3d_dict
        self.param_true = param_true
        self.prior = prior
        
    def select_data(self, summary_type, size, bins):
        '''
        This function selects the right dataset based on
        the summary type that you want

        Inputs:
            - summary_type: 'S1', 'PS', 'S2' or 'PS3d'

        Outputs:
            - data_summary: the selected data summary
            - data_obs: the selected "observed" data summary
        '''
        
        l1l2_summary_S1 = self.data_l1l2_S1[bins]
        l1l2_summary_S2 = self.data_l1l2_S2[bins]
        l1l2_summary_PS = self.data_l1l2_PS[bins]
        data_PS3d = self.data_PS3d[bins]
        
        data_obs_S1 = self.data_obs_l1l2_S1[bins]
        data_obs_S2 = self.data_obs_l1l2_S2[bins]
        data_obs_PS = self.data_obs_l1l2_PS[bins]
        data_obs_PS3d = self.data_obs_PS3d[bins]

        # Selecting the right datasets
        if summary_type == 'S1':
            data_summary = l1l2_summary_S1[:size]
            data_obs = data_obs_S1
        elif summary_type == 'PS':
            data_summary = l1l2_summary_PS[:size]
            data_obs = data_obs_PS
        elif summary_type == 'S2':
            data_summary = l1l2_summary_S2[:size]
            data_summary = np.reshape(data_summary[data_summary != 0], (len(data_summary), -1))
            data_obs = data_obs_S2[data_obs_S2 != 0]
        elif summary_type == 'PS3d':
            data_summary = data_PS3d[:size]
            data_obs = data_obs_PS3d
    
        return data_summary, data_obs


    def concatenate_summaries(self, summary_list, size, bins):
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
            data, data_obs = self.select_data(s, size, bins)
            data_summary_list.append(data)
            data_obs_list.append(data_obs)

        # Concatenating the datasets if more than one statistics is chosen
        # Otherwhise it selects the only chosen statistic
        if len(summary_list) != 1:
            data_summary = np.concatenate(data_summary_list, axis=1)
            data_obs = np.concatenate(data_obs_list)
        else:
            data_summary = data_summary_list[0]
            data_obs = data_obs_list[0]
        
        return torch.from_numpy(data_summary), torch.from_numpy(data_obs)

    
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


    def training(self, density_estimator, params, data, lr, epochs_max, val_size, patience):
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
        opt = Adam(list(density_estimator.parameters()), lr=lr)

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
        loss_val_min = np.inf
        loss_train_list = []
        loss_val_list = [np.inf]
        best_estimator = None

        # Training loop as long as the conditions are true
        while p < patience and e < epochs_max:
            opt.zero_grad()
            losses = density_estimator.loss(params_train, condition=data_train)
            loss_train = torch.mean(losses)
            loss_train_list.append(float(loss_train))
    
            # Calculates the validation loss
            loss_val = torch.mean(density_estimator.loss(params_val, condition=data_val))
            loss_val_list.append(float(loss_val))
            
            # Updates the min validation loss and the number of times in a row the validation loss didn't improve (p)
            if loss_val < loss_val_min:
                loss_val_min = loss_val
                p = 0
                best_estimator = density_estimator
            else:
                p += 1
    
            e += 1
    
            loss_train.backward()
            opt.step()
    
        if best_estimator is not None:
            return best_estimator, loss_train_list, loss_val_list
        else:
            return density_estimator, loss_train_list, loss_val_list


    def posterior_built_NDE(self,
                            nde, 
                            data_summary, 
                            param_arr, 
                            data_obs, 
                            lr, 
                            epochs_max, 
                            num_components, 
                            hidden_features = 50, 
                            val_size = 0.5, 
                            patience = 20,
                           ens = False):

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

        # Creating the NPE 
        if nde == 'maf':
            density_estimator = build_maf(param_arr, data_summary, hidden_features = hidden_features)
        elif nde == 'mdn':
            density_estimator = build_mdn(param_arr, data_summary, hidden_features = hidden_features, num_components = num_components)
        elif nde == 'nsf':
            density_estimator = build_nsf(param_arr, data_summary, hidden_features = hidden_features)
        else:
            print('Not a valid nde string')
            
        # Calling the function for the training process
        density_estimator, loss_train_list, loss_val_list = self.training(density_estimator, 
                                                                          param_arr, 
                                                                          data_summary, 
                                                                          lr=lr, 
                                                                          epochs_max = epochs_max, 
                                                                          val_size = val_size, 
                                                                          patience = patience)

        if not ens:
            # Sampling from the posterior only if the NDE is not part of an ensemble 
            samples = density_estimator.sample((100000,), condition=data_obs).detach()
            samples = samples.squeeze(dim=1)
        
            return samples, loss_train_list, loss_val_list, density_estimator
        else:
            return loss_train_list, loss_val_list, density_estimator


    def posterior_corner_n(self,
                           nde_list, 
                           lr, 
                           epochs_max, 
                           val_size = 0.2, 
                           patience = 20, 
                           loss_plot = False, 
                           corner_plot = False, 
                           loss_save = '', 
                           corner_save = ''):
        
        """
        This function calls the function that builds and trains the NDE.
        It then plots the training and validation loss and the posterior.
        It takes multiple NDEs as input so that you can compare the performance of different NDEs.

        Inputs:
            - nde_list: list of NDEs (the elements of the list are dictionaries. See tutorial 5.)
            - lr: learning rate
            - epochs_max: max number of epochs
            - val_size: validation size 
            - patience: number of epochs after which the training is stopped if the validation 
                        loss doesn't improve for this number of epochs consecutively 
            - loss_plot: (bool) if you want to plot the training and validation loss
            - corner_plot: (bool) if you want to plot the corner plot of the posterior 
            - loss_save: the string for the name of the file if you want to save the loss plot.
                         If loss_save == '' then it is not saved 
            - corner_save: the string for the name of the file if you want to save the corner plot.
                           If corner_save == '' then it is not saved 

        Outputs:
            - density_estimator_list: the trained density estimator
            - samples: the samples which are taken from the trained density estimator
        """
        
        cmap = colormaps['tab10']
        
        # Lists used to save the samples, limits of the corner plot and density estimators of all the NDEs
        samples_list = []
        limit_list = []
        density_estimator_list = []

        # For loop for each NDE
        for ind, nde_dict in tqdm(enumerate(nde_list)):
            # Checking that the data summary is actually a list
            assert type(nde_dict['stats']) is list, f'Element at index = {ind} of nde_list must be a list'

            # Concatenating the chosen data summaries
            data_summary, data_obs = self.concatenate_summaries(nde_dict['stats'], nde_dict['size'], nde_dict['bins'])
            
            # Training the NDE
            samples, loss_train_list, loss_val_list, density_estimator = self.posterior_built_NDE(nde_dict['nde'], 
                                                                          data_summary, 
                                                                          self.param_list[:nde_dict['size']], 
                                                                          data_obs, 
                                                                          lr, 
                                                                          epochs_max,
                                                                          nde_dict['num_components'],
                                                                          nde_dict['hidden_features'], 
                                                                          val_size = val_size, 
                                                                          patience = patience,
                                                                          )

            density_estimator_list.append(density_estimator)

            # Plotting the losses on the same figure
            if loss_plot:
                plt.plot(loss_train_list, color = cmap(ind), linestyle = '-', label='Train: ' + '-'.join(nde_dict['stats']))
                plt.plot(loss_val_list, color = cmap(ind), linestyle = ':', label='Val: ' + '-'.join(nde_dict['stats']))
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
            
            # Calculating the limits of the corner plot for each NDE
            limit = [self.set_limit(samples[:,0]), self.set_limit(samples[:,1])]
            samples_list.append(samples)
            limit_list.append(limit)
            
        # Only if you want to plot the loss
        if loss_plot:
            plt.legend()
            plt.grid()

            # Only if you want to save the plot of the losses
            if loss_save:
                plt.savefig(loss_save)
            plt.show()

        # Only if you want to plot the posteriors
        if corner_plot:
            self.corner_plot(nde_list, samples_list, limit_list, corner_save)
            
        return density_estimator_list, samples


    def corner_plot(self, nde_list, samples_list, limit_list, corner_save):
        """
        This function plots the corner plot for the posteriors of the NDEs.
        It is called by the function posterior_corner_n().

        Inputs:
        - nde_list: list of NDEs (the elements of the list are dictionaries. See tutorial 5.)
        - samples_list: list of the samples taken from the trained density estimator 
        - limit_list: list of the limits of each posterior
        - corner_save: the string for the name of the file if you want to save the corner plot.
                       If corner_save == '' then it is not saved
        """

        cmap = colormaps['tab10']

        # Calculating the limits for the corner plot (all the posteriors will be plotted in the same corner plot)
        plot_limit = [(min(x[0][0] for x in limit_list), max(x[0][1] for x in limit_list)), 
                      (min(x[1][0] for x in limit_list), max(x[1][1] for x in limit_list))]
        
        fig = plt.figure(figsize = (6, 6))
        lines_list = []

        # Plotting all the posteriors in the same cornerplot
        for ind, nde_dict in enumerate(nde_list):
            c = corner.corner(
                    np.array(samples_list[ind]),
                    color = colors.rgb2hex(cmap(ind), keep_alpha=True), 
                    fig=fig,
                    truths = self.param_true, 
                    levels = (0.95,),#(0.68, 0.95), 
                    plot_contours=True,
                    plot_density=False,
                    plot_datapoints=False,
                    fill_contours=False,
                    smooth = True,
                    bins = 20,
                    labels = ['$HII_{eff}$','$T_{vir}$'],
                    range=plot_limit)
        
            lines_list.append(mlines.Line2D([],[], color = cmap(ind), label='-'.join(nde_dict['stats'])))

        
        fig.legend(handles = lines_list, loc=(0.6,0.75))

        # Only if you want to save the plot
        if corner_save:
            fig.savefig(corner_save)
        plt.show()


    def ensemble_NDEs(self, NDEs, lr, epochs_max, val_size = 0.2, patience = 20):
        """
        This functions calculates the posterior of an ensemble of NDEs.
        Each NDE will be trained and the individual posterior will be combnined together 
        into one single posterior using the validation loss as a weight.

        Inputs:
            - NDEs: list of NDEs that comprise the enesmeble 
            - lr: learning rate 
            - epochs_max: max number of epochs 
            - val_size = 0.2: validation size 
            - patience: number of epochs after which the training is stopped if the validation 
                        loss doesn't improve for this number of epochs consecutively

        Outputs:
            - posterior_tot: the final posterior             
        """

        
        nde_stats = NDEs[0]['stats']

        # For loop for each NDE
        for nde_dict in NDEs:
            # Checking that nde_dict['stats'] is actually a list and is the same for all the NDEs of the ensemble
            assert type(nde_dict['stats']) is list, f'Elements of nde_list must be a list'
            assert nde_dict['stats'] == nde_stats, f'NDEs should have same statistics'
        
        # Concatenating the chosen data summaries
        data_summary, data_obs = self.concatenate_summaries(nde_dict['stats'], nde_dict['size'], nde_dict['bins'])
        
        # Lists used to save the samples, limits of the corner plot and density estimators of all the NDEs
        NDEs_list = []
        loss_train_list_all = []
        loss_val_list_all = []
        weights = []

        # For loop for each NDE
        for nde_dict in tqdm(NDEs):
            # Training the NDE
            loss_train_list, loss_val_list, density_estimator = self.posterior_built_NDE(nde_dict['nde'], 
                                                                          data_summary, 
                                                                          self.param_list[:nde_dict['size']], 
                                                                          data_obs, 
                                                                          lr, 
                                                                          epochs_max,
                                                                          nde_dict['num_components'],
                                                                          nde_dict['hidden_features'], 
                                                                          val_size = val_size, 
                                                                          patience = patience,
                                                                          ens = True)

            loss_train_list_all.append(loss_train_list)
            loss_val_list_all.append(loss_val_list)
            weights.append(torch.tensor((np.exp(-loss_train_list[-1]))))
            
            posterior = DirectPosterior(density_estimator, self.prior)
    
            NDEs_list.append(posterior)

        # Building the final posterior. If it doesn't work, copy and paste EnsemblePosterior from the package to your file/notebook
        posterior_tot = EnsemblePosterior(NDEs_list, weights)
        #posterior_tot = EnsemblePosterior_2(NDEs_list, weights)
    
        return posterior_tot
    
    
    def ensemble_n(self, ensemble_list, lr, epochs_max, val_size = 0.2, patience = 20, plot_ensembles=False, corner_save=''):
        """
        This functions traines multiple ensembles of NDEs by calling the function ensemble_NDEs() for each one fo them.
        See tutorial 5 on how to use it.

        Inputs:
            - ensemble_list: list of the ensembles 
            - lr: learning rate
            - epochs_max: max number of epochs
            - val_size: validation size
            - patience: number of epochs after which the training is stopped if the validation 
                        loss doesn't improve for this number of epochs consecutively 
            - plot_ensembles: (bool) if you want to plot the posterior of the ensembles 
            - corner_save: the string for the name of the file if you want to save the corner plot.
                           If corner_save == '' then it is not saved

        Output:
            - posterior_list: list of the posteriors
        """

        # Creating the list for the posterior and the "observed data"
        posterior_list = []
        data_obs_list = []

        # For loop for each ensemble
        for ens in ensemble_list:
            # Concatenating the chosen data summaries
            data_summary, data_obs = self.concatenate_summaries(ens[0]['stats'], ens[0]['size'], ens[0]['bins'])
            data_obs_list.append(data_obs)

            # Training each ensemble
            posterior_tot = self.ensemble_NDEs(ens, lr, epochs_max, val_size = val_size, patience = patience)
            posterior_list.append(posterior_tot)

        if plot_ensembles:
            self.plot_ensemble_n(posterior_list, data_obs_list, ensemble_list, corner_save)
    
        return posterior_list
    
    
    def plot_ensemble_n(self, posterior_list, data_obs_list, ensemble_list, corner_save):
        """
        This function plots the ensembles of NDEs.

        Inputs: 
            - posterior_list: list of the posteriors of each ensemble 
            - data_obs_list: list of the "observed data" 
            - ensemble_list: list of the ensembles 
            - corner_save: the string for the name of the file if you want to save the corner plot.
                           If corner_save == '' then it is not saved
        """

        cmap = colormaps['tab10']
        
        # List of the samples and the limits for each ensemble
        samples_list = []
        limit_list = []

        # For loop for each posterior
        for posterior, data_obs in zip(posterior_list, data_obs_list):
            data_obs = data_obs.reshape(1,data_obs.shape[0])
            samples = posterior.sample((100000,), x=data_obs, show_progress_bars = False)
            limit = [self.set_limit(samples[:,0]), self.set_limit(samples[:,1])]  
            samples_list.append(samples)
            limit_list.append(limit)
        
        # Calculatig the limits of the corner plot where alle the posteriors of the ensembles are plotted
        plot_limit = [(min(x[0][0] for x in limit_list), max(x[0][1] for x in limit_list)), 
                      (min(x[1][0] for x in limit_list), max(x[1][1] for x in limit_list))]
        
        fig = plt.figure(figsize = (6, 6))
        lines_list = []

        # Plotting the corner plot
        for ind, ens in enumerate(ensemble_list):
            c = corner.corner(
                    np.array(samples_list[ind]),
                    color = colors.rgb2hex(cmap(ind), keep_alpha=True), 
                    fig=fig,
                    truths = self.param_true, 
                    levels = (0.95,),#(0.68, 0.95), 
                    plot_contours=True,
                    plot_density=False,
                    plot_datapoints=False,
                    fill_contours=False,
                    smooth = True,
                    bins = 20,
                    labels = ['$HII_{eff}$','$T_{vir}$'],
                    range=None #plot_limit
                )
        
        
            lines_list.append(mlines.Line2D([],[], color = cmap(ind), label='-'.join(ens[0]['stats'])))
    
        fig.legend(handles = lines_list, loc=(0.6,0.75))
        if corner_save:
            fig.savefig(corner_save)
        plt.show()
        

    def set_limit(self, torch_arr):
        """
        This function calculates the limit of the posterior

        Inputs:
            - torch_arr: array for which to calculate the limits

        Outputs:
            - (arr_low, arr_high): tuple of the lower and upper limits
        """

        n = len(torch_arr)
        arr_high = torch.min(torch.topk(torch_arr, int(n*0.002), largest= True).values)
        arr_low = torch.max(torch.topk(torch_arr, int(n*0.0005), largest= False).values)
        return (arr_low, arr_high)


    def calc_med_std_factor_approx(self, gauss_x, gauss_y):
        """
        This functions finds the best initial values for the gaussian fit.

        Inputs:
            - gauss_x: x values
            - gauss_y: y values

        Ouputs:
            - list with the follwing values:
                - gauss_x[med_ind]: median
                - std_approx: approximate standard deviation
                - factor: scaling factor
        """

        # It is not perfect, but it does the job of getting good enough initial values
        med_ind = np.argmax(gauss_y)
        mask = gauss_y > gauss_y[med_ind]/2
        
        mask_ind_right = np.array([i>=med_ind for i in range(len(gauss_x))])
        mask_ind_left = np.array([i<=med_ind for i in range(len(gauss_x))])
        
        mask_left = mask &  mask_ind_left
        mask_right = mask &  mask_ind_right
        half_left_x = gauss_x[mask_left]
        half_right_x = gauss_x[mask_right]
        half_left_y = gauss_y[mask_left]
        half_right_y = gauss_y[mask_right]
    
        area_left = np.trapz(half_left_y, half_left_x)
        area_right = np.trapz(half_right_y, half_right_x)
    
        factor = max(area_left, area_right) * 2 / 0.76
        
        hwhm_left = half_left_x[-1] - half_left_x[0] 
        hwhm_right = half_right_x[-1] - half_right_x[0]
    
        fwhm = max(hwhm_left, hwhm_right) * 2
        
        std_approx = fwhm / (2*(2*np.log(2)))
        
        return [gauss_x[med_ind], std_approx, factor]
    
    
    def gaussian(self, x, mu, std, factor):
        """
        This is the equation for a gaussian with a scaling factor

        Inputs:
            - x: x variable
            - mu: average
            - std: standard deviation
            - factor: scaling factor

        Output:
            - gaussian with scaling factor
        """

        return (
            factor * 1.0 / (np.sqrt(2.0 * np.pi) * std) * np.exp(-0.5 * ((x - mu)**2) / (std**2))
        )
        
    def find_best_gaussian(self, x, y):
        """
        This function finds the best gaussian fit

        Inputs:
            - x: x data
            - y: y data
        
        Outputs:
            - mu: average of the best gaussian fit
            - std: standard deviation of the best gaussian fit
            - factor: scaling factor of the best gaussian fit
        """

        # Initial values
        p0 = self.calc_med_std_factor_approx(x, y)
        params = curve_fit(self.gaussian, x, y, p0 = p0)
    
        mu, std, factor = params[0]
    
        return mu, std, factor
    
    
    def calc_likelihood(self, nde_dict, posterior, n, parameter_grid, ens = False):
        """
        This function calculates the likelihood from the posterior.

        Inputs:
            - nde_dict: NDE dictionary (See tutorial 5)
            - posterior: posterior/density estimator
            - n: number of points for the likelihood
            - parameter_grid: grid of vaues in paramter space
            - ens: (bool) if you are considering an ensemble or an individual NDE

        Outputs:
            - like_eff: likelihood of the ionising efficiency
            - like_Tvir: likelihood of the virial temperature
        """

        if not ens:
            posterior = DirectPosterior(posterior, self.prior)
        
        # Concatenating the chosen data summaries
        data_summary, data_obs = self.concatenate_summaries(nde_dict['stats'], nde_dict['size'], nde_dict['bins'])

        # Calculating the log probability at each point in parameter space
        log_probability_samples = posterior.log_prob(parameter_grid, x=data_obs.unsqueeze(0)).exp()

        # Convert the likelihood list to a NumPy array
        likelihoods = torch.tensor(log_probability_samples).numpy()
        
        # Reshape likelihoods for plotting
        likelihoods = likelihoods.reshape(n, n)
    
        like_Tvir = np.sum(likelihoods, axis = 0)
        like_eff = np.sum(likelihoods, axis = 1)
    
        return like_eff, like_Tvir 
    
    
    def calc_skew(self, x,y):
        """
        This function calculates the skewness of the likelihood.

        Inputs:
            - x: x data
            - y: y data

        Outputs:
            - skewness
        """

        # Calculting the difference along the x direction between two consecutive points 
        dx = np.diff(x)[0]

        # Calculting the probability of finding values in an interval dx
        prob_vals = y * dx / np.sum(y*dx)

        # Sampling points from the likelihood
        data = np.random.choice(x, size = 1000000, p=prob_vals)

        # Returning the skewness of the sample
        return scipy.stats.skew(data)
            
            
                
                    