#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pywt

class Pietro_Wavelet_Transforms_1():
    '''
    Class for creating data summaries of the coeval cubes.
    It can be used to calculate PS, S1, S2 and PS3d of the coeval cube and
    the line of sight decomposition
    '''
    
    def __init__(self, box_size, n_pixels, bins, l1, l2, wavelet_type):
        '''
        Initialization of the object.

        Args:
            - box_size: size of the coeval cube in Mpc
            - n_pixels: number of pixel of the coeval cube
            - bins: number of bins for the statistics PS, S1, S2 and PS3d
            - l1: boolean value; True if you want to calculate l1 summary
            - l2: boolean value; True if you want to calculate l2 summary
            - wavelet_type: wavelet to be used for the line of sight decomposition
        '''
        
        self.box_size = box_size
        self.n_pixels = n_pixels
        self.bins = bins
        self.l1 = l1
        self.l2 = l2
        self.wavelet_type = wavelet_type

        # Calculating the k modes and the 2-dim box with the corresponding value of k
        self.ks_edges, self.ks, self.k_box = self._get_ks(box_size, n_pixels, bins)

        # Create array of family of window function 
        self.window_arr = self._create_window()

        # Same calculations but for the 3-dim case
        self.ks_edges_3d, self.ks_3d, self.k_box_3d = self._get_ks_3d(box_size, n_pixels, bins)
        self.window_arr_3d = self._create_window_3d()

        
    def _create_window(self):
        '''
        Function that creates the 2-dim gaussian window for each k_bin of PS, S1, S2

        Inputs: 

        Outputs:
            - window_arr: array of window functions
        
        '''

        # Creating a array of zeros for the windows
        window_arr = np.zeros((len(self.ks_edges) - 1, self.k_box.shape[0], self.k_box.shape[1]))

        # For loop which calculates guassian window for each k mode
        for i in range(len(self.ks_edges)-1):
            mean = (self.ks_edges[i] + self.ks_edges[i+1]) / 2
            std = (self.ks_edges[i+1] - self.ks_edges[i]) / 4 
            mean = np.ones(self.k_box.shape) * mean
            std = np.ones(self.k_box.shape) * std
            window_arr[i] = np.array(list(map(self._gauss_window, self.k_box, mean, std)))

        return window_arr

        
    def _create_window_3d(self):
        '''
        Function that creates the 3-dim gaussian window for each k_bin of PS, S1, S2

        Inputs: 

        Outputs:
            - window_arr: array of window functions
        
        '''

        # Creating a array of zeros for the windows
        window_arr = np.zeros((len(self.ks_edges_3d) - 1, self.k_box_3d.shape[0], self.k_box_3d.shape[1], self.k_box_3d.shape[2]))

        # For loop which calculates guassian window for each k mode
        for i in range(len(self.ks_edges_3d)-1):
            mean = (self.ks_edges_3d[i] + self.ks_edges_3d[i+1]) / 2
            std = (self.ks_edges_3d[i+1] - self.ks_edges_3d[i]) / 4 
            mean = np.ones(self.k_box_3d.shape) * mean
            std = np.ones(self.k_box_3d.shape) * std
            window_arr[i] = np.array(list(map(self._gauss_window, self.k_box_3d, mean, std)))

        return window_arr


    def S1(self, box_map, plot_fig = False):
        '''
        This function calculates S1 for one slice of the coeval cube.

        Inputs:
            - box_map: slice of the coeval cube
            - plot_fig: if you want to plot S1 (bool)

        Outputs:
            - ps_bins: binned values of S1 
        '''
        
        # Calculating number of cells, volume and pixel volume of the slice
        N_cells = np.prod(box_map.shape)
        box_volume = self.box_size**(len(box_map.shape))
        volume_pix = box_volume / N_cells

        # Taking the fourier transform of the slice
        fourier_image = np.fft.fft2(box_map) / N_cells
        fourier_image = np.fft.fftshift(fourier_image)

        # Calculation of S1 for each bin
        ps_bins = np.zeros(self.bins)
        for i in range(len(self.window_arr)):
            ps_bins[i] = np.sum(np.abs(fourier_image * self.window_arr[i]))

        #ps_bins = ps_bins * volume_pix**2/box_volume

        # Plotting S1
        if plot_fig:
            self._plot_PS(ps_bins, ks, box_map.shape)
            
        return ps_bins

    def PS(self, box_map, plot_fig = False): 
        '''
        This function calculates PS for one slice of the coeval cube.

        Inputs:
            - box_map: slice of the coeval cube
            - plot_fig: if you want to plot PS (bool)

        Outputs:
            - ps_bins: binned values of PS 
        '''

        # Calculating number of cells, volume and pixel volume of the slice/box
        N_cells = np.prod(box_map.shape)
        box_volume = self.box_size**(len(box_map.shape))
        volume_pix = box_volume / N_cells
        
        # Calculting the PS depending on whether box_map is 2-dim or 3-dim
        ps_bins = np.zeros(self.bins)
        if len(box_map.shape) == 2:
            # Taking the fourier transform of the slice
            fourier_image = np.fft.fft2(box_map) / N_cells
            fourier_image = np.fft.fftshift(fourier_image)

            # Calculation of PS for each bin
            for i in range(len(self.window_arr)):
                ps_bins[i] = np.sum(np.abs(fourier_image * self.window_arr[i])**2)
        elif len(box_map.shape) == 3:
            # Taking the fourier transform of the slice
            fourier_image = np.fft.fftn(box_map) / N_cells
            fourier_image = np.fft.fftshift(fourier_image)

            # Calculation of PS for each bin
            for i in range(len(self.window_arr_3d)):
                ps_bins[i] = np.sum(np.abs(fourier_image * self.window_arr_3d[i])**2)

        ps_bins = ps_bins * volume_pix**2/box_volume
        
        #if plot_fig:
        #    self._plot_PS(ps_bins, ks, box_map.shape)
            
        return ps_bins

    def S2(self, box_map, plot_fig = False):
        '''
        This function calculates S2 for one slice of the coeval cube.

        Inputs:
            - box_map: slice of the coeval cube
            - plot_fig: if you want to plot S2 (bool)

        Outputs:
            - ps_bins: binned values of S2
        '''
        
        # Calculating number of cells, volume and pixel volume of the slice/box
        N_cells = np.prod(box_map.shape)
        box_volume = self.box_size**(len(box_map.shape))
        volume_pix = box_volume / N_cells

        # Taking the fourier transform of the slice
        fourier_image = np.fft.fft2(box_map) / N_cells
        fourier_image = np.fft.fftshift(fourier_image)

        
        S2_arr = np.zeros((self.bins, self.bins), dtype = 'complex_')

        # Reversing the window arr, so that it is easier to 
        # follow the condition that \lambda_2 > \lambda_1
        window_arr_rev = self.window_arr[::-1, :, :]

        # Calculation of S2
        for i in range(len(window_arr_rev) - 1):
            f_x_psi1_fs = fourier_image * window_arr_rev[i]
            f_conv_psi1_real = np.fft.ifft2(np.fft.ifftshift(f_x_psi1_fs))
            f_conv_psi1_real_abs = np.abs(f_conv_psi1_real)
            fpsi1_fs = np.fft.fftshift(np.fft.fft2(f_conv_psi1_real_abs))

            for j in range(i+1, len(window_arr_rev)):
                fpsi1_x_psi2_fs = fpsi1_fs * window_arr_rev[j]
                S2_arr[i,j] = np.sum(fpsi1_x_psi2_fs)

        S2_arr = np.real(S2_arr) #* volume_pix**2/box_volume
            
        return S2_arr


    def _gauss_window(self, x, mean, std):
        '''
        This function returns the gaussian window function

        Inputs:
            - x: variable 
            - mean: mean of the gaussian
            - std: standard deviation of the gaussian
        '''
        
        return np.exp(-(x-mean)**2/(2*std**2))


    def _get_ks(self, box_size, n_pixels, bins):
        '''
        This function calculates the values of k modes for a slice of the cube

        Inputs:
            - box_size: dimesion in Mpc of the coeval cube
            - n_pixels: number of pixels of the coeval cube
            - bins: number of bins for the statistics

        Outputs:
            - ks_edges: edges of the k bins
            - ks: value of the k modes (centers of the bins)
            - k_box: k values in 2-dim box which will correspond to the 
                     fourier values calculated in the functions S1, S2, PS
        '''

        # min and max k value
        k_min = 2*np.pi/(box_size)
        k_max = 2*np.pi/(box_size/int(n_pixels/2 - 1))

        # Calculation of the k_x values
        k_x = np.fft.fftfreq(n_pixels, d=box_size/n_pixels)*2*np.pi
        k_x = np.fft.fftshift(k_x)

        # Calcultion of the k_box (2-dim)
        kx = k_x[:, np.newaxis]
        ky = k_x[np.newaxis, :]
        k_box = (kx**2 + ky**2)**0.5

        # Calcuting the log bins of k modes
        log_dist_k = (np.log10(np.max(k_box)) - np.log10(k_min))/bins
        #log_dist_k = (np.log10(k_max) - np.log10(k_min))/bins
        log_ks_edges = np.array([np.log10(k_min) + log_dist_k*i for i in range(bins+1)])
        ks_edges = 10**log_ks_edges
        ks = np.array([(ks_edges[i] + ks_edges[i+1])/2 for i in range(bins)])
    
        return ks_edges, ks, k_box

    
    def _get_ks_3d(self, box_size, n_pixels, bins):
        '''
        This function calculates the values of k modes for the entire cube

        Inputs:
            - box_size: dimesion in Mpc of the coeval cube
            - n_pixels: number of pixels of the coeval cube
            - bins: number of bins for the statistics

        Outputs:
            - ks_edges: edges of the k bins
            - ks: value of the k modes (centers of the bins)
            - k_box: k values in 2-dim box which will correspond to the 
                     fourier values calculated in the functions S1, S2, PS
        '''

        # min and max k value
        k_min = 2*np.pi/(box_size)
        k_max = 2*np.pi/(box_size/int(n_pixels/2 - 1))

        # Calculation of the k_x values
        k_x = np.fft.fftfreq(n_pixels, d=box_size/n_pixels)*2*np.pi
        k_x = np.fft.fftshift(k_x)

        # Calcultion of the k_box (3-dim)
        kz = k_x[:, np.newaxis, np.newaxis]
        ky = k_x[np.newaxis, :, np.newaxis]
        kx = k_x[np.newaxis, np.newaxis, :]
        k_box = (kx**2 + ky**2 + kz**2)**0.5

        # Calcuting the log bins of k modes
        log_dist_k = (np.log10(np.max(k_box)) - np.log10(k_min))/bins
        #log_dist_k = (np.log10(k_max) - np.log10(k_min))/bins
        log_ks_edges = np.array([np.log10(k_min) + log_dist_k*i for i in range(bins+1)])
        ks_edges = 10**log_ks_edges
        ks = np.array([(ks_edges[i] + ks_edges[i+1])/2 for i in range(bins)])
    
        return ks_edges, ks, k_box
        

    def _plot_PS(self, ps, ks, map_shape):
        '''
        Function that plots PS or S1.

        Inputs:
            - ps: values of PS or S1
            - ks: k modes
            - map_shape: shape of the map

        Output: no output, but it draws the plot
        '''
        
        # Plotting S1, S2. It is called directly from the S1 and S2 functions.
        plt.rcParams['figure.figsize'] = [7, 5] 
        #ptool, ktool = t2c.power_spectrum_1d(box_map, kbins=bins, box_dims=box_size)
        #plt.loglog(ktool, ptool*(ktool**(len(box_map.shape)))/2/np.pi**2)
        plt.loglog(ks, ps*(ks**(len(map_shape)))/2/np.pi**2)
        plt.xlabel('k (Mpc$^{-1}$)')
        plt.ylabel(f'P(k) k$^{len(map_shape)}$/$(2\pi^2)$')
        plt.title(f'2D Power spectrum (window: gaussian')
        plt.show()


    def l1_l2_summary(self, box_map, S_func, plot_fig = False):
        '''
        This function calculates the line of sight decomposition of the data statistics.
        It calculates the l1 and l2 summaries by doing the Wavelet Scattering Transforms
        with the Morlet wavelets.
        l1: sum(|values|)
        l2: sqrt(sum(values^2))
    
        Inputs:
            - box_map: array for which we want to calculate the line of sight decomposition
            - S_func: the function ('S1' or 'PS') for which you want to calculate the l1 and l2 summary (string)
            - plot_fig: if you want to plot the fig (bool)
    
        Outputs:
            - l1_summary: l1 summary of the data if l1 is true 
            - l2_summary: l2 summary of the data if l2 is true 
            - ks: k modes
            - scales: j scales used for the l1 and l2 summary
        '''

        # Calculating the S_func statistics for which you want to calculate the summary
        ps2d_list = np.array([getattr(self, S_func)(box_map[i], plot_fig = False) for i in range(len(box_map))])

        # j scales for the wavelet scattering transforms
        j_max = int(np.log2(box_map.shape[0]))
        scales = [pow(2,j) for j in range(1, j_max+1)]

        # Create array of zeros for l1 and l2 summaries
        if self.l1:
            l1_summary = np.zeros((len(scales), self.bins))

        if self.l2:
            l2_summary = np.zeros((len(scales), self.bins))

        # For every bin calculate the wavelet scattering transform
        for i in range(self.bins):
            cwtmatr, _ = pywt.cwt(ps2d_list[:, i], scales, self.wavelet_type)

            # Calculating the l1 and l2 summaries
            for l in range(len(cwtmatr)):
                if self.l1:
                    l1_summary[l, i] =  np.sum(np.abs(cwtmatr[l]))
                if self.l2:
                    l2_summary[l, i] =  np.linalg.norm(cwtmatr[l])
  

        # Return statements depending on l1 and l2 values
        # And if plot_fig == True, the plot is also shown
        if self.l1 and self.l2:
            if plot_fig:
                self._plot_l1_l2(self.ks, scales, l1_summary, l2_summary)
                
            return l1_summary, l2_summary, self.ks, scales
            
        elif self.l1:
            if plot_fig:
                self._plot_l1_l2(self.ks, scales, l1_summary=l1_summary)
            return l1_summary, self.ks, scales
        elif self.l2:
            if plot_fig:
                self._plot_l1_l2(self.ks, scales, l2_summary=l2_summary)
            return l2_summary, self.ks, scales


    def l1_l2_summary_S2(self, box_map):
        '''
        This function calculates the line of sight decomposition for S2.
        It calculates the l1 and l2 summaries by doing the Wavelet Scattering Transforms
        with the Morlet wavelets.
        l1: sum(|values|)
        l2: sqrt(sum(values^2))
    
        Inputs:
            - box_map: array for which we want to calculate the line of sight decomposition
    
        Outputs:
            - l1_summary: l1 summary of the data if l1 is true 
            - l2_summary: l2 summary of the data if l2 is true 
            - ks: k modes
            - scales: j scales used for the l1 and l2 summary
        '''

        # Calculating the S_func statistics for which you want to calculate the summary
        ps2d_list = np.array([getattr(self, 'S2')(box_map[i], plot_fig = False) for i in range(len(box_map))])

        # j scales for the wavelet scattering transforms
        j_max = int(np.log2(box_map.shape[0]))
        scales = [pow(2,j) for j in range(1, j_max+1)]

        # Create array of zeros for l1 and l2 summaries
        if self.l1:
            l1_summary = np.zeros((len(scales), self.bins, self.bins))

        if self.l2:
            l2_summary = np.zeros((len(scales), self.bins, self.bins))

        # For every bin calculate the wavelet scattering transform
        for i in range(self.bins):
            for j in range(self.bins):
                cwtmatr, _ = pywt.cwt(ps2d_list[:, i, j], scales, self.wavelet_type)

                # Calculating the l1 and l2 summaries
                for l in range(len(cwtmatr)):
                    if self.l1:
                        l1_summary[l, i, j] =  np.sum(np.abs(cwtmatr[l]))
                    if self.l2:
                        l2_summary[l, i, j] =  np.linalg.norm(cwtmatr[l])
  

        # Return statements depending on l1 and l2 values
        if self.l1 and self.l2:   
            return l1_summary, l2_summary, self.ks, scales
        elif self.l1:
            return l1_summary, self.ks, scales
        elif self.l2:
            return l2_summary, self.ks, scales


    def _plot_l1_l2(self, ks2d, scales, l1_summary = None, l2_summary = None):
        '''
        Function that plots the l1 and l2 summaries of PS or S1.

        Inputs:
            - ks2d: k modes
            - scales: the values of j used in the l1 and l2 summaries
            - l1_summary: values of the l1 summary (None if they were not calculated)
            - l2_summary: values of the l2 summary (None if they were not calculated)

        Output: no output, but it draws the plot
        '''
        n = self.l1 + self.l2
        fig2, ax2 = plt.subplots(1,2, figsize=(10,4), tight_layout = True)
        ax2 = ax2.flatten()
        
        for i in range(6):
            if self.l1:
                ax2[0].plot(np.log2(scales), l1_summary[:,i], label=f'Bin: {i+1}')
            if self.l2:
                ax2[1].plot(np.log2(scales), l2_summary[:,i], label=f'Bin: {i+1}')
        
        if self.l1:
            ax2[0].set_yscale('log')
            ax2[0].set_xlabel('j')
            ax2[0].set_title(f'$\ell^{1}$ summary')
            ax2[0].legend()
            
        if self.l2:
            ax2[1].set_yscale('log')
            ax2[1].set_xlabel('j')
            ax2[1].set_title(f'$\ell^{2}$ summary')
            ax2[1].legend()

        if not self.l1:
            ax2[0].set_visible(False)
        if not self.l2:
            ax2[1].set_visible(False)
        
        plt.show()


# class
# prove parameters change statistics
#
# 0.5 neutral fraction take 0.75 z around it
class Pietro_Wavelet_Transforms_3():
    # Class for creating data summaries of the coeval cubes
    def __init__(self, box_size, n_pixels, bins, l1, l2, wavelet_type = 'morl'):
        self.box_size = box_size
        self.n_pixels = n_pixels
        self.bins = bins
        self.l1 = l1
        self.l2 = l2
        self.wavelet_type = wavelet_type
        
        self.ks_edges, self.ks, self.k_box = self._get_ks(box_size, n_pixels, bins)
        self.window_arr = self._create_window()

        self.ks_edges_3d, self.ks_3d, self.k_box_3d = self._get_ks_3d(box_size, n_pixels, bins)
        self.window_arr_3d = self._create_window_3d()

    def _create_window(self):
        # Create the gaussian window for each k_bin of S1, S2, S3
        window_arr = np.zeros((len(self.ks_edges) - 1, self.k_box.shape[0], self.k_box.shape[1]))

        for i in range(len(self.ks_edges)-1):
            mean = (self.ks_edges[i] + self.ks_edges[i+1]) / 2
            std = (self.ks_edges[i+1] - self.ks_edges[i]) / 4 
            mean = np.ones(self.k_box.shape) * mean
            std = np.ones(self.k_box.shape) * std
            window_arr[i] = np.array(list(map(self._gauss_window, self.k_box, mean, std)))

        return window_arr

    def _create_window_3d(self):
        # Create the gaussian window for each k_bin of S1, S2, S3
        window_arr = np.zeros((len(self.ks_edges_3d) - 1, self.k_box_3d.shape[0], self.k_box_3d.shape[1], self.k_box_3d.shape[2]))

        for i in range(len(self.ks_edges_3d)-1):
            mean = (self.ks_edges_3d[i] + self.ks_edges_3d[i+1]) / 2
            std = (self.ks_edges_3d[i+1] - self.ks_edges_3d[i]) / 4 
            mean = np.ones(self.k_box_3d.shape) * mean
            std = np.ones(self.k_box_3d.shape) * std
            window_arr[i] = np.array(list(map(self._gauss_window, self.k_box_3d, mean, std)))

        return window_arr

    def load_sim(self, box_map):
        '''
        This function is used to load the coeval cube and calculate all the
        statistics: S1, S2, PS and PS3d.

        Inputs:
            - box_map: the coeval cube

        Outputs: no outputs; 
                 it just save attributes corresponding to the arrays of PS, S1, S2 and PS3d
        '''

        # Calculatin the four statistics PS3d, S1, PS and S2
        self.PS3d = self._PS(box_map)
        self.S1 = np.array([getattr(self, '_S1')(box_map[i], plot_fig = False) for i in range(len(box_map))])
        self.PS = np.array([getattr(self, '_PS')(box_map[i], plot_fig = False) for i in range(len(box_map))])
        self.S2 = np.array([getattr(self, '_S2')(box_map[i], plot_fig = False) for i in range(len(box_map))])

        # Normalization of S1 and S2
        self.S1 = self.S1 / np.sqrt(self.PS)
        S1_rev = self.S1[:,::-1]
        self.S2 =self.S2 / S1_rev[:,:,np.newaxis] 
        
        # Setting nan values to 0
        self.S1[np.isnan(self.S1)] = 0
        self.S2[np.isnan(self.S2)] = 0


    def _S1(self, box_map, plot_fig = False):
        # Similar to S2, but instead of squaring (fourier_image * window) you take the absolute value
        N_cells = np.prod(box_map.shape)
        box_volume = self.box_size**(len(box_map.shape))
        volume_pix = box_volume / N_cells
        
        fourier_image = np.fft.fft2(box_map) / N_cells
        fourier_image = np.fft.fftshift(fourier_image)
        #fourier_ampl = np.abs(fourier_image) * volume_pix**2/box_volume #**2

        
        ps_bins = np.zeros(self.bins)
        for i in range(len(self.window_arr)):
            ps_bins[i] = np.sum(np.abs(fourier_image * self.window_arr[i]))

        ps_bins = ps_bins * volume_pix**2/box_volume
       
        if plot_fig:
            self._plot_PS(ps_bins, ks, box_map.shape)
            
        return ps_bins

    def _PS(self, box_map, plot_fig = False): 
        # Similar to S1, but instead of taking the absolute value of (fourier_image * window) you take the square
        N_cells = np.prod(box_map.shape)
        box_volume = self.box_size**(len(box_map.shape))
        volume_pix = box_volume / N_cells
        
        ps_bins = np.zeros(self.bins)
        if len(box_map.shape) == 2:
            fourier_image = np.fft.fft2(box_map) / N_cells
            fourier_image = np.fft.fftshift(fourier_image)
            for i in range(len(self.window_arr)):
                ps_bins[i] = np.sum(np.abs(fourier_image * self.window_arr[i])**2)
        elif len(box_map.shape) == 3:
            fourier_image = np.fft.fftn(box_map) / N_cells
            fourier_image = np.fft.fftshift(fourier_image)
            for i in range(len(self.window_arr_3d)):
                ps_bins[i] = np.sum(np.abs(fourier_image * self.window_arr_3d[i])**2)

        ps_bins = ps_bins * volume_pix**2/box_volume
        
        #if plot_fig:
        #    self._plot_PS(ps_bins, ks, box_map.shape)
            
        return ps_bins

    def _S2(self, box_map, plot_fig = False):
        # Integral of |image * window_i| * window_j
        # The result depends on i, j and it has a number of bins equal to self.bins
        N_cells = np.prod(box_map.shape)
        box_volume = self.box_size**(len(box_map.shape))
        volume_pix = box_volume / N_cells
        
        fourier_image = np.fft.fft2(box_map) / N_cells
        fourier_image = np.fft.fftshift(fourier_image)

        S3_arr = np.zeros((self.bins, self.bins), dtype = 'complex_')
        window_arr_rev = self.window_arr[::-1, :, :]
        for i in range(len(window_arr_rev) - 1):
            f_x_psi1_fs = fourier_image * window_arr_rev[i]
            f_conv_psi1_real = np.fft.ifft2(np.fft.ifftshift(f_x_psi1_fs))
            f_conv_psi1_real_abs = np.abs(f_conv_psi1_real)
            fpsi1_fs = np.fft.fftshift(np.fft.fft2(f_conv_psi1_real_abs))

            for j in range(i+1, len(window_arr_rev)):
                fpsi1_x_psi2_fs = fpsi1_fs * window_arr_rev[j]
                S3_arr[i,j] = np.sum(fpsi1_x_psi2_fs)

        S3_arr = np.real(S3_arr) * volume_pix**2/box_volume
            
        return S3_arr


    def _gauss_window(self, x, mean, std):
        # Gaussian window
        return np.exp(-(x-mean)**2/(2*std**2))


    def _get_ks(self, box_size, n_pixels, bins):
        # Calculates the k values: edges, ks and k_box (a 3-dimensional array with same shape
        # as the map which contains the corresponding value of k for that cell
        
        k_min = 2*np.pi/(box_size)
        k_max = 2*np.pi/(box_size/int(n_pixels/2 - 1))
        k_x = np.fft.fftfreq(n_pixels, d=box_size/n_pixels)*2*np.pi
        k_x = np.fft.fftshift(k_x)
    
        kx = k_x[:, np.newaxis]
        ky = k_x[np.newaxis, :]
        k_box = (kx**2 + ky**2)**0.5
    
        log_dist_k = (np.log10(np.max(k_box)) - np.log10(k_min))/bins
        #log_dist_k = (np.log10(k_max) - np.log10(k_min))/bins
        log_ks_edges = np.array([np.log10(k_min) + log_dist_k*i for i in range(bins+1)])
        ks_edges = 10**log_ks_edges
        ks = np.array([(ks_edges[i] + ks_edges[i+1])/2 for i in range(bins)])
    
        return ks_edges, ks, k_box

    
    def _get_ks_3d(self, box_size, n_pixels, bins):
        # Calculates the k values: edges, ks and k_box (a 3-dimensional array with same shape
        # as the map which contains the corresponding value of k for that cell
        
        k_min = 2*np.pi/(box_size)
        k_max = 2*np.pi/(box_size/int(n_pixels/2 - 1))
        k_x = np.fft.fftfreq(n_pixels, d=box_size/n_pixels)*2*np.pi
        k_x = np.fft.fftshift(k_x)
    
        kz = k_x[:, np.newaxis, np.newaxis]
        ky = k_x[np.newaxis, :, np.newaxis]
        kx = k_x[np.newaxis, np.newaxis, :]
        k_box = (kx**2 + ky**2 + kz**2)**0.5
    
        log_dist_k = (np.log10(np.max(k_box)) - np.log10(k_min))/bins
        #log_dist_k = (np.log10(k_max) - np.log10(k_min))/bins
        log_ks_edges = np.array([np.log10(k_min) + log_dist_k*i for i in range(bins+1)])
        ks_edges = 10**log_ks_edges
        ks = np.array([(ks_edges[i] + ks_edges[i+1])/2 for i in range(bins)])
    
        return ks_edges, ks, k_box
        

    def _plot_PS(self, ps, ks, map_shape):
        # Plotting S1, S2. It is called directly from the S1 and S2 functions.
        plt.rcParams['figure.figsize'] = [7, 5] 
        #ptool, ktool = t2c.power_spectrum_1d(box_map, kbins=bins, box_dims=box_size)
        #plt.loglog(ktool, ptool*(ktool**(len(box_map.shape)))/2/np.pi**2)
        plt.loglog(ks, ps*(ks**(len(map_shape)))/2/np.pi**2)
        plt.xlabel('k (Mpc$^{-1}$)')
        plt.ylabel(f'P(k) k$^{len(map_shape)}$/$(2\pi^2)$')
        plt.title(f'2D Power spectrum (window: gaussian')
        plt.show()


    def l1_l2_summary(self, S_func, plot_fig = False):
        # Calculates the l1 and l2 summaries with the Morlet wavelets
        ps2d_list = getattr(self, S_func)
        #ps2d_list = np.array([self.S_func(box_map[i], plot_fig = False)[0] for i in range(len(box_map))])
        #ks2d = self.PS_window(box_map[0], dimension, bins, plot_fig = False)[1]
        
        j_max = int(np.log2(self.n_pixels))
        scales = [pow(2,j) for j in range(1, j_max+1)]

        if self.l1:
            l1_summary = np.zeros((len(scales), self.bins))

        if self.l2:
            l2_summary = np.zeros((len(scales), self.bins))
    
        for i in range(self.bins):
            cwtmatr, _ = pywt.cwt(ps2d_list[:, i], scales, self.wavelet_type)
            
            for l in range(len(cwtmatr)):
                if self.l1:
                    l1_summary[l, i] =  np.sum(np.abs(cwtmatr[l]))
                if self.l2:
                    l2_summary[l, i] =  np.linalg.norm(cwtmatr[l])
  

        if self.l1 and self.l2:
            if plot_fig:
                self._plot_l1_l2(self.ks, scales, l1_summary, l2_summary)
                
            return l1_summary, l2_summary, self.ks, scales
            
        elif self.l1:
            if plot_fig:
                self._plot_l1_l2(self.ks, scales, l1_summary=l1_summary)
            return l1_summary, self.ks, scales
        elif self.l2:
            if plot_fig:
                self._plot_l1_l2(self.ks, scales, l2_summary=l2_summary)
            return l2_summary, self.ks, scales


    def l1_l2_summary_S2(self):
        # Calculates the l1 and l2 summaries with the Morlet wavelets
        ps2d_list = getattr(self, 'S2')
        #ps2d_list = np.array([self.S_func(box_map[i], plot_fig = False)[0] for i in range(len(box_map))])
        #ks2d = self.PS_window(box_map[0], dimension, bins, plot_fig = False)[1]
        
        j_max = int(np.log2(self.n_pixels))
        scales = [pow(2,j) for j in range(1, j_max+1)]

        if self.l1:
            l1_summary = np.zeros((len(scales), self.bins, self.bins))

        if self.l2:
            l2_summary = np.zeros((len(scales), self.bins, self.bins))
    
        for i in range(self.bins):
            for j in range(self.bins):
                cwtmatr, _ = pywt.cwt(ps2d_list[:, i, j], scales, self.wavelet_type)
                
                for l in range(len(cwtmatr)):
                    if self.l1:
                        l1_summary[l, i, j] =  np.sum(np.abs(cwtmatr[l]))
                    if self.l2:
                        l2_summary[l, i, j] =  np.linalg.norm(cwtmatr[l])
  

        if self.l1 and self.l2:   
            return l1_summary, l2_summary, self.ks, scales
        elif self.l1:
            return l1_summary, self.ks, scales
        elif self.l2:
            return l2_summary, self.ks, scales


    def _plot_l1_l2(self, ks2d, scales, l1_summary = None, l2_summary = None):
        # Plots the l1 and l2 summaries
        n = self.l1 + self.l2
        fig2, ax2 = plt.subplots(1,2, figsize=(10,4), tight_layout = True)
        ax2 = ax2.flatten()
        
        for i in range(6):
            if self.l1:
                ax2[0].plot(np.log2(scales), l1_summary[:,i], label=f'Bin: {i+1}')
            if self.l2:
                ax2[1].plot(np.log2(scales), l2_summary[:,i], label=f'Bin: {i+1}')
        
        if self.l1:
            ax2[0].set_yscale('log')
            ax2[0].set_xlabel('j')
            ax2[0].set_title(f'$\ell^{1}$ summary')
            ax2[0].legend()
            
        if self.l2:
            ax2[1].set_yscale('log')
            ax2[1].set_xlabel('j')
            ax2[1].set_title(f'$\ell^{2}$ summary')
            ax2[1].legend()

        if not self.l1:
            ax2[0].set_visible(False)
        if not self.l2:
            ax2[1].set_visible(False)
        
        plt.show()
    