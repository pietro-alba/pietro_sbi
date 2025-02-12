#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pywt

class Pietro_Wavelet_Transforms():
    """
    This class creates data summaries of the coeval cubes
    """
    
    def __init__(self, box_size, n_pixels, bins, l1, l2, wavelet_type = 'morl', normed = True):
        """
        Initialisation of the class.

        Inputs:
            - box_size: size of the cube in Mpc
            - n_pixels: number of pixels of the side of the cube 
            - bins: number of bins for the summary statistics 
            - l1: (bool) if you want to calculate the \ell 1 norm for the line of sight decomposition
            - l2: (bool) if you want to calculate the \ell 2 norm for the line of sight decomposition 
            - wavelet_type: type of wavelet for the line of sight decomposition 
            - normed: (bool) if you want to normalise S2 (the second layer of the Wavelet Scattering Transform)
        """

        self.box_size = box_size
        self.n_pixels = n_pixels
        self.bins = bins
        self.l1 = l1
        self.l2 = l2
        self.wavelet_type = wavelet_type
        self.normed = normed
        
        self.ks_edges, self.ks, self.k_box = self._get_ks(box_size, n_pixels, bins)
        self.window_arr = self._create_window()

        self.ks_edges_3d, self.ks_3d, self.k_box_3d = self._get_ks_3d(box_size, n_pixels, bins)
        self.window_arr_3d = self._create_window_3d()

    def _create_window(self):
        """
        This functon creates the 2-dimensional gaussian window for each k_bin of S1, S2, S3
        
        Outputs:
            - window_arr: the array with the window function for each bin
        """
        
        window_arr = np.zeros((len(self.ks_edges) - 1, self.k_box.shape[0], self.k_box.shape[1]))

        for i in range(len(self.ks_edges)-1):
            mean = (self.ks_edges[i] + self.ks_edges[i+1]) / 2
            std = (self.ks_edges[i+1] - self.ks_edges[i]) / 4 
            mean = np.ones(self.k_box.shape) * mean
            std = np.ones(self.k_box.shape) * std
            window_arr[i] = np.array(list(map(self._gauss_window, self.k_box, mean, std)))

        return window_arr

    def _create_window_3d(self):
        """
        This functon creates the 3-dimensional gaussian window for each k_bin of S1, S2, S3
        
        Outputs:
            - window_arr: the array with the window function for each bin
        """

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

        if self.normed:
            # Normalization of S2
            S1_rev = self.S1[:,::-1]
            self.S2 =self.S2 / S1_rev[:,:,np.newaxis] 
            
            # Setting NaN values to 0
            self.S2[np.isnan(self.S2)] = 0


    def _S1(self, box_map):
        """
        This function calculates the first layer of the Wavelet Scattering Transform

        Input:
            - box_map: a 2-dimensional map
        """
        
        N_cells = np.prod(box_map.shape)
        
        fourier_image = np.fft.fft2(box_map) / N_cells
        fourier_image = np.fft.fftshift(fourier_image)

        ps_bins = np.zeros(self.bins)
        for i in range(len(self.window_arr)):
            f_conv_psi = fourier_image * self.window_arr[i]
            f_conv_psi_real = np.fft.ifft2(np.fft.ifftshift(f_conv_psi))
            ps_bins[i] = np.sum(np.abs(f_conv_psi_real))

            
        return ps_bins

    def _PS(self, box_map): 
        """
        This function calculates the 2- or 3-dimensional power spectrum

        Input:
            - box_map: a 2- or 3-dimensional map
        """
        
        N_cells = np.prod(box_map.shape)

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

        return ps_bins


    def _S2(self, box_map):
        """
        This function calculates the second layer of the Wavelet Scattering Transform

        Input:
            - box_map: a 2-dimensional map
        """
        
        N_cells = np.prod(box_map.shape)
        
        fourier_image = np.fft.fft2(box_map) / N_cells
        fourier_image = np.fft.fftshift(fourier_image)

        S2_arr = np.zeros((self.bins, self.bins))
        window_arr_rev = self.window_arr[::-1, :, :]
        for i in range(len(window_arr_rev) - 1):
            f_x_psi1_fs = fourier_image * window_arr_rev[i]
            f_conv_psi1_real = np.fft.ifft2(np.fft.ifftshift(f_x_psi1_fs))
            f_conv_psi1_real_abs = np.abs(f_conv_psi1_real)
            fpsi1_fs = np.fft.fftshift(np.fft.fft2(f_conv_psi1_real_abs) / N_cells)

            for j in range(i+1, len(window_arr_rev)):
                fpsi1_x_psi2_fs = fpsi1_fs * window_arr_rev[j]
                fpsi1_x_psi2_fs_real = np.fft.ifft2(np.fft.ifftshift(fpsi1_x_psi2_fs))
                S2_arr[i,j] = np.sum(np.abs(fpsi1_x_psi2_fs_real))
            
        return S2_arr


    def _gauss_window(self, x, mean, std):
        """
        This function gives the equation of the gaussian window

        Inputs:
            - x: x data
            - mean: the mean of the gaussian
            - std: the standard deviation of the gaussian
        """

        return np.exp(-(x-mean)**2/(2*std**2))


    def _get_ks(self, box_size, n_pixels, bins):
        """
        This function calculates the values of k-modes 

        Inputs:
            - box_size: size of the cube
            - n_pixels: number of pixels
            - bins: number of bins

        Outputs:
            - ks_edges: position of the edges of the bins in k space 
            - ks: values of the centre of the bins 
            - k_box: values of k for each pixel in the 2-dimensional map
        """
        
        k_min = 2*np.pi/(box_size)
        k_x = np.fft.fftfreq(n_pixels, d=box_size/n_pixels)*2*np.pi
        k_x = np.fft.fftshift(k_x)
    
        kx = k_x[:, np.newaxis]
        ky = k_x[np.newaxis, :]
        k_box = (kx**2 + ky**2)**0.5
    
        log_dist_k = (np.log10(np.max(k_box)) - np.log10(k_min))/bins
        log_ks_edges = np.array([np.log10(k_min) + log_dist_k*i for i in range(bins+1)])
        ks_edges = 10**log_ks_edges
        ks = np.array([(ks_edges[i] + ks_edges[i+1])/2 for i in range(bins)])
    
        return ks_edges, ks, k_box

    
    def _get_ks_3d(self, box_size, n_pixels, bins):
        """
        This function calculates the values of k-modes 

        Inputs:
            - box_size: size of the cube
            - n_pixels: number of pixels
            - bins: number of bins

        Outputs:
            - ks_edges: position of the edges of the bins in k space 
            - ks: values of the centre of the bins 
            - k_box: values of k for each pixel in the 3-dimensional map
        """
        
        k_min = 2*np.pi/(box_size)
        k_x = np.fft.fftfreq(n_pixels, d=box_size/n_pixels)*2*np.pi
        k_x = np.fft.fftshift(k_x)
    
        kz = k_x[:, np.newaxis, np.newaxis]
        ky = k_x[np.newaxis, :, np.newaxis]
        kx = k_x[np.newaxis, np.newaxis, :]
        k_box = (kx**2 + ky**2 + kz**2)**0.5
    
        log_dist_k = (np.log10(np.max(k_box)) - np.log10(k_min))/bins
        log_ks_edges = np.array([np.log10(k_min) + log_dist_k*i for i in range(bins+1)])
        ks_edges = 10**log_ks_edges
        ks = np.array([(ks_edges[i] + ks_edges[i+1])/2 for i in range(bins)])
    
        return ks_edges, ks, k_box
        


    def l1_l2_summary(self, S_func):
        """
        This function calculates the ell1 and ell2 norm of the line of sight decomposition 
        for S1 and PS.

        Inputs:
            - S_func: either 'PS' or 'S1'

        Outputs:
            - l1_summary: ell1 norm of the line of sight decomposition
            - l2_summary: ell1 norm of the line of sight decomposition
            - scales: the 2^j scales 
        """

        # Taking the values of the chosen summary
        ps2d_list = getattr(self, S_func)
        
        # Calculating the max. value of j and the scales
        j_max = int(np.log2(self.n_pixels))
        scales = [pow(2,j) for j in range(1, j_max+1)]

        if self.l1:
            l1_summary = np.zeros((len(scales), self.bins))

        if self.l2:
            l2_summary = np.zeros((len(scales), self.bins))

        # Calcualating the line of sight decomposition for each bin
        for i in range(self.bins):
            cwtmatr, _ = pywt.cwt(ps2d_list[:, i], scales, self.wavelet_type)
            
            for l in range(len(cwtmatr)):
                if self.l1:
                    l1_summary[l, i] =  np.sum(np.abs(cwtmatr[l]))
                if self.l2:
                    l2_summary[l, i] =  np.linalg.norm(cwtmatr[l])
  

        if self.l1 and self.l2:
            return l1_summary, l2_summary, scales
        elif self.l1:
            return l1_summary, scales
        elif self.l2:
            return l2_summary, scales


    def l1_l2_summary_S2(self):
        """
        This function calculates the ell1 and ell2 norm of the line of sight decomposition 
        for S2.

        Outputs:
            - l1_summary: ell1 norm of the line of sight decomposition
            - l2_summary: ell1 norm of the line of sight decomposition
            - scales: the 2^j scales 
        """

        # Taking the values of the chosen summary
        ps2d_list = getattr(self, 'S2')
        
        # Calculating the max. value of j and the scales
        j_max = int(np.log2(self.n_pixels))
        scales = [pow(2,j) for j in range(1, j_max+1)]

        if self.l1:
            l1_summary = np.zeros((len(scales), self.bins, self.bins))

        if self.l2:
            l2_summary = np.zeros((len(scales), self.bins, self.bins))

        # Calcualating the line of sight decomposition for each bin
        for i in range(self.bins):
            for j in range(self.bins):
                cwtmatr, _ = pywt.cwt(ps2d_list[:, i, j], scales, self.wavelet_type)
                
                for l in range(len(cwtmatr)):
                    if self.l1:
                        l1_summary[l, i, j] =  np.sum(np.abs(cwtmatr[l]))
                    if self.l2:
                        l2_summary[l, i, j] =  np.linalg.norm(cwtmatr[l])
  

        if self.l1 and self.l2:   
            return l1_summary, l2_summary, scales
        elif self.l1:
            return l1_summary, scales
        elif self.l2:
            return l2_summary, scales

    