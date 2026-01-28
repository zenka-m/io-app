import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from typing import Tuple, List
from config import Config
from DataManager import DataManager

class Preprocessor:
    def __init__(self, config: Config):
        self.cfg = config
        #self.dm = DataManager(config)

    def _apply_gauss_smoothing(self, spectral_arr: np.ndarray, sigma: Tuple[int] = None):
        """Smoothens spectral image applying Gaussian Filter with parameter sigma.

        Args:
            spectral_arr (np.ndarray): Spectral image data
            sigma (Tuple[int], optional): Gaussian sigma parameter for each axis. Defaults to None.

        Returns:
            np.ndarray: Smoothened spectral image
        """
        if sigma is None:
            sigma = self.cfg.SIGMA_PARAM 
        return gaussian_filter(spectral_arr, sigma=sigma)

    def _calculate_mean_spectrum(self, spectral_arr: np.ma.MaskedArray):
        """Calculates the mean spectrum by taking the average over spatial dimensions.

        Args:
            spectral_arr (np.ma.MaskedArray): Masked hyperspectral image

        Returns:
            np.ndarray: 1D array representing the meanspectrum for each band
        """
        mean_spectrum = spectral_arr.mean(axis=(1,2))

        if np.ma.is_masked(mean_spectrum):
            return None
        return mean_spectrum.data

    def _apply_sav_gol_filter(self, mean_spectrum: np.ndarray, sg_window: int = None, sg_poly: int = None, sg_deriv: int = None):
        """Applies Savitzky-Golay filter to the mean spectrum
        Also we can choose if we want to return a derivative of which order of the spectrum by adjusting the sg_deriv parameter. 

        Args:
            mean_spectrum (np.ndarray): Average reflectance of a spectral np array
            sg_window (int, optional): SG Filter window size. Defaults to None.
            sg_poly (int, optional): SG Filter polynomial order. Defaults to None.
            sg_deriv (int, optional): SG Filter derivative order. Defaults to None.

        Returns:
            np.ndarray: Smoothened spectrum or its derivative based on sg_deriv param
        """
        if sg_window is None:
            sg_window = self.cfg.SG_WINDOW
        if sg_poly is None:
            sg_poly = self.cfg.SG_POLY
        if sg_deriv is None:
            sg_deriv = self.cfg.SG_DERIV

        return savgol_filter(mean_spectrum, window_length=sg_window, polyorder=sg_poly, deriv=sg_deriv)

    def _calculate_higher_derivatives(self, sg_first_deriv: np.ndarray):
        """Calculates second and third order derivatives based on the first derivative.

        Args:
            sg_first_deriv (np.ndarray): First derivative of the average reflectance of a spectral image

        Returns:
            Tuple[np.ndarray]: A tuple containing second and third derivatives 
        """
        second_deriv = np.gradient(sg_first_deriv, axis=0)
        third_deriv = np.gradient(second_deriv, axis=0)

        return second_deriv, third_deriv

    def process_image(self, spectral_arr: np.ma.MaskedArray, include_higher_derivs:bool =False):
        """Processes a signe hyperspectral image through the full pipeline
        1. Spatial Gauss smoothing.
        2. Mean spectrum calculation
        3. Spectral smoothing & first derivative using SG filter
        4. Optional calculation of 2nd and 3rd deivatives

        Args:
            spectral_arr (np.ma.MaskedArray): Raw hyyperspectral image
            include_higher_derivs (bool, optional): If true, appends 2nd and 3rd derivatives to the feature vector. Defaults to False.

        Returns:
            np.ndarray: Extracted features
        """
        processed_data = spectral_arr.copy()

        blurred_data = self._apply_gauss_smoothing(spectral_arr.data)
        processed_data.data[:] = blurred_data
        mean_spectrum = self._calculate_mean_spectrum(processed_data)

        if mean_spectrum is None:
            return None
        
        first_deriv = self._apply_sav_gol_filter(mean_spectrum)

        if not include_higher_derivs:
            return first_deriv
        
        second_deriv, third_deriv = self._calculate_higher_derivatives(first_deriv)

        features = np.concatenate([first_deriv, second_deriv, third_deriv])
        return features      
    
    def process_dataset(self, data_manager: DataManager, image_ids: List, file_path: str, include_higher_derivs:bool =False):
        """Processes a batch of images identified by image_ids
        Iterates through the array and loads each image using the data manager and applies the preprocessing pipeline

        Args:
            data_manager (DataManager): Instance of DataManager
            image_ids List: List of image IDs to process
            file_path (str): Path to the directory where images are stored
            include_higher_derivs (bool, optional): If true inclues higher order derivatives. Defaults to False.

        Returns:
            _type_: _description_
        """
        X_features = []
        valid_indices = []

        image_count = len(image_ids)

        for id in image_ids:
            spectral_arr = data_manager.load_image(id, file_path)          

            if spectral_arr is None:
                continue

            processed_arr = self.process_image(spectral_arr, include_higher_derivs)

            if processed_arr is not None:
                X_features.append(processed_arr)
                valid_indices.append(id)
            
            if id >0 and id % 100 == 0:
                print(f'processed {id}/{image_count} images...')

        X = np.array(X_features)

        return X, valid_indices




    