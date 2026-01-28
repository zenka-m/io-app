#space for correlation, PCA and Spectra - diagrams

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



class Explorer:


    #notka przy gui zrobic sprawdzenie przed czy band ok jeżeli będzie implemetacja tego w gui bo sie wyswietla tu tylko w konsolce!
#validating methods
    @staticmethod
    def validate_band( band:int, total_bands: int) -> bool:
        if band < 0 or band >= total_bands:
            print(f"Wrong band, image has only {total_bands} bands")
            return False

        return True

    @staticmethod
    def validate_numer_images(band: int, total_bands: int) -> bool:
        if band < 0 or band >= total_bands:
            print(f"Wrong band, image has only {total_bands} bands")
            return False

        return True






#working methods
    @staticmethod
    def compare_masked_vs_not_masked( image_3d: np.ma.MaskedArray, band: int = 0):

        bands, height, width = image_3d.shape
        if not Explorer.validate_band(band, bands):

                print(f"Wrong band, image has only {bands} bands")

        else:

            fig, ax = plt.subplots(1, 2, figsize=(8, 8))

            ax[0].imshow(image_3d[band, :, :].data)
            ax[1].imshow(image_3d[band, :, :])

            plt.suptitle(f'Band {band} of spectral image ')
            plt.show()



    @staticmethod
    def plot_single_image_spectrum(image: np.ma.MaskedArray, wavelengths: pd.DataFrame):


        raw_curve = np.mean(image.data, axis=(1, 2))
        masked_curve = np.mean(image, axis=(1, 2))
        wl_axis = wavelengths.iloc[:, 0].values

        plt.figure(figsize=(10, 6))
        plt.plot(wl_axis, raw_curve, label="Raw data (with background)", color='red', linestyle='--')
        plt.plot(wl_axis, masked_curve, label="Masked data (clean)", color='green')

        title = f'Average reflectance'
        plt.title(f'Average reflectance for wavelengths')
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Reflectance")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    @staticmethod
    def plot_image_derivatives(image_3d: np.ma.MaskedArray, wave_lengths: pd.DataFrame):

        masked_curve = np.mean(image_3d, axis=(1, 2))
        first_der = np.gradient(masked_curve, axis=0)
        second_der = np.gradient(first_der, axis=0)
        third_der = np.gradient(second_der, axis=0)

        # plt.plot(wave_lengths['wavelength'], masked_curve, label = "Masked data")
        fig, ax = plt.subplots(1, 2, figsize=(14, 8))
        plt.suptitle(f"Average reflectance of image and its respective derivatives")
        ax[0].plot(wave_lengths['wavelength'], masked_curve, label="Masked curve")
        ax[1].plot(wave_lengths['wavelength'], first_der, label="First derivative")
        ax[1].plot(wave_lengths['wavelength'], second_der, label="Second derivative")
        ax[1].plot(wave_lengths['wavelength'], third_der, label="Third derivative")
        ax[1].legend()
        fig.supxlabel("Wavelength")
        plt.show()

    @staticmethod
    def plot_band_correlation_matrix(image_3d: np.ma.MaskedArray):

        bands, height, width = image_3d.shape
        image_flat = image_3d.reshape(bands, -1)
        corr = np.corrcoef(image_flat)

        plt.imshow(corr, cmap='viridis')
        plt.colorbar()
        plt.xticks(range(0, bands + 1, 25))
        plt.yticks(range(0, bands + 1, 25))
        plt.show()










