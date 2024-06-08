from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter


def pull_from_csv(file_path):
    # Load the data from the csv file and remove the last element since it is not captured
    data = np.genfromtxt(file_path, delimiter=',', dtype=float, usecols=[0], skip_header=1)[:-1]
    return data


def gaussian_2d(x, y, x0, y0, sigma_x, sigma_y):
    return np.exp(-(((x - x0) ** 2 / (2.0 * sigma_x ** 2)) + ((y - y0) ** 2 / (2.0 * sigma_y ** 2))))


def pixel_emission_wavelengths_array():
    # Coefficients for the polynomial equation to calculate the emission wavelength for each pixel
    coeffs = [314.089, 2.694, -9.355e-4, -9.793e-6, 1.829e-8, -8.756e-12]

    # Create a list to hold the 288 emission wavelengths for each pixel
    emission_wavelengths = []
    for p in range(1, 289):
        emission_wavelengths.append(
            coeffs[0] +
            (coeffs[1] * p) +
            (coeffs[2] * (p ** 2)) +
            (coeffs[3] * (p ** 3)) +
            (coeffs[4] * (p ** 4)) +
            (coeffs[5] * (p ** 5)))

    return emission_wavelengths


def plot_pixel_calibration():
    plt.plot(pixel_emission_wavelengths_array())
    plt.xlabel('Pixel')
    plt.ylabel('Wavelength (nm)')
    plt.title('Pixel Calibration')
    plt.show()


@dataclass
class LED:
    # Excitation wavelength in nm
    wavelength: float = 272.5
    fwhm: float = 20.0
    peak_intensity: float = 1.0

    def intensity_at_wavelength(self, wavelength, peak_intensity=1.0):
        sigma = self.fwhm / 2.355  # Convert FWHM to standard deviation
        intensity = peak_intensity * np.exp(-((wavelength - self.wavelength) ** 2) / (2 * sigma ** 2))
        return intensity


@dataclass
class FluorescenceMeasurement:
    # Excitation led
    led: LED

    # Emission wavelengths in nm
    emission_responses: np.ndarray = field(default_factory=lambda: np.zeros(288, dtype=np.float64))

    def normalize(self):
        max_response = np.max(self.emission_responses)
        if max_response > 0:
            self.emission_responses = self.emission_responses / max_response

    def remove_extranous_data(self):
        average_nm_per_pixel = 0.0

    def __post_init__(self):
        self.normalize()

    def plot(self):
        plt.plot(pixel_emission_wavelengths_array(), self.emission_responses)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity (a.u.)')
        plt.title('Emission Spectrum')
        plt.show()


class EEM:
    pixel_calibration: list[float]
    resampled_matrix: np.ndarray
    ife_matrix: np.ndarray
    grid_size: int

    def __init__(self, pixel_calibration=None, absorbance_data=None, grid_size=1024):
        if pixel_calibration is None:
            self.pixel_calibration = pixel_emission_wavelengths_array()

        self.grid_size = grid_size
        self.resampled_matrix: np.ndarray = np.zeros((grid_size, grid_size))
        self.ife_matrix: np.ndarray = np.zeros((grid_size, grid_size))

    def plot(self, graph_title='EEM Matrix', range_min_nm=0, range_max_nm=1000):
        colors = [(0, 0, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]  # Blue, Green, Yellow, Red
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)

        # Set the figure size to make columns wider
        fig, ax = plt.subplots(figsize=(20, 6))  # Adjust the size as needed
        cax = ax.matshow(self.resampled_matrix, cmap=cmap, interpolation='nearest', aspect='auto')

        plt.xlabel('Excitation Wavelength (nm)')
        plt.ylabel('Emission Wavelength (nm)')

        # Add a line through the diagonal with a thickness of 12 nm
        ax.plot([range_min_nm, range_max_nm], [range_min_nm, range_max_nm], color='black', linewidth=20)
        plt.title(graph_title)

    def get_normalized_matrix(self):
        return self.resampled_matrix / np.max(self.resampled_matrix)

    def add_flor_measurement(self, measurement: FluorescenceMeasurement):
        sigma_y = 12.0 / 2.355  # Standard deviation for the emission wavelength (12 nm FWHM which is typical from DS)
        sigma_x = measurement.led.fwhm / 2.355  # Standard deviation for the LED (excitation) based on its FWHM
        print("Adding fluorescence measurement")

        for x in range(self.grid_size):
            print(f"Processing row {x}")
            for y in range(self.grid_size):
                excitation_nm = x  # X-axis: Excitation wavelength
                emission_nm = y  # Y-axis: Emission wavelength

                for i in range(len(measurement.emission_responses)):
                    # Calculate Gaussian value
                    center_x = excitation_nm  # Example center on the excitation wavelength
                    center_y = int(self.pixel_calibration[i])  # Corresponding emission wavelength

                    # Apply Gaussian modulation based on the center wavelength and intensity
                    self.resampled_matrix[emission_nm][excitation_nm] += (
                            gaussian_2d(excitation_nm, emission_nm, center_x, center_y, sigma_x, sigma_y) *
                            measurement.led.peak_intensity * measurement.emission_responses[i])

    def add_flor_measurement_meshed(self, measurement):
        print("Adding fluorescence measurement... ", end='')

        sigma_y = 12.0 / 2.355  # Standard deviation for the emission wavelength (12 nm FWHM which is typical from DS)
        sigma_x = measurement.led.fwhm / 2.355  # Standard deviation for the LED (excitation) based on its FWHM
        x = np.arange(self.grid_size)
        X, Y = np.meshgrid(x, x, indexing='ij')

        for i, response in enumerate(measurement.emission_responses):
            center_x = measurement.led.wavelength
            center_y = int(self.pixel_calibration[i])
            peak_intensity = measurement.led.peak_intensity

            # Calculate the entire 2D Gaussian at once
            gaussian_distribution = gaussian_2d(Y, X, center_x, center_y, sigma_x, sigma_y)

            # Update the resampled matrix with the calculated Gaussian distribution scaled by the response
            self.resampled_matrix += gaussian_distribution * peak_intensity * response

        print("done.")

    def add_flor_measurement_meshed_interp(self, measurement):
        print("Adding fluorescence measurement... ", end='')
        sigma_x = measurement.led.fwhm / 2.355  # Standard deviation for the LED (excitation) based on its FWHM

        # Linearly interpolate the values across the relevant pixels
        toadd = np.zeros(self.grid_size)
        for w in range(round(self.pixel_calibration[0]), round(self.pixel_calibration[-1])):
            toadd[w] = np.interp(w, self.pixel_calibration, measurement.emission_responses)

        # Represent the Excitation LED FMWH as a guassian distribution across every col in the resampled matrix
        for w in range(self.grid_size):
            # Calculate the entire 2D Gaussian at once
            gaussian = (norm.pdf(np.arange(self.grid_size), loc=measurement.led.wavelength, scale=sigma_x)
                        * sigma_x * np.sqrt(2 * np.pi))

            # Update the resampled matrix with the calculated Gaussian distribution scaled by the response
            if min(self.pixel_calibration) <= w <= max(self.pixel_calibration):
                self.resampled_matrix[w] += (gaussian * measurement.led.peak_intensity *
                                             np.interp(w, self.pixel_calibration, measurement.emission_responses))

        print("done.")
