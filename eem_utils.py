import numpy as np
from dataclasses import dataclass, field


def pull_from_csv(file_path):
    # Load the data from the csv file and remove the last element since it is not captured
    data = np.genfromtxt(file_path, delimiter=',', dtype=float, usecols=[0], skip_header=1)[:-1]
    return data


def pixel_emission_wavelengths_array():
    # Create a list to hold the 288 data points
    emission_wavelengths = []
    for p in range(1, 289):
        emission_wavelengths.append(
            312.444 + (2.391 * p) + (-6.0e-04 * (p ** 2)) + (-8.0e-06 * (p ** 3)) + (-1.5e-08 * (p ** 4)) +
            (-8.5e-12 * (p ** 5)))

    return emission_wavelengths


@dataclass
class LED:
    # Excitation wavelength in nm
    wavelength: float = 272.5
    bandwidth: float = 15
    intensity: float = 1.0


@dataclass
class FluorescenceMeasurement:
    # Excitation wavelength in nm
    led: LED

    # Emission wavelength in nm
    emission_responses: np.ndarray = field(default_factory=lambda: np.zeros(288, dtype=np.float64))

    def normalize(self):
        self.emission_responses = self.emission_responses / np.max(self.emission_responses)


@dataclass
class EEM:
    def __init__(self, pixel_calibration=None, grid_size=1024):
        if pixel_calibration is None:
            self.pixel_calibration = pixel_emission_wavelengths_array()

        self.resampled_matrix: np.ndarray = np.zeros((grid_size, grid_size))

    def add_flor_measurement(self, measurement: FluorescenceMeasurement):
        # Compute the response of the fluorophore to the LED
        for i, pixel_value in enumerate(measurement.emission_responses):
            self.resampled_matrix[i] += pixel_value
