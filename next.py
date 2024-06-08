import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter

from eem_utils import pixel_emission_wavelengths_array, FluorescenceMeasurement, LED, pull_from_csv, EEM


def main():
    eem_iso = EEM()
    eem_trypto = EEM()

    absorption_iso = pull_from_csv('data/91isopropyl/LED6.csv')
    absorption_trypto = pull_from_csv('data/ltryptophan/LED6.csv')

    excitation_led_nm = [275, 365, 460, 525, 590, 635]
    flor_ltrp = None
    flor_iso = None

    for i in range(0, 6):
        flor_ltrp = FluorescenceMeasurement(LED(wavelength=excitation_led_nm[i]),
                                            emission_responses=pull_from_csv(
                                                f'data/ltryptophan/LED{str(i)}.csv'))
        flor_iso = FluorescenceMeasurement(LED(wavelength=excitation_led_nm[i]),
                                           emission_responses=pull_from_csv(
                                               f'data/91isopropyl/LED{str(i)}.csv'))

        eem_iso.add_flor_measurement_meshed_interp(flor_iso)
        eem_trypto.add_flor_measurement_meshed_interp(flor_ltrp)

    # TODO REMOVE normalization until the end
    # Subtract the background (Isopropanol) from the sample (L-Tryptophan)
    eem_trypto.resampled_matrix = eem_trypto.resampled_matrix - eem_iso.resampled_matrix
    # If the value is negative, set it to 0
    eem_trypto.resampled_matrix[eem_trypto.resampled_matrix < 0] = 0
    # Normalize the matrix to the maximum value
    eem_trypto.resampled_matrix = eem_trypto.get_normalized_matrix()

    eem_iso.plot("EEM Matrix for 91% Isopropyl")
    eem_trypto.plot("EEM Matrix for L-Tryptophan")

    plt.show()


if __name__ == "__main__":
    main()
