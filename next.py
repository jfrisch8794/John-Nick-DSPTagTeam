import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter

from eem_utils import pixel_emission_wavelengths_array, FluorescenceMeasurement, LED, pull_from_csv


def main():
    corrected = pixel_emission_wavelengths_array()
    # print first and last 5 values
    print(corrected[:5], corrected[-5:])

    eem_matrix_iso = np.array([])
    absorption_iso = pull_from_csv('data/91isopropyl/LED6.csv')

    eem_matrix_trypto = np.array([])
    absorption_trypto = pull_from_csv('data/ltryptophan/LED6.csv')

    for i in range(0, 6):
        emission_data_iso = pull_from_csv('data/91isopropyl/LED' + str(i) + '.csv')
        emission_data_trypto = pull_from_csv('data/ltryptophan/LED' + str(i) + '.csv')

        # Next append the 288 data points to the eem_matrix as a new column
        eem_matrix_iso = np.column_stack(
            (eem_matrix_iso, emission_data_iso)) if eem_matrix_iso.size else emission_data_iso
        eem_matrix_trypto = np.column_stack(
            (eem_matrix_trypto, emission_data_trypto)) if eem_matrix_trypto.size else emission_data_trypto

    eem_matrix_iso_corr = compute_absorbance_correction(corrected, eem_matrix_iso, absorption_iso)
    eem_matrix_trypto = compute_absorbance_correction(corrected, eem_matrix_trypto, absorption_trypto)

    plot_eem(corrected, eem_matrix_iso, "EEM Matrix for 91% Isopropyl")
    plot_eem(corrected, eem_matrix_trypto, "EEM Matrix for L-Tryptophan")

    plt.show()


if __name__ == "__main__":
    main()