import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter

from eem_utils import pixel_emission_wavelengths_array, FluorescenceMeasurement, LED, pull_from_csv


def generate_eem(fluorescence_collection: list[FluorescenceMeasurement], absorption_spectra: FluorescenceMeasurement):
    eem = np.full((1024, 1024), -1.0, dtype=float)

    for i, led in enumerate(fluorescence_collection):
        for j, em in enumerate(led.emission_responses):
            for k, ex in enumerate(pixel_emission_wavelengths_array()):
                eem[j, k] = ex * em * absorption_spectra[i]


def plot_emission(em, spectra):
    plt.plot(spectra, em)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity (a.u.)')
    plt.title('Emission Spectrum')
    plt.show()


def compute_absorbance_correction(pixel_to_wavelength, eem_matrix, absorbance_result):
    # iterate through each column of the eem_matrix
    corrected_eem_matrix = np.array([])
    for i in range(0, eem_matrix.shape[1]):
        corrected_eem_matrix = np.column_stack(
            (corrected_eem_matrix, eem_matrix[:, i] - absorbance_result)) if corrected_eem_matrix.size else eem_matrix[
                                                                                                            :, i]

    return eem_matrix


def plot_eem(pixel_to_wavelength, emission_matrix, graph_title='EEM Matrix'):
    colors = [(0, 0, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]  # Blue, Green, Yellow, Red
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)

    # Set the figure size to make columns wider
    fig, ax = plt.subplots(figsize=(20, 6))  # Adjust the size as needed
    cax = ax.imshow(emission_matrix, cmap=cmap, interpolation='nearest', aspect='auto')

    # Set custom x-axis values
    excitation_led_nm = [275, 365, 460, 525, 590, 635]
    ax.set_xticks(np.arange(len(excitation_led_nm)))
    ax.set_xticklabels(excitation_led_nm)

    def update_y_labels(ax):
        def dynamic_format(y, pos):
            y_int = int(y)
            if 0 <= y_int < len(pixel_to_wavelength):
                return f'{pixel_to_wavelength[y_int]:.1f}'
            else:
                return ''

        formatter = FuncFormatter(dynamic_format)
        ax.yaxis.set_major_formatter(formatter)
        ax.figure.canvas.draw_idle()

    # Initial setting of y-axis labels
    update_y_labels(ax)

    # Connect the event to dynamically update y-axis labels on zoom/pan
    def on_xlim_change(event_ax):
        update_y_labels(event_ax)

    ax.callbacks.connect('ylim_changed', on_xlim_change)

    plt.xlabel('Excitation Wavelength (nm)')
    plt.ylabel('Emission Wavelength (nm)')
    plt.title(graph_title)


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
