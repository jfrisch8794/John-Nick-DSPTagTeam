import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def pull_from_csv(file_path):
    # Load the data from the csv file and remove the last element since it is not captured
    data = np.genfromtxt(file_path, delimiter=',', dtype=float, usecols=[0], skip_header=1)[:-1]
    return data


def plot_emission(em, spectra):
    plt.plot(spectra, em)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity (a.u.)')
    plt.title('Emission Spectrum')
    plt.show()


def plot_eem(emission_matrix):
    colors = [(0, 0, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]  # Blue, Green, Yellow, Red
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)

    # Set the figure size to make columns wider
    fig, ax = plt.subplots(figsize=(20, 6))  # Adjust the size as needed
    cax = ax.imshow(emission_matrix, cmap=cmap, interpolation='nearest', aspect='auto')

    # Set custom x and y axis values
    response_spectra_nm = np.linspace(340, 850, 288)
    excitation_led_nm = [385, 450, 495, 540, 570]

    ax.set_xticks(np.arange(len(excitation_led_nm)))
    ax.set_xticklabels(excitation_led_nm)

    ax.set_yticks(np.linspace(0, len(response_spectra_nm) - 1, num=10).astype(int))
    ax.set_yticklabels(np.round(np.linspace(response_spectra_nm[0], response_spectra_nm[-1], num=10), 1))

    plt.xlabel('Excitation Wavelength (nm)')
    plt.ylabel('Emission Wavelength (nm)')
    plt.title('EEM Matrix')

    plt.show()


if __name__ == "__main__":
    eem_matrix = np.array([])

    for i in range(1, 6):
        emission_data = pull_from_csv('data/LED_' + str(i) + '_50.csv')
        # Next append the 288 data points to the eem_matrix as a new column
        eem_matrix = np.column_stack((eem_matrix, emission_data)) if eem_matrix.size else emission_data

    plot_eem(eem_matrix)
