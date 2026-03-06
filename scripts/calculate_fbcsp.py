import os
from matplotlib.pyplot import pause

from src.utils.parse_bci_iv_files import process_file
from src.utils.events import slice_epochs
from src.utils.save_helpers import make_unique_filename

from src.analysis.preprocessing import bandpass_filter
from src.analysis.CSP import calculate_CSP, calculate_robust_cov, compute_csp


from src.visualization.plot_csp_components import plot_10_csp_components
from src.visualization.spectr import plot_spectr


def calculate_csp_in_bands(eeg, Fs, idxs_1, idxs_2, xy, edges_ms=250, bands=[[8, 10], [10, 12], [12, 14], [8, 12], [10, 14]], output_filename=None, anatoly=False, spectr=False, show_plot=False):
    # ==== universal ====
    for band in bands:
        eeg_f = bandpass_filter(eeg, fs=Fs, low=band[0], high=band[1])
        epochs_1, epochs_2 = slice_epochs(eeg_f, idxs_1), slice_epochs(eeg_f, idxs_2)
        
        n = edges_ms // (1000 // Fs)

        if anatoly:
            cov1 = calculate_robust_cov(epochs_1[:, n:-n, :]).covariance_
            cov2 = calculate_robust_cov(epochs_2[:, n:-n, :]).covariance_

            projInverse, projForward, evals = calculate_CSP(cov1, cov2)
        else:
            projInverse, projForward, evals = compute_csp(epochs_1, epochs_2)
        
        fig = plot_10_csp_components(abs(evals), projForward, xy)
        fig.suptitle(f"CSP: Freq Band {band}", fontsize=16)

        if output_filename is None:
            output_filename = os.path.join("results", f"simple_cov_fbcsp_{band}.png")
        output_filename = make_unique_filename(output_filename)
        fig.savefig(output_filename, dpi=300, bbox_inches="tight")
        if show_plot:
            fig.show()
            pause(0.1)
        
        if spectr:
            eeg_csp = eeg @ projInverse
            epochs_1, epochs_2 = slice_epochs(eeg_csp, idxs_1), slice_epochs(eeg_csp, idxs_2)
            fig_spectr = plot_spectr(epochs_1, epochs_2, Fs)
            output_filename_spectr = os.path.join("results", f"simple_cov_spectr_{band}.png")
            output_filename = make_unique_filename(output_filename_spectr)
            
            if show_plot:
                fig_spectr.show()
                pause(0.1)
    

if __name__ == "__main__":
    # ==== BCI Comp IV ====
    data_folder = r"C:\Users\hodor\Documents\lab-MSU\диссер\Дупло белки\mu_clf\data\BCI Competition IV"
    records = os.listdir(data_folder)
    records = [record for record in os.listdir(data_folder) if record.find("calib") != -1]

    eeg, idxs_1, idxs_2, xy, Fs = process_file(os.path.join(data_folder, records[0]))


    # ==== Resonance Files ====


    # ==== universal part ==== 
    calculate_csp_in_bands(eeg, Fs, idxs_1, idxs_2, xy, edges_ms=1000, bands=[[8, 10], [10, 12], [12, 14], [8, 12], [10, 14]], spectr=True)
