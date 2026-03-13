import os
from matplotlib.pyplot import pause

from src.utils.parse_bci_iv_files import process_file_bci_comp
from src.utils.parse_resonance_files import process_file_resonance

from src.utils.events import slice_epochs
from src.utils.save_helpers import make_unique_filename

from src.analysis.preprocessing import bandpass_filter
from src.analysis.CSP import calculate_CSP, calculate_robust_cov, compute_csp


from src.visualization.plot_csp_components import plot_10_csp_components
from src.visualization.spectr import plot_spectr, plot_spectrograms


def calculate_csp_in_bands(eeg, Fs, idxs_1, idxs_2, xy, edges_ms=250, bands=[[8, 10], [10, 12], [12, 14], [8, 12], [10, 14]], folder_output=r"./results/csp_components", anatoly=False, spectr=False, show_plot=False):
    # ==== universal ====
    os.makedirs(folder_output, exist_ok=True)

    for band in bands:

        eeg_f, sos = bandpass_filter(eeg, fs=Fs, low=band[0], high=band[1])
        epochs_1, epochs_2 = slice_epochs(eeg_f, idxs_1), slice_epochs(eeg_f, idxs_2)
        
        n = edges_ms // (1000 // Fs)
        start_shift = 500

        if anatoly:
            cov1 = calculate_robust_cov(epochs_1[:, n+start_shift:-n, :]).covariance_
            cov2 = calculate_robust_cov(epochs_2[:, n+start_shift:-n, :]).covariance_

            projInverse, projForward, evals = calculate_CSP(cov1, cov2)
        else:
            projInverse, projForward, evals = compute_csp(epochs_1, epochs_2)
        
        fig = plot_10_csp_components(abs(evals), projForward, xy)
        fig.suptitle(f"CSP: Freq Band {band}", fontsize=16)

        output_filename = os.path.join(folder_output, f"COMPONENTS_{band}.png")
        output_filename = make_unique_filename(output_filename)
        fig.savefig(output_filename, dpi=300, bbox_inches="tight")
        if show_plot:
            fig.show()
            pause(0.1)
        
        if spectr:
            eeg_csp = eeg_f @ projInverse
            epochs_1, epochs_2 = slice_epochs(eeg_csp, idxs_1), slice_epochs(eeg_csp, idxs_2)
            fig_spectr = plot_spectrograms(epochs_1, epochs_2, Fs, title=f"Spectrogrammm: Freq Band {band}", baseline=(0, 300))

            output_filename_spectr = os.path.join(folder_output, f"SPECTROGRAMM_{band}.png")
            output_filename_spectr = make_unique_filename(output_filename_spectr)
            fig_spectr.savefig(output_filename_spectr, dpi=300, bbox_inches="tight")
            if show_plot:
                fig_spectr.show()
                pause(0.1)
        
        print(f"Band {band} is processed.")

if __name__ == "__main__":
    data = "fb_q"
    # ==== BCI Comp IV ====
    if data == "bci_comp":
        data_folder = r"C:\Users\hodor\Documents\lab-MSU\диссер\Дупло белки\mu_clf\data\BCI Competition IV"
        records = os.listdir(data_folder)
        records = [record for record in os.listdir(data_folder) if record.find("calib") != -1]

        eeg, idxs_1, idxs_2, xy, Fs = process_file_bci_comp(os.path.join(data_folder, records[1]))
    
    # ==== Resonance Files ====
    else:
        data_folder = r".\data\test\03_06 Artem"
        record = "01_222sec_112.hdf"
        # eeg, idxs_1, idxs_2, xy, Fs = process_file_resonance(os.path.join(data_folder, record))
        eeg, idxs_rest, idxs_right, idxs_left, xy, Fs = process_file_resonance(os.path.join(data_folder, record))       # 500 лишних сэмплов в начале

    mode = "right-rest" # 'right-rest' or 'left-rest'ff
    if mode == "left-right":
        idxs1 = idxs_left 
        idxs2 = idxs_right 
    elif mode == "right-rest":
        idxs1 = idxs_right
        idxs2 = idxs_rest
    elif mode == "left-rest":
        idxs1 = idxs_left
        idxs2 = idxs_rest

    # ==== universal part ==== 
    calculate_csp_in_bands(eeg, Fs, idxs1, idxs2, xy, edges_ms=250, bands=[[8, 10], [10, 12], [12, 14], [8, 12], [10, 14]], spectr=True,
                           folder_output=os.path.join(r"./results/csp_components/03_03 Artem", record[:-4], mode))
