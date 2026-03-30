import os
from matplotlib.pyplot import pause

from src.utils.parse_bci_iv_files import process_file_bci_comp
from src.utils.parse_resonance_files import process_file_resonance, get_idxs

from src.utils.events import slice_epochs
from src.utils.save_helpers import make_unique_filename
from src.utils.olivehawkins_robustcov import olivehawkins_robustcov

from src.analysis.preprocessing import bandpass_filter
from src.analysis.CSP import calculate_CSP, calculate_robust_cov, compute_csp


from src.visualization.plot_csp_components import plot_10_csp_components
from src.visualization.spectr import plot_spectr, plot_spectrograms, plot_log_ratio_psd

import numpy as np

def calculate_csp_in_bands(eeg, Fs, idxs_1, idxs_2, xy, 
                           edges_ms=250, start_shift=500, filter_epoch=True, 
                           bands=[[8, 10], [10, 12], [12, 14], [8, 12], [10, 14]], folder_output=r"./results/csp_components", anatoly=False, spectr=False, show_plot=False):
    # ==== universal ====
    os.makedirs(folder_output, exist_ok=True)
    eeg, _ = bandpass_filter(eeg, fs=Fs, low=1, high=40)
    log_var_ratio = True
    n = edges_ms // (1000 // Fs)
    # start_shift += 1000
    epochs_1, epochs_2 = slice_epochs(eeg, idxs_1)[:, n+start_shift:-n, :], slice_epochs(eeg, idxs_2)[:, n+start_shift:-n, :]
    if log_var_ratio:
        
        output_filename_spectr = os.path.join(folder_output, f"log_var_C3.png")
        idx_C3, idx_C4 = 9, 22
        plot_log_ratio_psd(epochs_2, epochs_1, sfreq=Fs, ch_idx=idx_C3, fmin=1, fmax=30, filename=output_filename_spectr)
        output_filename_spectr = os.path.join(folder_output, f"log_var_C4.png")
        plot_log_ratio_psd(epochs_1, epochs_2, sfreq=Fs, ch_idx=idx_C4, fmin=1, fmax=30, filename=output_filename_spectr)
    for band in bands:
        
        if True:
            if not filter_epoch:
                eeg, _ = bandpass_filter(eeg, fs=Fs, low=band[0], high=band[1])
            epochs_1, epochs_2 = slice_epochs(eeg, idxs_1)[:, n+start_shift:-n, :], slice_epochs(eeg, idxs_2)[:, n+start_shift:-n, :]
        
            if filter_epoch:
                epochs_1 = np.array([bandpass_filter(ep, fs=Fs, low=band[0], high=band[1])[0] for ep in epochs_1])
                epochs_2 = np.array([bandpass_filter(ep, fs=Fs, low=band[0], high=band[1])[0] for ep in epochs_2])

            if anatoly:
                projInverse, projForward, evals = calculate_CSP(epochs_1, epochs_2)
            else:
                projInverse, projForward, evals = compute_csp(epochs_1, epochs_2, config)
            
            fig = plot_10_csp_components(abs(evals), projForward, xy)
            fig.suptitle(f"CSP: Freq Band {band}", fontsize=16)

            output_filename = os.path.join(folder_output, f"COMPONENTS_{band}.png")
            output_filename = make_unique_filename(output_filename)
            fig.savefig(output_filename, dpi=300, bbox_inches="tight")
            if show_plot:
                fig.show()
                pause(0.1)
            
            if spectr:
                eeg_csp = eeg @ projInverse
                epochs_1, epochs_2 = slice_epochs(eeg_csp, idxs_1), slice_epochs(eeg_csp, idxs_2)
                fig_spectr = plot_spectrograms(epochs_1, epochs_2, Fs, title=f"Spectrogrammm: Freq Band {band}", baseline=(0, 300), start_shift=start_shift)

                output_filename_spectr = os.path.join(folder_output, f"SPECTROGRAMM_{band}.png")
                output_filename_spectr = make_unique_filename(output_filename_spectr)
                fig_spectr.savefig(output_filename_spectr, dpi=300, bbox_inches="tight")
                if show_plot:
                    fig_spectr.show()
                    pause(0.1)
              
            print(f"Band {band} is processed.")
        # except Exception as e:
        #     # Print the specific error message
        #     print(f"An error occurred: {e}") 

config = {
    "Fs": 1000, 
    "do_baseline": True,
    "baseline_shift": 500, 
    "edge": 250,
    "mode": "left-right",
    "epoch_len": 1000,
    "window_step": 1000, 
    "epoch_filter": True, 
    "bands": [[9, 13], [11, 13],[8, 12], [10, 14],  [16, 22]],
    "olivehawkins": False,
    "mean_cov": True,
    "shrinkage": True,
    "shrinkage_alpha": 0.01,
    "sel_comp": [1, 2, 54, 55]
}

if __name__ == "__main__":
    data = "fb_q"
    start_shift = 500  # 500 лишних сэмплов в начале для визуализации бейзлайна
    
    n = 250
    mode = 'left-right' #"left-right"  or 'left-rest'   
    # ==== Resonance Files ====

    all_epochs = {tuple(band): [] for band in config["bands"]}  # хранение эпох по диапазонам
    all_labels = {tuple(band): [] for band in config["bands"]}

    data_folder = r"./data/test/03_23 Artem"
    records = ["01_calib.hdf", "02_game.hdf"] # 4 sec 
    for record in records:
        eeg, idxs_rest, idxs_right, idxs_left, xy, Fs = process_file_resonance(os.path.join(data_folder, record), start_shift=start_shift)      
        idxs1, idxs2 = idxs_1, idxs_2 = get_idxs(mode, idxs_rest, idxs_right, idxs_left)
        eeg, _ = bandpass_filter(eeg, fs=Fs, low=1, high=40)
        
        for band in config["bands"]:
            eeg_band, _ = bandpass_filter(eeg, fs=Fs, low=band[0], high=band[1])

            if not config["epoch_filter"]:
                eeg, _ = bandpass_filter(eeg, fs=Fs, low=band[0], high=band[1])
            epochs_1, epochs_2 = slice_epochs(eeg, idxs_1)[:, n+start_shift:-n, :], slice_epochs(eeg, idxs_2)[:, n+start_shift:-n, :]
        
            if config["epoch_filter"]:
                epochs_1 = np.array([bandpass_filter(ep, fs=Fs, low=band[0], high=band[1])[0] for ep in epochs_1])
                epochs_2 = np.array([bandpass_filter(ep, fs=Fs, low=band[0], high=band[1])[0] for ep in epochs_2])

            # Собираем все эпохи в один список
            min_len = min(epochs_1.shape[1], epochs_2.shape[1])
            epochs_1 = epochs_1[:, :min_len, :]
            epochs_2 = epochs_2[:, :min_len, :]
            all_epochs[tuple(band)].append(np.concatenate([epochs_1, epochs_2], axis=0))
            all_labels[tuple(band)].append(np.concatenate([np.ones(len(epochs_1)), np.zeros(len(epochs_2))]))
    
    folder_output=os.path.join(r"./results/csp_components/03_23 Artem/30 03 tests/norobust", records[0][:-4] + "_" + records[1][:-4], mode)
    os.makedirs(folder_output, exist_ok=True)

    for band in config["bands"]:
        # список массивов epochs для одного диапазона
        epochs_list = all_epochs[tuple(band)]

        # находим минимальную длину по оси времени (axis=1)
        min_len = min(ep.shape[1] for ep in epochs_list)

        # усекаем все эпохи до min_len
        epochs_list_equal = [ep[:, :min_len, :] for ep in epochs_list]

        # теперь можно безопасно объединять по axis=0
        X_epochs = np.concatenate(epochs_list_equal, axis=0)
        # X_epochs = np.concatenate(all_epochs[tuple(band)], axis=0)
        y = np.concatenate(all_labels[tuple(band)], axis=0)
        
        # Вычисляем CSP
        projInverse, projForward, evals = compute_csp(X_epochs[y==1], X_epochs[y==0], config)
        
        # Визуализация
        fig = plot_10_csp_components(abs(evals), projForward, xy)
        fig.suptitle(f"CSP: Freq Band {band}", fontsize=16)
        output_filename = os.path.join(folder_output, f"COMPONENTS_{band}.png")
        fig.savefig(output_filename, dpi=300, bbox_inches="tight")