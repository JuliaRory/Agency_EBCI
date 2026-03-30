import os
from pathlib import Path
from numpy import mean, diff, array

from src.utils.parse_resonance_files import process_file_resonance, get_idxs
from src.utils.events import slice_epochs, get_sliding_epochs
from src.utils.save_helpers import make_unique_filename

from src.analysis.preprocessing import bandpass_filter, subtract_baseline
from src.analysis.CSP import compute_csp
from src.visualization.plot_csp_components import plot_10_csp_components
from src.visualization.spectr import plot_psd, plot_psd_diff
from src.visualization.ERD import plot_erd
from src.utils.montage_processing import find_ch_idx


"""
профильтрую сигнал 5-30 Гц
0) выделю эпохи по 4 сек
1) сделаю вычитание бейзлайна для эпохи
2) фильтрование эпохи в 8-12 ГЦ
3) потом среднюю ковариацию по эпохам
4) сами csp паттерны

затем применю пространственную фильтрацию получившимся фильтром на 0, 1 и -2, -1 компонентах. 

потом разделю все эпохи на окна по 1000 мс
выделю признаки log-var
обучу LDA

при применении в онлайне буду делать
частотную фильтрацию сигнала 5-30 Гц
применение пространственного фильтра получившимся фильтром на 0, 1 и -2, -1 компонентах. 
разделю сигнал на пересекающиеся окно 1000 мс с шагом 100 мс
для каждого буду считать log-var
применять LDA
"""

data_folder = r"./data/test/03_23 Artem"
records = os.listdir(data_folder) #["04_calib.hdf"] 
records = ["01_calib.hdf", "04_calib.hdf"] 

fl_montage = r"./resources/mks64_standard.ced"
idx_C4 = find_ch_idx("C4", fl_montage)
idx_C3 = find_ch_idx("C3", fl_montage)

config = {
    "Fs": 1000, 
    "do_baseline": True,
    "baseline_shift": 500, 
    "edge": 250,
    "mode": "right-rest", #"left-right",
    "epoch_len": 1000,
    "epoch_filter": True, 
    # "bands": [[8, 10], [8, 12], [10, 12], [10, 14]],
    "bands": [[8, 14], [9, 13]],
    "olivehawkins": False,
    "mean_cov": True,
    "shrinkage": True,
    "shrinkage_alpha": 0.01
}

def get_output_path(record_info):
    parts = Path(record_info).parts
    output_folder = os.path.join(r"results", parts[-2], parts[-1][:-4])
    os.makedirs(output_folder, exist_ok=True)
    return output_folder

def plot_10_comp(evals, projForward, xy, band, record_info, mode, config):
    fig = plot_10_csp_components(abs(evals), projForward, xy)

    filter_epoch = "in epochs" if config["epoch_filter"] else "in signal"
    robust = "robust" if config["olivehawkins"] else "norobust"
    shrinkage = "shrinkage"+str(config["shrinkage_alpha"]) if config["shrinkage"] else ""
    concat = "epochs_concat" if not config["mean_cov"] else "epochs_mean"
    fig.suptitle(f"CSP: {band} Hz {filter_epoch}, {robust}, {shrinkage}, {concat}", fontsize=16)

    output_folder = get_output_path(record_info)
    
    output_folder = os.path.join(output_folder, "CSP_components", mode)
    os.makedirs(output_folder, exist_ok=True)
    output_filename = os.path.join(output_folder, f"COMPONENTS_{band}.png")
    output_filename = make_unique_filename(output_filename)
    print(output_filename)
    fig.savefig(output_filename, dpi=300, bbox_inches="tight")
    
def process_band(eeg, idxs_1, idxs_2, band, config):
    Fs = config["Fs"]
    n = config["edge"]
    start_shift=config["baseline_shift"]
    
    if not config["epoch_filter"]:
        eeg, _ = bandpass_filter(eeg, fs=Fs, low=band[0], high=band[1])    
    
    epochs_1 = slice_epochs(eeg, idxs_1)
    epochs_2 = slice_epochs(eeg, idxs_2)

    if config["do_baseline"]:
        epochs_1 = subtract_baseline(epochs_1, baseline_samples=(0, config["baseline_shift"]))
        epochs_2 = subtract_baseline(epochs_2, baseline_samples=(0, config["baseline_shift"]))

    if config["epoch_filter"]:
        epochs_1 = array([bandpass_filter(ep, fs=Fs, low=band[0], high=band[1])[0] for ep in epochs_1])
        epochs_2 = array([bandpass_filter(ep, fs=Fs, low=band[0], high=band[1])[0] for ep in epochs_2])

    mask = lambda x: x[:, n+start_shift:-n, :]
    epochs_1 = mask(epochs_1)
    epochs_2 = mask(epochs_1)

    projInverse, projForward, evals = compute_csp(epochs_1, epochs_2, config)

    return projInverse, projForward, evals

def plot_spectr(eeg, idxs_1, idxs_2, Fs, mode, full_path):
    
    epochs_1 = slice_epochs(eeg, idxs_1)
    epochs_2 = slice_epochs(eeg, idxs_2)
    mode1 = mode[:mode.find("-")]
    mode2 = mode[mode.find("-")+1:]
    print(f"{mode1}: {epochs_1.shape}, {mode2}: {epochs_2.shape}")

    fig = plot_psd_diff(epochs_1, epochs_2, Fs, ch_names=None, picks=[idx_C3, idx_C4])
    fig.suptitle(f"PSD Difference: {mode}", fontsize=16)
    output_folder = get_output_path(full_path)
    output_filename = os.path.join(output_folder, f"PSD_diff_{mode}.png")
    fig.savefig(output_filename, dpi=300, bbox_inches="tight") 

    fig = plot_psd(epochs_1, epochs_2, Fs, ch_names=None, picks=[idx_C3, idx_C4])
    fig.suptitle(f"PSD: {mode}", fontsize=16)
    output_folder = get_output_path(full_path)
    output_filename = os.path.join(output_folder, f"PSD_{mode}.png")
    fig.savefig(output_filename, dpi=300, bbox_inches="tight") 

def process_record(full_path, config):
    mode=config["mode"]
    start_shift=config["baseline_shift"]
    epoch_len=config["epoch_len"]

    mode1 = mode[:mode.find("-")]
    mode2 = mode[mode.find("-")+1:]
    eeg, idxs_rest, idxs_right, idxs_left, xy, Fs = process_file_resonance(full_path, start_shift=start_shift)  
    eeg, _ = bandpass_filter(eeg, fs=Fs, low=1, high=40)    # basic bandpass filtering    

    idxs_1, idxs_2 = get_idxs(mode, idxs_rest, idxs_right, idxs_left)
    print(f"Epoch {mode1}: n={len(idxs_1)}, dur={round(mean(diff(idxs_1)))}. \nEpoch {mode2}: n={len(idxs_2)}, dur={round(mean(diff(idxs_2)))}.")
    
    plot_spectr(eeg, idxs_1, idxs_2, Fs, mode, full_path)

    for band in config["bands"]:
        projInverse, projForward, evals = process_band(eeg, idxs_1, idxs_2, band, config)

        plot_10_comp(evals, projForward, xy, band, full_path, mode, config)
        

if __name__ == "__main__":

    for record in records:
        print(f"Record {record}")
        process_record(full_path=os.path.join(data_folder, record), config=config)
        
