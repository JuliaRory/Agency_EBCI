import os
import sys
from numpy import concatenate, arange, mean, where, isin, array, ones, zeros, any
from matplotlib.pyplot import pause, ion, show, subplots

from src.utils.parse_bci_iv_files import process_file
from src.utils.events import slice_epochs
from src.utils.save_helpers import make_unique_filename

from src.analysis.preprocessing import bandpass_filter
from src.analysis.CSP import calculate_CSP, calculate_robust_cov, compute_csp, apply_csp
from src.analysis.spectral_analysis import get_fft
from src.analysis.features import csp_features, fft_feature

from src.visualization.plot_csp_components import plot_10_csp_components

from scripts.calculate_fbcsp import calculate_csp_in_bands

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_validate


def get_features(feature, window_size=100, step=10):
    block = array([feature[:, i:i+window_size] for i in range(0, feature.shape[1]-window_size+1, step)])
    return mean(block, axis=0).T


def train_clssifier(eeg, Fs, idxs_1, idxs_2, edges_ms=250, band=[8, 13], sel_comp=[0, 1], freq=[10, 11, 12], anatoly=False, features="csp", output_filename=None):
    # ==== universal ====

    eeg_f = bandpass_filter(eeg, fs=Fs, low=band[0], high=band[1])
    epochs_1, epochs_2 = slice_epochs(eeg_f, idxs_1), slice_epochs(eeg_f, idxs_2)
    
    if anatoly:
        n = edges_ms // (1000 // Fs)
        
        cov1 = calculate_robust_cov(epochs_1[:, n:-n, :]).covariance_
        cov2 = calculate_robust_cov(epochs_2[:, n:-n, :]).covariance_

        projInverse, projForward, evals = calculate_CSP(cov1, cov2)
    else:
        projInverse, projForward, evals = compute_csp(epochs_1, epochs_2)

    epochs_1_csp = apply_csp(epochs_1, projInverse, sel_components=sel_comp)
    epochs_2_csp = apply_csp(epochs_2, projInverse, sel_components=sel_comp)

    # features
    if features == "csp":
        X1 = csp_features(epochs_1_csp)
        X2 = csp_features(epochs_2_csp)
    elif features == 'fft':
        X1 = fft_feature(epochs_1_csp)
        X2 = fft_feature(epochs_2_csp)
    
    fig, ax = subplots(1, 1, figsize=(3, 3))
    ax.scatter(X1[:, 0], X1[:, 1], color='green')
    ax.scatter(X2[:, 0], X2[:, 1], color='red')
    fig.show()
    pause(0.1)
    
    X = concatenate([X1, X2], axis=0)
    y = concatenate([ones(len(X1)), zeros(len(X2))])
      
    lda = LDA()
    scores = cross_validate(lda, X, y, cv=5, scoring=('accuracy', 'balanced_accuracy'), return_train_score=True)
    print("Test accuracy:", round(mean(scores["test_accuracy"]), 2))


if __name__ == "__main__":
    # ==== BCI Comp IV ====
    data_folder = r"C:\Users\hodor\Documents\lab-MSU\диссер\Дупло белки\mu_clf\data\BCI Competition IV"
    records = os.listdir(data_folder)
    records = [record for record in os.listdir(data_folder) if record.find("calib") != -1]

    eeg, idxs_1, idxs_2, xy, Fs = process_file(os.path.join(data_folder, records[1]))
    
    # ==== Resonance Files ====


    # ==== universal part ==== 
    ion()
    
    choose = False       #<-------------------------- ввести вручную
    band = [8, 15]      #<-------------------------- ввести вручную

    # СДЕЛАЙ ЭТО ЧЕРЕЗ str = input("text") ПОПОЗЖЕ !!!!!!!!!!!!!!!!!!!!!!!
    if choose:
        output_filename = os.path.join("results", f"csp_{band}_final.png")
        # смотрим ещё раз что получилось
        calculate_csp_in_bands(eeg, Fs, idxs_1, idxs_2, xy, edges_ms=1000, bands=[band], 
                                spectr=True, show_plot=True, anatoly=False, 
                                output_filename=output_filename)

        show(block=True)

        sys.exit(0)
        
    sel_comp = [-1, -2] #<-------------------------- ввести вручную
    train_clssifier(eeg, Fs, idxs_1, idxs_2, edges_ms=250, band=band, 
                    sel_comp=sel_comp, freq=[10, 11, 12, 21, 22, 23], anatoly=False,
                    features="csp")
    show(block=True)
    
    

