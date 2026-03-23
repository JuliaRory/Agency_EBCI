import os
import sys
import json
import copy
from numpy import concatenate, arange, mean, where, isin, array, ones, zeros, any, unique, ndarray
from matplotlib.pyplot import pause, ion, show, subplots

from src.utils.parse_bci_iv_files import process_file_bci_comp
from src.utils.parse_resonance_files import process_file_resonance

from src.utils.events import slice_epochs, sliding_epochs
from src.utils.save_helpers import make_unique_filename

from src.analysis.preprocessing import bandpass_filter
from src.analysis.CSP import calculate_CSP, calculate_robust_cov, compute_csp, apply_csp
from src.analysis.spectral_analysis import get_fft
from src.analysis.features import csp_features, fft_feature, riemannian_features

from src.visualization.plot_csp_components import plot_10_csp_components

from scripts.step1_calculate_fbcsp import calculate_csp_in_bands

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
import numpy as np

def get_idxs(mode, idxs_rest, idxs_right, idxs_left):
    if mode == "left-right":
        idxs1 = idxs_left 
        idxs2 = idxs_right 
    elif mode == "right-rest":
        idxs1 = idxs_right
        idxs2 = idxs_rest
    elif mode == "left-rest":
        idxs1 = idxs_left
        idxs2 = idxs_rest
    return idxs1, idxs2


def get_epochs(eeg, Fs, idxs_1, idxs_2, edges_ms=250, start_shift=500, filtered=True, band=[8, 13]):
    eeg, _ = bandpass_filter(eeg, fs=Fs, low=1, high=40)

    filter_epoch = True
    n = edges_ms // (1000 // Fs)
    eeg_f = copy.copy(eeg)
    if not filter_epoch and filtered:
        eeg_f, sos = bandpass_filter(eeg, fs=Fs, low=band[0], high=band[1])
    epochs_1, epochs_2 = slice_epochs(eeg_f, idxs_1)[:, n+start_shift:-n, :], slice_epochs(eeg_f, idxs_2)[:, n+start_shift:-n, :]
    if filter_epoch and filtered:
        epochs_1 = np.array([bandpass_filter(ep, fs=Fs, low=band[0], high=band[1])[0] for ep in epochs_1])
        epochs_2 = np.array([bandpass_filter(ep, fs=Fs, low=band[0], high=band[1])[0] for ep in epochs_2])

    return epochs_1, epochs_2

def get_csp_filter(epochs_1, epochs_2, anatoly=False):
    if anatoly:
        projInverse, projForward, evals = calculate_CSP(epochs_1, epochs_2)
    else:
        projInverse, projForward, evals = compute_csp(epochs_1, epochs_2, robust=False)
    return projInverse, projForward, evals

def train_clssifier(epochs_1_csp, epochs_2_csp, freq=[10, 11, 12], features="csp", classifier=LDA()):

    if features == "csp":
        X1 = csp_features(epochs_1_csp)
        X2 = csp_features(epochs_2_csp)
        X = concatenate([X1, X2], axis=0)
    elif features == 'fft':
        X1 = fft_feature(epochs_1_csp, diff_ind_f=freq)
        X2 = fft_feature(epochs_2_csp, diff_ind_f=freq)
        X = concatenate([X1, X2], axis=0)
    elif features == "riemannian":
        n1, n2 = epochs_1_csp.shape[1], epochs_2_csp.shape[1]
        nmax =int(min(n1, n2))
        X = riemannian_features(concatenate([epochs_1_csp[:, :nmax, :], epochs_2_csp[:, :nmax, :]], axis=0))
        n = epochs_1_csp.shape[0]
        X1 = X[:n]
        X2 = X[n:]
        
    y = concatenate([ones(len(epochs_1_csp)), zeros(len(epochs_2_csp))])
      
    scores = cross_validate(classifier, X, y, cv=5, scoring=('accuracy', 'balanced_accuracy'), return_train_score=True)

    fig, ax = subplots(1, 1, figsize=(3, 3))
    ax.scatter(X1[:, 0], X1[:, -1], color='green')
    ax.scatter(X2[:, 0], X2[:, -1], color='red')
    ax.set_title(f"Test accuracy: {mean(scores["test_accuracy"]):.2f}")
    fig.show()
    pause(0.1)

    # Обучаем финальную модель на всех данных
    classifier.fit(X, y)
    # Веса LDA (для признаков, полученных из CSP)
    w_lda = classifier.coef_[0]  # вектор [компоненты] или [признаки]
    b_lda = classifier.intercept_[0]
    

# ---------- Предсказание ----------
# def predict_csp_riemannian(model, X):
#     W = model["W"]
#     Cref = model["Cref"]
#     clf = model["clf"]

#     X_csp = np.array([ep @ W for ep in X])
#     covs = np.array([compute_cov(ep) for ep in X_csp])
#     X_feat = tangent_space(covs, Cref)

#     return clf.predict(X_feat)

if __name__ == "__main__":
    data = "fb_q"
    start_shift = 500  # 500 лишних сэмплов в начале для визуализации бейзлайна

    # ==== BCI Comp IV ====
    if data == "bci_comp":
        data_folder = r"C:\Users\hodor\Documents\lab-MSU\диссер\Дупло белки\mu_clf\data\BCI Competition IV"
        records = os.listdir(data_folder)
        records = [record for record in os.listdir(data_folder) if record.find("calib") != -1]

        eeg, idxs_1, idxs_2, xy, Fs = process_file_bci_comp(os.path.join(data_folder, records[1]))
    
    # ==== Resonance Files ====
    else:
        # data_folder = r"R:\projects_FEEDBACK_QUASI\data\02 ES calibration session v2.0"
        # record = "02-OM.hdf"
        data_folder = r"./data/test/03_16 Artem"
        record = "05_calib.hdf"
        eeg, idxs_rest, idxs_right, idxs_left, xy, Fs = process_file_resonance(os.path.join(data_folder, record), start_shift=start_shift)    

    mode = "left-right" # 'right-rest' or 'left-rest'
    idxs_1, idxs_2 = get_idxs(mode, idxs_rest, idxs_right, idxs_left)

    # ==== universal part ==== 
    ion()
    
    choose = False      #<-------------------------- ввести вручную
    band = [8, 12]      #<-------------------------- ввести вручную

    # СДЕЛАЙ ЭТО ЧЕРЕЗ str = input("text") ПОПОЗЖЕ !!!!!!!!!!!!!!!!!!!!!!!
    if choose:
        output_filename = os.path.join("results", f"csp_{band}_final.png")
        # смотрим ещё раз что получилось
        calculate_csp_in_bands(eeg, Fs, idxs_1, idxs_2, xy, 
                           edges_ms=250, start_shift=start_shift, 
                           bands=[band], 
                           spectr=False, anatoly=False, 
                           folder_output=output_filename)

        show(block=True)

        sys.exit(0)
        
    sel_comp = [0, 55] #<-------------------------- ввести вручную
    window = 1000 # для обучения 
    filter_epoch = True

    epochs_1, epochs_2 = get_epochs(eeg, Fs, idxs_1, idxs_2, edges_ms=250, start_shift=start_shift,  filtered=True, band=band)

    projInverse, projForward, evals = get_csp_filter(epochs_1, epochs_2, anatoly=False)
    fig = plot_10_csp_components(abs(evals), projForward, xy)
    fig.suptitle(f"CSP: Freq Band {band}", fontsize=16)
    fig.show()
    pause(0.1)

    # epochs_1, epochs_2 = get_epochs(eeg, Fs, idxs_1, idxs_2, edges_ms=250, start_shift=start_shift,  filtered=False, band=band)

    epochs_1_csp = apply_csp(epochs_1, projInverse, sel_components=sel_comp)
    epochs_2_csp = apply_csp(epochs_2, projInverse, sel_components=sel_comp)

    # features = "riemannian"
    # classifier = LogisticRegression(max_iter=1000)

    features = "csp"
    classifier = LDA()
    train_clssifier(epochs_1_csp, epochs_2_csp, 
                    freq=[8, 9, 10, 11, 12, 13, 14, 15], 
                    features=features, classifier=classifier)
    
    #Сохраняем все необходимые параметры в JSON
    # output_filename=r"models/test_classifier.json"
    # if output_filename:
    #     # Создаем словарь со всеми параметрами
    #     classifier_data = {
    #         'spatialW': projInverse[:, sel_comp].tolist(),  # веса csp фильтра
    #         'sos': sos.tolist(),  # коэффициенты SOS фильтра [секции × 6]
    #         'features_type': features,  # тип признаков
    #         # 'fft_features': {"ch": arange(epochs_1_csp.shape), "freq":}
    #         # 'sel_comp': sel_comp,  # выбранные компоненты CSP

    #         # LDA веса
    #         'w_lda': w_lda.tolist(),  # веса LDA
    #         'b_lda': float(b_lda),    # смещение LDA
            
    #         'fs': Fs,  # частота дискретизации
    #         "n_components": len(sel_comp)
    #     }
        
    #     # Сохраняем в JSON файл
    #     with open(output_filename, 'w') as f:
    #         json.dump(classifier_data, f, indent=4)
        
    #     print(f"Классификатор сохранен в {output_filename}")

    show(block=True)
