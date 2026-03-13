import os
import sys
import json
from numpy import concatenate, arange, mean, where, isin, array, ones, zeros, any, unique, ndarray
from matplotlib.pyplot import pause, ion, show, subplots

from src.utils.parse_bci_iv_files import process_file_bci_comp
from src.utils.parse_resonance_files import process_file_resonance

from src.utils.events import slice_epochs
from src.utils.save_helpers import make_unique_filename

from src.analysis.preprocessing import bandpass_filter
from src.analysis.CSP import calculate_CSP, calculate_robust_cov, compute_csp, apply_csp
from src.analysis.spectral_analysis import get_fft
from src.analysis.features import csp_features, fft_feature

from src.visualization.plot_csp_components import plot_10_csp_components

from scripts.step1_calculate_fbcsp import calculate_csp_in_bands

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_validate


def get_features(feature, window_size=100, step=10):
    block = array([feature[:, i:i+window_size] for i in range(0, feature.shape[1]-window_size+1, step)])
    return mean(block, axis=0).T


def train_clssifier(eeg, Fs, idxs_1, idxs_2, edges_ms=250, band=[8, 13], sel_comp=[0, 1], freq=[10, 11, 12], anatoly=False, features="csp", output_filename=None):
    # ==== universal ====

    n = edges_ms // (1000 // Fs)
    start_shift = 500   # потому что индексы бралис с запасом 500 сэмплов на бейзлайн

    eeg_f, sos = bandpass_filter(eeg, fs=Fs, low=band[0], high=band[1])
    epochs_1, epochs_2 = slice_epochs(eeg_f, idxs_1)[:, n+start_shift:-n, :], slice_epochs(eeg_f, idxs_2)[:, n+start_shift:-n, :]
    
    if anatoly:
        cov1 = calculate_robust_cov(epochs_1).covariance_
        cov2 = calculate_robust_cov(epochs_2).covariance_

        projInverse, projForward, evals = calculate_CSP(cov1, cov2)
    else:
        projInverse, projForward, evals = compute_csp(epochs_1[:, n+start_shift:-n, :], epochs_2[:, n+start_shift:-n, :])

    epochs_1_csp = apply_csp(epochs_1, projInverse, sel_components=sel_comp)
    epochs_2_csp = apply_csp(epochs_2, projInverse, sel_components=sel_comp)

    # features
    if features == "csp":
        X1 = csp_features(epochs_1_csp)
        X2 = csp_features(epochs_2_csp)
    elif features == 'fft':
        X1 = fft_feature(epochs_1_csp)
        X2 = fft_feature(epochs_2_csp)
    
    X = concatenate([X1, X2], axis=0)
    y = concatenate([ones(len(X1)), zeros(len(X2))])
      
    lda = LDA()
    scores = cross_validate(lda, X, y, cv=5, scoring=('accuracy', 'balanced_accuracy'), return_train_score=True)

    fig, ax = subplots(1, 1, figsize=(3, 3))
    ax.scatter(X1[:, 0], X1[:, 1], color='green')
    ax.scatter(X2[:, 0], X2[:, 1], color='red')
    ax.set_title(f"Test accuracy: {mean(scores["test_accuracy"]):.2f}")
    fig.show()
    pause(0.1)

    # Обучаем финальную модель на всех данных
    lda.fit(X, y)
    # Веса LDA (для признаков, полученных из CSP)
    w_lda = lda.coef_[0]  # вектор [компоненты] или [признаки]
    b_lda = lda.intercept_[0]
    
    #Сохраняем все необходимые параметры в JSON
    if output_filename:
        # Создаем словарь со всеми параметрами
        classifier_data = {
            'spatialW': projInverse[:, sel_comp].tolist(),  # веса csp фильтра
            'sos': sos.tolist(),  # коэффициенты SOS фильтра [секции × 6]
            'features_type': features,  # тип признаков
            # 'fft_features': {"ch": arange(epochs_1_csp.shape), "freq":}
            # 'sel_comp': sel_comp,  # выбранные компоненты CSP

            # LDA веса
            'w_lda': w_lda.tolist(),  # веса LDA
            'b_lda': float(b_lda),    # смещение LDA
            
            'fs': Fs,  # частота дискретизации
            "n_components": len(sel_comp)
        }
        
        # Сохраняем в JSON файл
        with open(output_filename, 'w') as f:
            json.dump(classifier_data, f, indent=4)
        
        print(f"Классификатор сохранен в {output_filename}")



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
        # data_folder = r"R:\projects_FEEDBACK_QUASI\data\02 ES calibration session v2.0"
        # record = "02-OM.hdf"
        data_folder = r"R:\projects_FEEDBACK_QUASI\data\tests\01 Evgeny 13.03"
        record = "01 calibration session.hdf"
        # eeg, idxs_1, idxs_2, xy, Fs = process_file_resonance(os.path.join(data_folder, record))
        eeg, idxs_rest, idxs_right, idxs_left, xy, Fs = process_file_resonance(os.path.join(data_folder, record))

    mode = "right-rest" # 'right-rest' or 'left-rest'
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
    ion()
    
    choose = False       #<-------------------------- ввести вручную
    band = [10, 14]      #<-------------------------- ввести вручную

    # СДЕЛАЙ ЭТО ЧЕРЕЗ str = input("text") ПОПОЗЖЕ !!!!!!!!!!!!!!!!!!!!!!!
    if choose:
        output_filename = os.path.join("results", f"csp_{band}_final.png")
        # смотрим ещё раз что получилось
        calculate_csp_in_bands(eeg, Fs, idxs1, idxs2, xy, edges_ms=1000, bands=[band], 
                                spectr=True, show_plot=True, anatoly=False, 
                                output_filename=output_filename)

        show(block=True)

        sys.exit(0)
        
    sel_comp = [0, 63] #<-------------------------- ввести вручную
    train_clssifier(eeg, Fs, idxs1, idxs2, edges_ms=250, band=band, 
                    sel_comp=sel_comp, freq=[10, 11, 12, 21, 22, 23], anatoly=False,
                    features="csp", 
                    output_filename=r"models/test_classifier.json")
    show(block=True)
