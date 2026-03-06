import numpy as np

from src.analysis.spectral_analysis import get_fft

def csp_features(epochs_csp):
    """
    epochs_csp: [n_epochs, samples, components]
    """
    feats = []

    for ep in epochs_csp:
        var = np.var(ep, axis=0)
        if ep.shape[1] != 2:
            var /= np.sum(var)
        feats.append(np.log(var))

    return np.array(feats)

def fft_feature(epoch, diff_ind_f=[10, 11, 12, 21, 22, 23], Fs=1000):
    fft_res, fft_t = get_fft(epoch, Fs=Fs, hop=int(0.1 * Fs), window=int(1 * Fs))
    feature = np.mean(fft_res[diff_ind_f, :, :], axis=0)        # усредняем по частотам 
    return np.mean(feature, axis=1)                             # усредняем по времени