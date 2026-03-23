import numpy as np

from src.analysis.spectral_analysis import get_fft
from src.analysis.CSP import compute_cov
from src.analysis.Riemannian import riemannian_mean, tangent_space

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


# ---------- Riemannian features -------

def riemannian_features(epochs_csp):
    """
    epochs_csp: [n_epochs, samples, components]
    """
    covs = np.array([compute_cov(ep) for ep in epochs_csp])
    Cref = riemannian_mean(covs)
    X_feat = tangent_space(covs, Cref)

    eigvals_ref, eigvecs_ref = np.linalg.eigh(Cref)
    inv_sqrt = eigvecs_ref @ np.diag(1.0 / np.sqrt(np.maximum(eigvals_ref, 1e-10))) @ eigvecs_ref.T
    return X_feat, Cref, inv_sqrt
