import numpy as np
import scipy.linalg as la


# ---------- Riemannian mean ----------

def riemannian_mean_slow(covs, tol=1e-6, max_iter=50):
    C = np.mean(covs, axis=0)

    for _ in range(max_iter):
        sqrtC = la.sqrtm(C)
        inv_sqrtC = la.inv(sqrtC)

        logs = [la.logm(inv_sqrtC @ Ci @ inv_sqrtC) for Ci in covs]
        delta = np.mean(logs, axis=0)

        C_new = sqrtC @ la.expm(delta) @ sqrtC

        if np.linalg.norm(C_new - C) < tol:
            break

        C = C_new

    return C

def riemannian_mean(covs):
    """
    Быстрый Log-Euclidean mean
    covs: [n_trials, n_channels, n_channels]
    """
    logs = []
    for C in covs:
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-10)
        logs.append(eigvecs @ np.diag(np.log(eigvals)) @ eigvecs.T)

    mean_log = np.mean(logs, axis=0)

    eigvals, eigvecs = np.linalg.eigh(mean_log)
    Cref = eigvecs @ np.diag(np.exp(eigvals)) @ eigvecs.T

    return Cref

# ---------- Tangent space ----------
def tangent_space_slow(covs, Cref):
    sqrtC = la.sqrtm(Cref)
    inv_sqrtC = la.inv(sqrtC)

    feats = []
    for C in covs:
        T = la.logm(inv_sqrtC @ C @ inv_sqrtC)

        # векторизация (верхний треугольник)
        idx = np.triu_indices_from(T)
        feats.append(T[idx])

    return np.array(feats)


def riemannian_features_online(X_window, Cref, inv_sqrt):
    """
    X_window: (n_channels, n_samples) — одно окно EEG
    Cref: (n_channels, n_channels) — референсная ковариация
    Возвращает: вектор признаков (log-Euclidean)
    """
    # ковариация окна
    C = np.cov(X_window, rowvar=True)  # (n_channels, n_channels)

    M = inv_sqrt @ C @ inv_sqrt

    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 1e-10)
    logM = eigvecs @ np.diag(np.log(eigvals)) @ eigvecs.T

    # векторизуем верхний треугольник
    idx = np.triu_indices_from(logM)
    feats = logM[idx]

    return feats  # shape = (n_features,)

def tangent_space(covs, Cref):
    eigvals_ref, eigvecs_ref = np.linalg.eigh(Cref)
    eigvals_ref = np.maximum(eigvals_ref, 1e-10)

    inv_sqrt = eigvecs_ref @ np.diag(1.0 / np.sqrt(eigvals_ref)) @ eigvecs_ref.T

    feats = []
    for C in covs:
        M = inv_sqrt @ C @ inv_sqrt

        # вместо logm
        eigvals, eigvecs = np.linalg.eigh(M)
        eigvals = np.maximum(eigvals, 1e-10)

        logM = eigvecs @ np.diag(np.log(eigvals)) @ eigvecs.T

        idx = np.triu_indices_from(logM)
        feats.append(logM[idx])

    return np.array(feats)




def compute_covs_batch(X):
    # X: [trials, samples, channels]
    Xc = X - X.mean(axis=1, keepdims=True)
    covs = np.einsum('tse,tsf->tef', Xc, Xc)

    traces = np.trace(covs, axis1=1, axis2=2)
    covs /= traces[:, None, None]

    return covs