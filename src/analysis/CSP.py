import numpy as np
from scipy.linalg import eigh

from src.utils.olivehawkins_robustcov import olivehawkins_robustcov
from sklearn.covariance import MinCovDet
import scipy.linalg as la
from scipy.signal import butter, filtfilt

# ===================
# == CHAT GPT =======
# ===================


import scipy.linalg as la

def compute_cov(epoch):
    X = epoch - np.mean(epoch, axis=0, keepdims=True)
    C = X.T @ X
    return C / np.trace(C)



def compute_csp(epochs1, epochs2, robust=False, filter=True):
    """
    epochs1, epochs2 : [n_epochs, samples, channels]

    Returns
    -------
    W : spatial filters
    A : spatial patterns (для визуализации)
    eigvals : eigenvalues
    """

    def class_cov(epochs):
        covs = [compute_cov(ep) for ep in epochs]
        return np.mean(covs, axis=0)

    def class_robust_cov(epochs):
        covs = [olivehawkins_robustcov(ep)[0] for ep in epochs]
        return np.mean(covs, axis=0)
    
    calculate_cov = class_cov if not robust else class_robust_cov
    C1 = calculate_cov(epochs1)
    C2 = calculate_cov(epochs2)

    C = C1 + C2
    # regularization
    reg = 1e-3 * np.trace(C) / C.shape[0]
    C += reg * np.eye(C.shape[0])


    # whitening
    eigvals, eigvecs = la.eigh(C)
    eigvals = np.maximum(eigvals, 1e-10)

    P = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    S1 = P.T @ C1 @ P

    eigvals, B = la.eigh(S1)

    W = P @ B

    # сортировка
    order = np.argsort(eigvals)
    W = W[:, order]
    eigvals = eigvals[order]

    # spatial patterns
    A = la.pinv(W).T

    return W, A, eigvals


def apply_csp(epochs, W, sel_components=[0, 1]):
    """
    epochs: [n_epochs, samples, channels]
    """
    # sel = np.r_[0:n_components//2, -n_components//2:0]

    epochs_csp = np.array([
        ep @ W[:, sel_components] for ep in epochs
    ])

    return epochs_csp

# ===================
# == Анатолий-like ==
# ===================

def calculate_robust_cov(epochs):
    """
    epochs [n_trials, n_samples, n_channels]
    Return:
        cov [n_channels, n_channels]
    """
    data = np.concatenate(epochs, axis=0)   # [n_samples, n_channels]
    # MCD = MinCovDet(support_fraction=0.5, store_precision=False)
    # cov = MCD.fit(data)
    cov = olivehawkins_robustcov(data)
    return cov

def calculate_CSP(epochs_1, epochs_2):
    """
    c1, c2: covariance matrix
    Return:
        W_fixed:        
        projForward:    
        evals:          

    """
    с1 = calculate_robust_cov(epochs_1)
    с2 = calculate_robust_cov(epochs_2)
    
    R1 = с1 / np.trace(c1)
    R2 = c2 / np.trace(c2)
    L, W = la.eig(R1, R1+R2)
    order = np.argsort(L)
    L = L[order]
    W = W[:, order]
    fProj = np.dot(W.T, R1).T
    d, p = fProj.shape
    maxind = np.argmax(np.abs(fProj), axis=0)
    # maxinds = np.array([np.where(np.abs(W[:, i]) == np.max(np.abs(W[:, i])))[0][0] for i in range(W.shape[1])])
    max_magnitudes = np.array([fProj[maxind[i], i] for i in range(W.shape[1])])
    rowsign = np.sign(max_magnitudes)
    W_fixed = W * rowsign
    projForward = la.pinv(W_fixed).T
    evals = L
    return W_fixed, projForward, evals

# ======================
# == basic principles ==
# ======================


def cov_epoch(X):
    """
    X: (channels, time)
    """
    C = X @ X.T
    return C / np.trace(C)

def regularize(C, alpha=0.05):
    return (1 - alpha) * C + alpha * np.eye(C.shape[0])

def calculate_CSP_in_trials(epochs_motor, epochs_rest):
    covs_motor = np.array([cov_epoch(ep.T) for ep in epochs_motor])  # ep: (time, ch)
    C_motor  = covs_motor.mean(axis=0)
    C_motor  = regularize(C_motor,  alpha=0.05)

    covs_rest = np.array([cov_epoch(ep.T) for ep in epochs_rest])  # ep: (time, ch)
    C_rest  = covs_rest.mean(axis=0)
    C_rest = regularize(C_rest, alpha=0.05)
    
    C_sum = C_motor + C_rest
    eigvals, eigvecs = eigh(C_motor, C_sum)     # λ = 1 -> motor class
    
    # сортируем по убыванию собственных значений, первые - лучшие 
    ix = np.argsort(eigvals)[::-1]  # убывание λ
    eigvals = eigvals[ix]
    eigvecs = eigvecs[:, ix]

    ## spatial patterns 
    A = C_sum @ eigvecs
    A /= np.linalg.norm(A, axis=0, keepdims=True) # to normalize

    return eigvals, eigvecs, A