import matplotlib.pyplot as plt
from numpy import concatenate, mean, log

from src.analysis.spectral_analysis import get_fft_fast, compute_epoch_spectrogram
from src.visualization.plot_helpers import get_color_map

from matplotlib.colors import LinearSegmentedColormap

newcmp = get_color_map()

import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch


def plot_log_ratio_psd(epochs_left, epochs_right, sfreq, ch_idx, fmin=1, fmax=40, filename=None, show=False):
    """
    log-ratio PSD: log(P_left / P_right)
    """

    def compute_mean_psd(epochs):
        psds = []
        for ep in epochs:
            freqs, psd = welch(
                ep[:, ch_idx],
                fs=sfreq,
                nperseg=sfreq*2,
                noverlap=sfreq
            )
            psds.append(psd)
        return freqs, np.mean(psds, axis=0)

    freqs, psd_left = compute_mean_psd(epochs_left)
    _, psd_right = compute_mean_psd(epochs_right)

    # защита от нулей
    eps = 1e-12
    log_ratio = np.log(psd_left + eps) - np.log(psd_right + eps)

    # ограничение диапазона
    mask = (freqs >= fmin) & (freqs <= fmax)

    plt.figure(figsize=(8, 5))
    plt.plot(freqs[mask], log_ratio[mask])

    plt.axhline(0, linestyle="--")
    plt.axvline(8, linestyle="--", alpha=0.5)
    plt.axvline(12, linestyle="--", alpha=0.5)
    plt.xticks(np.arange(fmin, fmax, 2))

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("log(P_left / P_right)")
    plt.title(f"Log-Ratio PSD (channel {ch_idx})")
    plt.grid(True)
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    

def plot_psd_channel(epochs, sfreq, ch_idx, label=None, fmin=1, fmax=40):
    """
    epochs : [n_trials, samples, channels]
    sfreq  : частота дискретизации
    ch_idx : индекс канала
    label  : подпись (например 'left' / 'right')
    """

    psds = []

    for ep in epochs:
        signal = ep[:, ch_idx]

        freqs, psd = welch(
            signal,
            fs=sfreq,
            nperseg=sfreq * 2,   # окно ~2 сек
            noverlap=sfreq       # 50% overlap
        )

        psds.append(psd)

    psds = np.array(psds)
    mean_psd = psds.mean(axis=0)
    std_psd = psds.std(axis=0)

    # ограничение частот
    mask = (freqs >= fmin) & (freqs <= fmax)

    freqs = freqs[mask]
    mean_psd = mean_psd[mask]
    std_psd = std_psd[mask]

    # в dB
    mean_psd_db = 10 * np.log10(mean_psd)
    std_psd_db = 10 * np.log10(std_psd + 1e-12)

    plt.plot(freqs, mean_psd_db, label=label)
    plt.fill_between(
        freqs,
        mean_psd_db - std_psd_db,
        mean_psd_db + std_psd_db,
        alpha=0.2
    )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB)")
    plt.title(f"PSD (channel {ch_idx})")
    plt.grid(True)
    
def plot_spectrograms(
        epochs_1,
        epochs_2,
        Fs,
        channels=[0,1,2,3,4,-5,-4,-3,-2,-1],
        fmin=3,
        fmax=30,
        baseline=(0, 100),
        start_shift=500, 
        title=None):
    
    spec1, t, freqs = compute_epoch_spectrogram(epochs_1, Fs)
    baseline_idx = np.where((t*Fs >= baseline[0]) & (t*Fs <= baseline[1]))[0]
    base1 = spec1[:, :, baseline_idx].mean(axis=2, keepdims=True)

    spec2, t2, _     = compute_epoch_spectrogram(epochs_2, Fs)
    baseline_idx = np.where((t2*Fs >= baseline[0]) & (t2*Fs <= baseline[1]))[0]
    base2 = spec2[:, :, baseline_idx].mean(axis=2, keepdims=True)

    spec1 = 10*np.log10(spec1 / (base1 + 1e-12))
    spec2 = 10*np.log10(spec2 / (base2 + 1e-12))

    n_time = spec1.shape[2]
    d = (spec2.shape[2] - n_time) // 2
    if d != 0:
        spec2 = spec2[:, :, d:d+n_time]  # центральный участок

    spec = spec2 - spec1    

    # --- частотный диапазон ---
    mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[mask]
    spec = spec[mask]

    # корректируем отрицательные индексы каналов
    channels = [spec.shape[1]+c if c<0 else c for c in channels]

    # создаем сетку 2x5
    fig, ax = plt.subplots(2,5, figsize=(18,6), sharex=True, sharey=True)

    # симметричная шкала цвета
    vlim = np.percentile(np.abs(spec[:,channels,:]), 98)

    for i,ch in enumerate(channels):
        r = i//5
        c = i%5

        im = ax[r,c].imshow(
            spec[:,ch,:],
            origin="lower",
            aspect="auto",
            extent=[t[0], t[-1], freqs[0], freqs[-1]],
            cmap=newcmp,
            vmin=-vlim,
            vmax=vlim
        )

        ax[r,c].set_title(f"Ch {ch}")
        ax[r,c].axvline(start_shift-baseline[1], color="black", lw=1)  # линия события

    # подписи осей
    for a in ax[:,0]:
        a.set_ylabel("Frequency (Hz)")
    for a in ax[-1]:
        a.set_xlabel("Time (s)")

    # цветовая шкала
    # --- добавляем отдельную вертикальную colorbar справа от всех графиков ---
    fig.subplots_adjust(right=0.9)  # оставляем место справа
    cbar_ax = fig.add_axes([1.02, 0.15, 0.02, 0.7])  # [лево, низ, ширина, высота]
    fig.colorbar(im, cax=cbar_ax, label="ΔPower (dB vs baseline)")
    # fig.colorbar(im, ax=ax.ravel().tolist(), label="ΔPower (dB vs baseline)")

    if title is None:
        title = "Spectrogram difference"
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()

    return fig

def plot_spectr(epochs_1, epochs_2, Fs, channels=[0, 1, 2, 3, 4, -5, -4, -3, -2, -1], title=None):
    
    eeg_1= concatenate(epochs_1, axis=0)
    eeg_2 = concatenate(epochs_2, axis=0)
    
    fft_res_1, fft_t_1 = get_fft_fast(eeg_1, Fs=Fs, hop=int(0.1 * Fs), window=int(1 * Fs))
    fft_res_2, fft_t_2 = get_fft_fast(eeg_2, Fs=Fs, hop=int(0.1 * Fs), window=int(1 * Fs))

    fft_mean_1 = mean(10 * log(fft_res_1), axis=2)
    fft_mean_2 = mean(10 * log(fft_res_2), axis=2)

    fig, ax = plt.subplots(1, 2, figsize=(20, 4))
    
    ch1 = [0, 1, 2, 3, 4]
    ch2 = [-5, -4, -3, -2, -1]
    ch2 = [fft_mean_2.shape[1]+idx for idx in ch2]

    ax[0].plot(fft_mean_2[:, ch1]-fft_mean_1[:, ch1])
    ax[1].plot(fft_mean_2[:, ch2]-fft_mean_1[:, ch2])

    ax[0].legend(ch1, loc=[1,0])
    ax[1].legend(ch2, loc=[1,0])
    
    y_label = "PSD [µV²/Hz, dB]" 
    ax[0].set_ylabel(y_label)
    for axes in ax:
        axes.grid(alpha=.5)
        axes.set_xlabel("Frequency [Hz]")
        
        axes.set_xlim(0, 30)

    if title is None:
        title = "Power Spectral Density"
    fig.suptitle(title, fontsize=16, y=1.05)

    return fig

