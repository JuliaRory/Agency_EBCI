
from scipy.signal import ShortTimeFFT, windows
from numpy import abs, ones, float32, abs
import numpy as np

def get_fft(eeg, Fs=100, hop=10, window=100):
    SFT = ShortTimeFFT(win=ones(window), hop=hop, fs=Fs,  fft_mode='onesided')
    fft_res = abs(SFT.stft(eeg, axis=0)) ** 2
    fft_t = SFT.t(len(eeg))
    return fft_res, fft_t

def get_fft_fast(eeg, Fs=100, hop=10, window=100):

    eeg = eeg.astype(float32)

    SFT = ShortTimeFFT(
        win=windows.hann(window, sym=False),
        hop=hop,
        fs=Fs,
        fft_mode="onesided",
        scale_to="psd"
    )

    fft_res = abs(SFT.stft(eeg, axis=0))
    fft_res = fft_res**2

    fft_t = SFT.t(len(eeg))
    fft_f = SFT.f

    return fft_res, fft_t, fft_f

def compute_epoch_spectrogram(epochs, Fs):

    specs = []

    for ep in epochs:

        fft, t, freqs = get_fft_fast(
            ep,
            Fs,
            hop=int(0.1*Fs),
            window=int(Fs)
        )

        specs.append(fft)

    specs = np.stack(specs)

    # усреднение по эпохам
    spec = np.mean(specs, axis=0)

    return spec, t, freqs



def compute_psd_welch(data, fs, fmin=0.5, fmax=40.0, freq_res=0.5, nperseg=None):
    """
    Compute power spectral density (PSD) using Welch's method.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_channels)
        Continuous EEG/signal data.
    fs : float
        Sampling frequency (Hz).
    fmin : float
        Minimum frequency to keep (Hz).
    fmax : float
        Maximum frequency to keep (Hz).
    freq_res : float
        Desired frequency resolution (Hz). 
        Determines nfft: nfft = fs / freq_res.
    nperseg : int or None
        Length of each Welch segment. If None, defaults to min(256, n_samples).

    Returns
    -------
    freqs : ndarray
        Frequency values in [fmin, fmax].
    psd : ndarray, shape (n_channels, n_freqs)
        Power spectral density for each channel.
    """

    from scipy.signal import welch
    from numpy import asarray

    n_samples, n_channels = data.shape
    
    # Определяем nfft для нужного разрешения по частоте
    nfft = int(fs / freq_res)
    
    if nperseg is None:
        nperseg = min(256, n_samples)
    
    psd_list = []
    
    for ch in range(n_channels):
        freqs_all, psd_ch = welch(
            data[:, ch],
            fs=fs,
            nperseg=nperseg,
            nfft=nfft
        )
        # маска частот
        freq_mask = (freqs_all >= fmin) & (freqs_all <= fmax)
        psd_list.append(psd_ch[freq_mask])
    
    psd = asarray(psd_list)  # shape: (n_channels, n_freqs)
    freqs = freqs_all[freq_mask]
    
    return freqs, psd
  

def compute_windowed_fft(data, fs=1000, channels=None, nperseg=1000, noverlap=100, window='hann'):
    """
    Compute windowed FFT (STFT) for each channel.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_channels)
        EEG or other multichannel signal.
    fs : float
        Sampling frequency (Hz).
    channels : list or ndarray, optional
        Channels to include. Default: all channels.
    nperseg : int
        Segment length for STFT.
    noverlap : int, optional
        Overlap between segments. Default: nperseg//2.
    window : str
        Window type ('hann', 'hamming', etc.).

    Returns
    -------
    f : ndarray
        Frequencies corresponding to rows of spectrogram.
    t : ndarray
        Time points corresponding to columns of spectrogram.
    spectrograms : ndarray, shape (n_channels, n_freqs, n_times)
        Magnitude squared (power) of STFT for each channel.
    """
    from numpy import asarray, arange, abs
    from scipy.signal import stft

    data = asarray(data)
    n_samples, n_channels_total = data.shape
    
    if channels is None:
        channels = arange(n_channels_total)
    
    if noverlap is None:
        noverlap = nperseg // 2
    
    spectrograms = []

    for ch in channels:
        f, t, Zxx = stft(
            data[:, ch],
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nperseg,
            padded=False
        )
        psd = abs(Zxx)**2
        spectrograms.append(psd)

    spectrograms = asarray(spectrograms)  # shape: (n_channels, n_freqs, n_times)
    return f, t, spectrograms