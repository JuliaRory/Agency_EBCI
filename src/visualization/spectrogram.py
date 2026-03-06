import matplotlib.pyplot as plt
from numpy import arange, mean, ones_like, log10, finfo, max, abs

def plot_spectrogram(f, t, S, channels=None, average=True, fmin=None, fmax=None, 
                     to_db=True, 
                     title="EEG Spectrogram", 
                     xlabel="Time [s]", 
                     ylabel="Frequency [Hz]", 
                     cmap='jet', symmetric=False):
    """
    Plot spectrogram from precomputed FFT.

    Parameters
    ----------
    f : ndarray
        Frequencies from STFT.
    t : ndarray
        Time points from STFT.
    S : ndarray, shape (n_channels, n_freqs, n_times)
        Power spectra for each channel.
    channels : list, optional
        Channels to plot. Default: all.
    average : bool
        If True, average over selected channels.
    fmin, fmax : float, optional
        Frequency range to display.
    to_db : bool
        Convert power to dB.
    title, xlabel, ylabel : str
        Plot labels.
    cmap : str
        Colormap.
    symmetric : bool
        If True, set color scale symmetric around zero (or median if not in dB).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    """
    if channels is None:
        channels = arange(S.shape[0])
    
    # выбираем каналы
    data_plot = S[channels, :, :]
    
    if average:
        data_plot = mean(data_plot, axis=0)
    
    # маска частот
    freq_mask = ones_like(f, dtype=bool)
    if fmin is not None:
        freq_mask &= f >= fmin
    if fmax is not None:
        freq_mask &= f <= fmax
    f_plot = f[freq_mask]
    data_plot = data_plot[freq_mask, :]
    
    if to_db:
        data_plot = 10 * log10(data_plot + finfo(float).eps)
    
    fig, ax = plt.subplots(figsize=(10, 5))

    # симметричная шкала цвета
    if symmetric:
        abs_max = max(abs(data_plot))
        c = ax.pcolormesh(t, f_plot, data_plot, shading='gouraud', cmap=cmap, vmin=-abs_max, vmax=abs_max)
    else:
        c = ax.pcolormesh(t, f_plot, data_plot, shading='gouraud', cmap=cmap)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(f_plot[0], f_plot[-1])
    fig.colorbar(c, ax=ax, label="PSD [µV²/Hz, dB]" if to_db else "PSD [µV²/Hz]")
    fig.tight_layout()
    
    return fig, ax