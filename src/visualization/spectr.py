import matplotlib.pyplot as plt
from numpy import concatenate, mean, log

from src.analysis.spectral_analysis import get_fft

def plot_spectr(epochs_1, epochs_2, Fs, channels=[0, 1, 2, 3, 4, -5, -4, -3, -2, -1]):
    
    eeg_1= concatenate(epochs_1, axis=0)
    eeg_2 = concatenate(epochs_2, axis=0)
    
    fft_res_1, fft_t_1 = get_fft(eeg_1, Fs=Fs, hop=int(0.1 * Fs), window=int(1 * Fs))
    fft_res_2, fft_t_2 = get_fft(eeg_2, Fs=Fs, hop=int(0.1 * Fs), window=int(1 * Fs))

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

    fig.suptitle("Power Spectral Density", fontsize=16, y=1.05)

    return fig