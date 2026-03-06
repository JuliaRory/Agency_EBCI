import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

from numpy import mean

from src.utils.transformations import unit_to_db

def plot_spectr(freq, spectr, labels, plot_mean=True, 
               freq_min = 0, freq_max=20, y_min=0, y_max=20, to_db=True, plot=True):
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    plot_all_channels(spectr, ax, freq, labels)
    if plot_mean:
        plot_mean_of_channels(spectr, ax, freq, labels)

    ax.set_ylim(y_min, y_max)
    ax.set_xlim(freq_min, freq_max)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(alpha=.5)
    y_label = "PSD [µV²/Hz"
    y_label = y_label + ",dB]" if to_db else y_label + "]"
    
    ax.set_ylabel(y_label)
    ax.set_xlabel("Frequency [Hz]")

    ax.legend(loc=[1,0], title="Channels")
    ax.set_title("Power Spectral Density", fontsize=16, y=1.05)
    if plot:
        plt.show()


def plot_all_channels(data, ax, freq, labels):
        for data_ch, label_ch in zip(data, labels):
            ax.plot(freq, data_ch, label=label_ch, linewidth=.5)

def plot_mean_of_channels(data, ax, freq, labels):
        data_mean = mean(data, axis=0)
        ax.plot(freq, data_mean, label="mean", color="black", linewidth=2)

def plot_alpha_spectr(freq, opened_eyes, closed_eyes, labels, plot_mean=True, 
               freq_min = 0, freq_max=20, y_min=0, y_max=20,
               to_db=False):
    
    if to_db:
        opened_eyes = unit_to_db(opened_eyes)
        closed_eyes = unit_to_db(closed_eyes)

    fig = plt.figure(figsize=(12, 3))

    gs = gridspec.GridSpec(1, 3, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    plot_all_channels(opened_eyes, ax1, freq, labels)
    plot_all_channels(closed_eyes, ax2, freq, labels)
    plot_all_channels(closed_eyes-opened_eyes, ax3, freq, labels)
    
    if plot_mean:
        plot_mean_of_channels(opened_eyes, ax1, freq, labels)
        plot_mean_of_channels(closed_eyes, ax2, freq, labels)
        plot_mean_of_channels(closed_eyes-opened_eyes, ax3, freq, labels)
    
    ax1.set_title("Opened eyes")
    ax2.set_title("Closed eyes")
    ax3.set_title("Diff (closed eyes - opened eyes)")

    for ax in [ax1, ax2]:
        ax.set_ylim(y_min, y_max)
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(freq_min, freq_max)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.grid(alpha=.5)

    y_label = "PSD [µV²/Hz"
    y_label = y_label + ",dB]" if to_db else y_label + "]"
    
    ax1.set_ylabel(y_label)
    ax1.set_xlabel("Frequency [Hz]")

    ax3.legend(loc=[1,0], title="Channels")
    
    fig.suptitle("Power Spectral Density", fontsize=16, y=1.05)

    return fig

