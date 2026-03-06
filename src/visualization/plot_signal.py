import matplotlib.pyplot as plt
import numpy as np

def plot_signal(start_s, end_s, signal, s_to_idx, ch=None, plot=True):
    start_idx, end_idx = s_to_idx(start_s), s_to_idx(end_s)
    plt.figure(figsize=(15, 3))
    signal2plot = signal[start_idx:end_idx] if ch is None else signal[ch, start_idx:end_idx]
    plt.plot(signal2plot)
    ticks = np.arange(start_idx, end_idx, 1E4)
    step_s = (end_s - start_s) / ticks.shape[0]
    plt.xticks(ticks, np.arange(start_s, end_s, step_s).astype(int));
    plt.xlabel("Time [s]")
    plt.ylabel("Signal [uV]")
    if plot:
        plt.show()

