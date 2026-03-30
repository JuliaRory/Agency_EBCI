import matplotlib.pyplot as plt

from src.visualization.plot_helpers import get_color_map


newcmp = get_color_map()

import numpy as np

import numpy as np
import matplotlib.pyplot as plt


def compute_erd_epochs(epochs, baseline_samples):
    """
    epochs: [n_epochs, samples, channels]
    baseline_samples: (start, end)
    
    return:
        erd: [samples, channels]
    """
    erds = []

    for ep in epochs:
        baseline = ep[baseline_samples[0]:baseline_samples[1], :]
        task = ep

        power_base = np.mean(baseline**2, axis=0, keepdims=True)
        power_task = task**2

        erd = (power_task - power_base) / power_base
        erds.append(erd)

    erds = np.array(erds)  # [epochs, samples, channels]
    return np.mean(erds, axis=0)



def plot_erd(epochs_1, epochs_2, fs, baseline_samples, picks=None, ch_names=None):
    """
    Рисует ERD/ERS для двух классов на одной фигуре:
    - fig, ax = plt.subplots(1,2)
    - одинаковое время и масштаб
    """
    erd1 = compute_erd_epochs(epochs_1, baseline_samples)
    erd2 = compute_erd_epochs(epochs_2, baseline_samples)

    # Выравниваем длину по минимальной
    min_len = min(erd1.shape[0], erd2.shape[0])
    erd1 = erd1[:min_len, :]
    erd2 = erd2[:min_len, :]
    t = np.arange(min_len) / fs

    if picks is None:
        picks = range(erd1.shape[1])

    fig, ax = plt.subplots(1, 2, figsize=(20, 5), sharey=True, sharex=True)

    for ch in picks:
        label1 = f"class1-{ch_names[ch]}" if ch_names else f"class1-ch{ch}"
        label2 = f"class2-{ch_names[ch]}" if ch_names else f"class2-ch{ch}"

        ax[0].plot(t, erd1[:, ch], label=label1)
        ax[1].plot(t, erd2[:, ch], color='tab:red', label=label2)

    # Общие настройки
    for i, a in enumerate(ax):
        a.axhline(0, color='black', linewidth=1)
        a.set_xlabel("Time (s)")
        a.set_ylabel("ERD")
        a.grid(True)
        a.legend()
        a.set_title("Class 1" if i==0 else "Class 2")

    plt.suptitle("ERD/ERS Comparison (same scale, same time)")
    plt.show()