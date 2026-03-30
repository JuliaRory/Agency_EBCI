


from scipy.io import loadmat
import numpy as np

from src.utils.parse_h5df import load_h5df, ttl2binary, reverse_trigger
from src.utils.events import receive_epochs
from src.utils.montage_processing import get_topo_positions, get_channel_names, find_ch_idx

# ==== unique for resonance hdf files ====

EEG_CHANNELS = np.arange(64)
bad_channels = ["FT9", "TP9", "T7", "AF7", "AF8", "FT10", "TP10", "T8"]
labels = get_channel_names(r"./resources/mks64_standard.ced")
EEG_CHANNELS = np.array([find_ch_idx(ch, r"./resources/mks64_standard.ced") for ch in labels if not(ch in bad_channels)])

CED_FILE = r"./resources/mks64_standard.ced"

Fs = 1000 # Hz
s_to_idx = lambda x: int(x * Fs)
ms_to_idx = lambda x: int(x // 1000 * Fs)

def get_idxs(mode, idxs_rest, idxs_right, idxs_left):
    if mode == "left-right":
        idxs1 = idxs_left 
        idxs2 = idxs_right 
    elif mode == "right-rest":
        idxs1 = idxs_right
        idxs2 = idxs_rest
    elif mode == "left-rest":
        idxs1 = idxs_left
        idxs2 = idxs_rest
    return idxs1, idxs2

def process_file_resonance(filename, start_shift=100):
    """
    filename: str 
        absolute path
    """
    
    data, _ = load_h5df(filename)
    raw_eeg = data[:, EEG_CHANNELS] * 1E6 # uV

    trigger = reverse_trigger(ttl2binary(data[:, -1], bit_index=0))

    # events, _ = trigger_to_event_v1_1(trigger, window_size=600)        # 1 - motor, 2 - rest
    # idx_motor = receive_epochs(events, event_code=1)
    # idx_rest = receive_epochs(events, event_code=2)

    idxs_rest, idxs_right, idxs_left = parse_events(trigger, window_size=200, start_shift=start_shift, end_shift=0)
    
    xy = get_topo_positions(CED_FILE)[EEG_CHANNELS]

    Fs = 1000

    return raw_eeg, idxs_rest, idxs_right, idxs_left, xy, Fs

def define_label(dtrigger, idx, buff=600, labels={1: 0, 2: 1, 3: 2}):
        dtrig = dtrigger[idx-buff:idx-10] 
        n_shifts = np.where(dtrig == 1)[0].shape[0]
        for key in labels:
            if n_shifts == key:
                return labels[key]
        return np.nan

def parse_events(trigger, window_size=200, start_shift=100, end_shift=100):
    strigger = np.convolve(trigger, np.ones(window_size, dtype=int), 'valid')   # sum of trigger in window  
    
    start_idx = np.where((strigger == window_size) & (np.diff(strigger, prepend=0) == 1))[0].reshape((-1, 1))
    end_idx = np.where((strigger == 0) & (np.diff(strigger, prepend=0) == -1))[0].reshape((-1, 1))

    dtrigger = np.diff(trigger)
    labels = np.array([define_label(dtrigger, idx[0]) for idx in start_idx])

    events = np.concatenate([start_idx-start_shift, end_idx+end_shift], axis=1)

    idxs1 = events[labels == 0]
    idxs2 = events[labels == 1]
    idxs3 = events[labels == 2]
    
    # for quasi feedback
    # start_idx = (idxs2[:, 1] + 4000).reshape((-1, 1))
    # end_idx = (start_idx + 8000).reshape((-1, 1))
    # idxs1 = np.concatenate([start_idx, end_idx], axis=1)
    
    return idxs1, idxs2, idxs3


def parse_events_really_handle(trigger, window_size=600):
    trigger_sum = []
    idxs_start = []
    for start_idx in range(len(trigger)):
        how_much_left = len(trigger) - window_size
        end_idx = start_idx + window_size if window_size < how_much_left else start_idx + how_much_left
        tsum = np.sum(trigger[start_idx:end_idx])
        trigger_sum.append(tsum)

    sum_trigger = np.array(trigger_sum)
    dsumtrigger = np.diff(sum_trigger)
    
    # --------------------> HANDLE INPUT <---------------------------- 
    trial_len = 1800    # 2000-200
    additional_samples = 500    # 600-100
    max_value = 600 # dont know why

    idxs_end = np.where((sum_trigger[:-1] == max_value) & (dsumtrigger == -1))[0] + additional_samples
    idxs_start = idxs_end - trial_len
    
    photo_len = 700     
    idxs_photo = idxs_start - photo_len

    mask = np.ones(len(idxs_start))    # right
    for i, idx in enumerate(idxs_photo):
        sphoto = np.sum(trigger[idx:idxs_start[i]])
    
        if sphoto < 250:
            mask[i] = 0                 # rest
        if sphoto > 300:
            mask[i] = 2                 # left

    idxs = np.concatenate([idxs_start.reshape((-1, 1)), idxs_end.reshape((-1, 1))], axis=1)
    idxs1 = idxs[mask == 0]
    idxs2 = idxs[mask == 1]
    idxs3 = idxs[mask == 2]
    
    return idxs1, idxs2, idxs3



def trigger_to_event_v1_1(trigger, window_size=600):
    """
    Parse a photodiode trigger signal to detect motor and rest events.

    This function scans a binary photomark signal and identifies events
    based on the magnitude of fluctuations within a sliding window. 
    It returns an array of the same length as the input trigger, where
    each element indicates the type of event at that time.

    Parameters
    ----------
    trigger : array-like
        Binary (0 or 1) signal from a photodiode, representing stimulus fluctuations.
    window_size : int, optional, default=600
        Number of samples to consider in the sliding window when detecting changes.

    Returns
    -------
    events : array-like
        Array of the same length as `trigger`, containing:
        - 0 : no event
        - 1 : motor event
        - 2 : rest event
    trigger_sum : array-like
        Array of the same length as `trigger`, 
        containing sum of its elements in window_size. 
    """

    events = np.zeros(len(trigger)) 
    
    n_motor = 0
    wait_start = True
    idx_trial_start = None
    idx_rest = None
    trigger_sum = []
    pr_v = 0
    for start_idx in range(len(trigger)):
        how_much_left = len(trigger) - window_size
        end_idx = start_idx + window_size if window_size < how_much_left else start_idx + how_much_left
        tsum = sum(trigger[start_idx:end_idx])
        trigger_sum.append(tsum)

        tsummax = max(trigger_sum[-window_size:])                   # max value for the last window_size ms
        if (pr_v == tsummax) & (tsum < pr_v):                       # if new value is smaller than previous
            if wait_start:                                         # two  bursts  -> signal of a beginning 
                wait_start = False
                idx_trial_start = start_idx
            elif not(wait_start) and (n_motor < 4):                # three bursts -> signal of a motor trial 
                n_motor += 1
                if n_motor == 4:
                    events[idx_trial_start:start_idx] = 1
                    idx_rest = start_idx
            elif not(wait_start) and (n_motor == 4):               # four bursts -> signal of a rest trial
                events[idx_rest:start_idx] = 2
                n_motor = 0
                wait_start = True

        pr_v =  tsum
    
    return events, np.asarray(trigger_sum) 