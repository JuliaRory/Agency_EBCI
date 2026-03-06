from numpy import array, asarray, sum, diff

def slice_epochs(data, intervals):
    """
    Slice multi-channel data into epochs based on start and end indices.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_channels)
        Input signal array.
    intervals : list of [start, end]
        List of intervals specifying the start (inclusive) and end (exclusive)
        indices for each epoch.

    Returns
    -------
    epochs : ndarray, shape (n_epochs, n_samples_in_epoch, n_channels)
        Array containing the extracted epochs.
    """

    min_epoch_dur = min(diff(intervals))[0]

    epochs = []
    for start, end in intervals:
        epochs.append(data[end-min_epoch_dur:end])
    # print(array(epochs))
    return array(epochs)

def receive_epochs(events, event_code):
    return asarray(find_intervals(events, event_code))

def reveive_events_info(events, events_info=None):
    if events_info is None:
        assert "events_info is empty."

    for key in events_info:
        events_info[key]["num"] = count_any_transitions(events, events_info[key]["event_code"])
        events_info[key]["dur"] = get_duration(events_info[key]["trial_dur_ms"], events_info[key]["num"])

def get_duration(trial_dur, n_trial, degree=1):
    return float(round(trial_dur * n_trial / 1000 * degree, 1))

def count_any_transitions(arr, event_code=1):
    """
    Count transitions to bit in a discrete signal.

    Parameters
    ----------
    arr : array-like
        Input array of numbers.
    event_code: int
        Code of some event.

    Returns
    -------
    transitions : int
        Number of transitions.
    """
    arr = asarray(arr)
    # сдвинутый массив на 1 для сравнения текущего и предыдущего значения
    prev = arr[:-1]
    curr = arr[1:]
    return int(sum((prev != event_code) & (curr == event_code)))
    

def find_intervals(arr, value):
    """
    Find intervals where a specific value occurs consecutively in an array.

    Parameters
    ----------
    arr : array-like
    value : int
        Value to search for.

    Returns
    -------
    intervals : list of [start, end]
        List of intervals where the value occurs consecutively.
        Each interval is [start_index, end_index] (inclusive start, exclusive end).
    """
    arr = asarray(arr)
    intervals = []
    in_interval = False
    start_idx = None

    for i, v in enumerate(arr):
        if v == value:
            if not in_interval:
                start_idx = i
                in_interval = True
        else:
            if in_interval:
                intervals.append([start_idx, i])
                in_interval = False

    # если массив заканчивается значением value, закрываем последний интервал
    if in_interval:
        intervals.append([start_idx, len(arr)])

    return intervals
