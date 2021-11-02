import math

import numpy as np
import scipy


def choose_pmu_for_cardiac_gating(user_float):
    """Tries to automatically choose the best signal for cardiac gating.

    Args:
        user_float: 2D array. Shape (24, num_acq).

    Returns:
        chosen_ind: The index of the chosen PMU measurement in user_float.
    """
    if not user_float.shape[0] == 24:
        raise AssertionError('Expected array with 24 rows, '
                             f'got {user_float.shape[0]} instead.')

    # The PMU data was set to be stored in the last 6 rows (ref Eric Schrauben)
    num_meas = 6
    # Calculate the standard deviations to get a measure of spread
    standard_devs = np.empty(num_meas)
    standard_devs[:] = np.inf  # For finding the min later

    for a in range(num_meas):
        # Current measurement
        meas_number = 24 - num_meas + a
        meas = np.copy(user_float[meas_number])

        # Some measurements are flat
        if np.sum(np.abs(meas - np.mean(meas))) == 0:
            # Skip flat measurements
            continue

        # Try to deal with outliers from the measurement before normalizing
        # TODO: I have never looked at the distributions to validate this
        meas_mean = np.mean(meas)
        meas_std = np.std(meas)
        meas[meas > (meas_mean + 2*meas_std)] = meas_mean
        meas[meas < (meas_mean - 2*meas_std)] = meas_mean

        # Some measurements are flat but have large spikes. This will result in
        # very small standard deviations. Try to skip these.
        # Find the number of points that are close to the mean
        # Note these data spread parameters were chosen without validation
        meas_mean = np.mean(meas)
        all_less_than = meas < (meas_mean + 0.5*meas_std)
        all_greater_than = meas > (meas_mean - 0.5*meas_std)
        close_to_mean = np.logical_and(all_less_than, all_greater_than)
        # Percentage of total points that are within this range
        percentage = np.sum(close_to_mean) / meas.size
        if percentage > 0.9:  # If a high percentage
            # This should be a narrow distribution
            continue

        # Normalize the range
        meas = meas - meas.min()
        meas = meas / meas.max()

        standard_devs[a] = np.std(meas)

    # Find the measurement with the smallest standard deviation
    smallest_std_ind = np.argmin(standard_devs)
    chosen_ind = 24 - num_meas + smallest_std_ind
    return chosen_ind


def flip_upwards(meas):
    """Flip the signal so that the peaks are pointing upwards.

    Args:
        meas: 1D array. Cardiac measurement.

    Returns:
        flipped_meas: 1D array. Flipped so that peaks are pointing upwards.
    """
    meas_orig = meas
    meas = np.copy(meas)

    # Try to deal with outliers from the measurement before finding min/max
    # TODO: I have never looked at the distributions to validate this
    meas_mean = np.mean(meas)
    meas_std = np.std(meas)
    # Remove outliers
    meas[meas > (meas_mean + 2*meas_std)] = meas_mean
    meas[meas < (meas_mean - 2*meas_std)] = meas_mean

    meas_min = meas.min()
    meas_max = meas.max()
    meas_midpoint = (meas_min + meas_max) / 2
    meas_mean = np.mean(meas)  # Mean of the processed measurement

    if meas_mean > meas_midpoint:
        # Flip the original measurement (that has potential outliers)
        flipped_meas = - meas_orig + meas_max + meas_min
    else:
        flipped_meas = meas_orig

    return flipped_meas


def find_peaks(signal, fs, min_separation=270, window_width=2500, noise_level=1.25):
    """Find peaks from pulse oximeter or ECG.

    Args:
        signal: Pulse oximeter or ECG data.
        fs: Sampling frequency (Hz) of the signal.
        min_separation: Strict minimum time between peaks (ms).
        window_width: Width of sliding window for standard deviation (ms).
        noise_level: Scalar. Increase this if the height of erroneous peaks is
            high.

    Returns:
        peaks: 1D array. Indices of the signal array where peaks are found.
        height: The adaptive minimum height for determining peaks.
    """
    # Sliding window mean and standard deviation
    ww = int(np.round(window_width/1000*fs))
    means = np.zeros_like(signal)
    stds = np.zeros_like(signal)
    for a in range(len(stds)):
        # Current segment
        ww_ceil = int(np.ceil(ww/2))  # Upper ind
        ww_floor = int(np.floor(ww/2))  # Lower ind
        if a < ww_floor:
            curr_seg = signal[:(a+ww_ceil)]
        else:
            curr_seg = signal[(a-ww_floor):(a+ww_ceil)]

        means[a] = np.mean(curr_seg)
        stds[a] = np.std(curr_seg)

    height = means + noise_level*stds
    distance = min_separation/1000*fs
    peaks, _ = scipy.signal.find_peaks(signal, height=height,
                                       distance=distance)

    return peaks, height


def extrapolate_triggers(triggers, num_extrap=5):
    """Extrapolates some trigger times before and after.

    Args:
        triggers (array): 1D array containing cardiac trigger times.
        num_extrap (int): Number of trigger times to extrapolate.

    Returns:
        triggers (array): 1D array containing original cardiac trigger times,
            padded both sides with extrapolated trigger times.
    """
    rr = np.diff(triggers)
    rr_mean_beginning = np.mean(rr[:5])
    rr_mean_end = np.mean(rr[-5:])
    triggers_before = np.linspace(triggers[0] - num_extrap * rr_mean_beginning,
                                  triggers[0] - rr_mean_beginning,
                                  num_extrap)
    triggers_after = np.linspace(triggers[-1] + rr_mean_end,
                                 triggers[-1] + num_extrap * rr_mean_end,
                                 num_extrap)
    triggers = np.concatenate((triggers_before, triggers, triggers_after))
    return triggers


def perturb_triggers(triggers, eps=1):
    """Adds or subtracts a small, insignificant number to each trigger time.

    This is necessary when trigger times are determined from the same time
    stamps as k-space data. K-space data that fall right on a bin edge when
    sorting will all be sorted into one of two possible bins, when they should
    have equal probability of going into either bin.

    Args:
        triggers (array): 1D array containing cardiac trigger times in ms.
        eps (float): Time to add or subtract (ms).

    Returns:
        triggers (array): 1D array containing perturbed cardiac trigger times.
    """
    num_triggers = len(triggers)
    direction = np.ones(num_triggers)  # Vector of -1 or 1
    rnd = np.random.rand(num_triggers) > 0.5  # Standard uniform distribution
    direction[rnd] = -1
    eps = direction * eps
    triggers = triggers + eps
    return triggers


def hr_to_triggers(hr, first, last):
    """Calculates trigger times based on a constant heart rate.

    Args:
        hr (float): A constant heart rate (bpm).
        first (float): Time stamp of first k-space line (in ms).
        last (float): Time stamp of last k-space line (in ms).

    Returns:
        triggers (array): 1D array containing simulated cardiac trigger times.
    """
    # Convert heart rate (bpm) to RR interval (ms)
    rr = 1 / hr * 60 * 1000
    num_beats = math.ceil((last - first) / rr) + 1
    triggers = np.arange(num_beats) * rr + first
    return triggers


def read_trigger_log(filename):
    """Python translation of CWR_EXT_log.m.

    Reads peripheral monitor unit (PMU) measurements and identifies cardiac
    trigger times. At least that's what I think it does, no description was
    provided to me.

    Args:
        filename (str): Name of the file to be read and processed.

    Returns:
        triggers (array): Trigger times (ms).
    """
    # List of words in the file (delimited by spaces or newlines)
    words = []

    with open(filename, 'r') as f:
        # Reads file line by line
        for line in f:
            # Removes newlines  # TODO: split() with no arg might accomplish
            line = line.rstrip('\n')
            # Removes multiple spaces from the line
            # Checks to see if line contains double spaces
            remove_multiple_spaces = '  ' in line
            while remove_multiple_spaces:
                # Replaces double spaces with single space
                line = line.replace('  ', ' ')
                remove_multiple_spaces = '  ' in line

            # Appends words in this line to the list of words
            words += line.split(' ')

    # Finds the MDH start time (which is one index after the name)
    starttime = words[words.index('LogStartMDHTime:') + 1]
    starttime = float(starttime)

    # Now starts the part that makes no sense to me
    count = 1
    header = 0
    triggers = {}  # No idea what the values are that this holds
    num_skip = 5  # Number of words to skip
    for a in range(num_skip, len(words)):
        try:
            val = float(words[a])
        except ValueError:
            val = float('nan')  # Not all words are numbers

        if val in [5002, 5003]:
            header = 1
        elif val == 6002:
            header = 0

        if val == 5000:
            triggers[count-1] = 5000
        elif val == 6000:
            triggers[count-1] = 6000

        if (header == 0) and (val < 4999):
            triggers[count] = val
            count += 1

    # Convert dictionary to array
    triggers = np.array(list(triggers.values()))

    t = np.linspace(starttime, starttime + (len(triggers) - 1)*2.5,
                    num=len(triggers), endpoint=True)
    triggers = t[triggers == 5000]
    return triggers
