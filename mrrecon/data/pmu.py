import numpy as np


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
