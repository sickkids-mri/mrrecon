import numpy as np

import sigpy as sp


def arrays_to_device(arrs, device=-1):
    """Copies a list of arrays to the specified device.

    Args:
        arrs (list): List of NumPy or CuPy arrays.

    Returns:
        outs (list): New list.
    """
    if arrs is None:
        return None

    outs = []
    for arr in arrs:
        outs.append(sp.to_device(arr, device))

    return outs


def sinc_upsample(s, upsample_factor):
    """Upsamples 1D signals using Fourier zero-padding.

    Usually used to upsample flow curves or gating signals. The upsampled
    signal will be shifted so that the first point of the upsampled signal
    corresponds to the first point of the original signal. This convention is
    convenient for upsampling corresponding time stamps and plotting.

    Args:
        s (array): 1D array containing signal to be upsampled.
        upsample_factor (int): Positive integer.
    """
    xp = sp.get_array_module(s)
    dtype = s.dtype

    if upsample_factor == 1:
        return xp.copy(s)

    nx = s.shape[0]
    nx_new = nx * upsample_factor
    s = sp.fft(s)
    s = sp.util.resize(s, [nx_new])  # Zero-pad
    s = sp.ifft(s)
    if not np.issubdtype(dtype, np.complexfloating):  # If real
        s = s.real  # Imaginary part should be 0
    s = s.astype(dtype)  # Will sometimes convert float64 to float32
    s *= upsample_factor ** 0.5
    if nx % 2 == 1:  # If odd
        # Need to shift so that the first point of the upsampled signal
        # corresponds to the first point of the original signal
        shift = int(xp.floor(upsample_factor / 2))
        s = xp.roll(s, -shift)
    return s


def upsample_times(times, upsample_factor):
    """Calculates time stamps for an upsampled cyclic signal.

    Signal is assumed cyclic, e.g. flow curve.

    NOTE: If the time stamps did not start from 0, the upsampled time stamps
    may no longer be in ascending order required for plotting.

    Args:
        times (array): 1D array containing the time stamps of the original
            signal. Time stamps should be evenly spaced.
        upsample_factor (int): Positive integer.
    """
    xp = sp.get_array_module(times)

    if upsample_factor == 1:
        return xp.copy(times)

    dt = times[1] - times[0]
    dt = dt / upsample_factor
    nx = times.shape[0]
    nx_new = nx * upsample_factor
    # This gets confusing when time stamps have already been shifted and no
    # longer start from 0. But the following should be correct...
    shift = times[0]  # How far first time stamp shifted from 0
    rr = times[-1] - shift + (times[1] - times[0])
    times = (np.arange(nx_new) * dt + shift) % rr
    return times


def rot_from_quat(quat):
    """Generate rotation matrix from quaternion.

    TODO Has this been verified?
    Why is it different from scipy's implementation?
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
    """
    a, b, c, d = quat
    rot = np.array([[a * a + b * b - c * c - d * d, 2 * b * c - 2 * a * d, 2 * b * d + 2 * a * c],
                    [2 * b * c + 2 * a * d, a * a - b * b + c * c - d * d, 2 * c * d - 2 * a * b],
                    [2 * b * d - 2 * a * c, 2 * c * d + 2 * a * b, a * a - b * b - c * c + d * d]])
    return rot
