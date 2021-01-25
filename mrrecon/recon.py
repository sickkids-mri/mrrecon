import warnings

import numpy as np

import sigpy as sp


def phase_difference(img):
    """Performs a phase difference reconstruction.

    Works for both 2D phase contrast and 4D flow images.

    Args:
        img (complex array): Raw velocity encoded images. Shape (nv, ...).
            `img[0]` should be the phase reference.

    Returns:
        out (float32 array): Shape (nv, ...). `img[0]` is the magnitude image,
            calculated by averaging the magnitudes from each velocity encode.
            `img[1]` is the first phase image, `img[2]` is the second phase
            image, etc.
    """
    device = sp.get_device(img)
    xp = device.xp
    nv = img.shape[0]
    with device:
        out = xp.empty_like(img, dtype=xp.float32)
        out[0] = xp.mean(xp.abs(img), axis=0)
        for v in range(1, nv):
            out[v] = xp.angle(img[0] * xp.conj(img[v]))

    return out


def psf(traj, dcf=None, fov_scale=2, img_shape=None):
    """Calculates the point spread function.

    By default calculates the PSF at double the FOV so aliasing from
    sub-Nyquist sampling can be seen. By default the image size is not
    increased with FOV to save memory, therefore pixel widths would double.

    Args:
        traj (array): K-space trajectory with shape (..., ndim), where `ndim`
            is the number of spatial dimensions.
        dcf (array): Density compensation factor.
        fov_scale (float): Factor to increase or decrease the field of view of
            the point spread function.
        img_shape (list or tuple): Length of list/tuple should be `ndim`. Shape
            of the point spread function. Size affects the spatial resolution.

    Returns:
        psf (array): Point spread function.
    """
    device = sp.get_device(traj)
    xp = device.xp

    if img_shape is None:
        img_shape = sp.estimate_shape(traj)

    with device:
        traj = traj * fov_scale

        ones = xp.ones(traj.shape[:-1], dtype=xp.complex64)

        if dcf is not None:
            ones *= dcf

        psf = sp.nufft_adjoint(ones, traj, img_shape)

    return psf


def detect_steady_state(signal):
    """Detects when a signal reaches steady state.

    Args:
        signal (array): 1D float array containing signal that can be used to
            detect when the scan has reached steady state, e.g. magnitude of
            centre of k-space.

    Returns:
        ind (int): Point in signal where steady state was detected to begin.
    """
    # Find mean and spread of last half of signal,
    # which should be in steady state
    t = int(len(signal) / 2)  # Index at half
    last_half = signal[t:]
    last_half_mean = np.mean(last_half)
    last_half_std = np.std(last_half)

    ss_signal_thresh = last_half_mean + 6 * last_half_std
    if np.any(signal > ss_signal_thresh):
        # Then assumes signal contains non-steady state signal

        # Fit an exponential to the signal
        def exp(x, a, b, c, d):
            return a * np.exp(-b * (x - c)) + d

        x = np.linspace(0, 1, num=len(signal))
        # Estimate initial parameters
        p0 = np.ones(4, dtype=np.float64)
        p0[2] = 0  # Start c at 0
        p0[3] = last_half_mean
        p0[0] = signal[0] - p0[3]  # When x = 0
        p0[1] = - np.log((signal[1] - p0[3]) / p0[0]) / x[1]

        from scipy.optimize import curve_fit
        popt, pcov = curve_fit(exp, x, signal, p0=p0, maxfev=1000000)
        fitted = exp(x, *popt)

        # Find percentage of range
        drange = (signal.max() - fitted.min()) * 0.01
        deltas = fitted - fitted.min()
        conds = (deltas > drange)
        x_inds = np.argwhere(conds)
        # Could be empty if the exponential didn't fit the beginning
        if not (x_inds.size == 0):
            ind = np.max(x_inds)
        else:
            warnings.warn('Steady state signal fitting failed.')
            ind = 0

    else:
        # Entire signal is at steady state
        ind = 0

    # Multiply this number by a factor of 2, which heuristically moves the
    # spoke number to a better steady state point
    ind *= 2

    return ind
