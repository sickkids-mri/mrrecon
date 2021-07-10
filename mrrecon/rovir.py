"""Implementation of ROVir coils.

TODO: Add description of what ROVir can do. This may be long and complicated.

Reference:
    D. Kim et al. (2021). Region-optimized virtual (ROVir) coils: Localization
    and/or suppression of spatial regions using sensor-domain beamforming.
    Magnetic Resonance in Medicine, 86(1), 197-212.
"""
import numpy as np
import scipy.linalg

import sigpy as sp


def get_regions(img, indices):
    """Slices out regions from a set of coil images.

    Args:
        img (array): 2D, 3D, or 4D array. The first dimension should be the
            coil dimension, and the other dimensions should be (x,), (y, x), or
            (z, y, x) respectively.
        indices (list of lists of tuples): Contains the indices for slicing
            out sections of `img`.

    Returns:
        signal (array): 2D array with shape (ncoils, -1). The second dimension
            contains all the sliced pixels unraveled.

    Example:
        For a 3D image, the input array `img` should have shape
        (ncoils, nz, ny, nx). If two regions are to be sliced out, `indices`
        should look like the following.
        >>> # For region one
        >>> roi_1_x, roi_1_y, roi_1_z = (40, 120), (0, 320), (0, 160)
        >>> indices_1 = [roi_1_x, roi_1_y, roi_1_z]
        >>> # For region two
        >>> roi_2_x, roi_2_y, roi_2_z = (280, 320), (0, 320), (0, 160)
        >>> indices_2 = [roi_2_x, roi_2_y, roi_2_z]
        >>> indices = [indices_1, indices_2]
    """
    ncoils = img.shape[0]
    ndim = len(img.shape) - 1

    signal = []
    for roi in indices:
        if ndim == 3:
            (x0, x1), (y0, y1), (z0, z1) = roi
            signal.append(img[:, z0:z1, y0:y1, x0:x1])
        elif ndim == 2:
            (x0, x1), (y0, y1) = roi
            signal.append(img[:, y0:y1, x0:x1])
        elif ndim == 1:
            (x0, x1), = roi
            signal.append(img[:, x0:x1])

    signal = [arr.reshape((ncoils, -1)) for arr in signal]
    signal = np.concatenate(signal, axis=1)
    return signal


def coil_covariance(arr):
    """Calculates the inter-coil signal covariance matrix over a region.

    The reference calls this the inter-coil 'correlation' matrix but it should
    actually be the 'covariance' matrix.

    Args:
        arr (array): Array with shape (num_pix, ncoils), containing the
            multi-coil pixel values from a region.

    Returns:
        out (array): Array with shape (ncoils, ncoils). Coil covariance matrix.
    """
    xp = sp.get_array_module(arr)
    num_pix, ncoils = arr.shape

    # print(f'Coil means: {xp.mean(arr, axis=0)}')
    # print(f'Coil mean magnitudes: {xp.mean(xp.abs(arr), axis=0)}')

    out = xp.zeros((ncoils, ncoils), dtype=arr.dtype)

    for i in range(num_pix):
        # Why doesn't the spatial mean need to be subtracted?
        g = arr[i].reshape((1, ncoils))
        # Note: Broadcasting should be the same as matrix multiplication
        out += xp.conj(g).transpose() @ g

    return out


def solve(signal, interference=None):
    """Calculates ROVir coil-combination weights using calibration data.

    Args:
        signal (array): Array containing pixel values from the region of
            interest of an image of any shape. `signal` should be an array with
            two dimensions or more. The first dimension should be the coil
            dimension.
        interference (array or None): Array containing pixel values from the
            region(s) that should be suppressed. `interference` should be an
            array with two dimensions or more. The first dimension should be
            the coil dimension. If None, there will be no separation of signal
            of interest and interference in the virtual coils.

    Note: The region of interest and the interference regions should be
    separated by a gap.

    Note: `signal` and `interference` should be on the same device
    (CPU or GPU).

    Returns:
        W (array): Array with shape (ncoils, ncoils). The columns are unit
            eigenvectors corresponding to the associated eigenvalues arranged
            in descending order. NOTE: Possibly not currently in descending
            order. These eigenvectors can be used to create virtual coils.
            ROVir coil transformation matrix.
        A (array): Array with shape (ncoils, ncoils). Coil covariance matrix
            of the region of interest.
        B (array): Array with shape (ncoils, ncoils). Coil covariance matrix
            of the interference region(s).
    """
    xp = sp.get_array_module(signal)
    ncoils = signal.shape[0]
    signal = signal.transpose().reshape((-1, ncoils))
    if interference is not None:
        interference = interference.transpose().reshape((-1, ncoils))

    print(f'signal: {signal.dtype} {signal.shape}')
    print(f'interference: {interference.dtype} {interference.shape}')

    A = coil_covariance(signal)
    if interference is not None:
        B = coil_covariance(interference)
    else:
        B = xp.identity(ncoils, dtype=signal.dtype)

    A = sp.to_device(A)
    B = sp.to_device(B)

    # Check matrices are Hermitian
    assert np.array_equal(A, np.conj(A).transpose())
    assert np.array_equal(B, np.conj(B).transpose())

    # Check matrices positive-semidefinite
    print(f'eigenvalues of A:\n{np.linalg.eigvals(A)}')
    print(f'eigenvalues of B:\n{np.linalg.eigvals(B)}')
    print(f'eigenvalues of A (magnitude):\n{np.abs(np.linalg.eigvals(A))}')
    print(f'eigenvalues of B (magnitude):\n{np.abs(np.linalg.eigvals(B))}')

    # assert np.all(np.linalg.eigvals(sp.to_device(A)) >= 0)
    # assert np.all(np.linalg.eigvals(sp.to_device(B)) >= 0)

    # Check that B has full rank

    # Solve generalized eigenvalue problem
    eigvals, eigvecs = scipy.linalg.eig(A, b=B, left=False, right=True)
    print('Generalized eigenvalue problem:')
    print(f'eigenvalues: {eigvals}')
    print(f'eigenvalues (magnitude): {np.abs(eigvals)}')
    print(f'Check normalization: {np.linalg.norm(eigvecs, axis=0)}')
    # Some of the lower eigenvalues are out of order. Does this matter?
    # Also, the eigenvalues aren't exactly real. There is a small imaginary
    # component. Is this right?

    W = eigvecs
    return W, A, B


def total_signal_energy(W, A, dtype=np.float64):
    """Calculates the total regional signal energy for each ROVir coil.

    Calculated from the regional coil covariance matrix (A) and the ROVir coil
    transformation matrix (W).

    Args:
        W (array): Array with shape (ncoils, ncoils). ROVir coil transformation
            matrix.
        A (array): Array with shape (ncoils, ncoils). Coil covariance matrix
            of a region of interest (or interference region).

    Note: These arguments are usually from the outputs of `solve`.

    Returns:
        signal_energy (array): Array with shape (ncoils,). Total signal energy
            from the region for each ROVir coil.
    """
    ncoils = A.shape[0]
    signal_energy = np.zeros((ncoils,), dtype=A.dtype)
    for coil in range(ncoils):
        w = W[:, coil].reshape((ncoils, 1))
        signal_energy[coil] = w.conj().transpose() @ A @ w

    signal_energy = np.abs(signal_energy).astype(dtype)
    return signal_energy


def retained_signal_energy(W, A, dtype=np.float64):
    """Calculates the cumulative percentage retained signal energy.

    Calculated from the regional coil covariance matrix (A) and the ROVir coil
    transformation matrix (W).

    Args:
        W (array): Array with shape (ncoils, ncoils). ROVir coil transformation
            matrix.
        A (array): Array with shape (ncoils, ncoils). Coil covariance matrix
            of a region of interest (or interference region).

    Note: These arguments are usually from the outputs of `solve`.

    Returns:
        cumulative_energy (array): Array with shape (ncoils,).
    """
    ncoils = A.shape[0]
    cumulative_energy = np.zeros((ncoils,), dtype=dtype)
    for coil in range(ncoils):
        W_project = W[:, :(coil + 1)]
        WWH = W_project @ W_project.conj().transpose()
        cumulative_energy[coil] = \
            np.linalg.norm(WWH @ A @ WWH) / np.linalg.norm(A) * 100

    return cumulative_energy
