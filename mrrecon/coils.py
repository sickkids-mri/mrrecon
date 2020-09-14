import numpy as np

import sigpy as sp


def svd_compress(kspace, percent=0.95, k=None, calib_lines=50):
    """Compresses coils based on the SVD.

    Performs a singular value decomposition (SVD) on the input data matrix,
    where the rows of the data matrix are measurements at different points in
    k-space, and the columns are the measurements from each coil. The SVD
    calculates a new basis for the input matrix (which is used to calculate
    virtual coils), and the basis vectors corresponding to the lowest singular
    values are discarded (and thus discarding the corresponding virtual coils).

    Two different methods can be used to determine how many of the top virtual
    coils to keep. The first method calculates the cumulative percentage of
    total eigenvalues and keeps enough of the top virtual coils so that the
    cumulative percentage exceeds a certain threshold. The second method simply
    keeps the top k virtual coils. If a value for `k` is provided, the second
    method will be used instead of the first.

    This function works for both non-Cartesian 2D phase contrast and 4D flow
    data.

    Args:
        kspace (array): Multicoil k-space data to be compressed.
            Shape (ncoils, nv, na, ns).
        percent (float): Used to find the least number of virtual coils that
            will produce a cumulative percentage of total eigenvalues greater
            than `percent`.
        k (int): Top `k` number of virtual coils to keep.
        calib_lines (int): Number of lines of multicoil k-space to use in the
            SVD.

    Returns:
        compressed (array): Coil compressed k-space.
        s (array): Singular values.
    """
    xp = sp.get_array_module(kspace)
    ncoils, nv, na, ns = kspace.shape

    # Take a portion of the data for SVD
    a = kspace[:, :, :calib_lines, :]
    # Average the velocity encodes
    a = xp.mean(a, axis=1)  # Shape (ncoils, calib_lines, ns)
    a = xp.transpose(a, (1, 2, 0))  # Shape (calib_lines, ns, ncoils)
    a = xp.reshape(a, (-1, ncoils))

    u, s, vh = xp.linalg.svd(a, full_matrices=False)
    u = None  # noqa
    v = xp.conj(xp.transpose(vh))

    if k is None:
        eigenvalues = s**2
        conds = xp.cumsum(eigenvalues) / xp.sum(eigenvalues) > percent
        # TODO CuPy currently does not have cp.argwhere
        conds = sp.to_device(conds)
        # Number of virtual coils to keep
        num_virtual_coils = np.min(np.argwhere(conds)) + 1
    else:
        num_virtual_coils = k

    # Only calculate the virtual coils to be kept
    v = v[:, :num_virtual_coils]

    compressed = xp.empty((num_virtual_coils, nv, na, ns), kspace.dtype)

    # Compress input k-space
    # Note: Must use the same transpose and reshape as before
    for vel in range(nv):
        a = kspace[:, vel]
        a = xp.transpose(a, (1, 2, 0))
        a = xp.reshape(a, (-1, ncoils))
        av = xp.matmul(a, v)
        a = None
        av = xp.reshape(av, (na, ns, num_virtual_coils))
        compressed[:, vel] = xp.transpose(av, (2, 0, 1))
        av = None

    return compressed, s
