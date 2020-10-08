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
