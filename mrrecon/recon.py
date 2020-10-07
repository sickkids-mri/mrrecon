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
