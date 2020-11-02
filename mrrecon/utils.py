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
