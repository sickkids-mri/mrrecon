import sigpy as sp


def concat_arrays(arrays, axis, flatten=False):
    """Concatenates a list of arrays.

    This function is mainly used to convert a list of cardiac gated k-space
    into an array, which is the required input format of SigPy linear
    operators.

    Args:
        arrays (list): List of arrays.
        axis (int): Axis along which the arrays are stacked.
        flatten (bool): Whether to flatten each array before concatenation.

    Returns:
        array (array): Concatenated array.
        shapes (list): Shapes of the original arrays. Length of `shapes` should
            be the same as the length of the list of arrays.
        indices (list): Indices to slice the concatenated array back into the
            original list of arrays. Length of list should be one more than the
            number of arrays concatenated.
    """
    device = sp.get_device(arrays[0])
    xp = device.xp

    shapes = [arr.shape for arr in arrays]

    idx = 0
    indices = [idx]

    for shape in shapes:
        if flatten:
            idx += int(sp.prod(shape))
        else:
            idx += shape[axis]

        indices.append(idx)

    with device:
        if flatten:
            array = xp.concatenate(arrays, axis=None)
        else:
            array = xp.concatenate(arrays, axis=axis)

    return array, shapes, indices


def unconcat_arrays(array, shapes, indices, axis, flatten=False):
    """Slices and reshapes an array into a list of arrays.

    Performs the opposite of `concat_arrays`.

    Args:
        array (array): Concatenated array.
        shapes (list): Shapes of the original arrays. Length of `shapes` should
            be the same as the length of the list of arrays.
        indices (list): Indices to slice the concatenated array back into the
            original list of arrays. Length of list should be one more than the
            number of arrays concatenated.
        axis (int): Axis to slice.
        flatten (bool): Whether the array was flattened before concatenation.

    Returns:
        arrays (list): List of arrays.
    """
    device = sp.get_device(array)
    xp = device.xp

    ndim = array.ndim
    arrays = []

    if flatten:
        axis = 0

    for i, shape in enumerate(shapes):
        idx0 = indices[i]
        idx1 = indices[i + 1]

        slc = [slice(None)] * axis  # Slices axes before `axis`
        slc += [slice(idx0, idx1)]  # Slices the chosen axis
        slc += [slice(None)] * (ndim - axis - 1)  # Slices axes after `axis`
        slc = tuple(slc)

        sliced = array[slc].reshape(shape)
        arrays.append(sliced)

    return arrays
