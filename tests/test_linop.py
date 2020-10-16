import numpy as np
import numpy.testing as npt

import sigpy as sp

import mrrecon as mr


def test_concat_arrays(flatten=False, dtype=np.complex64, device=-1):
    device = sp.Device(device)
    xp = device.xp

    nt = 20  # List length
    ncoils = 8
    ns = 128

    with device:
        # Create original list of arrays
        arrays = []
        for _ in range(nt):
            na_t = np.random.randint(80, 120)
            arr = xp.random.randn(ncoils, na_t, ns).astype(dtype)
            arrays.append(arr)

        array, shapes, indices = \
            mr.linop.concat_arrays(arrays, axis=1, flatten=flatten)

        # Recreate list of arrays
        # I manually slice the array here instead of using unconcat_arrays in
        # case unconcat_arrays has errors
        new_arrays = []
        for t in range(nt):
            idx0 = indices[t]
            idx1 = indices[t + 1]
            if flatten:
                new_arrays.append(array[idx0:idx1].reshape(shapes[t]))
            else:
                new_arrays.append(array[:, idx0:idx1, :].reshape(shapes[t]))

        for a0, a1 in zip(arrays, new_arrays):
            a0 = sp.to_device(a0)
            a1 = sp.to_device(a1)
            npt.assert_equal(a1, a0)

        # Test unconcat_arrays
        new_arrays = mr.linop.unconcat_arrays(array, shapes, indices, axis=1,
                                              flatten=flatten)

        for a0, a1 in zip(arrays, new_arrays):
            a0 = sp.to_device(a0)
            a1 = sp.to_device(a1)
            npt.assert_equal(a1, a0)
