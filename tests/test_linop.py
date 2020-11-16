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


def test_finite_difference():
    """Compares to SigPy implementation of finite difference."""

    ndims = [1, 2, 3, 4, 5]
    shapes = [(1736,), (47, 99), (20, 18, 16), (5, 6, 7, 8), (12, 10, 5, 8, 9)]
    devices = [-1, 0]
    dtypes = [np.float32, np.complex64]

    for ndim, shape in zip(ndims, shapes):
        for device in devices:
            device = sp.Device(device)
            xp = device.xp
            with device:
                for dtype in dtypes:
                    x = sp.util.randn(shape, dtype=dtype, device=device)
                    for axis in range(ndim):
                        G = sp.linop.FiniteDifference(shape, axes=[axis])

                        # Test forward finite difference
                        y0 = G * x
                        y0 = y0[0]  # Remove singleton dimension
                        y = mr.linop.finite_difference(x, axis, adjoint=False)

                        y0, y = sp.to_device(y0), sp.to_device(y)
                        npt.assert_array_equal(y, y0)

                        # Test adjoint finite difference
                        y0 = G.H * xp.expand_dims(x, 0)
                        y = mr.linop.finite_difference(x, axis, adjoint=True)

                        y0, y = sp.to_device(y0), sp.to_device(y)
                        npt.assert_array_equal(y, y0)
