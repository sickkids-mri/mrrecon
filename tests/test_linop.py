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


def test_FiniteDifference():
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

                    for axes in ([None] + [[axis] for axis in range(ndim)]):
                        if axes is None:
                            z = sp.util.randn(
                                (ndim,) + shape, dtype=dtype, device=device)
                        else:
                            z = sp.util.randn(
                                (1,) + shape, dtype=dtype, device=device)

                        # Test forward
                        D = mr.linop.FiniteDifference(shape, axes=axes)
                        G = sp.linop.FiniteDifference(shape, axes=axes)

                        npt.assert_equal(D.ishape, G.ishape)
                        npt.assert_equal(D.oshape, G.oshape)

                        y = sp.to_device(D * x)
                        y0 = sp.to_device(G * x)
                        npt.assert_array_equal(y, y0)

                        # Test adjoint through forward
                        npt.assert_equal(D.H.ishape, G.H.ishape)
                        npt.assert_equal(D.H.oshape, G.H.oshape)

                        y = sp.to_device(D.H * z)
                        y0 = sp.to_device(G.H * z)
                        npt.assert_array_equal(y, y0)

                        # Test adjoint_adjoint through forward
                        npt.assert_equal(D.H.H.ishape, G.H.H.ishape)
                        npt.assert_equal(D.H.H.oshape, G.H.H.oshape)

                        y = sp.to_device(D.H.H * x)
                        y0 = sp.to_device(G.H.H * x)
                        npt.assert_array_equal(y, y0)

                        # Test adjoint
                        DH = mr.linop.FiniteDifferenceAdjoint(shape,
                                                              axes=axes)
                        GH = G.H

                        npt.assert_equal(DH.ishape, GH.ishape)
                        npt.assert_equal(DH.oshape, GH.oshape)

                        y = sp.to_device(DH * z)
                        y0 = sp.to_device(GH * z)
                        npt.assert_array_equal(y, y0)

                        # Test adjoint through adjoint
                        npt.assert_equal(DH.H.ishape, GH.H.ishape)
                        npt.assert_equal(DH.H.oshape, GH.H.oshape)

                        y = sp.to_device(DH.H * x)
                        y0 = sp.to_device(GH.H * x)
                        npt.assert_array_equal(y, y0)

                        # Test adjoint_adjoint through adjoint
                        npt.assert_equal(DH.H.H.ishape, GH.H.H.ishape)
                        npt.assert_equal(DH.H.H.oshape, GH.H.H.oshape)

                        y = sp.to_device(DH.H.H * z)
                        y0 = sp.to_device(GH.H.H * z)
                        npt.assert_array_equal(y, y0)
