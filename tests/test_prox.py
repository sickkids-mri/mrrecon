import numpy as np
import numpy.testing as npt

import sigpy as sp

import mrrecon as mr


def test_soft_thresh_inplace():
    shapes = [(47, 99), (20, 18, 16), (5, 6, 7, 8)]
    devices = [0]
    dtypes = [np.float32, np.complex64]

    for device in devices:
        device = sp.Device(device)
        xp = device.xp
        with device:
            for shape in shapes:
                for dtype in dtypes:
                    # Test when lamda is a scalar
                    lamda = 0.27
                    x = sp.util.randn(shape, dtype=dtype, device=device)

                    y0 = sp.thresh.soft_thresh(lamda, x)
                    y = mr.prox.soft_thresh_inplace(lamda, x)

                    y0, y = sp.to_device(y0), sp.to_device(y)
                    npt.assert_array_equal(y, y0)

                    x = sp.to_device(x)
                    npt.assert_array_equal(x, y)  # Test op was in-place

                    # Test when lamda is an array
                    lamda = sp.util.randn(
                        shape, dtype=np.float32, device=device)
                    x = sp.util.randn(shape, dtype=dtype, device=device)

                    y0 = sp.thresh.soft_thresh(lamda, x)
                    y = mr.prox.soft_thresh_inplace(lamda, x)

                    y0, y = sp.to_device(y0), sp.to_device(y)
                    npt.assert_array_equal(y, y0)

                    x = sp.to_device(x)
                    npt.assert_array_equal(x, y)  # Test op was in-place


test_soft_thresh_inplace()
