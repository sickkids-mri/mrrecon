import numpy as np

import sigpy as sp


def soft_thresh_inplace(lamda, x):
    """Soft thresholds in-place.

    Args:
        lamda (float or array): Threshold parameter.
        x (array): Input array.
    """
    device = sp.get_device(x)
    xp = device.xp
    if xp == np:
        raise NotImplementedError
    else:
        if np.isscalar(lamda):
            lamda = sp.to_device(lamda, device)

        _soft_thresh_inplace(lamda, x)

    return x


if sp.config.cupy_enabled:
    import cupy as cp

    _soft_thresh_inplace = cp.ElementwiseKernel(
        'S lamda',
        'T x',
        """
        S abs_x = abs(x);
        T sign;
        if (abs_x == 0)
            sign = 0;
        else
            sign = x / (T) abs_x;
        S mag = abs_x - lamda;
        mag = (abs(mag) + mag) / 2.;

        x = (T) mag * sign;
        """,
        name='soft_thresh_inplace')
