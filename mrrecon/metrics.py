import numpy as np


def electrostatic_potential(points):
    """Calculates the electrostatic potential.

    Args:
        points (array): Array with shape (num_points, n), where n is the
            dimensionality of each point.

    Returns:
        out (float): Electrostatic potential.

    References:
        S. S. Schauman et al. (2021). The Set Increment with Limited Views
        Encoding Ratio (SILVER) Method for Optimizing Radial Sampling of
        Dynamic MRI. bioRxiv 2020.06.25.171017.

        A. Katanforoush et al. (2003). Distributing Points on the Sphere, I.
        Experimental Mathematics, 12(2), 199-209.
    """
    out = 0

    # Loop over all points, excluding the last one
    for n, point in enumerate(points[:-1]):
        x = point - points[(n+1):]
        x = 1 / np.linalg.norm(x, ord=2, axis=1)
        out += x.sum()

    return out
