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


def std_heartbeats(triggers1, triggers2):
    """Calculates the standard deviation of the difference in RR intervals.

    Calculates the standard deviation of the difference in RR intervals between
    two methods. E.g. compares the triggers obtained from self-gating and ECG.

    Args:
        triggers1 (array): 1D array containing the trigger times from one
            method.
        triggers2 (array): 1D array containing the trigger times from the
            second method. The number of triggers provided by each method
            needs to be the same for the calculation of this metric.

    Returns:
        out (float): Value of the metric, in the same units as the triggers.
    """
    assert triggers1.shape == triggers2.shape

    rr1 = np.diff(triggers1)
    rr2 = np.diff(triggers2)
    out = np.std(rr1 - rr2)
    return out
