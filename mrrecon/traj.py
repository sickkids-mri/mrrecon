import numpy as np


def half_spoke(n):
    """Calculates the radial coordinates of a half spoke.

    Args:
        n (int): Number of elements along one side of the image.

    Returns:
        r (array): Radial coordinates of the half spoke.
    """
    r = np.arange(np.ceil(n / 2) + 1)
    return r


def full_spoke(n):
    """Calculates the radial coordinates of a full spoke.

    The first and last point of the spoke should be the same distance from the
    origin. Also, the full spoke should go out to the same distance as the half
    spoke. This helps simplify the analysis.

    Args:
        n (int): Number of elements along one side of the image.

    Returns:
        r (array): Radial coordinates of the full spoke.
    """
    r = np.arange(n + 1 + n % 2) - np.ceil(n / 2)
    return r


def radial_3d(r, angles, dtype=np.float64):
    """Calculates a 3D radial trajectory with the given angles.

    Args:
        r (array): 1D array. Radial coordinates of one spoke.
        angles (array): Polar and azimuthal angles for each spoke.
            Shape (na, 2). The first column contains the polar angles. The
            second column contains the azimuthal angles. Radians.

    Returns:
        traj (array): Trajectory with shape (na, ns, 3). traj[..., 0]
            corresponds to the x-axis, traj[..., 1] corresponds to the y-axis,
            and traj[..., 2] corresponds to the z-axis.
    """
    na = angles.shape[0]
    ns, = r.shape
    thetas, phis = angles[:, 0], angles[:, 1]

    traj = np.zeros((na, ns, 3), dtype=dtype)
    for a in range(na):
        theta, phi = thetas[a], phis[a]
        # Convert from spherical coordinates to Cartesian coordinates
        traj[a, :, 0] = r * np.sin(theta) * np.cos(phi)
        traj[a, :, 1] = r * np.sin(theta) * np.sin(phi)
        traj[a, :, 2] = r * np.cos(theta)

    return traj
