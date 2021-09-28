import numpy as np


def cardiac(triggers, nt, times, kspace, traj, angles=None, dcf=None):
    """Sorts data into cardiac phases.

    Args:
        triggers (array): 1D array containing cardiac trigger times (ms).
        nt (int): Number of cardiac phases to sort into.
        times (array): 1D array containing the time stamps for each line of
            k-space.
        kspace (array): Shape (ncoils, na, ns).
        traj (array): Shape (na, ns, ndim).
        angles (array): Shape (na, ...).
        dcf (array): Shape (na, ns).

    Returns:
        kspace_t (list): Length `nt`.
            Each item in the list is an array with shape (ncoils, na_t, ns).
        traj_t (list): Length `nt`.
            Each item in the list is an array with shape (na_t, ns, ndim).
        angles_t (list or None): Length `nt`.
            Each item in the list is an array with shape (na_t, ...).
        dcf_t (list or None): Length `nt`.
            Each item in the list is an array with shape (na_t, ns).
    """
    # Create bin edges
    edges = [triggers[0]]  # Very first edge
    for p in range(1, len(triggers)):
        # Edges for this RR interval
        e = list(np.linspace(triggers[p - 1], triggers[p], nt + 1))
        edges += e[1:]  # Exclude first edge

    # Find bin indices
    inds = np.digitize(times, edges)
    inds -= 1  # Start indices from 0 instead of 1
    inds %= nt

    kspace_t = []
    traj_t = []
    angles_t = [] if angles is not None else None
    dcf_t = [] if dcf is not None else None

    for t in range(nt):
        inds_t = (inds == t)
        kspace_t.append(kspace[:, inds_t, :])
        traj_t.append(traj[inds_t])
        if angles is not None:
            angles_t.append(angles[inds_t])

        if dcf is not None:
            dcf_t.append(dcf[inds_t])

    return kspace_t, traj_t, angles_t, dcf_t


def cardiac_vel(triggers, nt, times, kspace, traj, angles=None, dcf=None):
    """Sorts velocity encoded data into cardiac phases.

    Args:
        triggers (array): 1D array containing cardiac trigger times (ms).
        nt (int): Number of cardiac phases to sort into.
        times (array): 1D array containing the time stamps for each line of
            k-space.
        kspace (array): Shape (ncoils, nv, na, ns).
        traj (array): Shape (nv, na, ns, ndim).
        angles (array): Shape (nv, na, ...).
        dcf (array): Shape (nv, na, ns).

    Returns:
        kspace_vt (list): List of lists. List 'shape' is (nv, nt).
            Each item is an array with shape (ncoils, na_t, ns).
        traj_vt (list): List of lists. List 'shape' is (nv, nt).
            Each item is an array with shape (na_t, ns, ndim).
        angles_vt (list or None): List of lists. List 'shape' is (nv, nt).
            Each item is an array with shape (na_t, ...).
        dcf_vt (list or None): List of lists. List 'shape' is (nv, nt).
            Each item is an array with shape (na_t, ns).
    """
    nv = kspace.shape[1]
    times = np.reshape(times, (-1, nv))

    kspace_vt = []
    traj_vt = []
    angles_vt = [] if angles is not None else None
    dcf_vt = [] if dcf is not None else None

    for v in range(nv):
        angles_v = angles[v] if angles is not None else None
        dcf_v = dcf[v] if dcf is not None else None
        kspace_t, traj_t, angles_t, dcf_t = \
            cardiac(triggers, nt, times[:, v], kspace[:, v], traj[v], angles_v,
                    dcf_v)
        kspace_vt.append(kspace_t)
        traj_vt.append(traj_t)
        if angles is not None:
            angles_vt.append(angles_t)

        if dcf is not None:
            dcf_vt.append(dcf_t)

    return kspace_vt, traj_vt, angles_vt, dcf_vt


def sim_heartbeats(acq_time, rr_base=400, std=7, delta=0.1):
    """Simulates heart beats using a bounded random walk.

    The length of each step (heart beat) is normally distributed.

    References:
        M. S. Jansz et al. (2010). Metric Optimized Gating for Fetal Cardiac
        MRI. Magnetic Resonance in Medicine, 64(5), 1304-1314.

    Args:
        acq_time (float): Duration of data acquisition (ms).
        rr_base (float): Baseline RR interval (ms).
        std (float): Standard deviation of the RR intervals (ms).
        delta (float): Strength of the bias bounding the walk around the
            baseline value. Should be a value between 0 and 1.

    Returns:
        triggers (array): Trigger times (e.g. times at R-waves) (ms).
    """
    rr = []  # List to hold sequence of RR intervals
    rr.append(rr_base + np.random.normal(0, std))  # First RR value
    elapsed_time = rr[0]  # Holds elapsed time in the simulation (ms)
    triggers = [0, rr[0]]  # The trigger times (ms)
    b = 1  # Index for current beat
    while elapsed_time < acq_time:  # Ensures the heart beats longer than acq
        # Mean of the normally distributed step length
        mean = delta * (rr_base - rr[b - 1])
        # Random walk and append to list of RR intervals
        rr.append(rr[b - 1] + np.random.normal(mean, std))
        elapsed_time = elapsed_time + rr[b]
        triggers.append(elapsed_time)
        b = b + 1

    triggers = np.array(triggers, dtype=np.float64)
    return triggers
