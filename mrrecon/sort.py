import numpy as np


def cardiac(triggers, nt, times, kspace, traj, angles=None, dcf=None):
    """Sorts data into cardiac phases.

    Args:
        triggers (array): 1D array containing cardiac trigger times (ms).
        nt (int): Number of cardiac phases to sort into.
        times (array): 1D array containing the time stamps for each line of
            k-space. Shape (na,).
        kspace (array): Shape (ncoils, na, ns).
        traj (array): Shape (na, ns, ndim).
        angles (array or None): Shape (na, ...).
        dcf (array or None): Shape (na, ns).

    Returns:
        times_t (list): Length `nt`.
            Each item in the list is an array with shape (na_t,).
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

    times_t = []
    kspace_t = []
    traj_t = []
    angles_t = [] if angles is not None else None
    dcf_t = [] if dcf is not None else None

    for t in range(nt):
        inds_t = (inds == t)
        times_t.append(times[inds_t])
        kspace_t.append(kspace[:, inds_t, :])
        traj_t.append(traj[inds_t])
        if angles is not None:
            angles_t.append(angles[inds_t])

        if dcf is not None:
            dcf_t.append(dcf[inds_t])

    return times_t, kspace_t, traj_t, angles_t, dcf_t


def vel_card(triggers, nt, times, kspace, traj, angles=None, dcf=None):
    """Sorts velocity encoded data into cardiac phases.

    Args:
        triggers (array): 1D array containing cardiac trigger times (ms).
        nt (int): Number of cardiac phases to sort into.
        times (array): 1D array containing the time stamps for each line of
            k-space. Shape (na * nv,).
        kspace (array): Shape (ncoils, nv, na, ns).
        traj (array): Shape (nv, na, ns, ndim).
        angles (array or None): Shape (nv, na, ...).
        dcf (array or None): Shape (nv, na, ns).

    Returns:
        times_vt (list): List of lists. List 'shape' is (nv, nt).
            Each item is an array with shape (na_t,).
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

    times_vt = []
    kspace_vt = []
    traj_vt = []
    angles_vt = [] if angles is not None else None
    dcf_vt = [] if dcf is not None else None

    for v in range(nv):
        angles_v = angles[v] if angles is not None else None
        dcf_v = dcf[v] if dcf is not None else None
        times_t, kspace_t, traj_t, angles_t, dcf_t = \
            cardiac(triggers, nt, times[:, v], kspace[:, v], traj[v], angles_v,
                    dcf_v)
        times_vt.append(times_t)
        kspace_vt.append(kspace_t)
        traj_vt.append(traj_t)
        if angles is not None:
            angles_vt.append(angles_t)

        if dcf is not None:
            dcf_vt.append(dcf_t)

    return times_vt, kspace_vt, traj_vt, angles_vt, dcf_vt


def calc_resp_edges(resp_sig, nresp):
    """Calculates respiratory bin edges.

    This may include flipping the respiratory signal so that respiratory phases
    are ordered from max inspiration to max expiration. Estimation of which
    part of the signal corresponds to inspiration/expiration may be incorrect.

    Args:
        resp_sig (array): 1D array containing the respiratory signal. The
            respiratory signal should have length `na`.
        nresp (int): Number of respiratory phases to sort into.

    Returns:
        resp_edges (array): 1D array with shape (nresp + 1,).
        resp_sig (array): Returns the respiratory signal but it may have been
            flipped. Use this respiratory signal (rather than the input
            respiratory signal) with the returned bin edges.
    """
    mean = resp_sig.mean()

    # Calculate initial fine histogram
    num_small_bins = nresp * 4
    hist, edges = np.histogram(resp_sig, bins=num_small_bins)
    # Calculate bin centres
    centres = (edges[:-1] + edges[1:]) / 2

    # Determine first and last resp phase bin centres
    # Assume max expiration has the most data
    max_expiration = centres[np.argmax(hist)]
    last = max_expiration
    half_ind = int(num_small_bins / 2)
    if last < mean:  # If negative:
        first = centres[half_ind:][np.argmax(hist[half_ind:])]
    else:  # Else positive
        first = centres[:half_ind][np.argmax(hist[:half_ind])]

    # Make the first respiratory bin be negative, so that the order of the
    # respiratory phases after np.digitize is max inspiration to max expiration
    if first > mean:
        resp_sig = -resp_sig  # Flip respiratory signal
        first = -first
        last = -last

    # Respiratory bin width (width on signal axis)
    bin_width = np.abs(last - first) / (nresp - 1)

    # Create bin edges
    limits = [first - bin_width / 2, last + bin_width / 2]
    resp_edges = np.linspace(limits[0], limits[1], num=nresp+1)
    return resp_edges, resp_sig


def respiratory(resp_sig, resp_edges, times, kspace, traj,
                angles=None, dcf=None):
    """Sorts data into respiratory phases.

    Args:
        resp_sig (array): 1D array containing the respiratory signal. The
            respiratory signal should have length `na`.
        resp_edges (array): 1D array with shape (nresp + 1,).
            Respiratory bin edges.
        times (array): 1D array containing the time stamps for each line of
            k-space. Shape (na,).
        kspace (array): Shape (ncoils, na, ns).
        traj (array): Shape (na, ns, ndim).
        angles (array or None): Shape (na, ...).
        dcf (array or None): Shape (na, ns).

    Returns:
        times_r (list): Length `nresp`.
            Each item in the list is an array with shape (na_r,).
        kspace_r (list): Length `nresp`.
            Each item in the list is an array with shape (ncoils, na_r, ns).
        traj_r (list): Length `nresp`.
            Each item in the list is an array with shape (na_r, ns, ndim).
        angles_r (list or None): Length `nresp`.
            Each item in the list is an array with shape (na_r, ...).
        dcf_r (list or None): Length `nresp`.
            Each item in the list is an array with shape (na_r, ns).
    """
    nresp = len(resp_edges) - 1

    inds = np.digitize(resp_sig, resp_edges)
    inds -= 1  # Start indices from 0 instead of 1
    # Data falling outside the bin range will have indices of -1 and `nresp`.
    # Sort those data into corresponding nearest bin
    inds[inds == -1] = 0
    inds[inds == nresp] == nresp - 1

    times_r = []
    kspace_r = []
    traj_r = []
    angles_r = [] if angles is not None else None
    dcf_r = [] if dcf is not None else None

    for r in range(nresp):
        inds_r = (inds == r)
        times_r.append(times[inds_r])
        kspace_r.append(kspace[:, inds_r, :])
        traj_r.append(traj[inds_r])
        if angles is not None:
            angles_r.append(angles[inds_r])

        if dcf is not None:
            dcf_r.append(dcf[inds_r])

    return times_r, kspace_r, traj_r, angles_r, dcf_r


def resp_card(resp_sig, resp_edges, triggers, nt, times, kspace, traj,
              angles=None, dcf=None):
    """Sorts data into respiratory phases and then into cardiac phases.

    Args:
        resp_sig (array): 1D array containing the respiratory signal. The
            respiratory signal should have length `na`.
        resp_edges (array): 1D array with shape (nresp + 1,).
            Respiratory bin edges.
        triggers (array): 1D array containing cardiac trigger times (ms).
        nt (int): Number of cardiac phases to sort into.
        times (array): 1D array containing the time stamps for each line of
            k-space. Shape (na,).
        kspace (array): Shape (ncoils, na, ns).
        traj (array): Shape (na, ns, ndim).
        angles (array or None): Shape (na, ...).
        dcf (array or None): Shape (na, ns).

    Returns:
        times_rc (list): List of lists. List 'shape' is (nresp, nt).
            Each item in the list is an array with shape (na_rc,).
        kspace_rc (list): List of lists. List 'shape' is (nresp, nt).
            Each item in the list is an array with shape (ncoils, na_rc, ns).
        traj_rc (list): List of lists. List 'shape' is (nresp, nt).
            Each item in the list is an array with shape (na_rc, ns, ndim).
        angles_rc (list or None): List of lists. List 'shape' is (nresp, nt).
            Each item in the list is an array with shape (na_rc, ...).
        dcf_rc (list or None): List of lists. List 'shape' is (nresp, nt).
            Each item in the list is an array with shape (na_rc, ns).
    """
    # First sort data into respiratory phases
    times_rc, kspace_rc, traj_rc, angles_rc, dcf_rc = \
        respiratory(resp_sig, resp_edges, times, kspace, traj, angles, dcf)

    # Further sort data from each respiratory phase into cardiac phases
    nresp = len(times_rc)
    for r in range(nresp):
        angles_r = angles_rc[r] if angles is not None else None
        dcf_r = dcf_rc[r] if dcf is not None else None
        times_t, kspace_t, traj_t, angles_t, dcf_t = \
            cardiac(triggers, nt, times_rc[r], kspace_rc[r], traj_rc[r],
                    angles_r, dcf_r)

        # Replace the array with a list of arrays
        times_rc[r] = times_t
        kspace_rc[r] = kspace_t
        traj_rc[r] = traj_t
        if angles is not None:
            angles_rc[r] = angles_t
        if dcf is not None:
            dcf_rc[r] = dcf_t

    return times_rc, kspace_rc, traj_rc, angles_rc, dcf_rc


def vel_resp_card(resp_sig, resp_edges, triggers, nt, times, kspace, traj,
                  angles=None, dcf=None):
    """Sorts velocity encoded data into cardiac and respiratory phases.

    Args:
        resp_sig (array): 1D array containing the respiratory signal. The
            respiratory signal should have length `na * nv`.
        resp_edges (array): 1D array with shape (nresp + 1,).
            Respiratory bin edges.
        triggers (array): 1D array containing cardiac trigger times (ms).
        nt (int): Number of cardiac phases to sort into.
        times (array): 1D array containing the time stamps for each line of
            k-space. Shape (na * nv,).
        kspace (array): Shape (ncoils, nv, na, ns).
        traj (array): Shape (nv, na, ns, ndim).
        angles (array or None): Shape (nv, na, ...).
        dcf (array or None): Shape (nv, na, ns).

    Returns:
        times_vrc (list): List of lists of lists.
            List 'shape' is (nv, nresp, nt).
            Each item in the list is an array with shape (na_rc,).
        kspace_vrc (list): List of lists of lists.
            List 'shape' is (nv, nresp, nt).
            Each item in the list is an array with shape (ncoils, na_rc, ns).
        traj_vrc (list): List of lists of lists.
            List 'shape' is (nv, nresp, nt).
            Each item in the list is an array with shape (na_rc, ns, ndim).
        angles_vrc (list): List of lists of lists.
            List 'shape' is (nv, nresp, nt).
            Each item in the list is an array with shape (na_rc, ...).
        dcf_vrc (list): List of lists of lists.
            List 'shape' is (nv, nresp, nt).
            Each item in the list is an array with shape (na_rc, ns).
    """
    nv = kspace.shape[1]
    resp_sig = np.reshape(resp_sig, (-1, nv))
    times = np.reshape(times, (-1, nv))

    times_vrc = []
    kspace_vrc = []
    traj_vrc = []
    angles_vrc = [] if angles is not None else None
    dcf_vrc = [] if dcf is not None else None

    for v in range(nv):
        angles_v = angles[v] if angles is not None else None
        dcf_v = dcf[v] if dcf is not None else None
        times_rc, kspace_rc, traj_rc, angles_rc, dcf_rc = \
            resp_card(resp_sig[:, v], resp_edges, triggers, nt, times[:, v],
                      kspace[:, v], traj[v], angles=angles_v, dcf=dcf_v)

        times_vrc.append(times_rc)
        kspace_vrc.append(kspace_rc)
        traj_vrc.append(traj_rc)
        if angles is not None:
            angles_vrc.append(angles_rc)
        if dcf is not None:
            dcf_vrc.append(dcf_rc)

    return times_vrc, kspace_vrc, traj_vrc, angles_vrc, dcf_vrc


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
