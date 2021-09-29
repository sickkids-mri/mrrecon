import numpy as np
import numpy.testing as npt

import mrrecon as mr


def test_cardiac(num_triggers=100, tr=4.1, nt=16):
    ncoils = 1
    ns = 1

    # Random numbers
    rnd1 = np.random.uniform() * 100
    rnd2 = np.random.uniform() / 100

    # Create random trigger times
    beat_std = 17  # ms
    beat_mean = 800  # ms
    beats = np.random.randn(num_triggers) * beat_std + beat_mean
    triggers = np.cumsum(beats)
    triggers = triggers - triggers[0] + rnd1

    # K-space line time stamps
    acq_time = triggers[-1] - triggers[0]
    na = np.ceil(acq_time / tr).astype(int).item()
    times = np.arange(na) * tr + rnd1 + rnd2  # ms

    # Make the last heart beat a little bit longer
    triggers[-1] = triggers[-1] + rnd2

    # Create data arrays
    kspace = np.zeros((ncoils, na, ns), dtype=np.complex64)
    traj = np.zeros((na, ns, 3), dtype=np.float32)
    angles = np.zeros((na, 2), dtype=np.float32)
    dcf = np.zeros((na, ns), dtype=np.float32)

    # Create bin edges
    edges = np.array([triggers[0]], dtype=np.float64)
    for trigger in triggers[1:]:
        e = np.linspace(edges[-1], trigger, num=nt+1, dtype=np.float64)
        edges = np.concatenate((edges, e[1:]))

    # Calculate bin indices
    inds = np.digitize(times, edges)
    inds -= 1  # Start from 0
    inds = inds % nt

    for t in range(nt):
        kspace[:] = 0
        traj[:] = 0
        angles[:] = 0
        dcf[:] = 0

        # Set points corresponding to current cardiac phase to 1
        inds_t = inds == t
        kspace[:, inds_t, :] = 1
        traj[inds_t, :, :] = 1
        angles[inds_t, :] = 1
        dcf[inds_t, :] = 1

        kspace_t, traj_t, angles_t, dcf_t = \
            mr.sort.cardiac(triggers, nt, times, kspace, traj, angles, dcf)

        for tt in range(nt):
            # Data from current cardiac phase should be all 1s, else 0s
            if tt == t:
                npt.assert_equal(kspace_t[tt], 1)
                npt.assert_equal(traj_t[tt], 1)
                npt.assert_equal(angles_t[tt], 1)
                npt.assert_equal(dcf_t[tt], 1)
            else:
                npt.assert_equal(kspace_t[tt], 0)
                npt.assert_equal(traj_t[tt], 0)
                npt.assert_equal(angles_t[tt], 0)
                npt.assert_equal(dcf_t[tt], 0)
