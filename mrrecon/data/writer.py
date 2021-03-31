from pathlib import Path
import os
import shutil
import uuid

import numpy as np

import pydicom

import ndflow as nf


def normalize_pc(img, new_max=4096):
    """Normalizes and casts phase contrast image to uint16 for dicom writing.

    Args:
        img (float array): Phase contrast image with phase difference already
            calculated. `img[0]` should be the magnitude image, `img[1]` should
            be the first phase image, `img[2]` should be the second phase
            image, etc.
        new_max (int): Desired max of uint16 images.

    Returns:
        out (uint16 array): Normalized image with the same shape as the input.
    """
    nv = img.shape[0]
    out = np.empty_like(img, dtype=np.uint16)

    # Normalize magnitude image
    out[0] = img[0] / img[0].max() * new_max

    # Normalize phase image(s)
    for v in range(1, nv):
        out[v] = (img[v] / np.pi + 1) * (new_max / 2)

    return out


def _fix_matfile_format(d):
    """Fixes the formatting of loaded matfiles.

    Only meant to work for 4D flow matfiles.

    Everything in matfiles get turned into arrays. This function tries to take
    things out of arrays when appropriate.
    """
    d['dx'] = d['dx'].item()
    d['dy'] = d['dy'].item()
    d['dz'] = d['dz'].item()
    d['fovx'] = d['fovx'].item()
    d['fovy'] = d['fovy'].item()
    d['fovz'] = d['fovz'].item()
    d['rr_avg'] = d['rr_avg'].item()
    d['systemmodel'] = str(d['systemmodel'][0])
    d['acquisition_date'] = str(d['acquisition_date'][0])
    d['acquisition_time'] = str(d['acquisition_time'][0])
    d['PatientLOID'] = str(d['PatientLOID'][0])
    d['tr'] = d['tr'].item()
    d['te'] = d['te'].item()
    d['flipangle'] = d['flipangle'].item()

    k = d['slice_normal'][0][0].dtype.names[0]
    d['slice_normal'] = {k: d['slice_normal'][0][0][k].item()}

    d['slice_pos'] = np.array([d['slice_pos']['flSag'].item().item(),
                               d['slice_pos']['flCor'].item().item(),
                               d['slice_pos']['flTra'].item().item()])

    d['rot_quat'] = d['rot_quat'][0]

    d['venc'] = d['venc'].item()
    return d


def write_4d_flow_dicoms(img, data, outdir, slices_to_include=None):
    """Writes 4D flow dicoms.

    Dicoms work for the 4D flow module in cvi42.

    Args:
        img (array): Array with shape (nv, nt, nz, ny, nx). `nv` must have a
            value of 4. `img[0]` should be the magnitude image, `img[1]` should
            be the z velocity image, `img[2]` should be the x velocity image,
            and `img[3]` should be the y velocity image.
        data (dict): Output dictionary from the reconstruction pipeline.
        outdir (str): Folder where dicoms will be saved.
        slices_to_include (array): 1D array of integers indicating which slices
            should be written to dicoms.
    """
    assert img.ndim == 5, f'5D array required. Got {img.ndim}D array instead.'
    nv, nt, nz, ny, nx = img.shape
    assert nv == 4, f'4 image series required. Got {nv} instead.'

    # Transform image. TODO: Not sure if these belong here, or if they are
    # dependent on another parameter.
    # Transpose (nv, nt, nz, ny, nx) to (nv, nt, nz, nx, ny)
    img = np.transpose(img, (0, 1, 2, 4, 3))
    img = np.flip(img, axis=2)  # Flip superior-inferior axis
    # Redefine x and y. Let x correspond to left-right direction, and y
    # correspond to anterior-posterior direction
    nv, nt, nz, ny, nx = img.shape
    # Since x and y were swapped, also swap pixel spacings
    dx = data['dy']  # Column spacing
    dy = data['dx']  # Row spacing
    dz = data['dz']  # Slice spacing

    fovx = dx * nx
    fovy = dy * ny
    fovz = dz * nz

    thisdir = Path(__file__).parent

    outdir = Path(outdir)

    subdir_mag = outdir / 'mag'
    subdir_vx = outdir / 'vx'
    subdir_vy = outdir / 'vy'
    subdir_vz = outdir / 'vz'

    subdirs = [subdir_mag, subdir_vx, subdir_vy, subdir_vz]

    for subdir in subdirs:
        if subdir.is_dir():
            shutil.rmtree(subdir)
        subdir.mkdir()

    # Loop over each of the image series
    for v in range(nv):
        # Read dummy dicom file for current image series
        if v == 0:
            ds = pydicom.dcmread(thisdir / 'DummyDicoms' / '1.IMA')
        elif v == 1:
            ds = pydicom.dcmread(thisdir / 'DummyDicoms' / '2.IMA')
        elif v == 2:
            ds = pydicom.dcmread(thisdir / 'DummyDicoms' / '3.IMA')
        elif v == 3:
            ds = pydicom.dcmread(thisdir / 'DummyDicoms' / '4.IMA')

        startTime = 0
        ds.NominalInterval = int(round(data['rr_avg']))
        ds.CardiacNumberOfImages = nt
        ds.Rows = ny
        ds.Columns = nx
        ds.PixelSpacing = [dy, dx]
        ds.PercentSampling = 100
        ds.PercentPhaseFieldOfView = nx / ny * 100 #assuming square voxels here
        ds.SliceThickness = dz
        ds.NumberOfPhaseEncodingSteps = nx
        ds.AcquisitionMatrix = [0, ny, nx, nz]
        ds[(0x0051, 0x100b)].value = str(ny) + '*' + str(nx) + 's'
        ds[(0x0051, 0x100c)].value = 'FoV ' + str(fovy) + '*' + str(fovx)

        ds.ManufacturerModelName = data['systemmodel']
        ds.AcquisitionDate = data['acquisition_date']
        ds.SeriesDate = data['acquisition_date']
        ds.StudyDate = data['acquisition_date']
        ds.ContentDate = data['acquisition_date']
        ds.SeriesTime = data['acquisition_time']
        ds.StudyTime = data['acquisition_time']
        # ds.StudyInstanceUID = data['StudyLOID']
        # ds.SeriesInstanceUID = data['SeriesLOID']
        patient_name = data.get('PatientName', None)
        if patient_name is not None:
            ds['PatientName'].value = patient_name
        ds['PatientID'].value = data['PatientLOID']

        ds['RepetitionTime'].value = data['tr']
        ds['EchoTime'].value = data['te']
        ds['FlipAngle'].value = data['flipangle']

        # data['slice_normal'] is a dict with one key
        orientation = list(data['slice_normal'])[0][1:]  # Get part of the key
        ds[(0x0051, 0x100e)].value = orientation
        Sag_inc, Tra_inc, Cor_inc = 0, 0, 0

        R = nf.traj.rot_from_quat(data['rot_quat'])

        if orientation == 'Tra':
            Tra_inc = data['slice_normal']['dTra']
            ds.ImageOrientationPatient[:] = [1, 0, 0, 0, 1, 0]
        elif orientation == 'Sag':
            Sag_inc = data['slice_normal']['dSag']
            ds.ImageOrientationPatient[:] = [0, 1, 0, 0, 0, 1]
        elif orientation == 'Cor':
            Cor_inc = data['slice_normal']['dCor']
            ds.ImageOrientationPatient[:] = [1, 0, 0, 0, 0, 1]

        imPos = np.array(data['slice_pos'].tolist())

        imPos_edge = (imPos - fovy / 2 * R[:, 0] - fovx / 2 * R[:, 1]
                      - fovz / 2 * (np.array([Sag_inc, Cor_inc, Tra_inc])))

        if slices_to_include is None:
            slices_to_include = np.arange(nz)

        frame_array = np.arange(0, ds.NominalInterval, ds.NominalInterval / nt)

        for iframe in range(nt):
            ds.TriggerTime = frame_array[iframe]
            ds.InstanceCreationTime = str(startTime + frame_array[iframe]/1000)
            ds.ContentTime = str(startTime + frame_array[iframe]/1000)

            for islice in slices_to_include:
                imPos_slice = imPos_edge + dz * islice * np.array([Sag_inc, Cor_inc, Tra_inc])
                ds.SliceLocation = imPos_slice[-1]
                ds.ImagePositionPatient = np.ravel(imPos_slice).tolist()
                ds[(0x0019, 0x1015)].value[:] = imPos_slice.tolist()

                if v == 0:
                    savename = subdir_mag / f'frame{iframe}_slice{islice}.IMA'
                    ds.SeriesNumber = 1
                    ds.ImageType = ['ORIGINAL', 'PRIMARY', 'M', 'RETRO', 'DIS2D']
                    ds[(0x0051, 0x1016)].value = 'p2 M/RETRO/DIS2D'

                if v == 1:
                    ds.SequenceName = 'fl3d1_v' + str(int(data['venc'])) + 'in'
                    savename = subdir_vz / f'frame{iframe}_slice{islice}.IMA'
                    ds.SeriesNumber = 4
                    ds.ImageType = ['DERIVED', 'PRIMARY', 'P', 'RETRO', 'DIS2D']
                    ds[(0x0051, 0x1016)].value = 'p2 P/RETRO/DIS2D'

                if v == 3:
                    ds.SequenceName = 'fl3d1_v' + str(int(data['venc'])) + 'ap'
                    savename = subdir_vy / f'frame{iframe}_slice{islice}.IMA'
                    ds.SeriesNumber = 3
                    ds.ImageType = ['DERIVED', 'PRIMARY', 'P', 'RETRO', 'DIS2D']
                    ds[(0x0051, 0x1016)].value = 'p2 P/RETRO/DIS2D'

                if v == 2:
                    ds.SequenceName = 'fl3d1_v' + str(int(data['venc'])) + 'rl'
                    savename = subdir_vx / f'frame{iframe}_slice{islice}.IMA'
                    ds.SeriesNumber = 2
                    ds.ImageType = ['DERIVED', 'PRIMARY', 'P', 'RETRO', 'DIS2D']
                    ds[(0x0051, 0x1016)].value = 'p2 P/RETRO/DIS2D'

                tmpslice = img[v, iframe, islice, :, :]
                ds.PixelData = tmpslice.tobytes()
                ds.SOPInstanceUID = uuid.uuid4().hex  # Generate unique UID
                ds.save_as(savename)

    return
