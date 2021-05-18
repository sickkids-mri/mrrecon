from pathlib import Path
import os
import shutil
import random
import uuid
import math

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
    d['fovx_prescribed'] = d['fovx_prescribed'].item()
    d['fovy_prescribed'] = d['fovy_prescribed'].item()
    d['fovz_prescribed'] = d['fovz_prescribed'].item()
    d['rr_avg'] = d['rr_avg'].item()
    d['systemmodel'] = str(d['systemmodel'][0])
    d['acquisition_date'] = str(d['acquisition_date'][0])
    d['acquisition_time'] = str(d['acquisition_time'][0])
    d['StudyLOID'] = str(d['StudyLOID'][0])
    d['SeriesLOID'] = str(d['SeriesLOID'][0])
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
    d['weight'] = d['weight'].item()

    d['height'] = d.get('height', None)
    if d['height'] is not None:
        d['height'] = d['height'].item()

    return d


def invert_velocity(v, dcm_img_max=4096):
    """Inverts a velocity image that has been converted to uint16.

    Inverts the velocity direction in a phase image that has already been
    reference subtracted and converted to uint16.

    Args:
        v (uint16 array): Velocity image.
        dcm_img_max (int): Max of uint16 images.
    """
    v = dcm_img_max - v
    return v


def write_4d_flow_dicoms(img, data, outdir, save_as_unique_study=True):
    """Writes 4D flow dicoms.

    Dicoms work for the 4D flow module in cvi42.

    Args:
        img (array): Array with shape (nv, nt, nz, ny, nx). `nv` must have a
            value of 4. `img[0]` should be the magnitude image, `img[1]` should
            be the z velocity image, `img[2]` should be the x velocity image,
            and `img[3]` should be the y velocity image.
        data (dict): Output dictionary from the reconstruction pipeline.
        outdir (str): Folder where dicoms will be saved.
        save_as_unique_study (bool): Use this to make dicoms appear in their
            own study in the study list in cvi42.
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

    # Invert velocity directions
    # TODO Confirm this. Are these dependent on anything?
    img[1] = invert_velocity(img[1])  # Invert z velocities
    img[2] = invert_velocity(img[2])  # Invert x velocities

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

    if save_as_unique_study:
        # Random integer up to 28 digits (to be used to modify study ID)
        unique_study = random.randint(1, 9999999999999999999999999999)

    # Loop over each of the image series
    for v in range(nv):
        # Read dummy dicom file for current image series
        if v == 0:  # Magnitude image
            ds = pydicom.dcmread(thisdir / 'DummyDicoms' / '1.IMA')

            # Try to automatically calculate a good window centre
            window_center = _auto_window_centre(img[0])
            ds.WindowCenter = window_center
            ds.WindowWidth = 2 * window_center

        elif v == 1:  # z velocity image
            ds = pydicom.dcmread(thisdir / 'DummyDicoms' / '2.IMA')

            ds.WindowCenter = 0
            ds.WindowWidth = 4096

        elif v == 2:  # x velocity image
            ds = pydicom.dcmread(thisdir / 'DummyDicoms' / '3.IMA')

            ds.WindowCenter = 0
            ds.WindowWidth = 4096

        elif v == 3:  # y velocity image
            ds = pydicom.dcmread(thisdir / 'DummyDicoms' / '4.IMA')

            ds.WindowCenter = 0
            ds.WindowWidth = 4096

        startTime = 0
        ds.NominalInterval = int(round(data['rr_avg']))
        ds.CardiacNumberOfImages = nt
        ds.Rows = ny
        ds.Columns = nx
        ds.PixelSpacing = [dy, dx]
        ds.PercentSampling = 100
        ds.PercentPhaseFieldOfView = fovy / fovx * 100
        ds.SliceThickness = dz
        ds.NumberOfPhaseEncodingSteps = ny
        ds.AcquisitionMatrix = [0, ny, nx, nz]
        ds[(0x0051, 0x100b)].value = str(ny) + '*' + str(nx) + 's'
        ds[(0x0051, 0x100c)].value = 'FoV ' + str(fovy) + '*' + str(fovx)

        ds.Manufacturer = data['vendor']
        ds.ManufacturerModelName = data['systemmodel']
        ds.MagneticFieldStrength = None
        ds.AcquisitionDate = data['acquisition_date']
        ds.SeriesDate = data['acquisition_date']
        ds.StudyDate = data['acquisition_date']
        ds.ContentDate = data['acquisition_date']
        ds.SeriesTime = data['acquisition_time']
        ds.StudyTime = data['acquisition_time']

        if save_as_unique_study:
            # Modify last part of ID
            parts = ds.StudyInstanceUID.split('.')
            parts[-1] = str(int(parts[-1]) + unique_study)
            ds.StudyInstanceUID = '.'.join(parts)

        # ds.StudyInstanceUID = data['StudyLOID']
        # ds.SeriesInstanceUID = data['SeriesLOID']
        ds.PatientName = data.get('PatientName', 'anon nona')
        # ds['PatientID'].value = data['PatientLOID']
        ds.PatientID = None
        ds.PatientBirthDate = None
        ds.PatientSex = None
        ds.PatientAge = None
        ds.OperatorsName = None
        ds.PatientPosition = None

        ds.BodyPartExamined = None
        ds.ImageComments = None
        ds.LargestImagePixelValue = 4096
        ds.PatientSize = data.get('height', None)
        if ds.PatientSize is not None:
            ds.PatientSize = ds.PatientSize / 1000  # Convert mm to m
        ds.PatientWeight = data['weight']
        ds.PerformedProcedureStepDescription = None
        ds.PerformedProcedureStepID = None
        ds.PerformedProcedureStepStartDate = None
        ds.PerformedProcedureStepStartTime = None
        ds.PixelBandwidth = None
        ds.ProtocolName = None
        ds.SAR = None
        ds.SeriesDescription = None
        ds.SoftwareVersions = None
        ds.StationName = None
        ds.StudyDescription = None
        ds.TransmitCoilName = None

        ds.RepetitionTime = data['tr']
        ds.EchoTime = data['te']
        ds.FlipAngle = data['flipangle']

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

        imPos_edge = (imPos - fovy / 2 * R[:, 0] + fovx / 2 * R[:, 1]
                      - fovz / 2 * np.array([Sag_inc, Cor_inc, Tra_inc]))

        frame_array = np.arange(0, ds.NominalInterval, ds.NominalInterval / nt)

        for iframe in range(nt):
            ds.TriggerTime = frame_array[iframe]
            ds.InstanceCreationTime = str(startTime + frame_array[iframe]/1000)
            ds.ContentTime = str(startTime + frame_array[iframe]/1000)

            for islice in range(nz):
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


def _auto_window_centre(img):
    # Try to auto window magnitude image
    hist, bin_edges = np.histogram(img, bins=100)
    inds = bin_edges < (img.mean() * 1.25)
    inds = inds[:-1]
    hist[inds] = 0  # Try to avoid background peak
    ind = np.argmax(hist)
    window_center = bin_edges[ind + 1]
    return window_center


def crop(img, img_pos, x_inds=None, y_inds=None, z_inds=None,
         dx=None, dy=None, dz=None):
    """Crops a 3D image and calculates the new image position.

    Calculates the new image position for off-centre cropping as well.

    Args:
        img (array): Array with shape (..., nz, ny, nx).
        img_pos (array): Array with shape (3,). Position of the centre voxel in
            mm. Coordinate system should be the same as the slice position
            parameter from the Siemens twix data header.
        x_inds (tuple): 2-tuple containing indices to slice the input in x.
        y_inds (tuple): 2-tuple containing indices to slice the input in y.
        z_inds (tuple): 2-tuple containing indices to slice the input in z.
        dx (float): Pixel width in x.
        dy (float): Pixel width in y.
        dz (float): Pixel width in z.

    Returns:
        img_cropped (array): Input cropped according to the provided indices.
        img_pos_new (array): Position of the centre voxel in the cropped image
            in mm.
    """
    img_cropped = np.copy(img)
    img_pos_new = np.copy(img_pos)
    nz, ny, nx = img.shape[-3:]

    def calc_shift(num_pixels, inds):
        # Calculates shift in number of pixels
        old_centre = math.floor(num_pixels / 2)
        new_centre = math.floor((inds[1] + inds[0]) / 2)
        shift = new_centre - old_centre
        return shift

    if x_inds is not None:
        shift = calc_shift(nx, x_inds)
        img_pos_new[1] = img_pos_new[1] + shift * dx
        img_cropped = img_cropped[..., :, :, x_inds[0]:x_inds[1]]

    if y_inds is not None:
        shift = calc_shift(ny, y_inds)
        img_pos_new[0] = img_pos_new[0] + shift * dy
        img_cropped = img_cropped[..., :, y_inds[0]:y_inds[1], :]

    if z_inds is not None:
        shift = calc_shift(nz, z_inds)
        img_pos_new[2] = img_pos_new[2] - shift * dz
        # Not sure why subtracted here when others were added
        img_cropped = img_cropped[..., z_inds[0]:z_inds[1], :, :]

    return img_cropped, img_pos_new
