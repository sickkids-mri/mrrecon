from pathlib import Path
import os
import shutil
import random
import uuid
import math
import datetime

import numpy as np

import pydicom

import mrrecon as mr


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
    d['fovx_shift'] = d['fovx_shift'].item()
    d['fovy_shift'] = d['fovy_shift'].item()
    d['fovz_shift'] = d['fovz_shift'].item()
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
    d['recon_pos'] = d['recon_pos'][0]

    d['rot_quat'] = d['rot_quat'][0]

    d['venc'] = d['venc'].item()
    d['weight'] = d['weight'].item()

    d['height'] = d.get('height', None)
    if d['height'] is not None:
        d['height'] = d['height'].item()

    d['vendor'] = d['vendor'].item()
    d['protocol_name'] = d['protocol_name'].item()
    d['patient_orientation'] = d['patient_orientation'].item()

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


def write_4d_flow_dicoms(img, data, outdir, save_as_unique_study=True,
                         use_this_study_id=None):
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
            own study in the study list in cvi42. If False, the study ID from
            the dummy dicoms will be used.
        use_this_study_id (string): If a value is provided, it will be set as
            the dicom StudyInstanceUID (only if `save_as_unique_study` is
            True). The randomly generated StudyInstanceUID will not be used.
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

        ds.NominalInterval = int(round(data['rr_avg']))
        ds.CardiacNumberOfImages = nt
        ds.Rows = ny
        ds.Columns = nx
        ds.PixelSpacing = [np.round(dy, 3), np.round(dx, 3)]
        ds.PercentSampling = 100
        ds.PercentPhaseFieldOfView = fovy / fovx * 100
        ds.SliceThickness = np.round(dz, 1)
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
            if use_this_study_id is not None:
                ds.StudyInstanceUID = use_this_study_id

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
        ds.SoftwareVersions = 'syngo MR E11'
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

        R = mr.utils.rot_from_quat(data['rot_quat'])

        if orientation == 'Tra':
            Tra_inc = data['slice_normal']['dTra']
            ds.ImageOrientationPatient[:] = [1, 0, 0, 0, 1, 0]
        elif orientation == 'Sag':
            Sag_inc = data['slice_normal']['dSag']
            ds.ImageOrientationPatient[:] = [0, 1, 0, 0, 0, 1]
        elif orientation == 'Cor':
            Cor_inc = data['slice_normal']['dCor']
            ds.ImageOrientationPatient[:] = [1, 0, 0, 0, 0, 1]

        imPos = data['recon_pos']

        imPos_edge = (imPos - fovy / 2 * R[:, 0] + fovx / 2 * R[:, 1]
                      - fovz / 2 * np.array([Sag_inc, Cor_inc, Tra_inc]))

        frame_array = np.arange(0, ds.NominalInterval, ds.NominalInterval / nt)

        for iframe in range(nt):
            ds.TriggerTime = frame_array[iframe]
            dt = datetime.datetime.now()
            timeStr = dt.strftime('%H%M%S.%f')
            ds.InstanceCreationTime = timeStr
            ds.ContentTime = timeStr

            for islice in range(nz):
                imPos_slice = imPos_edge + dz * islice * np.array([Sag_inc, Cor_inc, Tra_inc])
                ds.SliceLocation = np.round(imPos_slice[-1], 3)
                ds.ImagePositionPatient = np.ravel(imPos_slice).tolist()
                ds[(0x0019, 0x1015)].value[:] = imPos_slice.tolist()
                ds.InstanceNumber = iframe * nz + islice + 1

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
                ds.SOPInstanceUID = pydicom.uid.generate_uid()
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


def generate_study_id():
    org_root = 11235813
    suffix1 = datetime.datetime.now().strftime('%Y%m%d')
    suffix2 = random.randint(10000000000000, 99999999999999)  # 14 digit
    StudyInstanceUID = f'{org_root}.300000{suffix1}{suffix2}'
    return StudyInstanceUID


def generate_series_id():
    org_root = 11235813
    suffix1 = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    suffix2 = random.randint(10000000000000, 99999999999999)  # 14 digit
    SeriesInstanceUID = f'{org_root}.{suffix1}{suffix2}'
    return SeriesInstanceUID


def write_2d_flow_dicoms(img, data, outdir, StudyInstanceUID=None,
                         SeriesInstanceUID=None):
    """Creates dicom files for 2D phase contrast images.

    Code was initially generated by pydicom's code_file function, which was
    provided example 2D phase contrast dicoms exported from cvi42. The code
    generated by code_file was then modified and turned into this function.

    Note: This function still first loads a dummy dicom for the magnitude
    dicoms and a dummy dicom for the phase dicoms. Writing dicoms from scratch
    completely currently does not work, for some unknown reason. Dicoms created
    work in Segment but not in cvi42.

    Args:
        img (array): Array with shape (nv, nt, ny, nx). `nv` must have a
            value of 2. `img[0]` should be the magnitude image and `img[1]`
            should be the phase image. Datatype should be uint16.
        data (dict): Output dictionary from the reconstruction pipeline.
        outdir (str): Folder where dicoms will be saved.
    """
    # Make these imports local because the names are a bit general
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.sequence import Sequence

    # Create directories
    outdir = Path(outdir)
    subdirs = [outdir / 'mag', outdir / 'phase']
    for subdir in subdirs:
        if subdir.is_dir():
            shutil.rmtree(subdir)
        subdir.mkdir(parents=True)

    # Local naming of some values
    yyyymmdd = data['acquisition_date']
    MRImageStorage = '1.2.840.10008.5.1.4.1.1.4'
    venc = data['venc']
    rr_avg = int(round(data['rr_avg']))

    if StudyInstanceUID is None:
        StudyInstanceUID = generate_study_id()
    if SeriesInstanceUID is None:
        SeriesInstanceUID = generate_series_id()

    nv, nt, ny, nx = img.shape
    # nx is the number of columns
    # ny is the number of rows
    dx = data['dx']  # Column spacing
    dy = data['dy']  # Row spacing
    dz = data['dz']  # Slice thickness

    fovx = dx * nx
    fovy = dy * ny
    fovz = dz

    # Calculate time stamps for each frame
    dt = rr_avg / nt
    timestamps = np.arange(nt) * dt

    # Directory of dummy dicoms
    dummydir = Path(__file__).parent / 'pc2d_dummydicoms'

    SOPInstanceUID = 0  # This should be different for every dicom
    for v in [0, 1]:
        if v == 0:
            ds = pydicom.dcmread(dummydir / 'mag_frame1.dcm')
        elif v == 1:
            ds = pydicom.dcmread(dummydir / 'phase_frame1.dcm')

        for t in range(nt):
            SOPInstanceUID = SOPInstanceUID + 1

            # File meta info data elements
            file_meta = FileMetaDataset()
            file_meta.FileMetaInformationGroupLength = 196
            file_meta.FileMetaInformationVersion = b'\x00\x01'
            file_meta.MediaStorageSOPClassUID = MRImageStorage
            file_meta.MediaStorageSOPInstanceUID = str(SOPInstanceUID)
            # I shouldn't need to change the following three...
            file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'
            file_meta.ImplementationClassUID = '1.3.6.1.4.1.53684.1.0.3.6.5'
            file_meta.ImplementationVersionName = 'CVI42_DCMTK_365'

            # Main data elements
            # ds = Dataset()
            ds.SpecificCharacterSet = 'ISO_IR 100'

            if v == 0:
                ds.ImageType = ['ORIGINAL', 'PRIMARY', 'M', 'RETRO', 'NORM',
                                'DIS2D']
            elif v == 1:
                ds.ImageType = ['DERIVED', 'PRIMARY', 'P', 'RETRO', 'DIS2D']

            ds.InstanceCreationDate = yyyymmdd
            ds.InstanceCreationTime = \
                datetime.datetime.now().strftime('%H%M%S.%f')

            ds.SOPClassUID = MRImageStorage
            ds.SOPInstanceUID = str(SOPInstanceUID)

            ds.StudyDate = yyyymmdd
            ds.SeriesDate = yyyymmdd
            ds.AcquisitionDate = yyyymmdd
            ds.ContentDate = yyyymmdd

            # Don't think these times matter
            ds.StudyTime = data['acquisition_time']
            ds.SeriesTime = data['acquisition_time']
            ds.AcquisitionTime = data['acquisition_time']
            ds.ContentTime = ds.InstanceCreationTime

            ds.AccessionNumber = ''
            ds.Modality = 'MR'
            ds.Manufacturer = data['vendor']
            ds.InstitutionName = ''
            ds.InstitutionAddress = ''
            ds.ReferringPhysicianName = ''
            ds.StationName = ''

            ds.StudyDescription = ''
            if v == 0:
                ds.SeriesDescription = data['protocol_name']
            if v == 1:
                ds.SeriesDescription = data['protocol_name'] + '_P'

            ds.InstitutionalDepartmentName = ''
            ds.PerformingPhysicianName = ''

            ds.ManufacturerModelName = data['systemmodel']

            # Referenced Image Sequence
            refd_image_sequence = Sequence()
            ds.ReferencedImageSequence = refd_image_sequence

            # Referenced Image Sequence: Referenced Image 1
            refd_image1 = Dataset()
            refd_image1.ReferencedSOPClassUID = MRImageStorage
            refd_image1.ReferencedSOPInstanceUID = '1.3.12.2.1107.5.2.43.67090.2021102911551153824202133'
            refd_image_sequence.append(refd_image1)

            # Referenced Image Sequence: Referenced Image 2
            refd_image2 = Dataset()
            refd_image2.ReferencedSOPClassUID = MRImageStorage
            refd_image2.ReferencedSOPInstanceUID = '1.3.12.2.1107.5.2.43.67090.2021102911480673819900359'
            refd_image_sequence.append(refd_image2)

            # Referenced Image Sequence: Referenced Image 3
            refd_image3 = Dataset()
            refd_image3.ReferencedSOPClassUID = MRImageStorage
            refd_image3.ReferencedSOPInstanceUID = '1.3.12.2.1107.5.2.43.67090.2021102912014086351903798'
            refd_image_sequence.append(refd_image3)

            ds.DerivationDescription = ''

            """
            Is this needed?
            """
            # Derivation Code Sequence
            derivation_code_sequence = Sequence()
            ds.DerivationCodeSequence = derivation_code_sequence

            # Derivation Code Sequence: Derivation Code 1
            derivation_code1 = Dataset()
            derivation_code1.CodeValue = '121327'
            derivation_code1.CodingSchemeDesignator = 'DCM'
            derivation_code1.CodeMeaning = 'Full fidelity image'
            derivation_code_sequence.append(derivation_code1)

            ds.PatientName = ''
            ds.PatientID = ''
            ds.PatientBirthDate = ''
            ds.PatientSex = ''
            ds.PatientAge = ''
            ds.PatientSize = data.get('height', None)
            if ds.PatientSize is not None:
                ds.PatientSize = ds.PatientSize / 1000  # Convert mm to m
            ds.PatientWeight = data['weight']
            ds.BodyPartExamined = ''
            ds.ScanningSequence = 'GR'
            ds.SequenceVariant = 'SP'
            ds.ScanOptions = ''
            ds.MRAcquisitionType = '2D'

            if v == 0:
                ds.SequenceName = '*fl2d1r2'
            elif v == 1:
                ds.SequenceName = f'*fl2d1_v{int(venc)}in'
                # I think it should be converted to integer here

            ds.AngioFlag = 'N'
            ds.SliceThickness = dz
            ds.RepetitionTime = data['tr']
            ds.EchoTime = data['te']
            ds.NumberOfAverages = ''
            ds.ImagingFrequency = ''
            ds.ImagedNucleus = ''
            ds.EchoNumbers = ''
            ds.MagneticFieldStrength = ''
            ds.NumberOfPhaseEncodingSteps = ''
            ds.EchoTrainLength = ''
            ds.PercentSampling = ''
            ds.PercentPhaseFieldOfView = ''
            ds.PixelBandwidth = ''
            ds.DeviceSerialNumber = ''
            ds.SoftwareVersions = 'syngo MR E11'

            # Same as SeriesDescription without _P
            ds.ProtocolName = data['protocol_name']

            ds.TriggerTime = timestamps[t]  # Different for each frame
            ds.NominalInterval = rr_avg
            ds.CardiacNumberOfImages = nt

            ds.TransmitCoilName = None
            ds.AcquisitionMatrix = None
            ds.InPlanePhaseEncodingDirection = None

            ds.FlipAngle = data['flipangle']

            ds.VariableFlipAngleFlag = None
            ds.SAR = None
            ds.dBdt = None

            ds.PatientPosition = data['patient_orientation']

            ds.StudyInstanceUID = StudyInstanceUID
            ds.SeriesInstanceUID = SeriesInstanceUID

            ds.StudyID = '1'
            ds.SeriesNumber = f'{v+1}'
            ds.AcquisitionNumber = '1'

            ds.InstanceNumber = f'{t+1}'  # Frame number (count from 1)

            ds.ImagePositionPatient = [0, 0, 0]
            ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]

            ds.FrameOfReferenceUID = None  # Not sure what this should be
            ds.PositionReferenceIndicator = ''  # Not sure what this should be

            ds.SliceLocation = 0

            ds.ImageComments = ''

            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = 'MONOCHROME2'

            ds.Rows = ny
            ds.Columns = nx
            ds.PixelSpacing = [dy, dx]

            ds.BitsAllocated = 16
            ds.BitsStored = 12
            ds.HighBit = 11
            ds.PixelRepresentation = 0

            if v == 0:
                ds.SmallestImagePixelValue = 0
                ds.LargestImagePixelValue = 4096
                ds.WindowCenter = f'{ds.LargestImagePixelValue / 2}'
                ds.WindowWidth = f'{ds.LargestImagePixelValue}'
            elif v == 1:
                # Couldn't really make sense of the values from dummy dicom
                ds.SmallestImagePixelValue = 0
                ds.LargestImagePixelValue = 4096
                ds.WindowCenter = '0'
                ds.WindowWidth = '4096'
                ds.RescaleIntercept = '-4096.0'
                ds.RescaleSlope = '2.0'
                ds.RescaleType = 'US'

            ds.WindowCenterWidthExplanation = 'Algo1'

            ds.PerformedProcedureStepStartDate = None
            ds.PerformedProcedureStepStartTime = None
            ds.PerformedProcedureStepID = None
            ds.PerformedProcedureStepDescription = None
            ds.CommentsOnThePerformedProcedureStep = ''

            ds.PixelData = img[v, t].tobytes()

            # pydicom stuff I believe
            ds.file_meta = file_meta
            ds.is_implicit_VR = False
            ds.is_little_endian = True

            savename = f'{subdirs[v]}/frame_{t+1}.dcm'
            ds.save_as(savename)

    return
