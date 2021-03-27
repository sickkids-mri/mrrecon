import os
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


def write_to_dicom(data, img, outdir, slices_to_include=None):
    """Writes 4D flow dicoms.

    Dicoms work for the 4D flow module in cvi42.

    Args:
        img (array): Array with shape (nv, nt, nz, ny, nx). `nv` must have a
            value of 4. `img[0]` should be the magnitude image, `img[1]` should
            be the z velocity image, `img[2]` should be the x velocity image,
            and `img[3]` should be the y velocity image.
        data (dict): Output dictionary from the reconstruction pipeline.
        outdir
        slices_to_include
    """
    assert img.ndim == 5, f'5D array required. Got {img.ndim}D array instead.'
    nv, nt, nz, ny, nx = img.shape
    assert nv == 4, f'4 image series required. Got {nv} instead.'

    thisdir = os.path.dirname(__file__)

    subdir_mag = outdir + '/I_MAG_ph'
    subdir_vx = outdir + '/I_Vx_ph'
    subdir_vy = outdir + '/I_Vy_ph'
    subdir_vz = outdir + '/I_Vz_ph'

    subdirs = [subdir_mag, subdir_vx, subdir_vy, subdir_vz]

    import shutil
    for dirstr in subdirs:
        if os.path.exists(dirstr):
            shutil.rmtree(dirstr)
        os.mkdir(dirstr)

    counter = 0
    # read in dummy dicom files for each flow encode
    for fe in range(nv):
        if fe == 0:
            ds = pydicom.dcmread(os.path.join(thisdir, 'DummyDicoms/1.IMA'))
        elif fe == 1:
            ds = pydicom.dcmread(os.path.join(thisdir, 'DummyDicoms/2.IMA'))
        elif fe == 2:
            ds = pydicom.dcmread(os.path.join(thisdir, 'DummyDicoms/3.IMA'))
        elif fe == 3:
            ds = pydicom.dcmread(os.path.join(thisdir, 'DummyDicoms/4.IMA'))

        startTime = 0
        ds.NominalInterval = data.get('rr_avg',1000)
        ds.CardiacNumberOfImages = nt
        ds.Rows = img.shape[-3]
        ds.Columns = img.shape[-2]
        ds.PixelSpacing = [data['dx'], data['dy']]
        ds.PercentSampling = 100
        ds.PercentPhaseFieldOfView = img.shape[-2] / img.shape[-3] * 100 #assuming square voxels here
        ds.SliceThickness = data['dz']
        ds.NumberOfPhaseEncodingSteps = img.shape[-2]
        ds.AcquisitionMatrix = [0, img.shape[-3], img.shape[-2], img.shape[-1]]
        ds[(0x0051, 0x100b)].value = str(img.shape[-3]) + '*' + str(img.shape[-2]) + 's'
        #ds[(0x0051, 0x100c)].value = 'FoV ' + str(data['fovx']) + '*' + str(data['fovy'])
        fovx = data['dx']*img.shape[-3]
        fovy = data['dy']*img.shape[-2]
        fovz = data['dz']*img.shape[-1]
        ds[(0x0051, 0x100c)].value = 'FoV ' + str(fovx) + '*' + str(fovy)

        ds.ManufacturerModelName = data['systemmodel']
        ds.AcquisitionDate = data['acquisition_date']
        ds.SeriesDate = data['acquisition_date']
        ds.StudyDate = data['acquisition_date']
        ds.ContentDate = data['acquisition_date']
        ds.SeriesTime = data['acquisition_time']
        ds.StudyTime = data['acquisition_time']
        # ds.StudyInstanceUID = data['StudyLOID']
        # ds.SeriesInstanceUID = data['SeriesLOID']
        patient_name = data.get('PatientName',None)
        if patient_name is not None:
            ds['PatientName'].value = patient_name
        ds['PatientID'].value = data['PatientLOID']

        ds['RepetitionTime'].value = data['tr']
        ds['EchoTime'].value = data['te']
        ds['FlipAngle'].value = data.get('flipangle',10)

        tmp = data['slice_normal']
        tmpstr = list(data['slice_normal'])[0][1::]
        ds[(0x0051, 0x100e)].value = tmpstr
        Sag_inc, Tra_inc, Cor_inc = 0, 0, 0

        R = nf.traj.rot_from_quat(data['rot_quat'])
        newR = np.matmul(R, np.array([[1, 0], [0, 1], [0, 0]]))  # take only first two columns

        if 'Tra' in tmpstr:
            Tra_inc = tmp['dTra']
            ds.ImageOrientationPatient[:] = [1, 0, 0, 0, 1, 0]
        elif 'Sag' in tmpstr:
            Sag_inc = tmp['dSag']
            ds.ImageOrientationPatient[:] = [0, 1, 0, 0, 0, 1]
        elif 'Cor' in tmpstr:
            Cor_inc = tmp['dCor']
            ds.ImageOrientationPatient[:] = [1, 0, 0, 0, 0, 1]

        imPos = np.array(data['slice_pos'].tolist())

        imPos_edge = (imPos - fovx / 2 * newR[:, 0] - fovy / 2 * newR[:, 1]
                     - fovz / 2 * (np.array([Sag_inc , Cor_inc , Tra_inc])))

        if slices_to_include is not None:
            slice_num_array = slices_to_include
            nSlices = len(slice_num_array)
        else:
            nSlices = img.shape[-1]
            slice_num_array = np.arange(nSlices)
        start_slice = slice_num_array[0]
        frame_array = np.arange(0, ds.NominalInterval, ds.NominalInterval / nt)

        for iframe in range(nt):
            ds.TriggerTime = frame_array[iframe]
            ds.InstanceCreationTime = str(startTime + frame_array[iframe]/1000)
            ds.ContentTime = str(startTime + frame_array[iframe]/1000)

            for islice in slice_num_array:
                imPos_slice = imPos_edge + data['dz'] * islice * np.array([Sag_inc , Cor_inc , Tra_inc])
                ds.SliceLocation = imPos_slice[-1]
                ds.ImagePositionPatient = np.ravel(imPos_slice).tolist()
                ds[(0x0019,0x1015)].value[:] = imPos_slice.tolist()

                if fe == 0:

                    outfilename = subdir_mag + '/im' + str(iframe) + '_' + str(islice - start_slice) + '.IMA'
                    ds.SeriesNumber = 1
                    ds.ImageType = ['ORIGINAL', 'PRIMARY', 'M', 'RETRO', 'DIS2D']
                    ds[(0x0051, 0x1016)].value = 'p2 M/RETRO/DIS2D'

                if fe == 1:
                    ds.SequenceName = 'fl3d1_v' + str(int(data['venc'])) + 'in'
                    outfilename = subdir_vz + '/im' + str(iframe) + '_' + str(islice - start_slice) + '.IMA'
                    ds.SeriesNumber = 4
                    ds.ImageType = ['DERIVED', 'PRIMARY', 'P', 'RETRO', 'DIS2D']
                    ds[(0x0051, 0x1016)].value = 'p2 P/RETRO/DIS2D'

                if fe == 3:
                    ds.SequenceName = 'fl3d1_v' + str(int(data['venc'])) + 'ap'
                    outfilename = subdir_vy + '/im' + str(iframe) + '_' + str(islice - start_slice) + '.IMA'
                    ds.SeriesNumber = 3
                    ds.ImageType = ['DERIVED', 'PRIMARY', 'P', 'RETRO', 'DIS2D']
                    ds[(0x0051, 0x1016)].value = 'p2 P/RETRO/DIS2D'

                if fe == 2:
                    ds.SequenceName = 'fl3d1_v' + str(int(data['venc'])) + 'rl'
                    outfilename = subdir_vx + '/im' + str(iframe) + '_' + str(islice - start_slice) + '.IMA'
                    ds.SeriesNumber = 2
                    ds.ImageType = ['DERIVED', 'PRIMARY', 'P', 'RETRO', 'DIS2D']
                    ds[(0x0051, 0x1016)].value = 'p2 P/RETRO/DIS2D'

                tmpslice = img[fe, iframe, :, :, islice]
                ds.PixelData = tmpslice.tobytes()   
                ds.SOPInstanceUID = uuid.uuid4().hex #generate unique UID
                ds.save_as(outfilename)

                counter += 1

    return(counter)
