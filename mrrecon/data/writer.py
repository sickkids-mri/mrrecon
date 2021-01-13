import os
thisdir = os.path.dirname(__file__)
import numpy as np
import ndflow as nf

def write_to_dicom(data, img, outdir, slices_to_include = None):
    import pydicom
    import numpy as np
    nv = img.shape[0]
    slTh = data['dz']
    if len(img.shape) < 5:
        nframes = 1
    else:
        nframes = img.shape[1]

    img_norm = normalize_pc(img)

    counter = 0
    # read in dummy dicom files for each flow encode
    for fe in np.arange(nv):
        if fe == 0:
            ds = pydicom.dcmread(os.path.join(thisdir, 'DummyDicoms/1.ima'))
        elif fe == 1:
            ds = pydicom.dcmread(os.path.join(thisdir, 'DummyDicoms/2.ima'))
        elif fe == 2:
            ds = pydicom.dcmread(os.path.join(thisdir, 'DummyDicoms/3.ima'))
        elif fe == 3:
            ds = pydicom.dcmread(os.path.join(thisdir, 'DummyDicoms/4.ima'))

        SOPInstanceUID_str = ds.SOPInstanceUID
        startTime = 0
        #timeres = 0.1 #dummy value for now. TODO: change to RR_avg/nframes
        #ds.NominalInterval = nframes * timeres * 1000 #TODO: change to RR_avg
        ds.NominalInterval = data.get('rr_avg',1000)
        ds.CardiacNumberOfImages = nframes
        ds.Rows = img.shape[-3]
        ds.Columns = img.shape[-2]
        ds.Width = img.shape[-3]
        ds.Height = img.shape[-2]
        ds.PixelSpacing = [data['dx'], data['dy']]
        ds.PercentSampling = 100
        ds.PercentPhaseFieldOfView = img.shape[-2] / img.shape[-3] * 100 #assuming square voxels here
        ds.SliceThickness = data['dz']
        ds.NumberOfPhaseEncodingSteps = img.shape[-2]
        ds.AcquisitionMatrix = [0, img.shape[-3], img.shape[-2], img.shape[-1]]
        ds[(0x0051, 0x100b)].value = str(img.shape[-3]) + '*' + str(img.shape[-2]) + 's'
        ds[(0x0051, 0x100c)].value = 'FoV ' + str(data['fovx']) + '*' + str(data['fovy'])

        tmp = data['slice_normal']
        tmpstr = list(data['slice_normal'])[0][1::]
        ds[(0x0051, 0x100e)].value = tmpstr
        Sag_inc, Tra_inc, Cor_inc = 0, 0, 0
        if 'Tra' in tmpstr:
            Tra_inc = tmp['dTra']
        elif 'Sag' in tmpstr:
            Sag_inc = tmp['dSag']
        elif 'Cor' in tmpstr:
            Cor_inc = tmp['dCor']

        R = nf.traj.rot_from_quat(data['rot_quat'])
        newR = np.matmul(R , np.array([[1,0],[0,1],[0,0]])) #take only first two columns
        ds.ImageOrientationPatient[:] = np.ravel(np.transpose(newR)).tolist()

        imPos = data['slice_pos']
        imPos_edge = (imPos - data['fovx'] / 2 * newR[:, 0] - data['fovy'] / 2 * newR[:, 1]
                     - data['fovz'] / 2 * (np.array([Sag_inc , Cor_inc , Tra_inc])))

        if slices_to_include is not None:
            slice_num_array = slices_to_include
            nSlices = len(slice_num_array)
        else:
            nSlices = img.shape[-1]
            slice_num_array = np.arange(nSlices)
        start_slice = slice_num_array[0]
        frame_array = np.arange(0, ds.NominalInterval, ds.NominalInterval/nframes)

        for iframe in np.arange(nframes):
            ds.TriggerTime = frame_array[iframe]
            ds.InstanceCreationTime = str(startTime + frame_array[iframe]/1000)
            ds.ContentTime = str(startTime + frame_array[iframe]/1000)

            for islice in slice_num_array:
                imPos_slice = imPos_edge + slTh*islice*np.array([Sag_inc , Cor_inc , Tra_inc])
                ds.SliceLocation = imPos_slice[1]
                ds.ImagePositionPatient = np.ravel(imPos_slice).tolist()
                ds[(0x0019,0x1015)].value[:] = imPos_slice.tolist()

                if fe == 0:
                    outfilename = outdir + '/I_MAG_ph' + str(iframe) + '_' + str(islice - start_slice) + '.ima'
                    ds.SeriesNumber = 1
                    ds.ImageType = ['ORIGINAL', 'PRIMARY', 'M', 'RETRO', 'DIS2D']
                    ds[(0x0051, 0x1016)].value = 'p2 M/RETRO/DIS2D'

                if fe == 1:
                    ds.SequenceName = 'fl3d1_v' + str(int(data['venc'])) + 'in'
                    outfilename = outdir + '/I_Vz_ph' + str(iframe) + '_' + str(islice - start_slice) + '.ima'
                    ds.SeriesNumber = 4
                    ds.ImageType = ['DERIVED', 'PRIMARY', 'P', 'RETRO', 'DIS2D']
                    ds[(0x0051, 0x1016)].value = 'p2 P/RETRO/DIS2D'

                if fe == 3:
                    ds.SequenceName = 'fl3d1_v' + str(int(data['venc'])) + 'ap'
                    outfilename = outdir + '/I_Vy_ph' + str(iframe) + '_' + str(islice - start_slice) + '.ima'
                    ds.SeriesNumber = 3
                    ds.ImageType = ['DERIVED', 'PRIMARY', 'P', 'RETRO', 'DIS2D']
                    ds[(0x0051, 0x1016)].value = 'p2 P/RETRO/DIS2D'

                if fe == 2:
                    ds.SequenceName = 'fl3d1_v' + str(int(data['venc'])) + 'rl'
                    outfilename = outdir + '/I_Vx_ph' + str(iframe) + '_' + str(islice) + '.ima'
                    ds.SeriesNumber = 2
                    ds.ImageType = ['DERIVED', 'PRIMARY', 'P', 'RETRO', 'DIS2D']
                    ds[(0x0051, 0x1016)].value = 'p2 P/RETRO/DIS2D'

                tmpslice = img[fe, iframe, :, :, islice]
                ds.PixelData = tmpslice.tobytes()
                #ds.PixelData = np.ascontiguousarray(img[fe , iframe, : , : , islice])
                tmpstr = SOPInstanceUID_str.rsplit('.', 1)[0]
                tmpstr = tmpstr + str(counter)
                ds.SOPInstanceUID = tmpstr  #need a different UID for each image
                ds.save_as(outfilename)

                counter += 1


    return(counter)

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

