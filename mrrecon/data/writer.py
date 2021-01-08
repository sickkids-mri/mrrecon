import os
thisdir = os.path.dirname(__file__)
import numpy as np

def write_to_dicom(data, img, outdir):
    import pydicom
    import numpy as np
    nv = img.shape[0]
    img_norm = normalize_pc(img)

    # read in dummy dicom files for each flow encode
    for fe in np.arange(nv):
        if fe == 0:
            ds = pydicom.dcmread(os.path.join(thisdir, '1.ima'))
        elif fe == 1:
            ds = pydicom.dcmread(os.path.join(thisdir, '2.ima'))
        elif fe == 2:
            ds = pydicom.dcmread(os.path.join(thisdir, '3.ima'))
        elif fe == 3:
            ds = pydicom.dcmread(os.path.join(thisdir, '4.ima'))


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

