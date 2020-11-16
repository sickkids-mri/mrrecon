import numpy as np

import twixtools


def read_twix(filename, keep_syncdata_and_acqend=True):
    """Wraps `twixtools.read_twix` with some fixes to the reader.

    This wrapper can be removed once they fix these things.
    """
    scan_list = twixtools.read_twix(filename, keep_syncdata_and_acqend=keep_syncdata_and_acqend)  # noqa

    print('')  # Fixes absence of newline

    # Parse other headers
    for scan in scan_list:
        if not isinstance(scan, dict):
            # Then it is the raidfile_hdr (not needed)
            continue

        scan['hdr']['Config'] = _make_dict_from_hdr(scan['hdr']['Config'])
        scan['hdr']['Dicom'] = _make_dict_from_hdr(scan['hdr']['Dicom'])

    return scan_list


def _make_dict_from_hdr(dict_string):
    """Generates a dictionary from a portion of the header.

    Works for Config and Dicom.
    """
    import re
    pattern = re.compile(
            '<Param(Long|String|Double)\\."([^"]+)">  { ([^}]+)  }')
    out = {}
    for dtype, name, data in pattern.findall(dict_string):
        if dtype == "String":
            out[name] = data[1:-1]
        if dtype == "Long":
            if " " in data:
                out[name] = [int(x) for x in data.split()]
            else:
                out[name] = int(data)
        if dtype == "Double":
            out[name] = float(data.rstrip().split(" ")[-1])
    return out


class DataLoader:
    """Handles raw data loading and processing.

    Depends on twixtools for reading the Siemens MRI raw data file. Since the
    output of twixtools is a bit 'raw', data and parameters are processed and
    placed in a convenient dictionary called `data`.

    The whole process consists of 5 main steps:

    1) Reading the Siemens raw data file with twixtools.
    2) Storing scan data in NumPy arrays (e.g. noise and k-space measurements).
    3) Picking out relevant scan parameters from the header (e.g. image
        resolution, TR, VENC).
    4) Reading the mini data header for per line data values (e.g. time
        stamps, custom user-defined data).
    5) Reformatting the data.

    The original data structure output from twixtools can also be accessed
    (attribute name is `scan_list`).

    By default, scalars are stored as either Python `int`s or `float`s, and
    arrays are stored as either `np.float32` or `np.complex64`.

    Args:
        filename (str): Full name of raw data file.

    Attributes:
        data (dictionary): Contains loaded and processed items.
        scan_list (list): Output from twixtools.read_twix().
    """
    def __init__(self, filename):
        self.filename = filename
        self.data = {}

    def run(self):
        scan_list = self._load()
        image_scans = self._read_scan_data(scan_list)
        self._read_header(image_scans)
        self._read_minidataheader(image_scans)
        self._reformat()
        return self.data

    def _load(self):
        """Reads file and returns a list of scans."""
        scan_list = read_twix(self.filename, keep_syncdata_and_acqend=False)

        self.scan_list = scan_list
        return scan_list

    def _read_scan_data(self, scan_list):
        """Reads each scan/measurement and stores in NumPy arrays."""
        image_scans = []  # For collecting image scans for header reading
        self.data['kspace'] = []  # List of arrays
        self.data['calib'] = []  # List of arrays

        for scan in scan_list:
            if not isinstance(scan, dict):
                # Then it is the raidfile_hdr (not needed)
                continue

            array = self._fill_array(scan)

            first_line = scan['mdb'][0]
            if first_line.is_image_scan():
                self.data['kspace'].append(array)
                image_scans.append(scan)
            else:
                # Calibration scan
                self.data['calib'].append(array)

        return image_scans

    def _fill_array(self, scan):
        """Reads acquired data line by line and fills a 3D NumPy array.

        The shape of the array is (ncoils, nlines, nro).
        """
        # Get array shape
        nlines = len(scan['mdb'])
        ncoils, nro = scan['mdb'][0].data.shape  # Check first line

        array = np.empty((ncoils, nlines, nro),
                         dtype=np.complex64)

        for idx, line in enumerate(scan['mdb']):  # Looping over a list
            array[:, idx, :] = line.data

        return array

    def _read_header(self, image_scans):
        """Picks out relevant reconstruction parameters from the header."""
        # If there is more than one image scan, reads the header from the first
        hdr = image_scans[0]['hdr']

        # Only 'MeasYaps' was parsed and values stored dictionary
        # TODO: What happens when field/value does not exist?
        config = hdr['Config']
        dicom = hdr['Dicom']

        self.data['nx'] = config['ImageColumns']
        self.data['ny'] = config['ImageLines']

        meas = hdr['Meas'].split('\n')  # Not yet making dict out of 'Meas'
        for n, line in enumerate(meas):
            if 'i3DFTLength' in line:
                if int(meas[n + 2]) == 1:
                    self.data['nz'] = 1
                else:
                    self.data['nz'] = int(hdr['MeasYaps']['sKSpace']['lImagesPerSlab'])  # noqa
                break

        # In millimetres
        self.data['fovx'] = float(hdr['MeasYaps']['sSliceArray']['asSlice'][0]['dReadoutFOV'])  # noqa
        self.data['fovy'] = float(hdr['MeasYaps']['sSliceArray']['asSlice'][0]['dPhaseFOV'])  # noqa
        self.data['fovz'] = float(hdr['MeasYaps']['sSliceArray']['asSlice'][0]['dThickness'])  # noqa

        self.data['dx'] = self.data['fovx'] / self.data['nx']
        self.data['dy'] = self.data['fovy'] / self.data['ny']
        self.data['dz'] = self.data['fovz'] / self.data['nz']

        # Converts to milliseconds
        self.data['tr'] = float(hdr['MeasYaps']['alTR'][0]) / 1000  # noqa
        self.data['te'] = float(hdr['MeasYaps']['alTE'][0]) / 1000  # noqa
        self.data['ti'] = float(hdr['MeasYaps']['alTI'][0]) / 1000  # noqa

        # In degrees
        self.data['flipangle'] = float(hdr['MeasYaps']['adFlipAngleDegree'][0])  # noqa

        # VENC in (cm/s)
        self.data['venc'] = float(hdr['MeasYaps']['sAngio']['sFlowArray']['asElm'][0]['nVelocity'])  # noqa
        self.data['veldir'] = int(hdr['MeasYaps']['sAngio']['sFlowArray']['asElm'][0]['nDir'])  # noqa

        self.data['weight'] = dicom['flUsedPatientWeight']

        # Convert from nanoseconds to microseconds
        self.data['dwelltime'] = float(hdr['MeasYaps']['sRXSPEC']['alDwellTime'][0]) / 1000  # noqa

        self.data['vendor'] = dicom['Manufacturer']
        self.data['systemmodel'] = dicom['ManufacturersModelName']

        # Field strength
        self.data['field_strength'] = dicom['flMagneticFieldStrength']

        # Grad performance params (rise time and max grad)
        # Using dictionaries to look up values
        grad_mode = hdr['MeasYaps']['sGRADSPEC']['ucMode']

        # Dictionary values depend on system field strength
        if self.data['field_strength'] < 2:

            self.data['rise_time'] = {  # Rise time in usec/(mT/m)
                1: 5.88,  # FAST
                2: 10.0,  # NORMAL
                0: 10.0,  # Also NORMAL
                4: 20.0   # WHISPER
            }.get(grad_mode)  # Returns None if there is no value for grad_mode

            self.data['grad_max'] = {   # Max grad strength in mT/m
                1: 28,  # FAST
                2: 22,  # NORMAL
                0: 22,  # Also NORMAL
                4: 22   # WHISPER
            }.get(grad_mode)

        else:

            self.data['rise_time'] = {  # Rise time in usec/(mT/m)
                8: 5.3,    # PERFORMANCE
                1: 5.55,   # FAST
                2: 10.0,   # NORMAL
                0: 10.0,   # Also NORMAL
                4: 20.0    # WHISPER
            }.get(grad_mode)

            self.data['grad_max'] = {  # Max grad strength in mT/m
                8: 37,  # PERFORMANCE
                1: 24,  # FAST
                2: 22,  # NORMAL
                0: 22,  # Also NORMAL
                4: 22   # WHISPER
            }.get(grad_mode)

        self.data['readout_os_factor'] = config['ReadoutOversamplingFactor']
        self.data['seq_filename'] = config['SequenceFileName']
        try:
            self.data['fe_nav_flag'] = hdr['MeasYaps']['sWipMemBlock']['alFree'][2] #fe nav collection flag
        except:
            self.data['fe_nav_flag'] = 0
        return

    def _read_minidataheader(self, image_scans):
        """Reads mini data headers (MDH)."""
        # If there is more than one image scan, reads the mdh from the first
        scan = image_scans[0]

        nlines = len(scan['mdb'])

        times = np.zeros((nlines), dtype=np.float32)
        user_float = np.zeros((nlines, 24), dtype=np.float32)

        for idx, line in enumerate(scan['mdb']):
            times[idx] = line.mdh['ulTimeStamp'] * 2.5
            user_float[idx] = line.mdh['aushIceProgramPara']

        self.data['times'] = times
        self.data['user_float'] = np.copy(user_float.transpose())
        self.data['rot_quat'] = line.mdh[22][1]  # logical to physical rotation quaternion
        return

    def _reformat(self):
        """Reformatting steps that may be sequence-specific."""

        self.data['kspace'] = self.data['kspace'][0]  # Take out of list
        self.data['calib'] = self.data['calib'][0]  # Take out of list

        self.data['noise'] = self.data.pop('calib')  # Rename

        nv = 2  # Number of velocity encodes

        # Reshape
        (ncoils, nlines, nro) = self.data['kspace'].shape

        tmp = np.empty((ncoils, nv, int(nlines/nv), nro), dtype=np.complex64)

        for v in range(nv):
            tmp[:, v, :, :] = self.data['kspace'][:, v::nv, :]

        self.data['kspace'] = tmp
        tmp = None

        # Recalculate times at higher precision
        time0 = self.data['times'][0]
        times = np.linspace(time0,
                            time0 + (nlines - 1) * (self.data['tr'] / nv),
                            num=nlines, dtype=np.float32)
        self.data['times'] = times
        return


def print_dict_summary(data):
    """Summarizes attributes of items in the input dictionary.

    Args:
        data (dict): Contains raw data and scan parameters.
    """
    print(f'{"NAME":<18} {"TYPE":<36} {"SHAPE OR VALUE"}')
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            s = f'{type(v)} {v.dtype}'
            print(f'{k:<18} {s:<36} {v.shape}')
        else:
            print(f'{k:<18} {str(type(v)):<36} {v}')


class Flow4DLoader(DataLoader):
    """Data loader for 3D centre-out radial 4D flow."""
    def _read_scan_data(self, scan_list):
        if not len(scan_list) in [2, 3]:
            raise RuntimeError('Expected the length of scan_list to be either '
                               '2 or 3.')

        # Check datatypes
        assert isinstance(scan_list[0], np.void)  # raidfile_hdr
        assert isinstance(scan_list[1], dict)  # noise or k-space
        if len(scan_list) == 3:
            assert isinstance(scan_list[2], dict)  # k-space

        image_scans = []  # For collecting image scans for header reading

        for scan in scan_list:
            if not isinstance(scan, dict):
                # Then it is the raidfile_hdr (not needed)
                continue

            # Check first two lines to see if this is a k-space or noise scan.
            # If it's a noise scan, is_image_scan() should return False for
            # both lines. If it's k-space, the first line should return False
            # since it's the data collected during the flow encoding gradient,
            # and the second should return True.
            first_line = scan['mdb'][0]
            second_line = scan['mdb'][1]

            if first_line.is_image_scan() or second_line.is_image_scan(): #FE navs are not always collected

                image_scans.append(scan)

                if not first_line.is_image_scan(): #FE nav ON
                    # Data should alternate between flow nav and k-space
                    # Number of lines of k-space and flow navigators
                    nlines = int(len(scan['mdb']) / 2)
                    ncoils, nro = first_line.data.shape
                    self.data['flownav'] = np.empty((ncoils, nlines, nro),
                                                    dtype=np.complex64)
                    # Check second line for size of k-space array
                    ncoils, nro = second_line.data.shape
                    self.data['kspace'] = np.empty((ncoils, nlines, nro),
                                                   dtype=np.complex64)
                else:

                    nlines = int(len(scan['mdb']))
                    ncoils, nro = first_line.data.shape
                    self.data['kspace'] = np.empty((ncoils, nlines, nro),
                                                   dtype=np.complex64)

                # Loads and stores each line, and checks that flow nav and
                # k-space alternate
                f = 0
                k = 0
                for line in scan['mdb']:
                    if line.is_flag_set('RTFEEDBACK'):
                        assert f == k
                        self.data['flownav'][:, f, :] = line.data
                        f += 1
                    elif line.is_image_scan():
                        assert f == (k + 1)
                        self.data['kspace'][:, k, :] = line.data
                        k += 1
                    else:
                        raise RuntimeError('Data line has unidentified flag.')
            else: #no image scan in first 2 lines
            # It is noise scan
                nlines = len(scan['mdb'])
                ncoils, nro = first_line.data.shape

                self.data['noise'] = np.empty((ncoils, nlines, nro),
                                              dtype=np.complex64)

                for idx, line in enumerate(scan['mdb']):
                    self.data['noise'][:, idx, :] = line.data

        return image_scans

    def _reformat(self):
        """Reformatting steps that may be sequence-specific."""

        nv = 4  # Number of velocity encodes

        # Reshape
        (ncoils, nlines, nro) = self.data['kspace'].shape

        tmp = np.empty((ncoils, nv, int(nlines/nv), nro), dtype=np.complex64)

        for v in range(nv):
            tmp[:, v, :, :] = self.data['kspace'][:, v::nv, :]

        self.data['kspace'] = tmp
        tmp = None

        # Recalculate times at higher precision
        # Take the second time stamp, the first is flow navigator
        time0 = self.data['times'][1]
        times = np.linspace(time0,
                            time0 + (nlines - 1) * (self.data['tr'] / nv),
                            num=nlines, dtype=np.float32)
        self.data['times'] = times

        # Discard the user-defined measurements from flow navigators
        self.data['user_float'] = self.data['user_float'][:, 1::2]
        return
