import re

import numpy as np


def read_2dflow(reportfile):
    """Reads the text report from the cvi42 2D flow module.

    The report can contain measurements from a number of 2D phase contrast
    images. In cvi42, give an appropriate name to each ROI to simplify this
    part of the analysis.

    NOTE: This function was designed to read reports consisting of data from
    the 2D flow module only. If items were added to the report from other cvi42
    modules, there will likely be errors.

    Args:
        reportfile (str): Filename of the scientific report (.txt) consisting
            of data from the 2D flow module.

    Returns:
        outs (dict): Dictionary containing dictionaries, one sub-dictionary
            for each ROI. Each sub-dictionary contains data from the ROI as a
            function of time. Data are kept as strings.
    """
    outs = {}
    with open(reportfile, 'r', encoding='utf-16-le') as f:
        lines = f.readlines()

    # Find number of acquisitions/images compiled into this report
    where_next_image_starts = []
    for line_num, line in enumerate(lines):
        if 'Flow Analysis' in line:
            where_next_image_starts.append(line_num)
    num_meas = len(where_next_image_starts)

    # Divide the file into sections, one for each image
    sections = []
    for meas in range(num_meas):
        start = where_next_image_starts[meas]
        if meas == num_meas - 1:
            stop = len(lines)
        else:
            stop = where_next_image_starts[meas + 1]
        sections.append(lines[start:stop])

    for section in sections:
        for line_num, line in enumerate(section):
            # Skip params and general header info.
            # A number of ROIs can be analyzed for each image, although
            # usually it is just 1.
            # Loop through to find where each ROI starts.
            if 'Flow ROI' in line:
                vessel_name = section[line_num + 1].rstrip('\n')
                # Rename if name already used
                i = 1
                while vessel_name in outs.keys():
                    vessel_name = vessel_name + f'_{i}'
                    i += 1

                outs[vessel_name] = {}

                # Loop through the rest of the section.
                # There should be a subsection with statistics, then the
                # raw data will follow, then the next section will start.
                # Find where the raw data is, get it, then move on.
                subsection = section[line_num:]
                for line_num, line in enumerate(subsection):
                    if 'Time(ms)' in line:
                        headings = line.rstrip('\n').split('\t')
                        num_headings = len(headings)
                        data = []
                        for line in subsection[line_num+1:]:
                            vals = line.rstrip('\n').split(' \t')
                            if len(vals) != num_headings:
                                # Probably all data lines are read
                                break
                            data.append(vals)

                        # Array of strings
                        arr = np.array(data)  # (nt, num_headings)

                        for c, heading in enumerate(headings):
                            outs[vessel_name][heading] = arr[:, c]

                        break

    return outs


def read_4dflow(reportfile):
    """Reads the text report from the cvi42 4D flow module.

    NOTE: This function was designed to read reports consisting of data from
    the 4D flow module only. If items were added to the report from other cvi42
    modules, there will likely be errors.

    Args:
        reportfile (str): Filename of the scientific report (.txt) consisting
            of data from the 4D flow module.

    Returns:
        outs (dict):
    """
    outs = {}
    with open(reportfile, 'r', encoding='utf-16-le') as f:
        lines = f.readlines()

    # The report is divided by the segmentations made in cvi42
    segment_starts = []
    for line_num, line in enumerate(lines):
        if '4D Flow Analysis' in line:
            segment_starts.append(line_num)
    num_segs = len(segment_starts)

    # Divide the file into sections, one for each segmentation
    sections = []
    for seg in range(num_segs):
        start = segment_starts[seg]
        if seg == num_segs - 1:
            stop = len(lines)
        else:
            stop = segment_starts[seg + 1]
        sections.append(lines[start:stop])

    for section in sections:
        for line_num, line in enumerate(section):
            # A number of flow planes can be set for each segmentation.
            # Loop through to find where each flow plane starts.
            regex = r' - Complete Report'
            match = re.search(regex, line)
            if match:
                # Get name of flow plane. Flow planes should be renamed from
                # 'Flow 1', 'Flow 2', etc.
                flow_plane_name = line.rstrip(' - Complete Report\n')

                outs[flow_plane_name] = {}

                # Loop through the rest of the section.
                # There should be a subsection with statistics, then the
                # raw data will follow, then the next section will start.
                # Find where the raw data is, get it, then move on.
                subsection = section[line_num:]
                for line_num, line in enumerate(subsection):
                    if 'Flow Rate' in line:
                        headings = line.rstrip('\n').split('\t')
                        num_headings = len(headings)
                        # The units for each heading is in next line. Combine
                        # the units into the heading names, like the 2D flow
                        # report
                        units = subsection[line_num+1].rstrip('\n').split('\t')
                        for u in range(len(units)):
                            headings[u] = headings[u] + units[u]

                        data = []
                        for line in subsection[line_num+2:]:
                            vals = line.rstrip('\n').split(' \t')
                            if len(vals) != num_headings:
                                # Probably all data lines are read
                                break
                            data.append(vals)

                        # Array of strings
                        arr = np.array(data)  # (nt, num_headings)

                        for c, heading in enumerate(headings):
                            outs[flow_plane_name][heading] = arr[:, c]

                        break

    return outs
