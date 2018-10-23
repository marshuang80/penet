import dateutil.parser as date_parser
import numpy as np
import os
import re
import util


class CTSeries(object):
    def __init__(self, study_name, root_dir):
        """Series of CT slices. A CTSeries belongs to a CTStudy.
        There are usually multiple CTSeries in a single CTStudy.

        Args:
            study_name: Name of CT study containing this series.
            root_dir: Root directory holding DICOM files for this series.

        Raises:
            RuntimeError: If we can't find any DICOMs in the root directory, or if there's an unexpected file name.
        """

        # Collected
        self.dcm_dir = root_dir         # Original dir where DICOMs were stored
        self.study_name = study_name    # Anonymized ID for study
        self.series_number = None       # Series number within study
        self.dset_path = None           # Path to series pixels dataset in HDF5 file
        self.aneurysm_mask_path = None  # Path to aneurysm mask dataset in HDF5 file
        self.brain_mask_path = None     # Path to brain mask dataset in HDF5 file
        self.slice_names = None
        self.absolute_range = None
        self.is_bottom_up = None
        self.scanner_make = None
        self.brain_bbox = None
        self.brain_range = None
        self.dcm_thicknesses = []

        # Annotated
        self.is_aneurysm = None
        self.mode = None
        self.phase = None

        # Annotated (spreadsheet columns)
        self.anonymized_id = None
        self.medical_record_number = None
        self.date = None
        self.accession_number = None
        self.slice_thickness = None
        self.aneurysm_size = None
        self.aneurysm_bounds = None   # Minimal range containing all the ranges
        self.aneurysm_ranges = None  # List of ranges, some have multiple aneurysms

        self._initialize()

    def __len__(self):
        """Length of series is total number of slices."""
        return len(self.slice_names)

    def __iter__(self):
        """Exclude private members for converting to a dict."""
        for key in dir(self):
            if not key.startswith('_'):
                value = getattr(self, key)
                if not callable(value):
                    yield key, value

    def _initialize(self):
        """Initialize CT series. Collect info about this series from the DICOMs.

        Raises:
            RuntimeWarning: If we can't find any DICOMs in the root directory
            RuntimeError: If there's an unexpected file name.
        """
        dcm_paths = sorted([os.path.join(self.dcm_dir, f) for f in os.listdir(self.dcm_dir) \
                            if f.endswith('.dcm')])
        if len(dcm_paths) == 0:
            raise RuntimeWarning('Did not find any DICOMs in {}'.format(self.dcm_dir))
        self.slice_names = [os.path.basename(f)[:-4] for f in dcm_paths]

        # Read a DICOM as an example
        dcm = util.read_dicom(dcm_paths[0])
        self.series_number = int(dcm.SeriesNumber)
        if 'SliceThickness' in dcm:
            self.dcm_thicknesses.append(dcm.SliceThickness)
        if 'ContentDate' in dcm:
            self.date = date_parser.parse(dcm.ContentDate)

        # Record scan direction
        if len(dcm_paths) == 1:
            raise RuntimeWarning('Only found a single DICOM file in {}'.format(self.dcm_dir))
        dcm_second = util.read_dicom(dcm_paths[1])
        if 'AnatomicalOrientationType' in dcm:
            raise RuntimeWarning('Series {} has Anatomical Orientation Type {}, unable to fetch scan direction.' \
                                 .format(self.dcm_dir, dcm.AnatomicalOrientationType))
        # The z-axis of ImagePositionPatient is increasing toward the head of the patient
        elif 'ImagePositionPatient' not in dcm:
            raise RuntimeWarning('{}: No ImagePositionPatient attribute, unable to fetch scan direction.' \
                                 .format(self.dcm_dir))
        else:
            ipp1 = dcm.ImagePositionPatient
            ipp2 = dcm_second.ImagePositionPatient
            self.is_bottom_up = ipp1[2] < ipp2[2]

        # Record last DICOM slice thickness for possibly multiple slice thicknesses
        dcm_last = util.read_dicom(dcm_paths[-1])
        if 'SliceThickness' in dcm_last:
            self.dcm_thicknesses.append(dcm_last.SliceThickness)
        self.dcm_thicknesses = list(set(self.dcm_thicknesses))

        # Record scanner manufacturer
        if self.scanner_make is None and 'Manufacturer' in dcm:
            self.scanner_make = str(dcm.Manufacturer).lower()

        # Save mask path if mask exists
        aneurysm_mask_path = os.path.join(self.dcm_dir, 'aneurysm_mask.npy')
        if os.path.exists(aneurysm_mask_path):
            self.aneurysm_mask_path = aneurysm_mask_path
        brain_mask_path = os.path.join(self.dcm_dir, 'brain_mask.npy')
        if os.path.exists(brain_mask_path):
            self.brain_mask_path = brain_mask_path

        dcm_scan_num = None  # First number in name IM-####-####.dcm
        for dcm_path in dcm_paths:
            # Check if image is in annotated range
            m = re.search(r'(\d+)-(\d+).dcm', os.path.basename(dcm_path))
            if m is None:
                raise RuntimeError('Unexpected DICOM name: {}'.format(os.path.basename(dcm_path)))

            # Make sure the folder doesn't contain files from multiple series
            if dcm_scan_num is None:
                dcm_scan_num = int(m.group(1))
            elif dcm_scan_num != int(m.group(1)):
                raise RuntimeError('Folder {} might contain multiple series'.format(self.dcm_dir))

            # Keep track of absolute start and end scan numbers
            dcm_num = int(m.group(2))
            if self.absolute_range is None:
                self.absolute_range = [dcm_num, dcm_num]
            elif dcm_num < self.absolute_range[0]:
                self.absolute_range[0] = dcm_num
            elif dcm_num > self.absolute_range[1]:
                self.absolute_range[1] = dcm_num

        # Make sure start and end are consistent with number of slices found
        if self.absolute_range[1] - self.absolute_range[0] + 1 != len(self):
            raise RuntimeError('Start and end do not match number of slices: {} (start={}, end={}, slices={}'
                               .format(self.dcm_dir, self.absolute_range[0], self.absolute_range[1], len(self)))

    @staticmethod
    def _parse_size(size_str):
        """Parse a size annotation. Return the largest value as a float in mm units.

        Only consider string up to first 'CM' or 'MM' (case insensitive).
        Take the max of all floats up to that point.
        """
        m = re.search(r'(cm|mm)', size_str, re.IGNORECASE)
        if m is not None:
            # Find all floats or ints before the first MM or CM
            sizes = re.findall(r'(\d*\.\d+|\d+)', size_str[:m.start()])
            if len(sizes) > 0:
                scale_to_mm = 10. if m.group(1).lower() == 'cm' else 1.
                return scale_to_mm * max((float(size) for size in sizes))

        return None

    def annotate(self, is_aneurysm, mode, ann_dict, require_aneurysm_range=True):
        """Add annotation info to a series.

        Args:
            is_aneurysm: True iff series contains an aneurysm.
            mode: One of 'contrast', 'non_contrast', or 'post_contrast'.
            ann_dict: Dictionary of other annotation info.
            require_aneurysm_range: If True, require aneurysm studies to have aneurysm range annotation.

        Raises:
            RuntimeWarning if annotations were invalid.
        """
        self.is_aneurysm = is_aneurysm
        self.mode = mode

        self.anonymized_id = str(ann_dict['AnonID'])
        self.medical_record_number = str(ann_dict['MRN'])
        if self.date is None and len(ann_dict['Date']) > 0:
            self.date = date_parser.parse(str(ann_dict['Date']))

        self.accession_number = util.try_parse(ann_dict['Acc'], type_fn=int)

        study_type = 'CTA' if self.mode == 'contrast' else 'CT'
        if is_aneurysm:
            self.aneurysm_size = self._parse_size(str(ann_dict['size']))
            try:
                aneurysm_start = int(ann_dict['{} image # start'.format(study_type)])
                aneurysm_end = int(ann_dict['{} image # end'.format(study_type)])
                self.aneurysm_bounds = [aneurysm_start, aneurysm_end]
                self.aneurysm_ranges = [self.aneurysm_bounds]
            except ValueError:
                if require_aneurysm_range:
                    raise RuntimeWarning('Invalid aneurysm annotation for study {}.'.format(self.study_name))

        try:
            brain_start = int(float(ann_dict['cta_brain_start'.format(study_type)]))
            brain_end = int(float(ann_dict['cta_brain_end'.format(study_type)]))
            self.brain_range = [brain_start, brain_end]
        except ValueError:
            pass

        annotated_thickness = float(ann_dict['{} ST (mm)'.format(study_type)])
        if np.isnan(annotated_thickness):
            self.slice_thickness = min(self.dcm_thicknesses)
        elif not np.isnan(annotated_thickness) and annotated_thickness in self.dcm_thicknesses:
            self.slice_thickness = annotated_thickness
        elif not np.isnan(annotated_thickness) and annotated_thickness not in self.dcm_thicknesses:
            raise RuntimeWarning('Study {}: Annotated thickness {}, DICOM thicknesses {}.'
                                 .format(self.study_name, annotated_thickness, self.dcm_thicknesses))

    def slice_num_to_idx(self, slice_num):
        """Convert a slice number to an index in the volume.

        Args:
            slice_num: Number of slice as seen in DICOM viewer.

        Returns:
            Index into volume to get the corresponding slice.
        """
        if self.is_bottom_up:
            slice_idx = slice_num - 1
        else:
            slice_idx = len(self) - slice_num

        return slice_idx

    def slice_idx_to_num(self, slice_idx):
        """Convert a slice index into a slice number as seen in the DICOM viewer.

        Args:
            slice_idx: Index of slice to convert to slice number.

        Returns:
            Slice number (in DICOM viewer) of slice at corresponding index in volume.
        """
        if self.is_bottom_up:
            slice_num = slice_idx + 1
        else:
            slice_num = len(self) - slice_idx - 1

        return slice_num
