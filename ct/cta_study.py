import json
import numpy as np
import os
import re
import util

from collections import defaultdict
from imageio import imsave
from tqdm import tqdm


class CTAStudy(object):
    def __init__(self, root_dir, annotation, modes, ignore_range=False):
        """CT Angiography Study. A CTAStudy contains one or more series.

        Args:
            root_dir: Root directory for pre-processed studies.
            annotation: Dictionary with info from an annotation file.
            modes: Modes to convert (options are contrast, non_contrast, post_contrast).
            ignore_range: If true, collect all images ignoring the range specified by the annotation.
            Still keep the range on the `CTAStudy` object.
        """
        self.name = annotation['anonymized_id'].lower()
        self.modes = modes

        # Each is a dict mapping mode to int or list
        self.paths = {}
        self.series = {}
        self.ends = {}
        self.starts = {}

        for mode in modes:
            self.paths[mode] = []
            self.series[mode] = int(annotation['{}_series'.format(mode)])
            if not ignore_range:
                self.starts[mode] = int(annotation['{}_start'.format(mode)])
                self.ends[mode] = int(annotation['{}_end'.format(mode)])

        self.manufacturer = None  # Company that makes the scanner
        self.scan_direction = defaultdict(bool) # Map mode to scan direction: true for bottom-up, false for top-down

        self._collect_paths(root_dir, ignore_range=ignore_range)

    def __len__(self):
        """Length of study is total number of images (all modes)."""
        return sum([len(self.paths[mode]) for mode in self.modes])

    def num_slices(self, mode):
        return len(self.paths[mode]) if mode in self.modes else 0

    def _collect_paths(self, root_dir, ignore_range=False):
        """Collect the paths for images in this CTA study.

        Args:
            root_dir: Root directory for pre-processed CTA studies.

        Raises:
            RuntimeError: If we can't find the right number of images.
        """
        cta_path = os.path.join(root_dir, self.name.capitalize())
        for base_path, dirs, filenames in os.walk(cta_path):
            dcm_paths = sorted([os.path.join(base_path, f) for f in filenames if f.endswith('.dcm')])
            if len(dcm_paths) > 0:
                # Check one example to see if this is either our contrast or non-contrast series
                dcm = util.read_dicom(dcm_paths[0])
                dcm_second = util.read_dicom(dcm_paths[1])
                assert hasattr(dcm, 'AnatomicalOrientationType') == False, 'Check anatomical orientation type'
                # The z-axis of ImagePositionPatient is increasing toward the head of the patient.
                ipp1 = getattr(dcm, 'ImagePositionPatient')
                ipp2 = getattr(dcm_second, 'ImagePositionPatient')

                paths, start, end = None, None, None
                for mode, series_num in self.series.items():
                    if int(dcm.SeriesNumber) == series_num:
                        # Record scan direction
                        self.scan_direction[mode] = ipp1[2] < ipp2[2]
                        # Collect mode series
                        paths = self.paths[mode]
                        if not ignore_range:
                            start = self.starts[mode]
                            end = self.ends[mode]

                if paths is not None:
                    for dcm_path in dcm_paths:
                        if ignore_range:
                            paths.append(dcm_path)
                        else:
                            # Check if image is in annotated range
                            m = re.search(r'(\d+)-(\d+)', os.path.basename(dcm_path))
                            if m is not None:
                                dcm_num = int(m.group(2))
                                if start <= dcm_num <= end:
                                    paths.append(dcm_path)

                # Save scanner manufacturer
                if self.manufacturer is None and 'Manufacturer' in dcm:
                    self.manufacturer = dcm.Manufacturer

        # Verify that we found the right number of images given the annotated range
        for mode in self.modes:
            if not ignore_range and len(self.paths[mode]) != self.ends[mode] - self.starts[mode] + 1:
                raise RuntimeError('{}: {} range contains {} images, but actually collected {}.'
                                   .format(self.name, mode.capitalize(), self.ends[mode] - self.starts[mode] + 1,
                                           len(self.paths[mode])))

    def _get_output_name(self):
        """Convert CTA directory name to a name for the study.

        Returns:
            Name of the format cta_NN, which should be used for naming the output directory.
        """
        m = re.search(r'\d+', self.name)
        if m:
            return 'cta_{:04d}'.format(int(m.group(0)))
        else:
            raise RuntimeError('Unexpected CTA study name: {}'.format(self.name))

    def _get_meta_dict(self):
        """Get dictionary of metadata about this study.

        This metadata gets written to a JSON file in the top-level directory for the post-processed study.
        """
        meta_dict = {
            'name': self.name,
            'num_contrast': self.num_slices('contrast'),
            'num_non_contrast': self.num_slices('non_contrast'),
            'num_post_contrast': self.num_slices('post_contrast')
        }

        if self.manufacturer is not None:
            meta_dict['manufacturer'] = self.manufacturer

        for key in self.scan_direction:
            meta_dict['scan_direction_' + key] = self.scan_direction[key]
        
        return meta_dict

    def process_and_write(self, output_fmt, output_dir, c_window=None, n_window=None):
        """Process this CTA study and write all images to the output dir.

        Args:
            output_fmt: Format for converted images (options are 'png' or 'raw').
            output_dir: Directory for post-processed studies.
            c_window: Tuple (center, width) for contrast images (only if `output_fmt` is 'png').
            n_window: Tuple (center, width) for non-con. images (only if `output_fmt` is 'png').
        """

        # Create directory structure
        output_name = self._get_output_name()
        mode_dirs = [(mode, self.paths[mode]) for mode in self.modes]
        for mode_dir, _ in mode_dirs:
            for file_type_dir in (output_fmt, 'dcm'):
                path = os.path.join(output_dir, output_name, mode_dir, file_type_dir)
                os.makedirs(path, exist_ok=True)

        # Write meta-dict of study info
        with open(os.path.join(output_dir, output_name, 'metadata.json'), 'w') as meta_file:
            json.dump(self._get_meta_dict(), meta_file, indent=4, sort_keys=True)

        # Process and write DICOMs
        print('Writing study {}...'.format(output_name))
        with tqdm(total=len(self),  unit=' DICOMs') as progress_bar:
            for mode_dir, src_paths in mode_dirs:
                for src_path in src_paths:
                    dcm = util.read_dicom(src_path)
                    tgt_path = os.path.join(output_dir, output_name, mode_dir)
                    dcm_name = os.path.basename(src_path)[:-4]

                    # Save as dcm
                    dcm_path = os.path.join(tgt_path, 'dcm', dcm_name + '.dcm')
                    dcm.save_as(dcm_path)

                    # Save as raw or png
                    if output_fmt == 'png':
                        w_center = c_window[0] if mode_dir == 'contrast' else n_window[0]
                        w_width = c_window[1] if mode_dir == 'contrast' else n_window[1]
                        png_np = util.dcm_to_png(dcm, w_center=w_center, w_width=w_width)
                        png_path = os.path.join(tgt_path, 'png', dcm_name + '.png')
                        imsave(png_path, png_np)
                    elif output_fmt == 'raw':
                        raw_np = util.dcm_to_raw(dcm)
                        raw_path = os.path.join(tgt_path, 'raw', dcm_name + '.npy')
                        np.save(raw_path, raw_np)
                    else:
                        raise ValueError('Invalid output format: {}'.format(output_fmt))

                    # Update progress bar
                    progress_bar.update()
