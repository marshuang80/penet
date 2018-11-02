"""Merge three pickle files corresponding to the three PE subsets:
  1. Central
  2. Segmental
  3. Sub-segmental
"""
import argparse
import h5py
import numpy as np
import pickle

from collections import Counter
from tqdm import tqdm


def main(args):
    np.random.seed(7)

    print('Loading central PEs...')
    with open(args.central_path, 'rb') as pkl_fh:
        central_pes = pickle.load(pkl_fh)
    print('Got {} central PEs.'.format(len(central_pes)))

    print('Loading segmental PEs...')
    with open(args.segmental_path, 'rb') as pkl_fh:
        segmental_pes = pickle.load(pkl_fh)
    print('Got {} segmental PEs.'.format(len(segmental_pes)))

    print('Loading subsegmental PEs...')
    with open(args.subsegmental_path, 'rb') as pkl_fh:
        subsegmental_pes = pickle.load(pkl_fh)
    print('Got {} subsegmental PEs.'.format(len(subsegmental_pes)))

    hdf5_fh = h5py.File(args.hdf5_path, 'r')

    all_pes = central_pes + segmental_pes + subsegmental_pes
    with tqdm(total=len(all_pes)) as progress_bar:
        for subset, pe_type in [(central_pes, 'central'), (segmental_pes, 'segmental'), (subsegmental_pes, 'subsegmental')]:
            for pe in subset:
                # Mark with type and num_slices
                pe.type = pe_type
                pe.num_slices = hdf5_fh[str(pe.study_num)].shape[0]

                # All studies with slice labels are 'central' and positive, so put them in train
                if pe.type == 'central':
                    pe.phase = 'train'
                elif not pe.is_positive and np.random.random() < 0.05:
                    # Add some negatives to make prevalence constant
                    pe.phase = 'train'
                else:
                    pe.phase = np.random.choice(('val', 'test'), p=(0.5, 0.5))

                progress_bar.update()

    hdf5_fh.close()

    main_pes = [pe for pe in all_pes if pe.type == 'central' or pe.type == 'segmental']
    # Check the splits
    for phase in ('train', 'val', 'test'):
        print('Phase: {}'.format(phase))
        print('Count: {}'.format(sum(1 for pe in main_pes if pe.phase == phase)))
        print('Prev.: {}'.format(sum(1 for pe in main_pes if pe.phase == phase and pe.is_positive) /
                                 sum(1 for pe in main_pes if pe.phase == phase)))
        print('Slice thicknesses: {}'.format(Counter(pe.slice_thickness for pe in main_pes if pe.phase == phase)))

    # Write combined list
    with open(args.output_path, 'wb') as pkl_fh:
        pickle.dump(all_pes, pkl_fh)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge PE subsets.')

    parser.add_argument('--hdf5_path', type=str,
                        default='/deep/group/aihc-bootcamp-winter2018/medical-imaging/ct_chest_pe/cc_data/data.hdf5')
    parser.add_argument('--central_path', type=str,
                        default='/deep/group/aihc-bootcamp-winter2018/medical-imaging/ct_chest_pe/cc_data/series_list_orig_1007.pkl')
    parser.add_argument('--segmental_path', type=str,
                        default='/deep/group/aihc-bootcamp-winter2018/medical-imaging/ct_chest_pe/tanay_test_data/data.pkl')
    parser.add_argument('--subsegmental_path', type=str,
                        default='/deep/group/aihc-bootcamp-winter2018/medical-imaging/ct_chest_pe/subsegmental_test_data/data.pkl')
    parser.add_argument('--output_path', type=str,
                        default='/deep/group/aihc-bootcamp-winter2018/medical-imaging/ct_chest_pe/cc_data/series_list.pkl')

    main(parser.parse_args())
