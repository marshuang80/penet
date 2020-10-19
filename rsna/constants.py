from pathlib import Path

RSNA_DIR = Path('/data4/rsna')
TRAIN_DIR = RSNA_DIR / 'train'
TEST_DIR = RSNA_DIR / 'test'

TRAIN_CSV = RSNA_DIR / 'train.csv'
TEST_CSV = RSNA_DIR / 'test.csv'

PARSED_DIR = RSNA_DIR / 'parsed_data'
TRAIN_PARSED_DIR = PARSED_DIR / 'train'
TEST_PARSED_DIR = PARSED_DIR / 'test'

TRAIN_PATHS = {
    'csv': TRAIN_CSV,
    'dicoms': TRAIN_DIR,
    'output_dir': TRAIN_PARSED_DIR,
    'hdf5': TRAIN_PARSED_DIR / 'data.hdf5',
    'series_list': TRAIN_PARSED_DIR / 'series_list.pkl'
}

TEST_PATHS = {
    'csv': TEST_CSV,
    'dicoms': TEST_DIR,
    'output_dir': TEST_PARSED_DIR,
    'hdf5': TEST_PARSED_DIR / 'data.hdf5',
    'series_list': TEST_PARSED_DIR / 'series_list.pkl'
}

SPLIT_PATHS = {
    'train': TRAIN_PATHS,
    'test': TEST_PATHS
}

TRAIN_PERCENT = 0.9