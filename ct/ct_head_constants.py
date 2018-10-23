"""Constants related to the HEAD CT dataset."""

# Hounsfield Units for Air
AIR_HU_VAL = -1000.

# Statistics for Hounsfield Units
CONTRAST_HU_MIN = -300.     # Min value for loading con
CONTRAST_HU_MAX = 700.      # Max value for loading con
CONTRAST_HU_MEAN = 0.15899  # Mean pixel value after normalization and clipping

NON_CON_HU_MIN = -1000.     # Min value for loading non-con.
NON_CON_HU_MAX = 400.       # Max value for loading non-con.
NON_CON_HU_MEAN = 0.21411   # Mean pixel value after normalization and clipping

W_CENTER_DEFAULT = 40.      # Default window center for viewing
W_WIDTH_DEFAULT = 400.      # Default window width

MAX_HEAD_HEIGHT_MM = 240    # 99th percentile head height in millimeters
