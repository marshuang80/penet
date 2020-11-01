"""Constants related to the CT PE dataset."""

# Hounsfield Units for Air
AIR_HU_VAL = -1000.


# Statistics for Hounsfield Units
CONTRAST_HU_MIN = -100.     # Min value for loading contrast
CONTRAST_HU_MAX = 900.      # Max value for loading contrast
CONTRAST_HU_MEAN = 0.15897  # Mean voxel value after normalization and clipping
CONTRAST_HU_STD = 0.19974   # Standard deviation of voxel values after normalization and clipping
"""
CONTRAST_HU_MIN = -250#-250.     # Min value for loading contrast
CONTRAST_HU_MAX = 450      # Max value for loading contrast
CONTRAST_HU_MEAN = 0.15897 #0.3  # Mean voxel value after normalization and clipping
CONTRAST_HU_STD = 0.19974   # Standard deviation of voxel values after normalization and clipping
"""


W_CENTER_DEFAULT = 400.     # Default window center for viewing
W_WIDTH_DEFAULT = 1000.     # Default window width
