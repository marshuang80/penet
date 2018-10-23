"""Constants related to the SPINE CT dataset."""

# CT-CSPINE specific constants
PNG_SAGITTAL_MEAN = 70.523430355     # Average channel value for CT-CSPINE sagittal scans     
PNG_AXIAL_MEAN = 97.2124152182       # Average channel value for CT-CSPINE axial scans     

ANNOTATION_GRID_DIM_X = 3          # Annotation grid dimension in the X dimension
ANNOTATION_GRID_DIM_Y = 3          # Annotation grid dimension in the Y dimension

HU_MIN = -500.      # Min value for loading cspine
HU_MAX = 1300.      # Max value for loading cspine
HU_SAGITTAL_MEAN = 0.172698651196   # Mean pixel value after normalization and clipping
HU_AXIAL_MEAN = None  # <Calculated value! >   # Mean pixel value after normalization and clipping

W_CENTER_DEFAULT = 400.      # Default window center for viewing
W_WIDTH_DEFAULT = 1800.       # Default window width for viewing
