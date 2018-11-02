
class CTPE(object):
    """CT for Pulmonary Embolism."""

    def __init__(self, study_num, slice_thickness, pe_slice_nums, is_positive, phase=None):
        self.study_num = study_num
        self.slice_thickness = float(slice_thickness)
        self.is_positive = is_positive
        self.num_slices = len(pe_slice_nums)
        self.pe_idxs = [n - 1 for n in pe_slice_nums]
        self.phase = phase

    def __len__(self):
        return self.num_slices
