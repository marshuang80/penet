class PaddedInputs(object):
    """Wrapper class for sending padded inputs to a model."""
    def __init__(self, inputs, length):
        self.inputs = inputs
        self.length = length

    def to(self, device):
        self.inputs.to(device)
        self.length.to(device)
