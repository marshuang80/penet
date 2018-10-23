from torch.nn import DataParallel, init


def init_model(model, init_method='normal', gpu_ids=()):
    # # Move model to GPU if available
    # if len(gpu_ids) > 0:
    #     model.to('cuda')
    #     model = DataParallel(model, device_ids=gpu_ids)

    # Initialize model parameters
    _init_params(model, init_method)

    return model


def _init_params(model, init_method='xavier'):
    """Initialize model parameters.
    Args:
        model: Model to initialize.
        init_method: Name of initialization method: 'normal' or 'xavier'.
    """
    if init_method == 'normal':
        model.apply(_normal_init)
    elif init_method == 'xavier':
        model.apply(_xavier_init)
    else:
        raise NotImplementedError('Invalid weights initializer: {}'.format(init_method))


def _normal_init(model):
    """Apply normal initializer to all model weights."""
    class_name = model.__class__.__name__
    if hasattr(model, 'weight') and class_name.find('Conv') != -1:
        init.normal_(model.weight.data, 0.0, 0.02)
    elif hasattr(model, 'weight') and class_name.find('Linear') != -1:
        init.normal_(model.weight.data, 0.0, 0.02)
    elif class_name.find('BatchNorm2d') != -1:
        init.normal_(model.weight.data, 1.0, 0.02)
        init.constant_(model.bias.data, 0.0)


def _xavier_init(model):
    """Apply Xavier initializer to all model weights."""
    class_name = model.__class__.__name__
    if hasattr(model, 'weight') and class_name.find('Conv') != -1:
        init.xavier_normal(model.weight.data, gain=0.02)
    elif hasattr(model, 'weight') and class_name.find('Linear') != -1:
        init.xavier_normal(model.weight.data, gain=0.02)
    elif class_name.find('BatchNorm2d') != -1:
        init.normal_(model.weight.data, 1.0, 0.02)
        init.constant_(model.bias.data, 0.0)