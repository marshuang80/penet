import torch.utils.data as data
import util.transforms as transforms

from datasets import KineticsDataset


class KineticsDataLoader(data.DataLoader):

    def __init__(self, args, phase, is_training=True):

        self.phase = phase
        self.batch_size_ = args.batch_size

        # Normalization
        norm_value = 255
        mean = [110.63666788 / norm_value, 103.16065604 / norm_value, 96.29023126 / norm_value]
        std = [38.7568578 / norm_value, 37.88248729 / norm_value, 40.02898126 / norm_value]
        norm_method = transforms.Normalize(mean, std)

        # Transforms
        if is_training:
            assert args.crop_shape is not None and args.crop_shape[0] == args.crop_shape[1]
            crop_method = transforms.MultiScaleRandomCrop([1., 0.84, 0.71, 0.59, 0.49], args.crop_shape[0])

            spatial_transform = transforms.Compose([
                crop_method,
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                norm_method
            ])
            temporal_transform = transforms.TemporalRandomCrop(args.num_slices)
            target_transform = transforms.ClassLabel()
            n_samples = 1
        else:
            assert args.crop_shape is not None and args.crop_shape[0] == args.crop_shape[1]
            assert args.resize_shape is not None and args.resize_shape[0] == args.resize_shape[1]
            spatial_transform = transforms.Compose([
                transforms.Scale(args.crop_shape[0]),
                transforms.CenterCrop(args.crop_shape[0]),
                transforms.ToTensor(),
                norm_method
            ])
            temporal_transform = transforms.LoopPadding(args.num_slices)
            target_transform = transforms.ClassLabel()
            n_samples = 3

        dataset = KineticsDataset(args, phase, n_samples, spatial_transform, temporal_transform, target_transform)
        super(KineticsDataLoader, self).__init__(dataset,
                                                 batch_size=args.batch_size,
                                                 shuffle=is_training,
                                                 num_workers=args.num_workers,
                                                 pin_memory=True)
