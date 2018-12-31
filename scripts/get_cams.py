import moviepy.editor as mpy
import numpy as np
import os
import torch
import util

from args import TestArgParser
from cams import GradCAM
from cams import GuidedBackPropagation
from data_loader import CTDataLoader
from saver import ModelSaver


def get_cams(args):
    print('Loading model...')
    model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args.gpu_ids)
    model = model.to(args.device)
    args.start_epoch = ckpt_info['epoch'] + 1

    print('Last layer in model.features is named "{}"...'.format([k for k in model.module.encoders._modules.keys()][-1]))
    print('Extracting feature maps from layer named "{}"...'.format(args.target_layer))

    grad_cam = GradCAM(model, args.device, is_binary=True, is_3d=True)
    gbp = GuidedBackPropagation(model, args.device, is_binary=True, is_3d=True)

    num_generated = 0
    data_loader = CTDataLoader(args, phase=args.phase, is_training=True)
    for inputs, target_dict in data_loader:
        labels = target_dict['is_abnormal']
        if labels.item() == 0:
            # Keep going until we get an aneurysm study
            print('Skipping a normal example...')
            continue

        print('Generating CAM...')
        with torch.set_grad_enabled(True):
            probs, idx = grad_cam.forward(inputs)
            # TODO: Need to change if we change the threshold
            if probs < 0.5:
                print('Skipping false negatives...')
                continue
            print('Generating CAM...')
            grad_cam.backward(idx=idx[0])  # Just take top prediction
            cam = grad_cam.get_cam(args.target_layer)

            guided_backprop = None
            if args.use_gbp:
                inputs2 = torch.autograd.Variable(inputs, requires_grad=True)
                probs2, idx2 = gbp.forward(inputs2)
                gbp.backward(idx=idx2[0])
                guided_backprop = np.squeeze(gbp.generate())

        print('Overlaying CAM...')
        new_cam = util.resize(cam, inputs[0])

        input_np = util.un_normalize(inputs[0], args.img_format, data_loader.dataset.pixel_dict)
        input_np = np.transpose(input_np, (1, 2, 3, 0))
        input_frames = list(input_np)

        input_normed = np.float32(input_np) / 255
        cam_frames = list(util.add_heat_map(input_normed, new_cam))

        gbp_frames = None
        if args.use_gbp:
            gbp_np = util.normalize_to_image(guided_backprop * new_cam)
            gbp_frames = []
            for dim in range(gbp_np.shape[0]):
                slice_ = gbp_np[dim, :, :]
                gbp_frames.append(slice_[..., None])

        # Write to a GIF file
        output_path_input = os.path.join(os.path.join(args.cam_dir, 'input_{}.gif'.format(num_generated + 1)))
        output_path_cam = os.path.join(args.cam_dir, 'cam_{}.gif'.format(num_generated + 1))
        output_path_combined = os.path.join(args.cam_dir, 'combined_{}.gif'.format(num_generated + 1))

        print('Writing set {}/{} of CAMs to {}...'.format(num_generated + 1, args.num_cams, args.cam_dir))
        input_clip = mpy.ImageSequenceClip(input_frames, fps=4)
        input_clip.write_gif(output_path_input, verbose=False)
        cam_clip = mpy.ImageSequenceClip(cam_frames, fps=4)
        cam_clip.write_gif(output_path_cam, verbose=False)
        combined_clip = mpy.clips_array([[input_clip, cam_clip]])
        combined_clip.write_gif(output_path_combined, verbose=False)

        if args.use_gbp:
            output_path_gcam = os.path.join(args.cam_dir, 'gbp_{}.gif'.format(num_generated + 1))
            gbp_clip = mpy.ImageSequenceClip(gbp_frames, fps=4)
            gbp_clip.write_gif(output_path_gcam, verbose=False)

        num_generated += 1
        if num_generated == args.num_cams:
            return


if __name__ == '__main__':
    parser = TestArgParser()
    parser.parser.add_argument('--test_2d', action='store_true', help='Test CAMs on a pretrained 2D VGG net.')
    parser.parser.add_argument('--target_layer', type=str, default='module.encoders.3',
                               help='Name of target layer for extracting feature maps.')
    parser.parser.add_argument('--cam_dir', type=str, default='data/', help='Directory to write CAM outputs.')
    parser.parser.add_argument('--num_cams', type=int, default=1, help='Number of CAMs to generate.')
    parser.parser.add_argument('--use_gbp', type=util.str_to_bool, default=False,
                               help='If True, use guided backprop. Else just regular CAMs.')

    # Hard-coded settings (tested for PE dataset only)
    args_ = parser.parse_args()
    args_.do_hflip = False
    args_.do_vflip = False
    args_.do_center_pe = True
    args_.do_jitter = False
    args_.do_rotate = False
    args_.use_hem = False
    args_.abnormal_prob = 1.

    get_cams(args_)
