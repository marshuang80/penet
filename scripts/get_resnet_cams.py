import cv2
import numpy as np
import os
import torch
import util

from args import TestArgParser
from cams import GradCAM
from data_loader import CTDataLoader
from imageio import imsave
from saver import ModelSaver
from torchvision import models

# TODO: generalize to last layer of features
MODEL_CONFIGS = {
    'Resnet50': {
        'target_layer': 'module.model.layer4'
    },
    'VGG_19': {
        'target_layer': 'features.36'
    }
}


def get_cams(args):
    print('Loading model...')
    model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args.gpu_ids)
    model = model.to(args.device)
    args.start_epoch = ckpt_info['epoch'] + 1

    config = MODEL_CONFIGS['{}'.format(args.model)]
    print('Extracting feature maps from layer named "{}"...'.format(config['target_layer']))

    grad_cam = GradCAM(model, args.device, is_binary=True, is_3d=False)
    
    num_generated = 0
    data_loader = CTDataLoader(args, phase=args.phase, is_training=True)
    for inputs, labels in data_loader:
        if labels.item() == 0:
            # Keep going until we get an abnormal study
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
            cam = grad_cam.get_cam(config['target_layer'])

            inputs2 = torch.autograd.Variable(inputs, requires_grad=True)

        print('Overlaying CAM...')
        cam = np.expand_dims(cam, axis=0)
        new_cam = util.resize(cam, inputs[0])
        
        np_cam = util.un_normalize(inputs[0], args.img_format, data_loader.dataset.pixel_dict)
        np_cam = np.transpose(np_cam, (1, 2, 3, 0))

        np_cam = np.float32(np_cam) / 255
        final_cam = util.add_heat_map(np_cam, new_cam).astype(np.float32)

        # Write to a PNG file
        output_path_cam = os.path.join(args.cam_dir, 'cam_{}.png'.format(num_generated + 1))
        output_path_origin = os.path.join(args.cam_dir, 'orig_{}.png'.format(num_generated + 1))

        print('Writing Original to {}...'.format(output_path_origin))
        orig = util.normalize_image(np_cam)
        orig = np.squeeze(orig)
        imsave(output_path_origin, orig.astype(np.uint8))
        
        print('Writing CAM to {}...'.format(output_path_cam))
        cam_png = util.normalize_image(final_cam)
        cam_png = np.squeeze(cam_png)
        imsave(output_path_cam, cam_png.astype(np.uint8))
        
        num_generated += 1
        if num_generated == args.num_cams:
            return


def test_2d(args):
    """Sanity check for CAMs (2D VGG19 on ImageNet)"""
    def preprocess_image(img):
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]

        preprocessed_img = img.copy()[:, :, ::-1]
        for i in range(3):
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
        preprocessed_img = np.ascontiguousarray(np.transpose(img, (2, 0, 1)))

        return preprocessed_img

    config = MODEL_CONFIGS['VGG_19']

    # Load test image
    print('Loading image...')
    img = cv2.imread('data/test_image.jpg', 1)
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    img = preprocess_image(img)
    img_tensor = torch.from_numpy(img)
    img_tensor.unsqueeze_(0)
    img_tensor.requires_grad_(True)
    img_tensor.to(args.device)

    # Wrap VGG19 in a grad CAM generator
    print('Loading model...')
    model = models.vgg19(pretrained=True)
    model.to(args.device)
    grad_cam = GradCAM(model, args.device, is_binary=False, is_3d=False)
    if args.gbp:
        gbp = GuidedBackPropagation(model, args.device, is_binary=False, is_3d=False)

    # Get CAM for highest scoring class
    print('Generating CAM...')
    with torch.set_grad_enabled(True):
        probs, idx = grad_cam.forward(img_tensor)
        if args.gbp:
            probs2, idx2 = gbp.forward(img_tensor)

        img = np.transpose(img, (1, 2, 0))  # Put channels last
        for i in range(3):                  # Look at top 3 predictions
            grad_cam.backward(idx=idx[i])
            cam = grad_cam.get_cam(config['target_layer'])
            if args.gbp:
                gbp.backward(idx=idx2[i])
                guided_backprop = np.transpose(gbp.generate(), (1, 2, 0))

            # Overlay CAM on the image and save it
            print('Overlaying CAM...')
            cam = cv2.resize(cam, img.shape[:2])

            if args.gbp:
                g_cam = util.normalize_image(guided_backprop * np.expand_dims(cam, axis=2))
                output_path = os.path.join(args.cam_dir, 'gcam_{}.png'.format(i + 1))
                imsave(output_path, g_cam.astype(np.uint8))

            else:
                img_with_cam = util.add_heat_map(img, cam)
                output_path = os.path.join(args.cam_dir, 'cam_{}.png'.format(i + 1))
                imsave(output_path, img_with_cam)


if __name__ == '__main__':
    parser = TestArgParser()
    parser.parser.add_argument('--test_2d', action='store_true', help='Test CAMs on a pretrained 2D VGG net.')
    parser.parser.add_argument('--cam_dir', type=str, default='data/', help='Directory to write CAM outputs.')
    parser.parser.add_argument('--num_cams', type=int, default=1, help='Number of CAMs to generate.')
    args_ = parser.parse_args()

    if args_.test_2d:
        test_2d(args_)
    else:
        get_cams(args_)
