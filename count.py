import cv2
import json
import pickle
import numpy as np
import os
import sklearn.metrics as sk_metrics
import torch
import torch.nn.functional as F
import util

from args import TestArgParser
from data_loader import CTDataLoader
from collections import defaultdict
from logger import TestLogger
from PIL import Image
from saver import ModelSaver
from tqdm import tqdm

def test(args):
    model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args.gpu_ids)

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    print(pytorch_total_params)

if __name__ == '__main__':
    util.set_spawn_enabled()
    parser = TestArgParser()
    args_ = parser.parse_args()
    test(args_)
