import os
from io import BytesIO
import pydicom
import torch
import numpy as np
from custom_gradcam import CustomGradCAM
from saver import ModelSaver
import util
from helper import preprocess, compute_gradcam_gif

DEFAULT_PROBABILITY_THRESHOLD = "0.5"
DEFAULT_INPUT_SLICE_NUMBER = "24"
GRADCAM_OFF = "0"
GRADCAM_MAX_PROBABILITY = "1"


class MDAIModel:
    def __init__(self):
        root_path = os.path.dirname(os.path.dirname(__file__))

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            gpu_ids = list(range(torch.cuda.device_count()))
        else:
            self.device = torch.device("cpu")
            gpu_ids = []

        self.model, _ = ModelSaver.load_model(
            os.path.join(root_path, "penet_best.pth.tar"), gpu_ids
        )
        self.model = self.model.to(self.device)
        self.grad_cam = CustomGradCAM(
            self.model, self.device, is_binary=True, is_3d=True
        )

    def predict(self, data):
        """
        The input data has the following schema:

        {
            "instances": [
                {
                    "file": "bytes"
                    "tags": {
                        "StudyInstanceUID": "str",
                        "SeriesInstanceUID": "str",
                        "SOPInstanceUID": "str",
                        ...
                    }
                },
                ...
            ],
            "args": {
                "arg1": "str",
                "arg2": "str",
                ...
            }
        }

        Model scope specifies whether an entire study, series, or instance is given to the model.
        If the model scope is 'INSTANCE', then `instances` will be a single instance (list length of 1).
        If the model scope is 'SERIES', then `instances` will be a list of all instances in a series.
        If the model scope is 'STUDY', then `instances` will be a list of all instances in a study.

        The additional `args` dict supply values that may be used in a given run.

        For a single instance dict, `files` is the raw binary data representing a DICOM file, and
        can be loaded using: `ds = pydicom.dcmread(BytesIO(instance["file"]))`.

        The results returned by this function should have the following schema:

        [
            {
                "type": "str", // 'NONE', 'ANNOTATION', 'IMAGE', 'DICOM', 'TEXT'
                "study_uid": "str",
                "series_uid": "str",
                "instance_uid": "str",
                "frame_number": "int",
                "class_index": "int",
                "data": {},
                "probability": "float",
                "explanations": [
                    {
                        "name": "str",
                        "description": "str",
                        "content": "bytes",
                        "content_type": "str",
                    },
                    ...
                ],
            },
            ...
        ]

        The DICOM UIDs must be supplied based on the scope of the label attached to `class_index`.
        """
        input_instances = data["instances"]
        input_args = data["args"]

        x_arrays = []
        for instance in input_instances:
            tags = instance["tags"]
            ds = pydicom.dcmread(BytesIO(instance["file"]))
            x_orig = ds
            x_arrays.append(x_orig)

        input_slice_number = int(
            input_args.get("input_slice_number", DEFAULT_INPUT_SLICE_NUMBER)
        )
        # Handles inputs with small slices
        if len(x_arrays) < input_slice_number:
            input_slice_number = len(x_arrays)

        x, x_un_normalized = preprocess(x_arrays, input_slice_number)
        self.model.eval()

        best_window = 0
        probability = 0.0
        with torch.no_grad():
            for i, window in enumerate(x):
                cls_logits = self.model.forward(
                    window.to(self.device, dtype=torch.float)
                )
                cls_probs = torch.sigmoid(cls_logits).to("cpu").numpy()
                if cls_probs[0][0] > probability:
                    probability = cls_probs[0][0]
                    best_window = i

        if not probability >= float(
            input_args.get("probability_threshold", DEFAULT_PROBABILITY_THRESHOLD)
        ):
            result = {
                "type": "NONE",
                "study_uid": tags["StudyInstanceUID"],
                "series_uid": tags["SeriesInstanceUID"],
                "instance_uid": tags["SOPInstanceUID"],
                "frame_number": None,
            }
        else:
            result = {
                "type": "ANNOTATION",
                "study_uid": tags["StudyInstanceUID"],
                "series_uid": tags["SeriesInstanceUID"],
                "frame_number": None,
                "class_index": 0,
                "data": None,
                "probability": float(probability),
            }

            if input_args.get("gradcam", GRADCAM_OFF) == GRADCAM_MAX_PROBABILITY:
                self.grad_cam.register_hooks()

                with torch.set_grad_enabled(True):
                    probs, idx = self.grad_cam.forward(x[best_window])
                    self.grad_cam.backward(idx=idx[0])
                    cam = self.grad_cam.get_cam("module.encoders.3")

                self.grad_cam.remove_hooks()

                gradcam_output_buffer = compute_gradcam_gif(
                    cam, x[best_window], x_un_normalized[best_window]
                )
                gradcam_explanation = [
                    {
                        "name": "Grad-CAM",
                        "description": "Visualize how parts of the image affects neural network’s output by looking into the activation maps. From _Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization_ (https://arxiv.org/abs/1610.02391)",
                        "content": gradcam_output_buffer.getvalue(),
                        "content_type": "image/gif",
                    }
                ]

                result["explanations"] = gradcam_explanation

        return [result]
