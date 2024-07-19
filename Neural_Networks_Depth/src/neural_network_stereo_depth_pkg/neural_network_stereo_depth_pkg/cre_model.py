# cre_model.py

import cv2
import numpy as np
import onnxruntime
from dataclasses import dataclass

@dataclass
class CameraConfig:
    baseline: float
    f: float

# DEFAULT_CONFIG = CameraConfig(0.546, 120)  # Rough estimate from the original calibration

class CREStereoModel:
    def __init__(self, model_path, camera_config, max_dist):
        self.camera_config = camera_config
        self.max_dist = max_dist
        self.session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.get_input_details()
        self.get_output_details()
        self.has_flow = len(self.input_names) > 2

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[-1].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
        self.output_shape = model_outputs[0].shape

    def prepare_input(self, img, half=False):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if half:
            img_input = cv2.resize(img, (self.input_width // 2, self.input_height // 2), cv2.INTER_AREA)
        else:
            img_input = cv2.resize(img, (self.input_width, self.input_height), cv2.INTER_AREA)
        img_input = img_input.transpose(2, 0, 1)
        img_input = img_input[np.newaxis, :, :, :]
        return img_input.astype(np.float32)

    def inference(self, left_tensor, right_tensor):
        return self.session.run(self.output_names, {self.input_names[0]: left_tensor, self.input_names[1]: right_tensor})[0]

    def inference_with_flow(self, left_tensor_half, right_tensor_half, left_tensor, right_tensor):
        return self.session.run(self.output_names, {self.input_names[0]: left_tensor_half, self.input_names[1]: right_tensor_half, self.input_names[2]: left_tensor, self.input_names[3]: right_tensor})[0]

    def process_output(self, output):
        return np.squeeze(output[:, 0, :, :])

    def get_depth_from_disparity(self, disparity_map):
        return self.camera_config.f * self.camera_config.baseline / disparity_map

    def estimate_depth(self, left_img, right_img):
        self.img_height, self.img_width = left_img.shape[:2]
        left_tensor = self.prepare_input(left_img)
        right_tensor = self.prepare_input(right_img)
        if self.has_flow:
            left_tensor_half = self.prepare_input(left_img, half=True)
            right_tensor_half = self.prepare_input(right_img, half=True)
            outputs = self.inference_with_flow(left_tensor_half, right_tensor_half, left_tensor, right_tensor)
        else:
            outputs = self.inference(left_tensor, right_tensor)
        self.disparity_map = self.process_output(outputs)
        self.depth_map = self.get_depth_from_disparity(self.disparity_map)
        return self.depth_map

    def draw_depth(self):
        return self.util_draw_depth(self.depth_map, (self.img_width, self.img_height), self.max_dist)

    @staticmethod
    def util_draw_depth(depth_map, img_shape, max_dist):
        norm_depth_map = 255 * (1 - depth_map / max_dist)
        norm_depth_map[norm_depth_map < 0] = 0
        norm_depth_map[norm_depth_map >= 255] = 0
        norm_depth_map = cv2.resize(norm_depth_map, img_shape)
        return cv2.applyColorMap(cv2.convertScaleAbs(norm_depth_map, 1), cv2.COLORMAP_MAGMA)
