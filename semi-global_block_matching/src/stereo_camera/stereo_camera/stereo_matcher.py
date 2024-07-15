#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import yaml
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class StereoMatcherNode(Node):
    def __init__(self):
        super().__init__('stereo_matcher_node')
        self.subscription_left = self.create_subscription(
            Image,
            '/left_camera/image_raw',
            self.left_callback,
            10)
        self.subscription_right = self.create_subscription(
            Image,
            '/right_camera/image_raw',
            self.right_callback,
            10)
        self.publisher_disparity = self.create_publisher(Image, 'sgbm_algorithm_disparity_map', 10)
        self.get_logger().info("Subscriptions and publisher set up.")
        self.declare_parameter('calibration_file', '/opt/ros/humble/share/sgbm_init/config/calibration-camchain.yaml')
        calibration_file = self.get_parameter('calibration_file').get_parameter_value().string_value
        self.load_calibration(calibration_file)
        self.bridge = CvBridge()
        self.left_image = None
        self.right_image = None
        self.get_logger().info("Stereo Matcher Node has started.")

        self.exit_flag = False  # Add an exit flag
        self.sideBySide = False  # flag for camera layers view

    def load_calibration(self, filename):
        with open(filename, 'r') as file:
            calib = yaml.safe_load(file)
        self.camera_matrix_left = np.array([
            [calib['cam0']['intrinsics'][0], 0, calib['cam0']['intrinsics'][2]],
            [0, calib['cam0']['intrinsics'][1], calib['cam0']['intrinsics'][3]],
            [0, 0, 1]
        ])
        self.dist_coeffs_left = np.array(calib['cam0']['distortion_coeffs'])
        self.camera_matrix_right = np.array([
            [calib['cam1']['intrinsics'][0], 0, calib['cam1']['intrinsics'][2]],
            [0, calib['cam1']['intrinsics'][1], calib['cam1']['intrinsics'][3]],
            [0, 0, 1]
        ])
        self.dist_coeffs_right = np.array(calib['cam1']['distortion_coeffs'])
        T_cn_cnm1 = np.array(calib['cam1']['T_cn_cnm1'])
        t = T_cn_cnm1[:3, 3]
        R_mat = T_cn_cnm1[:3, :3]
        image_size = (1280, 720)
        self.R1, self.R2, self.P1, self.P2, self.Q, self.ROI_L, self.ROI_R = cv2.stereoRectify(
            self.camera_matrix_left, self.dist_coeffs_left,
            self.camera_matrix_right, self.dist_coeffs_right,
            image_size, R_mat, t)
        self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(
            self.camera_matrix_left, self.dist_coeffs_left, self.R1, self.P1, image_size, cv2.CV_16SC2)
        self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(
            self.camera_matrix_right, self.dist_coeffs_right, self.R2, self.P2, image_size, cv2.CV_16SC2)

        # Print the loaded calibration data
        print("Left Camera Matrix:\n", self.camera_matrix_left)
        print("Left Distortion Coefficients:", self.dist_coeffs_left)
        print("Right Camera Matrix:\n", self.camera_matrix_right)
        print("Right Distortion Coefficients:", self.dist_coeffs_right)
        print("Rotation Matrix:\n", R_mat)
        print("Translation Vector:", t)
        print("Rectification Matrix Left:\n", self.R1)
        print("Projection Matrix Left:\n", self.P1)
        print("Rectification Matrix Right:\n", self.R2)
        print("Projection Matrix Right:\n", self.P2)
        print("Disparity-to-depth Mapping Matrix (Q):\n", self.Q)

    def left_callback(self, data):
        try:
            left_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='mono8')
            rectified_left = cv2.remap(left_image, self.map1_left, self.map2_left, cv2.INTER_LANCZOS4)
            self.left_image = rectified_left
        except CvBridgeError as e:
            self.get_logger().error(f"Error converting left image: {str(e)}")

    def right_callback(self, data):
        try:
            right_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='mono8')
            rectified_right = cv2.remap(right_image, self.map1_right, self.map2_right, cv2.INTER_LANCZOS4)
            self.right_image = rectified_right
            if self.left_image is not None and self.right_image is not None:
                self.compute_and_publish_disparity()
                self.left_image = None  # Reset images after processing
                self.right_image = None
        except CvBridgeError as e:
            self.get_logger().error(f"Error converting right image: {str(e)}")

    def compute_left_disparity_SGBM(self):
        window_size = 4
        min_disp = 2
        nDispFactor = 19
        num_disp = 16 * nDispFactor - min_disp

        stereo_left = cv2.StereoSGBM.create(minDisparity=min_disp,
                                            numDisparities=num_disp,
                                            blockSize=window_size,
                                            P1=8 * 3 * window_size ** 2,
                                            P2=32 * 3 * window_size ** 2,
                                            disp12MaxDiff=1,
                                            uniquenessRatio=15,
                                            speckleWindowSize=0,
                                            speckleRange=2,
                                            preFilterCap=5,
                                            mode=cv2.StereoSGBM_MODE_SGBM_3WAY)

        if self.left_image is not None and self.right_image is not None:
            left_disparity = stereo_left.compute(self.left_image, self.right_image).astype(np.float32)
            left_disparity = (left_disparity / 16.0)  # Divide by 16 to convert to proper scale
            # self.get_logger().info(f"Disparity Min: {np.min(left_disparity)}, Max: {np.max(left_disparity)}")
        else:
            self.get_logger().error("Images not available for disparity calculation.")
            return None

        return left_disparity

    def apply_morphological_ops(self, disparity, kernelVal, iterations):
        disparity_8u = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        kernel = np.ones((kernelVal, kernelVal), np.uint8)
        filtered_disparity = cv2.medianBlur(disparity_8u, 1)

        morphed = filtered_disparity
        for _ in range(iterations):
            morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel)
            morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel)

        return morphed

    def compute_and_publish_disparity(self):
        disparity = self.compute_left_disparity_SGBM()
        if disparity is not None:
            disparity_processed = self.apply_morphological_ops(disparity, 14, 2)
            disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            disparity_filtered = cv2.bilateralFilter(disparity_normalized, 7, 30, 100)
            disparity_colormap = cv2.applyColorMap(disparity_filtered, cv2.COLORMAP_JET)

            try:
                disparity_image_msg = self.bridge.cv2_to_imgmsg(disparity_colormap, "bgr8")
                self.publisher_disparity.publish(disparity_image_msg)
            except CvBridgeError as e:
                self.get_logger().error(f"Error converting disparity image: {str(e)}")
        else:
            self.get_logger().error("Disparity computation failed.")

def main(args=None):
    rclpy.init(args=args)
    stereo_matcher_node = StereoMatcherNode()
    while rclpy.ok():
        rclpy.spin_once(stereo_matcher_node, timeout_sec=0.1)
        if stereo_matcher_node.exit_flag:
            break
    stereo_matcher_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
