import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge
import yaml
from datetime import datetime
from collections import deque
from neural_network_stereo_depth_pkg.hitnet_model import HitNet, ModelType, CameraConfig as HitNetCameraConfig
from neural_network_stereo_depth_pkg.cre_model import CREStereoModel, CameraConfig as CRECameraConfig

class NeuralNetworkDepthEstimationNode(Node):
    def __init__(self):
        super().__init__('neural_network_depth_estimation_node')
        self.subscription_left = self.create_subscription(
            Image,
            '/left_camera/image_raw',
            self.left_image_callback,
            10)
        self.subscription_right = self.create_subscription(
            Image,
            '/right_camera/image_raw',
            self.right_image_callback,
            10)
        
        self.publisher_hitnet_depth = self.create_publisher(Image, '/hitnet/depth', 10)
        self.publisher_cre_depth = self.create_publisher(Image, '/cre/depth', 10)
        
        self.bridge = CvBridge()
        self.declare_parameter('calibration_file', '/opt/ros/humble/share/depth_estimation/config/calibration-camchain.yaml')
        calibration_file = self.get_parameter('calibration_file').get_parameter_value().string_value
        self.load_calibration(calibration_file)
        self.get_logger().info("Calibration Data Loaded Successfully.")

        self.left_image = None
        self.right_image = None
        self.exit_flag = False
        self.depth_history = deque(maxlen=7)  # Store the last 7 frames for temporal smoothing

        # Extract baseline and focal length from calibration data
        baseline = abs(self.T[0])
        focal_length = self.camera_matrix_left[0, 0]  # Assuming fx from intrinsics

        hitnet_model_path = '/root/ros2_ws/src/neural_network_stereo_depth_pkg/models/ONNX-HITNET-Stereo-Depth-estimation/models/middlebury_d400/saved_model_480x640/model_float32.onnx'
        hitnet_model_type = ModelType.middlebury
        hitnet_camera_config = HitNetCameraConfig(baseline, focal_length)
        self.hitnet_model = HitNet(hitnet_model_path, hitnet_model_type, hitnet_camera_config, max_dist=5)

        cre_model_path = '/root/ros2_ws/src/neural_network_stereo_depth_pkg/models/ONNX-CREStereo-Depth-Estimation/models/crestereo_init_iter10_360x640.onnx'
        cre_camera_config = CRECameraConfig(baseline, focal_length)
        self.cre_model = CREStereoModel(cre_model_path, cre_camera_config, max_dist=10)

        self.get_logger().info('Subscriptions set up.')
        self.get_logger().info('Neural Network Depth Estimation Node has started.')

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
        self.T = T_cn_cnm1[:3, 3]
        R_mat = T_cn_cnm1[:3, :3]
        image_size = (1280, 720)
        self.R1, self.R2, self.P1, self.P2, self.Q, self.ROI_L, self.ROI_R = cv2.stereoRectify(
            self.camera_matrix_left, self.dist_coeffs_left,
            self.camera_matrix_right, self.dist_coeffs_right,
            image_size, R_mat, self.T)
        self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(
            self.camera_matrix_left, self.dist_coeffs_left, self.R1, self.P1, image_size, cv2.CV_16SC2)
        self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(
            self.camera_matrix_right, self.dist_coeffs_right, self.R2, self.P2, image_size, cv2.CV_16SC2)

    def left_image_callback(self, msg):
        self.left_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        rectified_left = cv2.remap(self.left_image, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
        self.left_image = rectified_left
        if self.right_image is not None:
            self.process_images()

    def right_image_callback(self, msg):
        self.right_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8').reshape(msg.height, msg.width, -1)
        rectified_right = cv2.remap(self.right_image, self.map1_right, self.map2_right, cv2.INTER_LANCZOS4)
        self.right_image = rectified_right
        if self.left_image is not None:
            self.process_images()

    def process_images(self):
        # HITNET model depth estimation
        try:
            start_time = datetime.now()
            hitnet_disparity = self.hitnet_model(self.left_image, self.right_image)
            hitnet_depth = self.hitnet_model.get_depth_from_disparity(hitnet_disparity, self.hitnet_model.camera_config)
            
            # Temporal smoothing
            self.depth_history.append(hitnet_depth)
            smoothed_depth = np.mean(self.depth_history, axis=0)
            
            end_time = datetime.now()
            duration = end_time - start_time
            self.get_logger().info(f'HITNET inference time: {duration.total_seconds()} seconds')
            self.get_logger().info(f'HITNET depth map range: min {smoothed_depth.min()}, max {smoothed_depth.max()}')
            self.publish_depth(smoothed_depth, "hitnet")
        except Exception as e:
            self.get_logger().error(f'Error in HITNET model: {e}')

        # CRE model depth estimation
        try:
            start_time = datetime.now()
            cre_depth = self.cre_model.estimate_depth(self.left_image, self.right_image)
            end_time = datetime.now()
            duration = end_time - start_time
            self.get_logger().info(f'CRE inference time: {duration.total_seconds()} seconds')
            self.get_logger().info(f'CRE depth map range: min {cre_depth.min()}, max {cre_depth.max()}')
            self.publish_depth(cre_depth, "cre")
        except Exception as e:
            self.get_logger().error(f'Error in CRE model: {e}')

    def publish_depth(self, depth_map, model_type):
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_map_colored = cv2.applyColorMap(cv2.convertScaleAbs(255 - depth_map_normalized, 1), cv2.COLORMAP_JET)
        depth_msg = self.bridge.cv2_to_imgmsg(depth_map_colored, encoding='bgr8')
        
        if model_type == "hitnet":
            self.publisher_hitnet_depth.publish(depth_msg)
        elif model_type == "cre":
            self.publisher_cre_depth.publish(depth_msg)

def main(args=None):
    rclpy.init(args=args)
    node = NeuralNetworkDepthEstimationNode()
    try:
        while rclpy.ok() and not node.exit_flag:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
