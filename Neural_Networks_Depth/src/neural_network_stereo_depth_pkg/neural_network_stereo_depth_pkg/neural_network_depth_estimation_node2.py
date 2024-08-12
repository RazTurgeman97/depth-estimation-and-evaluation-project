import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge
from datetime import datetime
import time
from neural_network_stereo_depth_pkg.hitnet_model import HitNet, ModelType, CameraConfig as HitNetCameraConfig
from neural_network_stereo_depth_pkg.cre_model import CREStereoModel, CameraConfig as CRECameraConfig
import subprocess

class NeuralNetworkDepthEstimationNode(Node):
    def __init__(self):
        super().__init__('neural_network_depth_estimation_node')
        
        self.get_logger().info('Initializing NeuralNetworkDepthEstimationNode...')

        # Start playing the ROS bag as soon as the node starts
        self.start_rosbag_playback()

        # Subscribe to the topics
        self.subscription_left = self.create_subscription(
            Image,
            '/camera/camera/infra1/image_rect_raw',
            self.left_image_callback,
            10)
        
        self.subscription_right = self.create_subscription(
            Image,
            '/camera/camera/infra2/image_rect_raw',
            self.right_image_callback,
            10)
        
        self.subscription_depth = self.create_subscription(
            Image,
            '/camera_triangulation/raw_depth_map',
            self.triangulation_depth_callback,
            10)

        self.publisher_hitnet_depth = self.create_publisher(Image, '/HITNET/depth', 10)
        self.publisher_cre_depth = self.create_publisher(Image, '/CRE/depth', 10)
        self.publisher_hitnet_raw_depth = self.create_publisher(Image, '/HITNET/raw_depth', 10)
        self.publisher_cre_raw_depth = self.create_publisher(Image, '/CRE/raw_depth', 10)

        self.bridge = CvBridge()
        self.load_calibration()

        # Store images and depth maps
        self.left_image = None
        self.right_image = None
        self.triangulation_depth_image = None
        self.hitnet_depth_maps = []
        self.cre_depth_maps = []
        self.exit_flag = False
        
        self.frame_size_key = None
        self.hitnet_model = None
        self.cre_model = None

        self.get_logger().info('Neural Network Depth Estimation Node has started.')

    def start_rosbag_playback(self):
        try:
            self.get_logger().info('Starting ROS bag playback...')
            subprocess.Popen(['ros2', 'bag', 'play', '/workspaces/depth-estimation-project/Neural_Networks_Depth/src/recordings/indoor_recording/indoor_recording_0.db3'])
            self.get_logger().info('ROS bag playback started.')
        except Exception as e:
            self.get_logger().error(f'Failed to start ROS bag playback: {e}')

    def left_image_callback(self, msg):
        self.left_image = self.bridge.imgmsg_to_cv2(msg, 'mono8')
        self.get_logger().info('Received left image.')
        self.check_and_initialize_models(self.left_image)

    def right_image_callback(self, msg):
        self.right_image = self.bridge.imgmsg_to_cv2(msg, 'mono8')
        self.get_logger().info('Received right image.')
        self.check_and_initialize_models(self.right_image)

    def triangulation_depth_callback(self, msg):
        self.triangulation_depth_image = self.bridge.imgmsg_to_cv2(msg, '32FC1')
        self.get_logger().info('Received triangulation depth image.')

    def check_and_initialize_models(self, image):
        if self.frame_size_key is None:
            self.get_logger().info('Selecting frame size key...')
            if not self.select_frame_size_key(image):
                self.get_logger().error("Frame size not found in calibration data.")
                return
            self.get_logger().info('Initializing models...')
            self.initialize_models()

    def process_images(self):
        if self.left_image is None or self.right_image is None:
            self.get_logger().info('Waiting for both left and right images.')
            return
        
        self.get_logger().info('Starting depth estimation...')

        if self.hitnet_model is None or self.cre_model is None:
            self.get_logger().error("Models not initialized.")
            return

        # HITNET model depth estimation
        try:
            self.get_logger().info('Estimating depth using HITNET model...')
            hitnet_disparity = self.hitnet_model(self.left_image, self.right_image)
            hitnet_depth = self.hitnet_model.get_depth_from_disparity(hitnet_disparity, self.hitnet_model.camera_config)
            hitnet_depth /= 1000.0  # Convert from millimeters to meters
            hitnet_depth = np.clip(hitnet_depth, 0, 10.0)  # Clip depth values to a reasonable range in meters
            self.hitnet_depth_maps.append(hitnet_depth)
            self.get_logger().info('HITNET depth estimation completed.')
        except Exception as e:
            self.get_logger().error(f'Error in HITNET model: {e}')

        # CRE model depth estimation
        try:
            self.get_logger().info('Estimating depth using CRE model...')
            cre_depth = self.cre_model.estimate_depth(self.left_image, self.right_image)
            cre_depth /= 1000.0  # Convert from millimeters to meters
            cre_depth is np.clip(cre_depth, 0, 10.0)  # Clip depth values to a reasonable range in meters
            self.cre_depth_maps.append(cre_depth)
            self.get_logger().info('CRE depth estimation completed.')
        except Exception as e:
            self.get_logger().error(f'Error in CRE model: {e}')

    def publish_stored_depth_maps(self):
        self.get_logger().info('Publishing stored depth maps at 60 fps...')
        frame_duration = 1.0 / 60.0  # For 60 fps
        
        for hitnet_depth, cre_depth in zip(self.hitnet_depth_maps, self.cre_depth_maps):
            self.publish_depth(hitnet_depth, "HITNET")
            self.publish_raw_depth_map(hitnet_depth, "HITNET")
            self.publish_depth(cre_depth, "CRE")
            self.publish_raw_depth_map(cre_depth, "CRE")
            
            self.get_logger().info('Published a set of depth maps.')
            time.sleep(frame_duration)

        self.get_logger().info('Completed publishing all stored depth maps.')

    def publish_depth(self, depth_map, model_type):
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_map_colored = cv2.applyColorMap(cv2.convertScaleAbs(depth_map_normalized, 1), cv2.COLORMAP_JET)
        depth_msg = self.bridge.cv2_to_imgmsg(depth_map_colored, encoding='bgr8')
        
        if model_type == "HITNET":
            self.publisher_hitnet_depth.publish(depth_msg)
        elif model_type == "CRE":
            self.publisher_cre_depth.publish(depth_msg)

    def publish_raw_depth_map(self, depth_map, model_type):
        depth_map_float = depth_map.astype(np.float32)
        depth_msg = self.bridge.cv2_to_imgmsg(depth_map_float, encoding='32FC1')
        
        if model_type == "HITNET":
            self.publisher_hitnet_raw_depth.publish(depth_msg)
        elif model_type == "CRE":
            self.publisher_cre_raw_depth.publish(depth_msg)

    def select_frame_size_key(self, image):
        height, width = image.shape[:2]
        for key, params in self.calib["rectified"].items():
            if params["width"] == width and params["height"] == height:
                self.frame_size_key = key
                self.fx = params["fx"]
                self.fy = params["fy"]
                self.ppx = params["ppx"]
                self.ppy = params["ppy"]
                self.image_size = (width, height)
                self.camera_matrix_left = np.array([
                    [self.fx, 0, self.ppx],
                    [0, self.fy, self.ppy],
                    [0, 0, 1]
                ])
                self.get_logger().info(f"Frame size key {self.frame_size_key} selected with size {self.image_size}")
                return True
        return False

    def load_calibration(self):
        self.get_logger().info('Loading calibration data...')
        # Calibration data remains the same as in your original implementation.
        self.calib = {
            "baseline": -95.044,
            "intrinsic_left": [
                [0.506, 0.809, 0.503],
                [0.493, -0.062, 0.070],
                [-0.000, 0.000, -0.023]
            ],
            "intrinsic_right": [
                [0.505, 0.806, 0.505],
                [0.496, -0.059, 0.071],
                [-0.000, 0.001, -0.023]
            ],
            "rectified": {
                "0": {"fx": 968.857, "fy": 968.857, "width": 1920, "height": 1080, "ppx": 970.559, "ppy": 533.393},
                "1": {"fx": 645.905, "fy": 645.905, "width": 1280, "height": 720, "ppx": 647.039, "ppy": 355.595},
                "2": {"fx": 387.543, "fy": 387.543, "width": 640, "height": 480, "ppx": 324.224, "ppy": 237.357},
                "3": {"fx": 427.912, "fy": 427.912, "width": 848, "height": 480, "ppx": 428.664, "ppy": 237.082},
                "4": {"fx": 322.952, "fy": 322.952, "width": 640, "height": 360, "ppx": 323.52, "ppy": 177.798},
                "5": {"fx": 213.956, "fy": 213.956, "width": 424, "height": 240, "ppx": 214.332, "ppy": 118.541},
                "6": {"fx": 193.771, "fy": 193.771, "width": 320, "height": 240, "ppx": 162.112, "ppy": 118.679},
                "7": {"fx": 242.214, "fy": 242.214, "width": 480, "height": 270, "ppx": 242.64, "ppy": 133.299},
                "8": {"fx": 645.905, "fy": 645.905, "width": 1280, "height": 800, "ppx": 647.039, "ppy": 395.595},
                "9": {"fx": 484.429, "fy": 484.429, "width": 960, "height": 540, "ppx": 485.279, "ppy": 266.696},
                "10": {"fx": 581.314, "fy": 581.314, "width": 0, "height": 0, "ppx": 366.335, "ppy": 356.036},
                "11": {"fx": 465.052, "fy": 465.052, "width": 0, "height": 0, "ppx": 293.068, "ppy": 284.829},
                "12": {"fx": 645.905, "fy": 645.905, "width": 640, "height": 400, "ppx": 647.039, "ppy": 395.595},
                "13": {"fx": 4.70255e-37, "fy": 0, "width": 576, "height": 576, "ppx": 0, "ppy": 0},
                "14": {"fx": 0, "fy": 0, "width": 720, "height": 720, "ppx": 0, "ppy": 0},
                "15": {"fx": 0, "fy": 0, "width": 1152, "height": 1152, "ppx": 0, "ppy": 0}
            },
            "world2left_rot": [
                [1.000, -0.002, 0.007],
                [0.002, 1.000, 0.001],
                [-0.007, -0.001, 1.000]
            ],
            "world2right_rot": [
                [1.000, 0.004, -0.006],
                [-0.004, 1.000, -0.001],
                [0.006, 0.001, 1.000]
            ]
        }
        self.baseline = np.abs(self.calib["baseline"])
        self.R_mat = np.array(self.calib["world2left_rot"])
        self.get_logger().info('Calibration data loaded.')

    def initialize_models(self):
        self.get_logger().info('Initializing HITNET and CRE models...')
        baseline = self.baseline
        focal_length = self.fx  # Use fx as the focal length

        hitnet_model_path = '/root/ros2_ws/src/neural_network_stereo_depth_pkg/models/ONNX-HITNET-Stereo-Depth-estimation/models/middlebury_d400/saved_model_480x640/model_float32.onnx'
        hitnet_model_type = ModelType.middlebury
        hitnet_camera_config = HitNetCameraConfig(baseline, focal_length)
        self.hitnet_model = HitNet(hitnet_model_path, hitnet_model_type, hitnet_camera_config, max_dist=5)

        cre_model_path = '/root/ros2_ws/src/neural_network_stereo_depth_pkg/models/ONNX-CREStereo-Depth-Estimation/models/crestereo_init_iter10_480x640.onnx'
        cre_camera_config = CRECameraConfig(baseline, focal_length)
        self.cre_model = CREStereoModel(cre_model_path, cre_camera_config, max_dist=10)
        self.get_logger().info('Models initialized.')

    def run_depth_evaluation(self):
        self.get_logger().info('Starting depth evaluation...')
        # Replace this with the command or method to start your depth evaluation node
        subprocess.Popen(['ros2', 'run', 'depth_eval_pkg', 'depth_eval_node'])
        self.get_logger().info('Depth evaluation started.')

def main(args=None):
    rclpy.init(args=args)
    node = NeuralNetworkDepthEstimationNode()
    try:
        # Process the entire ROS bag
        while rclpy.ok() and not node.exit_flag:
            rclpy.spin_once(node, timeout_sec=0.1)
            node.process_images()
        
        node.get_logger().info('Processing completed. Now publishing stored depth maps...')
        
        # Publish the stored depth maps
        node.publish_stored_depth_maps()
        
        # Start the depth evaluation
        node.run_depth_evaluation()
        
    except KeyboardInterrupt:
        node.get_logger().info('Node interrupted by keyboard.')
    finally:
        node.destroy_node()
        rclpy.shutdown()
        node.get_logger().info('Node shutdown.')

if __name__ == '__main__':
    main()
