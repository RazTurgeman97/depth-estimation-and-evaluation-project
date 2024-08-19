import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge
from datetime import datetime
import message_filters
from neural_network_stereo_depth_pkg.hitnet_model import HitNet, ModelType, CameraConfig as HitNetCameraConfig
from neural_network_stereo_depth_pkg.cre_model import CREStereoModel, CameraConfig as CRECameraConfig

class NeuralNetworkDepthEstimationNode(Node):
    def __init__(self):
        super().__init__('neural_network_depth_estimation_node')

        # Subscribers with message filters
        self.left_image_sub = message_filters.Subscriber(self, Image, '/camera/camera/infra1/image_rect_raw')
        self.right_image_sub = message_filters.Subscriber(self, Image, '/camera/camera/infra2/image_rect_raw')

        # Synchronize the left and right images
        self.ts = message_filters.ApproximateTimeSynchronizer([self.left_image_sub, self.right_image_sub], queue_size=1, slop=0.1)
        self.ts.registerCallback(self.image_callback)

        self.get_logger().info("Subscriptions and synchronization set up.")
        
        self.publisher_hitnet_depth = self.create_publisher(Image, '/HITNET/depth', 2)
        self.publisher_cre_depth = self.create_publisher(Image, '/CRE/depth', 2)
        self.publisher_hitnet_raw_depth = self.create_publisher(Image, '/HITNET/raw_depth', 2)
        self.publisher_cre_raw_depth = self.create_publisher(Image, '/CRE/raw_depth', 2)

        self.bridge = CvBridge()
        self.load_calibration()

        self.left_image = None
        self.right_image = None

        self.exit_flag = False
        
        self.frame_size_key = None
        self.hitnet_model = None
        self.cre_model = None

        self.get_logger().info('Subscriptions set up.')
        self.get_logger().info('Neural Network Depth Estimation Node has started.')

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
                self.initialize_models()
                return True
        return False

    def initialize_models(self):
        baseline = self.baseline/1000 # convert to m from mm
        focal_length = self.fx  # Use fx as the focal length

        hitnet_model_path = '/root/ros2_ws/src/neural_network_stereo_depth_pkg/models/ONNX-HITNET-Stereo-Depth-estimation/models/middlebury_d400/saved_model_480x640/model_float32.onnx'
        hitnet_model_type = ModelType.middlebury
        hitnet_camera_config = HitNetCameraConfig(baseline, focal_length)
        self.hitnet_model = HitNet(hitnet_model_path, hitnet_model_type, hitnet_camera_config, max_dist=5)

        cre_model_path = '/root/ros2_ws/src/neural_network_stereo_depth_pkg/models/ONNX-CREStereo-Depth-Estimation/models/crestereo_combined_iter10_480x640.onnx'
        cre_camera_config = CRECameraConfig(baseline, focal_length)
        self.cre_model = CREStereoModel(cre_model_path, cre_camera_config, max_dist=10)

    def load_calibration(self):
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

    # def left_image_callback(self, msg):
    #     self.left_image = self.bridge.imgmsg_to_cv2(msg, 'mono8')
    #     self.get_logger().info(f'Received left image of shape: {self.left_image.shape}')
    #     if self.frame_size_key is None:
    #         if not self.select_frame_size_key(self.left_image):
    #             self.get_logger().error("Frame size not found in calibration data.")
    #             return
    #     self.left_image = cv2.cvtColor(self.left_image, cv2.COLOR_GRAY2BGR)
    #     self.process_images()

    # def right_image_callback(self, msg):
    #     self.right_image = self.bridge.imgmsg_to_cv2(msg, 'mono8')
    #     self.get_logger().info(f'Received right image of shape: {self.right_image.shape}')
    #     if self.frame_size_key is None:
    #         if not self.select_frame_size_key(self.right_image):
    #             self.get_logger().error("Frame size not found in calibration data.")
    #             return
    #     self.right_image = cv2.cvtColor(self.right_image, cv2.COLOR_GRAY2BGR)
    #     self.process_images()

    def image_callback(self, left_msg, right_msg):
        self.left_image = self.bridge.imgmsg_to_cv2(left_msg, 'bgr8')
        self.get_logger().info(f'Received left image of shape: {self.left_image.shape}')
        self.right_image = self.bridge.imgmsg_to_cv2(right_msg, 'bgr8')
        self.get_logger().info(f'Received right image of shape: {self.right_image.shape}')

        if self.frame_size_key is None:
            if not self.select_frame_size_key(self.left_image):
                self.get_logger().error("Frame size not found in calibration data.")
                return
            if not self.select_frame_size_key(self.right_image):
                self.get_logger().error("Frame size not found in calibration data.")
                return
        self.process_images()

    def process_images(self):
        if self.hitnet_model is None or self.cre_model is None:
            self.get_logger().error("Models not initialized.")
            return

        if self.left_image is None or self.right_image is None:
            self.get_logger().error("Left or right image is None.")
            return
        
        self.get_logger().info(f'Processing images of shape: left={self.left_image.shape}, right={self.right_image.shape}')

        # CRE model depth estimation
        try:
            start_time = datetime.now()
            cre_depth = self.cre_model.estimate_depth(self.left_image, self.right_image)
            #cre_depth /= 1000.0  # Convert from millimeters to meters
            
            # Clipping values
            cre_depth = np.clip(cre_depth, 0, 10.0)  # Clip depth values to a reasonable range in meters
            
            end_time = datetime.now()
            duration = end_time - start_time
            self.get_logger().info(f'CRE inference time: {duration.total_seconds()} seconds')
            self.get_logger().info(f'CRE depth map range: min {cre_depth.min()}, max {cre_depth.max()}')
            self.publish_depth(cre_depth, "CRE")
            self.publish_raw_depth_map(cre_depth, "CRE")
        except Exception as e:
            self.get_logger().error(f'Error in CRE model: {e}')


        # HITNET model depth estimation
        try:
            start_time = datetime.now()
            hitnet_disparity = self.hitnet_model(self.left_image, self.right_image)
            self.get_logger().info(f'HITNET disparity map range: min {hitnet_disparity.min()}, max {hitnet_disparity.max()}')
            
            hitnet_depth = self.hitnet_model.get_depth_from_disparity(hitnet_disparity, self.hitnet_model.camera_config)
            #hitnet_depth /= 1000.0  # Convert from millimeters to meters
            
            self.get_logger().info(f'HITNET depth map before normalization range: min {hitnet_depth.min()}, max {hitnet_depth.max()}')
            
            # Clipping values
            hitnet_depth = np.clip(hitnet_depth, 0, 10.0)  # Clip depth values to a reasonable range in meters
            
            # Post-processing filters (optional)
            # hitnet_depth = cv2.medianBlur(hitnet_depth, 5)
            
            end_time = datetime.now()
            duration = end_time - start_time
            self.get_logger().info(f'HITNET inference time: {duration.total_seconds()} seconds')
            self.publish_depth(hitnet_depth, "HITNET")
            self.publish_raw_depth_map(hitnet_depth, "HITNET")
        except Exception as e:
            self.get_logger().error(f'Error in HITNET model: {e}')


    def publish_depth(self, depth_map, model_type):
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_map_colored = cv2.applyColorMap(cv2.convertScaleAbs(depth_map_normalized, 1), cv2.COLORMAP_JET)
        depth_msg = self.bridge.cv2_to_imgmsg(depth_map_colored, encoding='bgr8')
        depth_msg.header.stamp = self.get_clock().now().to_msg()
        
        if model_type == "HITNET":
            self.publisher_hitnet_depth.publish(depth_msg)
        elif model_type == "CRE":
            self.publisher_cre_depth.publish(depth_msg)

    def publish_raw_depth_map(self, depth_map, model_type):
        # Ensure depth_map is in floating-point format with meters as units
        depth_map_float = depth_map.astype(np.float32)
        
        # Publish as single-channel depth map
        depth_msg = self.bridge.cv2_to_imgmsg(depth_map_float, encoding='32FC1')
        depth_msg.header.stamp = self.get_clock().now().to_msg()
        
        if model_type == "HITNET":
            self.publisher_hitnet_raw_depth.publish(depth_msg)
        elif model_type == "CRE":
            self.publisher_cre_raw_depth.publish(depth_msg)


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

