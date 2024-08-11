import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class StereoTriangulationNode(Node):
    def __init__(self):
        super().__init__('stereo_triangulation_node')
        self.bridge = CvBridge()

        # Load camera parameters
        self.load_calibration()
        self.get_logger().info("Calibration Data Loaded Successfully.")

        # Subscribers
        self.create_subscription(Image, '/camera/camera/infra1/image_rect_raw', self.left_image_callback, 10)
        self.create_subscription(Image, '/camera/camera/infra2/image_rect_raw', self.right_image_callback, 10)
        self.get_logger().info("Subscriptions set up.")
        
        # Publishers
        self.left_depth_publisher = self.create_publisher(Image, '/camera_triangulation/depth_image', 1)
        self.raw_depth_publisher = self.create_publisher(Image, '/camera_triangulation/raw_depth_map', 1)

        self.get_logger().info("Triangulation Node has started.")

        self.left_image = None
        self.right_image = None

        self.exit_flag = False
        self.frame_size_key = None

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
        self.baseline = np.array([self.calib["baseline"], 0, 0])
        self.R_mat = np.array(self.calib["world2left_rot"])

        ## Extracted d455 Intel realsense camera parameters using pyrealsense2 pipeline
        # Intrinsic parameters for infra1 and infra2
        self.K_infra1 = np.array([
            [387.54296875, 0, 324.22357178],
            [0, 387.54296875, 237.35708618],
            [0, 0, 1]
        ])
        
        self.K_infra2 = np.array([
            [387.54296875, 0, 324.22357178],
            [0, 387.54296875, 237.35708618],
            [0, 0, 1]
        ])

        # Distortion coefficients (assuming zero distortion for this example)
        self.dist_coeffs_infra1 = np.zeros(5)
        self.dist_coeffs_infra2 = np.zeros(5)

        # Extrinsic parameters
        self.R = np.eye(3)  # Identity matrix
        self.T = np.array([-0.09504391, 0.0, 0.0])

        # Rectification (assuming already rectified images)
        self.R1 = np.eye(3)
        self.R2 = np.eye(3)
        
        # Projection matrices
        self.P1 = np.hstack((self.K_infra1, np.zeros((3, 1))))
        self.P2 = np.hstack((self.K_infra2, np.dot(self.K_infra2, self.T.reshape(3, 1))))

        # Disparity-to-depth mapping matrix
        self.Q = np.array([
            [1, 0, 0, -self.K_infra1[0, 2]],
            [0, 1, 0, -self.K_infra1[1, 2]],
            [0, 0, 0, self.K_infra1[0, 0]],
            [0, 0, -1/self.T[0], 0]
        ])

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

    def left_image_callback(self, msg):
        self.left_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        if self.frame_size_key is None:
            if not self.select_frame_size_key(self.left_image):
                self.get_logger().error("Frame size not found in calibration data.")
                return
        self.process_images()

    def right_image_callback(self, msg):
        self.right_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        if self.frame_size_key is None:
            if not self.select_frame_size_key(self.right_image):
                self.get_logger().error("Frame size not found in calibration data.")
                return
        self.process_images()

    def process_images(self):
        if self.left_image is not None and self.right_image is not None:
            orb = cv2.ORB.create()
            kp1, des1 = orb.detectAndCompute(self.left_image, None)
            kp2, des2 = orb.detectAndCompute(self.right_image, None)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)

            # Select matched points
            points_left = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
            points_right = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)

            # Triangulate points
            points_4d_hom = cv2.triangulatePoints(self.P1, self.P2, points_left.T, points_right.T)
            points_4d = points_4d_hom[:3] / points_4d_hom[3]
            depth = points_4d[2]

            # Filter out invalid depths
            min_depth_threshold = 0.1
            max_depth_threshold = 100.0
            valid_mask = (depth > min_depth_threshold) & (depth < max_depth_threshold)
            valid_points = points_left[valid_mask]
            valid_depth = depth[valid_mask]

            # Publish the color-coded depth image
            self.publish_depth_image(self.left_image, valid_points, valid_depth, self.left_depth_publisher)
            
            # Publish the raw depth map
            self.publish_raw_depth_map(valid_points, valid_depth)

    def publish_depth_image(self, image, points, depth, publisher):
        if depth.size > 0:
            depth_min, depth_max = np.min(depth), np.max(depth)
            if depth_max == depth_min:  # Avoid division by zero
                depth_max += 1.0
            for i, (point, d) in enumerate(zip(points, depth)):
                x, y = int(point[0]), int(point[1])
                if np.isfinite(d) and d > 0:  # Ensure valid depth values
                    normalized_depth = int(255 * (d - depth_min) / (depth_max - depth_min))
                    color = (0, 0, 255 - normalized_depth)  # Blue to Red
                    cv2.circle(image, (x, y), 5, color, -1)  # Blue to Red
                else:
                    cv2.circle(image, (x, y), 5, (0, 0, 0), -1)  # Black for invalid depth

            depth_image_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
            self.left_depth_publisher.publish(depth_image_msg)
    
    def publish_raw_depth_map(self, points, depth):
        if depth.size > 0:
            # Create a blank image to store depth values
            depth_map = np.zeros(self.left_image.shape[:2], dtype=np.float32)
            
            for i, (point, d) in enumerate(zip(points, depth)):
                x, y = int(point[0]), int(point[1])
                if np.isfinite(d) and d > 0:  # Ensure valid depth values
                    depth_map[y, x] = d
            
            depth_image_msg = self.bridge.cv2_to_imgmsg(depth_map, encoding="32FC1")
            self.raw_depth_publisher.publish(depth_image_msg)


def main(args=None):
    rclpy.init(args=args)
    node = StereoTriangulationNode()
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.exit_flag:
            break
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
