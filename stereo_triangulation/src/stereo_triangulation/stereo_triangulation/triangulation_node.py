import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import yaml
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class StereoTriangulationNode(Node):
    def __init__(self):
        super().__init__('stereo_triangulation_node')
        self.bridge = CvBridge()

        # Load camera parameters
        self.declare_parameter('calibration_file', '/opt/ros/humble/share/stereo_triangulation/config/calibration-camchain.yaml')
        calibration_file = self.get_parameter('calibration_file').get_parameter_value().string_value
        self.load_calibration(calibration_file)
        self.get_logger().info("Calibration Data Loaded Successfully.")

        # Subscribers
        self.create_subscription(Image, '/left_camera/image_raw', self.left_image_callback, 10)
        self.create_subscription(Image, '/right_camera/image_raw', self.right_image_callback, 10)
        self.get_logger().info("Subscriptions set up.")
        
        # Publishers
        self.left_depth_publisher = self.create_publisher(Image, '/left_camera_triangulation/depth_image', 10)
        self.right_depth_publisher = self.create_publisher(Image, '/right_camera_triangulation/depth_image', 10)
        
        self.get_logger().info("Triangulation Node has started.")

        self.left_image = None
        self.right_image = None

        self.exit_flag = False

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
        self.process_images()

    def right_image_callback(self, msg):
        self.right_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        rectified_right = cv2.remap(self.right_image, self.map1_right, self.map2_right, cv2.INTER_LANCZOS4)
        self.right_image = rectified_right
        self.process_images()

    def process_images(self):
        if self.left_image is not None and self.right_image is not None:
            # Match keypoints (for demonstration, using ORB)
            orb = cv2.ORB.create(nfeatures=500, scaleFactor=1.2, nlevels=8)
            kp1, des1 = orb.detectAndCompute(self.left_image, None)
            kp2, des2 = orb.detectAndCompute(self.right_image, None)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)

            # Select matched points
            points_left = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
            points_right = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)

            # # Filter matches using RANSAC
            # E, mask = cv2.findEssentialMat(points_left, points_right, self.camera_matrix_left, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            # points_left = points_left[mask.ravel() == 1]
            # points_right = points_right[mask.ravel() == 1]

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
            valid_points_right = points_right[valid_mask]
            valid_depth_right = depth[valid_mask]

            # Publish depth images
            self.publish_depth_image(self.left_image, valid_points, valid_depth, self.left_depth_publisher)
            self.publish_depth_image(self.right_image, valid_points_right, valid_depth_right, self.right_depth_publisher)

    def publish_depth_image(self, image, points, depth, publisher):
        depth_min, depth_max = np.min(depth), np.max(depth)
        for i, (point, d) in enumerate(zip(points, depth)):
            x, y = int(point[0]), int(point[1])
            if np.isfinite(d) and d > 0:  # Ensure valid depth values
                normalized_depth = int(255 * (d - depth_min) / (depth_max - depth_min))
                color = (0, 0, 255 - normalized_depth)  # Blue to Red
                cv2.circle(image, (x, y), 5, color, -1)  # Blue to Red
            else:
                cv2.circle(image, (x, y), 5, (0, 0, 0), -1)  # Black for invalid depth

        depth_image_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        publisher.publish(depth_image_msg)

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

