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
        self.left_depth_publisher = self.create_publisher(Image, '/left_camera_triangulation/depth_image', 10)
        self.right_depth_publisher = self.create_publisher(Image, '/right_camera_triangulation/depth_image', 10)
        
        self.get_logger().info("Triangulation Node has started.")

        self.left_image = None
        self.right_image = None

        self.exit_flag = False

    def load_calibration(self):
        # Provided JSON-like data
        calib = {
            "baseline": -95.0439,
            "intrinsic_left": [
                [0.505839, 0.808565, 0.503078],
                [0.492961, -0.0618031, 0.0696098],
                [-0.000149548, 0.000388624, -0.0226]
            ],
            "intrinsic_right": [
                [0.504375, 0.805967, 0.505367],
                [0.49573, -0.0594383, 0.0707053],
                [-0.000174214, 0.000842824, -0.0233214]
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
                [0.999971, -0.00226072, 0.00731733],
                [0.00225269, 0.999997, 0.00110529],
                [-0.00731981, -0.00108877, 0.999973]
            ],
            "world2right_rot": [
                [0.999976, 0.00354033, -0.00597893],
                [-0.00354689, 0.999993, -0.00108644],
                [0.00597504, 0.00110762, 0.999982]
            ]
        }

        self.camera_matrix_left = np.array(calib['intrinsic_left'])
        self.dist_coeffs_left = np.zeros(5)  # Assuming no distortion for simplification
        self.camera_matrix_right = np.array(calib['intrinsic_right'])
        self.dist_coeffs_right = np.zeros(5)  # Assuming no distortion for simplification
        self.T = np.array([calib['baseline'], 0, 0])
        R_mat = np.array(calib['world2left_rot'])

        image_size = (1280, 720)
        self.R1, self.R2, self.P1, self.P2, self.Q, self.ROI_L, self.ROI_R = cv2.stereoRectify(
            self.camera_matrix_left, self.dist_coeffs_left,
            self.camera_matrix_right, self.dist_coeffs_right,
            image_size, R_mat, self.T)

    def left_image_callback(self, msg):
        self.left_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.process_images()

    def right_image_callback(self, msg):
        self.right_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
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

            # Publish depth images
            self.publish_depth_image(self.left_image, valid_points, valid_depth, self.left_depth_publisher)

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
