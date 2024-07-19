import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

class DepthEvaluationNode(Node):
    def __init__(self):
        super().__init__('depth_evaluation_node')

        self.bridge = CvBridge()
        self.triangulation_depth = None
        self.hitnet_depth = None
        self.cre_depth = None
        self.d455_depth = None

        self.create_subscription(Image, '/left_camera_triangulation/depth_image', self.triangulation_callback, 10)
        self.create_subscription(Image, '/hitnet/depth', self.hitnet_callback, 10)
        self.create_subscription(Image, '/cre/depth', self.cre_callback, 10)
        self.create_subscription(Image, '/camera/depth/image_rect_raw', self.d455_callback, 10)

        self.get_logger().info("Depth Evaluation Node has started.")

    def triangulation_callback(self, msg):
        self.triangulation_depth = self.bridge.imgmsg_to_cv2(msg, '32FC1')
        self.evaluate_depth()

    def hitnet_callback(self, msg):
        self.hitnet_depth = self.bridge.imgmsg_to_cv2(msg, '32FC1')
        self.evaluate_depth()

    def cre_callback(self, msg):
        self.cre_depth = self.bridge.imgmsg_to_cv2(msg, '32FC1')
        self.evaluate_depth()

    def d455_callback(self, msg):
        self.d455_depth = self.bridge.imgmsg_to_cv2(msg, '32FC1')
        self.evaluate_depth()

    def evaluate_depth(self):
        if self.triangulation_depth is not None and self.hitnet_depth is not None and self.cre_depth is not None and self.d455_depth is not None:
            valid_mask = np.isfinite(self.triangulation_depth) & (self.triangulation_depth > 0)

            metrics = {
                "MAE_HITNET": self.compute_mae(self.hitnet_depth, self.triangulation_depth, valid_mask),
                "RMSE_HITNET": self.compute_rmse(self.hitnet_depth, self.triangulation_depth, valid_mask),
                "MAE_CRE": self.compute_mae(self.cre_depth, self.triangulation_depth, valid_mask),
                "RMSE_CRE": self.compute_rmse(self.cre_depth, self.triangulation_depth, valid_mask),
                "MAE_D455": self.compute_mae(self.d455_depth, self.triangulation_depth, valid_mask),
                "RMSE_D455": self.compute_rmse(self.d455_depth, self.triangulation_depth, valid_mask)
            }

            for key, value in metrics.items():
                self.get_logger().info(f"{key}: {value:.4f}")

    def compute_mae(self, predicted, ground_truth, mask):
        return np.mean(np.abs(predicted[mask] - ground_truth[mask]))

    def compute_rmse(self, predicted, ground_truth, mask):
        return np.sqrt(np.mean((predicted[mask] - ground_truth[mask])**2))

def main(args=None):
    rclpy.init(args=args)
    node = DepthEvaluationNode()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

