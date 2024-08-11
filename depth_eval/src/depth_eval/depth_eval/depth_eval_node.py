import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import sys
import select

class DepthEvaluationNode(Node):
    def __init__(self):
        super().__init__('depth_eval_node')

        self.bridge = CvBridge()
        self.triangulation_depth = None
        self.hitnet_depth = None
        self.cre_depth = None
        self.d455_depth = None
        self.frame_size = None

        # For accumulating metrics
        self.metrics = {
            "MAE_HITNET": [],
            "RMSE_HITNET": [],
            "MSE_HITNET": [],
            "MAE_CRE": [],
            "RMSE_CRE": [],
            "MSE_CRE": [],
            "MAE_D455": [],
            "RMSE_D455": [],
            "MSE_D455": []
        }

        self.create_subscription(Image, '/camera_triangulation/depth_image', self.triangulation_callback, 10)
        self.create_subscription(Image, '/HITNET/depth', self.hitnet_callback, 10)
        self.create_subscription(Image, '/CRE/depth', self.cre_callback, 10)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.d455_callback, 10)

        self.get_logger().info("Depth Evaluation Node has started.")

    def set_frame_size(self, depth_image):
        # Set the frame size based on the first image received
        if self.frame_size is None:
            self.frame_size = depth_image.shape[:2]  # (height, width)

    def triangulation_callback(self, msg):
        depth_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        self.set_frame_size(depth_image)
        self.triangulation_depth = cv2.resize(depth_image, (self.frame_size[1], self.frame_size[0]))
        self.evaluate_depth()

    def hitnet_callback(self, msg):
        depth_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        self.set_frame_size(depth_image)
        self.hitnet_depth = cv2.resize(depth_image, (self.frame_size[1], self.frame_size[0]))
        self.evaluate_depth()

    def cre_callback(self, msg):
        depth_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        self.set_frame_size(depth_image)
        self.cre_depth = cv2.resize(depth_image, (self.frame_size[1], self.frame_size[0]))
        self.evaluate_depth()

    def d455_callback(self, msg):
        depth_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        self.set_frame_size(depth_image)
        self.d455_depth = cv2.resize(depth_image, (self.frame_size[1], self.frame_size[0]))
        self.evaluate_depth()

    def evaluate_depth(self):
        if self.triangulation_depth is not None and self.hitnet_depth is not None and self.cre_depth is not None and self.d455_depth is not None:
            valid_mask = np.isfinite(self.triangulation_depth) & (self.triangulation_depth > 0)

            metrics = {
                "MAE_HITNET": self.compute_mae(self.hitnet_depth, self.triangulation_depth, valid_mask),
                "RMSE_HITNET": self.compute_rmse(self.hitnet_depth, self.triangulation_depth, valid_mask),
                "MSE_HITNET": self.compute_mse(self.hitnet_depth, self.triangulation_depth, valid_mask),
                "MAE_CRE": self.compute_mae(self.cre_depth, self.triangulation_depth, valid_mask),
                "RMSE_CRE": self.compute_rmse(self.cre_depth, self.triangulation_depth, valid_mask),
                "MSE_CRE": self.compute_mse(self.cre_depth, self.triangulation_depth, valid_mask),
                "MAE_D455": self.compute_mae(self.d455_depth, self.triangulation_depth, valid_mask),
                "RMSE_D455": self.compute_rmse(self.d455_depth, self.triangulation_depth, valid_mask),
                "MSE_D455": self.compute_mse(self.d455_depth, self.triangulation_depth, valid_mask)
            }

            # Accumulate metrics
            for key, value in metrics.items():
                self.metrics[key].append(value)

    def compute_mae(self, predicted, ground_truth, mask):
        return np.mean(np.abs(predicted[mask] - ground_truth[mask]))

    def compute_rmse(self, predicted, ground_truth, mask):
        return np.sqrt(np.mean((predicted[mask] - ground_truth[mask])**2))

    def compute_mse(self, predicted, ground_truth, mask):
        return np.mean((predicted[mask] - ground_truth[mask])**2)

    def calculate_averages(self):
        averages = {}
        for key, values in self.metrics.items():
            averages[key] = np.mean(values) if values else None
        return averages

    def print_averages(self):
        averages = self.calculate_averages()
        print("Averaged Metrics:")
        
        for key, value in averages.items():
            if value is not None:
                print(f"{key}: {value:.4f}")
        
        # Determine the best method based on the lowest MAE, RMSE, and MSE
        mae_values = { 'HITNET': averages["MAE_HITNET"], 'CRE': averages["MAE_CRE"], 'D455': averages["MAE_D455"] }
        rmse_values = { 'HITNET': averages["RMSE_HITNET"], 'CRE': averages["RMSE_CRE"], 'D455': averages["RMSE_D455"] }
        mse_values = { 'HITNET': averages["MSE_HITNET"], 'CRE': averages["MSE_CRE"], 'D455': averages["MSE_D455"] }
        
        best_mae = min(mae_values, key=mae_values.get)
        best_rmse = min(rmse_values, key=rmse_values.get)
        best_mse = min(mse_values, key=mse_values.get)

        print("\nConclusion:")
        if best_mae == best_rmse == best_mse:
            print(f"Overall, {best_mae} is the best-performing depth estimation method.")
        else:
            print(f"MAE Best: {best_mae}")
            print(f"RMSE Best: {best_rmse}")
            print(f"MSE Best: {best_mse}")

def main(args=None):
    rclpy.init(args=args)
    node = DepthEvaluationNode()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

            # Check if 'e' key is pressed
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.read(1)
                if line == 'e':
                    print("Evaluation key pressed. Stopping data collection and evaluating...")
                    node.print_averages()
                    break

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
