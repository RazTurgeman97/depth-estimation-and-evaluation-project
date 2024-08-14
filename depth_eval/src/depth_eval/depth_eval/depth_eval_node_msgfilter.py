import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import message_filters
import sys
import select
import subprocess
import matplotlib.pyplot as plt

class DepthEvaluationNode(Node):
    def __init__(self):
        super().__init__('depth_eval_node')

        # Start playing the ROS bag as soon as the node starts
        self.start_rosbag_playback()

        self.bridge = CvBridge()
        self.triangulation_depth = None
        self.hitnet_depth = None
        self.cre_depth = None
        self.d455_depth = None
        self.frame_size = None

        # Initialize flags for each depth map
        self.triangulation_updated = False
        self.hitnet_updated = False
        self.cre_updated = False
        self.d455_updated = False

        # Initialize flags for each depth map
        self.frame_count = {
            "Triangulation": 0,
            "HITNET": 0,
            "CRE": 0,
            "D455": 0
        }

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

        # Create message filters subscribers
        self.triangulation_sub = message_filters.Subscriber(self, Image, '/camera_triangulation/raw_depth_map')
        self.hitnet_sub = message_filters.Subscriber(self, Image, '/HITNET/raw_depth')
        self.cre_sub = message_filters.Subscriber(self, Image, '/CRE/raw_depth')
        self.d455_sub = message_filters.Subscriber(self, Image, '/camera/camera/depth/image_rect_raw')

        # Synchronize the topics using ApproximateTimeSynchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.triangulation_sub, self.hitnet_sub, self.cre_sub, self.d455_sub],
            queue_size=200,
            slop=0.3  # Allowable time difference between messages (in seconds)
        )
        self.ts.registerCallback(self.callback)

        self.get_logger().info("Depth Evaluation Node has started.")

    def start_rosbag_playback(self):
        try:
            self.get_logger().info('Starting ROS bag playback...')
            self.bag_process = subprocess.Popen(['ros2', 'bag', 'play', '/mnt/data/recordings/indoor_recording/indoor_recording_0.db3'])
            self.get_logger().info('ROS bag playback started.')
        except Exception as e:
            self.get_logger().error(f'Failed to start ROS bag playback: {e}')

    def set_frame_size(self, depth_image):
        if self.frame_size is None:
            self.frame_size = depth_image.shape[:2]  # (height, width)

    def callback(self, triangulation_msg, hitnet_msg, cre_msg, d455_msg):

        try:
            self.get_logger().info("Synchronized messages received.")
            self.get_logger().info(f"Received messages: Triangulation type: {type(triangulation_msg)}, HITNET type: {type(hitnet_msg)}, CRE type: {type(cre_msg)}, D455 type: {type(d455_msg)}")
            self.get_logger().info(f"Triangulation header: {triangulation_msg.header.stamp}, HITNET header: {hitnet_msg.header.stamp}, CRE header: {cre_msg.header.stamp}, D455 header: {d455_msg.header.stamp}")


            # Convert the ROS Image messages to OpenCV images
            triangulation_depth = self.bridge.imgmsg_to_cv2(triangulation_msg, 'passthrough')
            hitnet_depth = self.bridge.imgmsg_to_cv2(hitnet_msg, 'passthrough')
            cre_depth = self.bridge.imgmsg_to_cv2(cre_msg, 'passthrough')
            d455_depth = self.bridge.imgmsg_to_cv2(d455_msg, 'passthrough')

            # Set frame size if not set already
            self.set_frame_size(triangulation_depth)

            # Resize depth maps to match the triangulation depth map's size
            self.triangulation_depth = cv2.resize(triangulation_depth, (self.frame_size[1], self.frame_size[0]))
            self.hitnet_depth = cv2.resize(hitnet_depth, (self.frame_size[1], self.frame_size[0]))
            self.cre_depth = cv2.resize(cre_depth, (self.frame_size[1], self.frame_size[0]))
            self.d455_depth = cv2.resize(d455_depth, (self.frame_size[1], self.frame_size[0]))


            # Convert D455 depth image to meters if in 16UC1 format
            if d455_depth.dtype == np.uint16:
                d455_depth = d455_depth.astype(np.float32) * 0.001  # Convert to meters
            self.d455_depth = d455_depth

            self.triangulation_updated = True
            self.hitnet_updated = True
            self.cre_updated = True
            self.d455_updated = True

            # Ensure that depth images are single-channel
            if triangulation_depth.ndim == 3:
                triangulation_depth = cv2.cvtColor(triangulation_depth, cv2.COLOR_BGR2GRAY)
            if hitnet_depth.ndim == 3:
                hitnet_depth = cv2.cvtColor(hitnet_depth, cv2.COLOR_BGR2GRAY)
            if cre_depth.ndim == 3:
                cre_depth = cv2.cvtColor(cre_depth, cv2.COLOR_BGR2GRAY)
            if d455_depth.ndim == 3:
                d455_depth = cv2.cvtColor(d455_depth, cv2.COLOR_BGR2GRAY)

            self.get_logger().info("Processing synchronized depth maps.")

            self.evaluate_depth(triangulation_depth, hitnet_depth, cre_depth, d455_depth)

        except Exception as e:
            self.get_logger().error(f"Error in callback: {e}")


    def evaluate_depth(self, triangulation_depth, hitnet_depth, cre_depth, d455_depth):

        self.frame_count["Triangulation"] += 1
        self.frame_count["HITNET"] += 1
        self.frame_count["CRE"] += 1
        self.frame_count["D455"] += 1

        # Reset the flags
        self.triangulation_updated = False
        self.hitnet_updated = False
        self.cre_updated = False
        self.d455_updated = False

        valid_mask = np.isfinite(triangulation_depth) & (triangulation_depth > 0)

        metrics = {
            "MAE_HITNET": self.compute_mae(hitnet_depth, triangulation_depth, valid_mask),
            "RMSE_HITNET": self.compute_rmse(hitnet_depth, triangulation_depth, valid_mask),
            "MSE_HITNET": self.compute_mse(hitnet_depth, triangulation_depth, valid_mask),
            "MAE_CRE": self.compute_mae(cre_depth, triangulation_depth, valid_mask),
            "RMSE_CRE": self.compute_rmse(cre_depth, triangulation_depth, valid_mask),
            "MSE_CRE": self.compute_mse(cre_depth, triangulation_depth, valid_mask),
            "MAE_D455": self.compute_mae(d455_depth, triangulation_depth, valid_mask),
            "RMSE_D455": self.compute_rmse(d455_depth, triangulation_depth, valid_mask),
            "MSE_D455": self.compute_mse(d455_depth, triangulation_depth, valid_mask)
        }

        for key, value in metrics.items():
            self.metrics[key].append(value)

    def compute_mae(self, predicted, ground_truth, mask):
        # Debugging: print shapes
        print(f"Predicted shape: {predicted.shape}, Ground truth shape: {ground_truth.shape}")
        print(f"Masked predicted shape: {predicted[mask].shape}, Masked ground truth shape: {ground_truth[mask].shape}")
        
        # Ensure that both arrays have the same shape after masking
        assert predicted[mask].shape == ground_truth[mask].shape, "Shape mismatch after applying mask"

        return np.mean(np.abs(predicted[mask] - ground_truth[mask]))

    def compute_rmse(self, predicted, ground_truth, mask):
        return np.sqrt(np.mean((predicted[mask] - ground_truth[mask]) ** 2))

    def compute_mse(self, predicted, ground_truth, mask):
        return np.mean((predicted[mask] - ground_truth[mask]) ** 2)

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

        mae_values = {'HITNET': averages["MAE_HITNET"], 'CRE': averages["MAE_CRE"], 'D455': averages["MAE_D455"]}
        rmse_values = {'HITNET': averages["RMSE_HITNET"], 'CRE': averages["RMSE_CRE"], 'D455': averages["RMSE_D455"]}
        mse_values = {'HITNET': averages["MSE_HITNET"], 'CRE': averages["MSE_CRE"], 'D455': averages["MSE_D455"]}

        best_mae = min(mae_values, key=mae_values.get)
        best_rmse = min(rmse_values, key=rmse_values.get)
        best_mse = min(mse_values, key=mse_values.get)

        print("\nConclusion:")
        print(f"Total frames processed:")
        print(f"Triangulation: {self.frame_count['Triangulation']}")
        print(f"HITNET: {self.frame_count['HITNET']}")
        print(f"CRE: {self.frame_count['CRE']}")
        print(f"D455: {self.frame_count['D455']}")

        print("\nThe best depth estimation method was determined based on three key metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Squared Error (MSE).")
        print("\nMAE is the average of the absolute differences between predicted and actual depth values, providing a measure of overall accuracy.")
        print("\nRMSE and MSE give more weight to larger errors, with RMSE being in the same units as the depth measurements, while MSE amplifies larger errors even more.")

        if best_mae == best_rmse == best_mse:
            print(f"\nOverall, {best_mae} is the best-performing depth estimation method across all metrics, meaning it consistently produced the most accurate depth estimates with fewer large deviations.")
        else:
            print(f"MAE Best: {best_mae} - This method showed the least average error in depth estimation.")
            print(f"RMSE Best: {best_rmse} - This method had the lowest squared error, indicating fewer large deviations.")
            print(f"MSE Best: {best_mse} - This method minimized the squared error the most, reducing the impact of large discrepancies.")

    def plot_metrics(self):
        frames = range(1, len(self.metrics["MAE_HITNET"]) + 1)

        plt.figure(figsize=(14, 8))

        # Plot MAE
        plt.subplot(3, 1, 1)
        plt.plot(frames, self.metrics["MAE_HITNET"], label='MAE HITNET', color='red')
        plt.plot(frames, self.metrics["MAE_CRE"], label='MAE CRE', color='blue')
        plt.plot(frames, self.metrics["MAE_D455"], label='MAE D455', color='green')
        plt.ylabel('MAE')
        plt.legend()
        plt.title('Error as a function of frame')

        # Plot RMSE
        plt.subplot(3, 1, 2)
        plt.plot(frames, self.metrics["RMSE_HITNET"], label='RMSE HITNET', color='red')
        plt.plot(frames, self.metrics["RMSE_CRE"], label='RMSE CRE', color='blue')
        plt.plot(frames, self.metrics["RMSE_D455"], label='RMSE D455', color='green')
        plt.ylabel('RMSE')
        plt.legend()

        # Plot MSE
        plt.subplot(3, 1, 3)
        plt.plot(frames, self.metrics["MSE_HITNET"], label='MSE HITNET', color='red')
        plt.plot(frames, self.metrics["MSE_CRE"], label='MSE CRE', color='blue')
        plt.plot(frames, self.metrics["MSE_D455"], label='MSE D455', color='green')
        plt.ylabel('MSE')
        plt.legend()

        plt.xlabel('Frame')
        plt.tight_layout()
        plt.show()


def main(args=None):
    rclpy.init(args=args)
    node = DepthEvaluationNode()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            
            # Check for 'e' key press to evaluate
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.read(1)
                if line == 'e':
                    print("Evaluation key pressed. Stopping data collection and evaluating...")
                    node.print_averages()
                    node.plot_metrics()
                    break

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()