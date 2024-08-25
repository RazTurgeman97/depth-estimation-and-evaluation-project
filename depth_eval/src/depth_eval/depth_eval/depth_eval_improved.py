import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import message_filters
import subprocess
import matplotlib.pyplot as plt
import matplotlib
import os
import shutil
import time
import pandas as pd
import signal
import re

class DepthEvaluationNode(Node):
    def __init__(self):
        super().__init__('depth_eval_node')

        matplotlib.use('Agg')

        # Create directory for saving frames
        self.create_frame_analysis_directory()

        self.bridge = CvBridge()

        self.bag_process = None  # Initialize bag_process as None
        self.declare_parameter('stop_processing', False)


        # Initialize the flag
        self.processing_complete = False
        self.bagfile_path = '/mnt/data/recordings/indoor_recording/indoor_recording_0.db3'
        self.CRE_topic = '/CRE/raw_depth'
        self.HITNET_topic = '/HITNET/raw_depth'

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

        # Initialize frame count and rolling averages
        self.frame_count = {"Triangulation": 0, "HITNET": 0, "CRE": 0, "D455": 0}
        self.rolling_mae = {"HITNET": [], "CRE": [], "D455": []}
        self.rolling_mse = {"HITNET": [], "CRE": [], "D455": []}
        self.rolling_rmse = {"HITNET": [], "CRE": [], "D455": []}
        self.rolling_iou = {"HITNET": [], "CRE": [], "D455": []}
        self.rolling_epe = {"HITNET": [], "CRE": [], "D455": []}
        self.rolling_d1 = {"HITNET": [], "CRE": [], "D455": []}
        # self.runtime = {"HITNET": [], "CRE": [], "D455": []}

        # For accumulating metrics
        self.metrics = {
            "MAE_HITNET": [],
            "RMSE_HITNET": [],
            "MSE_HITNET": [],
            "IoU_HITNET": [],
            "EPE_HITNET": [],
            "D1_HITNET": [],
            "MAE_CRE": [],
            "RMSE_CRE": [],
            "MSE_CRE": [],
            "IoU_CRE": [],
            "EPE_CRE": [],
            "D1_CRE": [],
            "MAE_D455": [],
            "RMSE_D455": [],
            "MSE_D455": [],
            "IoU_D455": [],
            "EPE_D455": [],
            "D1_D455": []
        }

        # Start playing the ROS bag, stopping any previous instance first
        self.stop_previous_rosbag_playback()
        self.start_rosbag_playback()
        self.initialize_frame_counts()

        # Create message filters subscribers
        self.triangulation_sub = message_filters.Subscriber(self, Image, '/camera_triangulation/raw_depth_map')
        self.hitnet_sub = message_filters.Subscriber(self, Image, '/HITNET/raw_depth')
        self.cre_sub = message_filters.Subscriber(self, Image, '/CRE/raw_depth')
        self.d455_sub = message_filters.Subscriber(self, Image, '/camera/camera/depth/image_rect_raw')

        # Synchronize the topics using ApproximateTimeSynchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.triangulation_sub, self.hitnet_sub, self.cre_sub, self.d455_sub],
            queue_size=200,
            slop=0.2
        )
        self.ts.registerCallback(self.callback)

        self.get_logger().info("Depth Evaluation Node has started.")

    def create_frame_analysis_directory(self):
        # Check if the directory "frame_analysis" exists
        if os.path.exists("frame_analysis"):
            # Remove the existing directory and its contents
            shutil.rmtree("frame_analysis")
            self.get_logger().info("Removed existing 'frame_analysis' directory.")

        # Create a new directory
        os.makedirs("frame_analysis") # models_difference
        self.get_logger().info("Created new 'frame_analysis' directory.")
        os.makedirs("frame_analysis/Visual_Inspection")
        self.get_logger().info("Created new 'Visual_Inspection' directory.")
        os.makedirs("frame_analysis/depth_comparison")
        self.get_logger().info("Created new 'depth_comparison' directory.")
        os.makedirs("frame_analysis/error_distribution")
        self.get_logger().info("Created new 'error_distribution' directory.")
        os.makedirs("frame_analysis/error_progression")
        self.get_logger().info("Created new 'error_progression' directory.")
        os.makedirs("frame_analysis/models_difference")
        self.get_logger().info("Created new 'models_difference' directory.")
        os.makedirs("frame_analysis/disagreement_maps")
        self.get_logger().info("Created new 'disagreement_maps' directory.")

    def stop_previous_rosbag_playback(self):
        try:
            # Find all processes with 'ros2 bag play' in the command line
            output = subprocess.check_output(['ps', 'aux'], universal_newlines=True)
            for line in output.splitlines():
                if 'ros2 bag play' in line and '/mnt/data/recordings/indoor_recording/indoor_recording_0.db3' in line:
                    # Extract the process ID (PID)
                    pid = int(line.split()[1])
                    self.get_logger().info(f'Stopping existing ROS bag playback process with PID {pid}...')
                    os.kill(pid, signal.SIGTERM)
                    self.get_logger().info(f'ROS bag playback process {pid} stopped.')
        except Exception as e:
            self.get_logger().error(f'Failed to stop previous ROS bag playback: {e}')

    def start_rosbag_playback(self):
        try:
            self.get_logger().info('Starting ROS bag playback...')
            self.bag_process = subprocess.Popen(
                ['ros2', 'bag', 'play', '/mnt/data/recordings/indoor_recording/indoor_recording_0.db3']
            )
            self.get_logger().info('ROS bag playback started.')
        except Exception as e:
            self.get_logger().error(f'Failed to start ROS bag playback: {e}')

    def set_frame_size(self, depth_image):
        if self.frame_size is None:
            self.frame_size = depth_image.shape[:2]  # (height, width)

    def callback(self, triangulation_msg, hitnet_msg, cre_msg, d455_msg):

        try:
            self.get_logger().info("Synchronized messages received.")

            # Convert the ROS Image messages to OpenCV images
            self.triangulation_depth = self.bridge.imgmsg_to_cv2(triangulation_msg, 'passthrough')
            self.hitnet_depth = self.bridge.imgmsg_to_cv2(hitnet_msg, 'passthrough')
            self.cre_depth = self.bridge.imgmsg_to_cv2(cre_msg, 'passthrough')
            self.d455_depth = self.bridge.imgmsg_to_cv2(d455_msg, 'passthrough')

            # Set frame size if not set already
            self.set_frame_size(self.triangulation_depth)

            # Resize depth maps to match the triangulation depth map's size
            self.triangulation_depth = cv2.resize(self.triangulation_depth, (self.frame_size[1], self.frame_size[0]))
            self.hitnet_depth = cv2.resize(self.hitnet_depth, (self.frame_size[1], self.frame_size[0]))
            self.cre_depth = cv2.resize(self.cre_depth, (self.frame_size[1], self.frame_size[0]))
            self.d455_depth = cv2.resize(self.d455_depth, (self.frame_size[1], self.frame_size[0]))

            # Convert D455 depth image to meters if in 16UC1 format
            if self.d455_depth.dtype == np.uint16:
                self.d455_depth = self.d455_depth.astype(np.float32) * 0.001  # Convert to meters

            self.triangulation_updated = True
            self.hitnet_updated = True
            self.cre_updated = True
            self.d455_updated = True

            # Ensure that depth images are single-channel
            self.triangulation_depth = self.ensure_single_channel(self.triangulation_depth)
            self.hitnet_depth = self.ensure_single_channel(self.hitnet_depth)
            self.cre_depth = self.ensure_single_channel(self.cre_depth)
            self.d455_depth = self.ensure_single_channel(self.d455_depth)

            # Mask to apply metrics only on valid triangulation points
            valid_mask = np.isfinite(self.triangulation_depth) & (self.triangulation_depth > 0)

            # Ensure the mask is resized to match the dimensions of the depth maps
            valid_mask = cv2.resize(valid_mask.astype(np.uint8), (self.frame_size[1], self.frame_size[0])).astype(bool)

            self.get_logger().info("Processing synchronized depth maps.")
            self.evaluate_depth(self.triangulation_depth, self.hitnet_depth, self.cre_depth, self.d455_depth)

            # Perform analysis on the current frame
            frame_idx = self.frame_count["CRE"]

            self.visualize_and_save_frame(frame_idx, "frame_analysis")
            self.analyze_error_distribution(frame_idx)
            self.analyze_frame_content_correlation(frame_idx)
            self.analyze_error_localization(frame_idx)
            self.compare_model_outputs(frame_idx)
            self.analyze_model_disagreement(frame_idx)

        except Exception as e:
            self.get_logger().error(f"Error in callback: {e}")

    def ensure_single_channel(self, depth_image):
        if depth_image.ndim == 3:
            depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
        return depth_image

    def evaluate_depth(self, triangulation_depth, hitnet_depth, cre_depth, d455_depth):
        self.frame_count["Triangulation"] += 1
        self.frame_count["HITNET"] += 1
        self.frame_count["CRE"] += 1
        self.frame_count["D455"] += 1

        # Mask to apply metrics only on valid triangulation points
        valid_mask = np.isfinite(triangulation_depth) & (triangulation_depth > 0)

        metrics = {
            "MAE_HITNET": self.compute_mae(hitnet_depth, triangulation_depth, valid_mask),
            "MAE_CRE": self.compute_mae(cre_depth, triangulation_depth, valid_mask),
            "MAE_D455": self.compute_mae(d455_depth, triangulation_depth, valid_mask),
            "RMSE_HITNET": self.compute_rmse(hitnet_depth, triangulation_depth, valid_mask),
            "RMSE_CRE": self.compute_rmse(cre_depth, triangulation_depth, valid_mask),
            "RMSE_D455": self.compute_rmse(d455_depth, triangulation_depth, valid_mask),            
            "MSE_HITNET": self.compute_mse(hitnet_depth, triangulation_depth, valid_mask),
            "MSE_CRE": self.compute_mse(cre_depth, triangulation_depth, valid_mask),
            "MSE_D455": self.compute_mse(d455_depth, triangulation_depth, valid_mask),
            "IoU_HITNET": self.compute_iou(hitnet_depth, triangulation_depth, valid_mask),            
            "IoU_CRE": self.compute_iou(cre_depth, triangulation_depth, valid_mask),
            "IoU_D455": self.compute_iou(d455_depth, triangulation_depth, valid_mask),
            "EPE_HITNET": self.compute_epe(hitnet_depth, triangulation_depth, valid_mask),            
            "EPE_CRE": self.compute_epe(cre_depth, triangulation_depth, valid_mask),
            "EPE_D455": self.compute_epe(d455_depth, triangulation_depth, valid_mask),
            "D1_HITNET": self.compute_d1(hitnet_depth, triangulation_depth, valid_mask),            
            "D1_CRE": self.compute_d1(cre_depth, triangulation_depth, valid_mask),
            "D1_D455": self.compute_d1(d455_depth, triangulation_depth, valid_mask)
        }

        for key, value in metrics.items():
            self.metrics[key].append(value)
            if "MAE" in key:
                self.rolling_mae[key.split('_')[1]].append(value)
            if "MSE" in key:
                self.rolling_mse[key.split('_')[1]].append(value)
            if "RMSE" in key:
                self.rolling_rmse[key.split('_')[1]].append(value)
            if "IoU" in key:
                self.rolling_iou[key.split('_')[1]].append(value)
            if "EPE" in key:
                self.rolling_epe[key.split('_')[1]].append(value)
            if "D1" in key:
                self.rolling_d1[key.split('_')[1]].append(value)

        # Save depth map visualizations
        self.save_depth_map_visualizations(triangulation_depth, hitnet_depth, cre_depth, d455_depth)

        # Stop if MAE values converge
        if self.has_converged("MAE_HITNET", "MAE_CRE", "MAE_D455"):
            self.print_averages()
            self.plot_metrics()
            self.generate_analysis_report()
            return

    def last_frame_processed(self, frame_idx, margin):
        self.min_frame_count = min(self.total_CRE_frames, self.total_HITNET_frames)

        if frame_idx < self.min_frame_count:
            return False
        elif frame_idx >= (self.min_frame_count - margin):
            return True
        else:
            return False

    def wait_for_first_frame(self):
        """Wait until at least one frame is received."""
        self.get_logger().info('Waiting for the first frame...')
        while not self.frame_received:
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.1)  # Add a small sleep to avoid busy waiting
        self.get_logger().info('First frame received, continuing...')

    def get_total_messages_from_bag(self, topic_name):
        try:
            # Run the ros2 bag info command and capture the output
            output = subprocess.check_output(['ros2', 'bag', 'info', '/mnt/data/recordings/indoor_recording/indoor_recording_0.db3']).decode('utf-8')

            # Use a regular expression to match the line with the topic of interest
            pattern = rf'\s*Topic:\s+{re.escape(topic_name)}\s+\|\s+Type:\s+\S+\s+\|\s+Count:\s+(\d+)'
            match = re.search(pattern, output)

            if match:
                total_messages = int(match.group(1))
                return total_messages
            else:
                raise ValueError(f"Topic {topic_name} not found in bag file info.")

        except subprocess.CalledProcessError as e:
            print(f"Error running ros2 bag info: {e}")
            return 0

    def initialize_frame_counts(self):
        self.total_CRE_frames = self.get_total_messages_from_bag(self.CRE_topic)
        print(f"Total messages for /CRE/raw_depth: {self.total_CRE_frames}")
        self.total_HITNET_frames = self.get_total_messages_from_bag(self.HITNET_topic)
        print(f"Total messages for /HITNET/raw_depth: {self.total_HITNET_frames}")
            

    def compute_mae(self, predicted, ground_truth, mask):
        assert predicted[mask].shape == ground_truth[mask].shape, "Shape mismatch after applying mask"
        return np.mean(np.abs(predicted[mask] - ground_truth[mask]))

    def compute_rmse(self, predicted, ground_truth, mask):
        return np.sqrt(np.mean((predicted[mask] - ground_truth[mask]) ** 2))

    def compute_mse(self, predicted, ground_truth, mask):
        return np.mean((predicted[mask] - ground_truth[mask]) ** 2)

    def compute_iou(self, predicted, ground_truth, mask, threshold=0.5):
        intersection = np.sum((predicted[mask] > threshold) & (ground_truth[mask] > threshold))
        union = np.sum((predicted[mask] > threshold) | (ground_truth[mask] > threshold))
        return intersection / union if union != 0 else 0
    
    def compute_epe(self, predicted, ground_truth, mask): #########################################################################################
        """Compute End-Point Error (EPE)."""
        return np.mean(np.abs(predicted[mask] - ground_truth[mask]))

    def compute_d1(self, predicted, ground_truth, mask, threshold=3.0):
        """Compute Percentage of Erroneous Pixels (D1)."""
        error = np.abs(predicted[mask] - ground_truth[mask])
        return np.mean(error > threshold) * 100
    
    def calculate_averages(self):
        averages = {}
        for key, values in self.metrics.items():
            averages[key] = np.mean(values) if values else None
        return averages

    def print_averages(self):
        averages = self.calculate_averages()
        with open("frame_analysis/summary_report.txt", "a") as report_file:
            report_file.write("\nAveraged Metrics:\n")
            for key, value in averages.items():
                if value is not None:
                    report_file.write(f"{key}: {value:.4f}\n")
                    print(f"{key}: {value:.4f}")

            mae_values = {'HITNET': averages["MAE_HITNET"], 'CRE': averages["MAE_CRE"], 'D455': averages["MAE_D455"]}
            rmse_values = {'HITNET': averages["RMSE_HITNET"], 'CRE': averages["RMSE_CRE"], 'D455': averages["RMSE_D455"]}
            mse_values = {'HITNET': averages["MSE_HITNET"], 'CRE': averages["MSE_CRE"], 'D455': averages["MSE_D455"]}
            iou_values = {'HITNET': averages["IoU_HITNET"], 'CRE': averages["IoU_CRE"], 'D455': averages["IoU_D455"]}
            epe_values = {'HITNET': averages["EPE_HITNET"], 'CRE': averages["EPE_CRE"], 'D455': averages["EPE_D455"]}
            d1_values = {'HITNET': averages["D1_HITNET"], 'CRE': averages["D1_CRE"], 'D455': averages["D1_D455"]}

            best_mae = min(mae_values, key=mae_values.get)
            best_rmse = min(rmse_values, key=rmse_values.get)
            best_mse = min(mse_values, key=mse_values.get)
            best_iou = max(iou_values, key=iou_values.get)
            best_epe = min(epe_values, key=epe_values.get)
            best_d1 = min(d1_values, key=d1_values.get)

            report_file.write("\nConclusion:\n")
            report_file.write(f"Best MAE: {best_mae}\n")
            report_file.write(f"Best RMSE: {best_rmse}\n")
            report_file.write(f"Best MSE: {best_mse}\n")
            report_file.write(f"Best IoU: {best_iou}\n")
            report_file.write(f"Best EPE: {best_epe}\n")
            report_file.write(f"Best D1: {best_d1}\n")


        print("\nConclusion:")
        print(f"Total frames processed:")
        print(f"Triangulation: {self.frame_count['Triangulation']}")
        print(f"HITNET: {self.frame_count['HITNET']}")
        print(f"CRE: {self.frame_count['CRE']}")
        print(f"D455: {self.frame_count['D455']}")

        print("\nThe best depth estimation method was determined based on three key metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Squared Error (MSE).")
        print("\nMAE is the average of the absolute differences between predicted and actual depth values, providing a measure of overall accuracy.")
        print("\nRMSE and MSE give more weight to larger errors, with RMSE being in the same units as the depth measurements, while MSE amplifies larger errors even more.")

        if best_mae == best_rmse == best_mse == best_iou == best_epe == best_d1:
            print(f"\nOverall, {best_mae} is the best-performing depth estimation method across all metrics, meaning it consistently produced the most accurate depth estimates with fewer large deviations.\n")
        else:
            print(f"MAE Best: {best_mae} - This method showed the least average error in depth estimation.")
            print(f"RMSE Best: {best_rmse} - This method had the lowest squared error, indicating fewer large deviations.")
            print(f"MSE Best: {best_mse} - This method minimized the squared error the most, reducing the impact of large discrepancies.")
            print(f"Iou Best: {best_iou} - This method ***Enter Promt***.")
            print(f"EPE Best: {best_epe} - This method ***Enter Promt***.")
            print(f"D1 Best: {best_d1} - This method ***Enter Promt***.")


    def plot_metrics(self):
        frames = range(1, len(self.metrics["MAE_HITNET"]) + 1)

        for idx, frame in enumerate(frames):
            plt.figure(figsize=(14, 8))

            # Plot MAE
            plt.subplot(6, 1, 1)
            plt.plot(frames[:idx+1], self.metrics["MAE_HITNET"][:idx+1], label='MAE HITNET', color='red')
            plt.plot(frames[:idx+1], self.metrics["MAE_CRE"][:idx+1], label='MAE CRE', color='blue')
            plt.plot(frames[:idx+1], self.metrics["MAE_D455"][:idx+1], label='MAE D455', color='green')
            plt.ylabel('MAE')
            plt.legend()
            plt.title(f'Error as a function of frame {idx+1}')

            # Plot RMSE
            plt.subplot(6, 1, 2)
            plt.plot(frames[:idx+1], self.metrics["RMSE_HITNET"][:idx+1], label='RMSE HITNET', color='red')
            plt.plot(frames[:idx+1], self.metrics["RMSE_CRE"][:idx+1], label='RMSE CRE', color='blue')
            plt.plot(frames[:idx+1], self.metrics["RMSE_D455"][:idx+1], label='RMSE D455', color='green')
            plt.ylabel('RMSE')
            plt.legend()

            # Plot MSE
            plt.subplot(6, 1, 3)
            plt.plot(frames[:idx+1], self.metrics["MSE_HITNET"][:idx+1], label='MSE HITNET', color='red')
            plt.plot(frames[:idx+1], self.metrics["MSE_CRE"][:idx+1], label='MSE CRE', color='blue')
            plt.plot(frames[:idx+1], self.metrics["MSE_D455"][:idx+1], label='MSE D455', color='green')
            plt.ylabel('MSE')
            plt.legend()

            # Plot IoU
            plt.subplot(6, 1, 4)
            plt.plot(frames[:idx+1], self.metrics["IoU_HITNET"][:idx+1], label='IoU HITNET', color='red')
            plt.plot(frames[:idx+1], self.metrics["IoU_CRE"][:idx+1], label='IoU CRE', color='blue')
            plt.plot(frames[:idx+1], self.metrics["IoU_D455"][:idx+1], label='IoU D455', color='green')
            plt.ylabel('IoU')
            plt.legend()

            # Plot EPE
            plt.subplot(6, 1, 5)
            plt.plot(frames[:idx+1], self.metrics["EPE_HITNET"][:idx+1], label='EPE HITNET', color='red')
            plt.plot(frames[:idx+1], self.metrics["EPE_CRE"][:idx+1], label='EPE CRE', color='blue')
            plt.plot(frames[:idx+1], self.metrics["EPE_D455"][:idx+1], label='EPE D455', color='green')
            plt.ylabel('EPE')
            plt.legend()

            # Plot D1
            plt.subplot(6, 1, 6)
            plt.plot(frames[:idx+1], self.metrics["D1_HITNET"][:idx+1], label='D1 HITNET', color='red')
            plt.plot(frames[:idx+1], self.metrics["D1_CRE"][:idx+1], label='D1 CRE', color='blue')
            plt.plot(frames[:idx+1], self.metrics["D1_D455"][:idx+1], label='D1 D455', color='green')
            plt.ylabel('D1')
            plt.legend()

            plt.xlabel('Frame')
            plt.tight_layout()

            # Save the figure
            plt.savefig(f'frame_analysis/error_progression/error_comp_plot_frame_{idx+1}.png')
            plt.close()


    def has_converged(self, *metric_keys, threshold=1e-4, window_size=5):
        for key in metric_keys:
            values = self.metrics[key]
            if len(values) >= window_size:
                recent_values = values[-window_size:]
                if np.max(recent_values) - np.min(recent_values) < threshold:
                    return True
        return False

    def save_depth_map_visualizations(self, triangulation_depth, hitnet_depth, cre_depth, d455_depth):
        plt.figure()

        plt.subplot(221)
        plt.imshow(triangulation_depth, cmap='jet')
        plt.title('Triangulation Depth')
        plt.colorbar()

        plt.subplot(222)
        plt.imshow(hitnet_depth, cmap='jet')
        plt.title('HITNET Depth')
        plt.colorbar()

        plt.subplot(223)
        plt.imshow(cre_depth, cmap='jet')
        plt.title('CRE Depth')
        plt.colorbar()

        plt.subplot(224)
        plt.imshow(d455_depth, cmap='jet')
        plt.title('D455 Depth')
        plt.colorbar()

        plt.savefig(f'frame_analysis/depth_comparison/depth_comparison_frame_{self.frame_count["Triangulation"]}.png')
        plt.close()

    def generate_analysis_report(self):
        """Generate a summary report explaining the results."""
        averages = self.calculate_averages()

        # Create a DataFrame for the table
        data = {
            'MAE': [averages['MAE_HITNET'], averages['MAE_CRE'], averages['MAE_D455']],
            'RMSE': [averages['RMSE_HITNET'], averages['RMSE_CRE'], averages['RMSE_D455']],
            'MSE': [averages['MSE_HITNET'], averages['MSE_CRE'], averages['MSE_D455']],
            'IoU': [averages['IoU_HITNET'], averages['IoU_CRE'], averages['IoU_D455']],
            'EPE': [averages['EPE_HITNET'], averages['EPE_CRE'], averages['EPE_D455']],
            'D1 (%)': [averages['D1_HITNET'], averages['D1_CRE'], averages['D1_D455']],
            # 'Runtime (s)': [np.mean(self.runtime['HITNET']), np.mean(self.runtime['CRE']), np.mean(self.runtime['D455'])]
        }
        df = pd.DataFrame(data, index=['HITNET', 'CRE', 'D455'])

        # Save the DataFrame to a CSV file for easy access
        df.to_csv("frame_analysis/metrics_summary.csv")

        
        report = []
        report.append("Summary of Depth Estimation Results:\n")
        
        # MAE Analysis
        report.append("1. Mean Absolute Error (MAE):\n")
        report.append("   - MAE represents the average error between the predicted depth and the actual depth.\n")
        report.append(f"   - HITNET MAE: {averages['MAE_HITNET']:.4f}\n")
        report.append(f"   - CRE MAE: {averages['MAE_CRE']:.4f}\n")
        report.append(f"   - D455 MAE: {averages['MAE_D455']:.4f}\n")
        report.append("   The lower the MAE, the better the model is at estimating depth accurately.\n")

        # RMSE Analysis
        report.append("\n2. Root Mean Squared Error (RMSE):\n")
        report.append("   - RMSE is similar to MAE but gives more weight to larger errors.\n")
        report.append(f"   - HITNET RMSE: {averages['RMSE_HITNET']:.4f}\n")
        report.append(f"   - CRE RMSE: {averages['RMSE_CRE']:.4f}\n")
        report.append(f"   - D455 RMSE: {averages['RMSE_D455']:.4f}\n")
        report.append("   Lower RMSE values indicate fewer large discrepancies in depth estimation.\n")

        # MSE Analysis
        report.append("\n3. Mean Squared Error (MSE):\n")
        report.append("   - MSE is similar to RMSE but doesn't convert back to original units.\n")
        report.append(f"   - HITNET MSE: {averages['MSE_HITNET']:.4f}\n")
        report.append(f"   - CRE MSE: {averages['MSE_CRE']:.4f}\n")
        report.append(f"   - D455 MSE: {averages['MSE_D455']:.4f}\n")
        report.append("   Like RMSE, lower MSE values indicate better performance.\n")

        # IoU Analysis
        report.append("\n4. Intersection over Union (IoU):\n")
        report.append("   - IoU measures how well the predicted depth overlaps with the ground truth.\n")
        report.append(f"   - HITNET IoU: {averages['IoU_HITNET']:.4f}\n")
        report.append(f"   - CRE IoU: {averages['IoU_CRE']:.4f}\n")
        report.append(f"   - D455 IoU: {averages['IoU_D455']:.4f}\n")
        report.append("   Higher IoU values indicate better spatial agreement between the prediction and ground truth.\n")

        # EPE Analysis
        report.append("\n5. End-Point Error (EPE):\n")
        report.append("   - EPE measures the average disparity error in pixels between the predicted and ground truth depth maps.\n")
        report.append(f"   - HITNET EPE: {averages['EPE_HITNET']:.4f} pixels\n")
        report.append(f"   - CRE EPE: {averages['EPE_CRE']:.4f} pixels\n")
        report.append(f"   - D455 EPE: {averages['EPE_D455']:.4f} pixels\n")
        report.append("   A lower EPE value indicates that the depth estimation model is more accurate in predicting the depth values, which is critical for applications where precision is key.\n")
        
        # D1 Analysis
        report.append("\n6. Percentage of Erroneous Pixels (D1):\n")
        report.append("   - D1 calculates the percentage of pixels where the disparity error exceeds a threshold (e.g., 3 pixels).\n")
        report.append(f"   - HITNET D1: {averages['D1_HITNET']:.2f}%\n")
        report.append(f"   - CRE D1: {averages['D1_CRE']:.2f}%\n")
        report.append(f"   - D455 D1: {averages['D1_D455']:.2f}%\n")
        report.append("   A lower D1 percentage means fewer significant errors, indicating a more reliable depth estimation model, especially in real-world scenarios where accuracy is crucial.\n")

        # # Runtime Analysis
        # report.append("\n7. Runtime:\n")
        # report.append("   - Runtime measures the time taken to process each frame, which is particularly important for real-time applications.\n")
        # report.append(f"   - HITNET Runtime: {np.mean(self.runtime['HITNET']):.4f} seconds\n")
        # report.append(f"   - CRE Runtime: {np.mean(self.runtime['CRE']):.4f} seconds\n")
        # report.append(f"   - D455 Runtime: {np.mean(self.runtime['D455']):.4f} seconds\n")
        # report.append("   A lower runtime indicates better performance, making the model more suitable for time-sensitive applications like autonomous driving or real-time robotics.\n")


        # Conclusion
        report.append("\nConclusion:\n")
        best_mae = min(['HITNET', 'CRE', 'D455'], key=lambda x: averages[f"MAE_{x}"])
        best_rmse = min(['HITNET', 'CRE', 'D455'], key=lambda x: averages[f"RMSE_{x}"])
        best_mse = min(['HITNET', 'CRE', 'D455'], key=lambda x: averages[f"MSE_{x}"])
        best_iou = max(['HITNET', 'CRE', 'D455'], key=lambda x: averages[f"IoU_{x}"])
        best_epe = min(['HITNET', 'CRE', 'D455'], key=lambda x: averages[f"EPE_{x}"])
        best_d1 = min(['HITNET', 'CRE', 'D455'], key=lambda x: averages[f"D1_{x}"])
        # best_runtime = min(['HITNET', 'CRE', 'D455'], key=lambda x: np.mean(self.runtime[x]))

        report.append(f"   - {best_mae} had the best overall performance in terms of average accuracy (lowest MAE).\n")
        report.append(f"   - {best_rmse} was the most consistent, with the lowest squared errors (lowest RMSE).\n")
        report.append(f"   - {best_mse} minimized large errors the most (lowest MSE).\n")
        report.append(f"   - {best_iou} had the best spatial overlap with the ground truth (highest IoU).\n")
        report.append(f"   - {best_epe} showed the most precise disparity predictions (lowest EPE).\n")
        report.append(f"   - {best_d1} had the fewest significant errors (lowest D1 percentage).\n")
        # report.append(f"   - {best_runtime} was the fastest model, processing frames the quickest (lowest runtime).\n")

        # Print and save the table as a part of the report
        print(df)
        with open("frame_analysis/summary_report.txt", "a") as report_file:
            report_file.write("\n\nMetrics Summary Table:\n")
            report_file.write(df.to_string())
            report_file.write("\n\n")
            report_file.write("\n".join(report))
            self.get_logger().info("\nSaved 'summary_report.txt' in 'frame_analysis' directory.")


    def analyze_error_distribution(self, frame_idx):
        """Analyze the error distribution for a specific frame."""
        hitnet_error = np.abs(self.hitnet_depth - self.triangulation_depth)
        cre_error = np.abs(self.cre_depth - self.triangulation_depth)
        d455_error = np.abs(self.d455_depth - self.triangulation_depth)

        plt.figure(figsize=(10, 6))
        plt.hist(hitnet_error.ravel(), bins=50, alpha=0.5, label='HITNET')
        plt.hist(cre_error.ravel(), bins=50, alpha=0.5, label='CRE')
        plt.hist(d455_error.ravel(), bins=50, alpha=0.5, label='D455')
        plt.legend()
        plt.title(f"Error Distribution for Frame {frame_idx}")
        plt.xlabel("Error")
        plt.ylabel("Frequency")
        plt.savefig(f"frame_analysis/error_distribution/error_distribution_frame_{frame_idx}.png")
        plt.close()

        self.get_logger().info(f"Error distribution saved for frame {frame_idx}.")

    def analyze_error_distribution(self, frame_idx):
        """Analyze the error distribution for a specific frame."""
        hitnet_error = np.abs(self.hitnet_depth - self.triangulation_depth)
        cre_error = np.abs(self.cre_depth - self.triangulation_depth)
        d455_error = np.abs(self.d455_depth - self.triangulation_depth)

        plt.figure(figsize=(10, 6))
        plt.hist(hitnet_error.ravel(), bins=50, alpha=0.5, label='HITNET')
        plt.hist(cre_error.ravel(), bins=50, alpha=0.5, label='CRE')
        plt.hist(d455_error.ravel(), bins=50, alpha=0.5, label='D455')
        plt.legend()
        plt.title(f"Error Distribution for Frame {frame_idx}")
        plt.xlabel("Error")
        plt.ylabel("Frequency")
        plt.savefig(f"frame_analysis/error_distribution/error_distribution_frame_{frame_idx}.png")
        plt.close()

        # Save analysis to the summary report
        with open("frame_analysis/summary_report.txt", "a") as report_file:
            report_file.write(f"\nFrame {frame_idx} - Error Distribution Analysis:\n")
            report_file.write(f"HITNET Mean Error: {np.mean(hitnet_error):.4f}\n")
            report_file.write(f"CRE Mean Error: {np.mean(cre_error):.4f}\n")
            report_file.write(f"D455 Mean Error: {np.mean(d455_error):.4f}\n")

        self.get_logger().info(f"Error distribution saved for frame {frame_idx}.")

    def analyze_frame_content_correlation(self, frame_idx):
        """Basic analysis of correlation between frame content and errors."""
        # Regions of interest (ROI) based on distance.
        depth_ranges = [0, 1, 2, 5]  # Example depth ranges in meters
        hitnet_error = np.abs(self.hitnet_depth - self.triangulation_depth)
        cre_error = np.abs(self.cre_depth - self.triangulation_depth)
        d455_error = np.abs(self.d455_depth - self.triangulation_depth)

        roi_errors = {}
        for start, end in zip(depth_ranges[:-1], depth_ranges[1:]):
            mask = (self.triangulation_depth >= start) & (self.triangulation_depth < end)
            roi_errors[f"{start}-{end}m"] = {
                "HITNET": np.mean(hitnet_error[mask]),
                "CRE": np.mean(cre_error[mask]),
                "D455": np.mean(d455_error[mask]),
            }
        
        # Save ROI errors to summary report
        with open("frame_analysis/summary_report.txt", "a") as report_file:
            report_file.write(f"\nFrame {frame_idx} - ROI Error Analysis:\n")
            for roi, errors in roi_errors.items():
                report_file.write(f"ROI {roi}m - HITNET: {errors['HITNET']:.4f}, CRE: {errors['CRE']:.4f}, D455: {errors['D455']:.4f}\n")
                self.get_logger().info(f"ROI {roi}m - HITNET: {errors['HITNET']}, CRE: {errors['CRE']}, D455: {errors['D455']}")


    def track_error_progression(self):
        plt.figure(figsize=(10, 6))

        plt.plot(self.metrics['MAE_HITNET'], label='MAE HITNET')
        plt.plot(self.metrics['MAE_CRE'], label='MAE CRE')
        plt.plot(self.metrics['MAE_D455'], label='MAE D455')
        plt.title("MAE Error Progression Over Time")
        plt.xlabel("Frame Index")
        plt.ylabel("MAE")
        plt.legend()
        plt.savefig("frame_analysis/error_progression/MAE_error_progression_over_time.png")
        plt.close()

        plt.plot(self.metrics['MSE_HITNET'], label='MSE HITNET')
        plt.plot(self.metrics['MSE_CRE'], label='MSE CRE')
        plt.plot(self.metrics['MSE_D455'], label='MSE D455')
        plt.title("MSE Error Progression Over Time")
        plt.xlabel("Frame Index")
        plt.ylabel("MSE")
        plt.legend()
        plt.savefig("frame_analysis/error_progression/MSE_error_progression_over_time.png")
        plt.close()

        plt.plot(self.metrics['RMSE_HITNET'], label='RMSE HITNET')
        plt.plot(self.metrics['RMSE_CRE'], label='RMSE CRE')
        plt.plot(self.metrics['RMSE_D455'], label='RMSE D455')
        plt.title("RMSE Error Progression Over Time")
        plt.xlabel("Frame Index")
        plt.ylabel("RMSE")
        plt.legend()
        plt.savefig("frame_analysis/error_progression/RMSE_error_progression_over_time.png")
        plt.close()

        plt.plot(self.metrics['IoU_HITNET'], label='IoU HITNET')
        plt.plot(self.metrics['IoU_CRE'], label='IoU CRE')
        plt.plot(self.metrics['IoU_D455'], label='IoU D455')
        plt.title("IoU Error Progression Over Time")
        plt.xlabel("Frame Index")
        plt.ylabel("IoU")
        plt.legend()
        plt.savefig("frame_analysis/error_progression/IoU_error_progression_over_time.png")
        plt.close()

        plt.plot(self.metrics['EPE_HITNET'], label='EPE HITNET')
        plt.plot(self.metrics['EPE_CRE'], label='EPE CRE')
        plt.plot(self.metrics['EPE_D455'], label='EPE D455')
        plt.title("EPE Error Progression Over Time")
        plt.xlabel("Frame Index")
        plt.ylabel("EPE")
        plt.legend()
        plt.savefig("frame_analysis/error_progression/EPE_error_progression_over_time.png")
        plt.close()

        plt.plot(self.metrics['D1_HITNET'], label='D1 HITNET')
        plt.plot(self.metrics['D1_CRE'], label='D1 CRE')
        plt.plot(self.metrics['D1_D455'], label='D1 D455')
        plt.title("D1 Error Progression Over Time")
        plt.xlabel("Frame Index")
        plt.ylabel("D1")
        plt.legend()
        plt.savefig("frame_analysis/error_progression/D1_error_progression_over_time.png")
        plt.close()

        self.get_logger().info("Error progression over time plot saved.\n")

    def analyze_error_localization(self, frame_idx):
        """Analyze how errors vary across different regions of the image."""
        
        height, width = self.triangulation_depth.shape
        center_y, center_x = height // 2, width // 2
        
        # Define the central region (keeping it the same size as the original image)
        center_region = np.zeros_like(self.triangulation_depth)
        center_region[center_y-height//4:center_y+height//4, center_x-width//4:center_x+width//4] = \
            self.triangulation_depth[center_y-height//4:center_y+height//4, center_x-width//4:center_x+width//4]

        # Define the periphery region
        periphery_region = np.copy(self.triangulation_depth)
        periphery_region[center_y-height//4:center_y+height//4, center_x-width//4:center_x+width//4] = 0

        # Calculate the errors for these regions
        center_error = np.abs(self.hitnet_depth - center_region)
        periphery_error = np.abs(self.hitnet_depth - periphery_region)

        # Log the error statistics to the terminal
        self.get_logger().info(f"Center error for frame {frame_idx}: {np.mean(center_error)}")
        self.get_logger().info(f"Periphery error for frame {frame_idx}: {np.mean(periphery_error)}")

        # Save the error statistics to the summary report
        with open("frame_analysis/summary_report.txt", "a") as report_file:
            report_file.write(f"\nFrame {frame_idx} - Error Localization Analysis:\n")
            report_file.write(f"Center error: {np.mean(center_error):.4f}\n")
            report_file.write(f"Periphery error: {np.mean(periphery_error):.4f}\n")


    def compare_model_outputs(self, frame_idx):
        """Compare outputs between models for a specific frame."""
        diff_hitnet_cre = np.abs(self.hitnet_depth - self.cre_depth)
        diff_hitnet_d455 = np.abs(self.hitnet_depth - self.d455_depth)
        diff_cre_d455 = np.abs(self.cre_depth - self.d455_depth)

        # Normalize the differences for better visualization
        diff_hitnet_cre = cv2.normalize(diff_hitnet_cre, None, 0, 255, cv2.NORM_MINMAX)
        diff_hitnet_d455 = cv2.normalize(diff_hitnet_d455, None, 0, 255, cv2.NORM_MINMAX)
        diff_cre_d455 = cv2.normalize(diff_cre_d455, None, 0, 255, cv2.NORM_MINMAX)


        plt.figure(figsize=(14, 8))
        plt.subplot(3, 1, 1)
        plt.imshow(diff_hitnet_cre, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title(f"Difference between HITNET and CRE at frame {frame_idx}")

        plt.subplot(3, 1, 2)
        plt.imshow(diff_hitnet_d455, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title(f"Difference between HITNET and D455 at frame {frame_idx}")

        plt.subplot(3, 1, 3)
        plt.imshow(diff_cre_d455, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title(f"Difference between CRE and D455 at frame {frame_idx}")

        plt.tight_layout()
        plt.savefig(f"frame_analysis/models_difference/difference_models_frame_{frame_idx}.png")
        plt.close()

        # Save summary to the report
        with open("frame_analysis/summary_report.txt", "a") as report_file:
            report_file.write(f"\nFrame {frame_idx} - Model Output Comparison:\n")
            report_file.write(f"Average difference between HITNET and CRE: {np.mean(diff_hitnet_cre):.4f}\n")
            report_file.write(f"Average difference between HITNET and D455: {np.mean(diff_hitnet_d455):.4f}\n")
            report_file.write(f"Average difference between CRE and D455: {np.mean(diff_cre_d455):.4f}\n")

        self.get_logger().info(f"Cross-model comparison plots saved for frame {frame_idx}.")


    def analyze_model_disagreement(self, frame_idx):
        """Quantify disagreement between models for a specific frame."""
        depth_stack = np.stack([self.hitnet_depth, self.cre_depth, self.d455_depth], axis=0)
        disagreement_map = np.var(depth_stack, axis=0)

        # Normalize the disagreement map for visualization
        disagreement_map = cv2.normalize(disagreement_map, None, 0, 255, cv2.NORM_MINMAX)


        plt.figure(figsize=(10, 6))
        plt.imshow(disagreement_map, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title(f"Model Disagreement at frame {frame_idx}")
        plt.savefig(f"frame_analysis/disagreement_maps/disagreement_map_frame_{frame_idx}.png")
        plt.close()

        # Save disagreement statistics to the report
        with open("frame_analysis/summary_report.txt", "a") as report_file:
            report_file.write(f"\nFrame {frame_idx} - Model Disagreement Analysis:\n")
            report_file.write(f"Average disagreement (variance) among models: {np.mean(disagreement_map):.4f}\n")

        self.get_logger().info(f"Model disagreement analysis saved for frame {frame_idx}.")


    def calculate_statistics(self):
        statistics = {}
        for key, values in self.metrics.items():
            statistics[key] = {
                "average": np.mean(values) if values else None,
                "median": np.median(values) if values else None,
                "std_dev": np.std(values) if values else None,
            }
        return statistics

    def print_statistics(self):
        statistics = self.calculate_statistics()
        with open("frame_analysis/summary_report.txt", "a") as report_file:
            report_file.write("\nMetrics Statistics:\n")
            for key, stats in statistics.items():
                if stats['average'] is not None:
                    report_file.write(f"{key} - Average: {stats['average']:.4f}, Median: {stats['median']:.4f}, Std Dev: {stats['std_dev']:.4f}\n")
                    print(f"{key} - Average: {stats['average']:.4f}, Median: {stats['median']:.4f}, Std Dev: {stats['std_dev']:.4f}")

    def compare_segments(self):
        # Define segments
        segment_size = len(self.metrics["MAE_HITNET"]) // 2  # Split into two halves

        first_segment_metrics = {}
        second_segment_metrics = {}

        for key, values in self.metrics.items():
            first_segment_metrics[key] = np.mean(values[:segment_size])
            second_segment_metrics[key] = np.mean(values[segment_size:])

        with open("frame_analysis/summary_report.txt", "a") as report_file:
            report_file.write("\nSegment Comparison:\n")
            for key in first_segment_metrics:
                report_file.write(f"{key} - First Segment: {first_segment_metrics[key]:.4f}, Second Segment: {second_segment_metrics[key]:.4f}\n")
                print(f"{key} - First Segment: {first_segment_metrics[key]:.4f}, Second Segment: {second_segment_metrics[key]:.4f}")


    def visualize_and_save_frame(self, frame_idx, metric_name):
        """Visualize the frame with the largest error and save the overlay image."""
        # Normalize depth maps before overlay
        norm_triangulation = cv2.normalize(self.triangulation_depth, None, 0, 255, cv2.NORM_MINMAX)
        norm_hitnet = cv2.normalize(self.hitnet_depth, None, 0, 255, cv2.NORM_MINMAX)
        norm_cre = cv2.normalize(self.cre_depth, None, 0, 255, cv2.NORM_MINMAX)
        norm_d455 = cv2.normalize(self.d455_depth, None, 0, 255, cv2.NORM_MINMAX)

        # Get the specific depth maps for this frame
        triangulation = norm_triangulation
        hitnet = norm_hitnet
        cre = norm_cre
        d455 = norm_d455

        # Create an overlay image for comparison
        overlay_hitnet = cv2.addWeighted(triangulation, 0.5, hitnet, 0.5, 0)
        overlay_cre = cv2.addWeighted(triangulation, 0.5, cre, 0.5, 0)
        overlay_d455 = cv2.addWeighted(triangulation, 0.5, d455, 0.5, 0)

        # Save these overlay images
        cv2.imwrite(f"frame_analysis/Visual_Inspection/{metric_name}_hitnet_overlay_frame_{frame_idx}.png", overlay_hitnet)
        cv2.imwrite(f"frame_analysis/Visual_Inspection/{metric_name}_cre_overlay_frame_{frame_idx}.png", overlay_cre)
        cv2.imwrite(f"frame_analysis/Visual_Inspection/{metric_name}_d455_overlay_frame_{frame_idx}.png", overlay_d455)

        self.get_logger().info(f"Overlay visualizations saved for frame {frame_idx}.")


    def generate_summary_report(self):
        with open("frame_analysis/summary_report.txt", "a") as report_file:
            report_file.write("Depth Estimation Analysis Report\n")
            report_file.write("="*50 + "\n")

            # Write general statistics
            statistics = self.calculate_statistics()
            report_file.write("\nMetrics Statistics:\n")
            for key, stats in statistics.items():
                if stats['average'] is not None:
                    report_file.write(f"{key} - Average: {stats['average']:.4f}, Median: {stats['median']:.4f}, Std Dev: {stats['std_dev']:.4f}\n")

            # Write segment comparison
            report_file.write("\nSegment Comparison:\n")
            first_segment_metrics = {}
            second_segment_metrics = {}
            segment_size = len(self.metrics["MAE_HITNET"]) // 2
            for key, values in self.metrics.items():
                first_segment_metrics[key] = np.mean(values[:segment_size])
                second_segment_metrics[key] = np.mean(values[segment_size:])
                report_file.write(f"{key} - First Segment: {first_segment_metrics[key]:.4f}, Second Segment: {second_segment_metrics[key]:.4f}\n")

            # # Write spike analysis
            # report_file.write("\nSpike Analysis:\n")
            # spike_threshold = 1.5
            # for metric_name, metric_values in self.metrics.items():
            #     spikes = [i for i, value in enumerate(metric_values) if value > spike_threshold * np.mean(metric_values)]
            #     if spikes:
            #         report_file.write(f"Spikes detected in {metric_name} at frames: {spikes}\n")

        print("Summary report generated in frame_analysis/summary_report.txt")


def main(args=None):
    rclpy.init(args=args)
    node = DepthEvaluationNode()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=10)

            # Check if the ROS bag playback has finished and processing is complete
            frame_idx = node.frame_count["CRE"]
            # Set processing_complete to True after processing the last frame
            if node.bag_process.poll() is None and node.last_frame_processed(frame_idx, 0):
                node.processing_complete = True        
            elif node.bag_process.poll() is not None and node.last_frame_processed(frame_idx, 100):
                node.processing_complete = True
            elif node.bag_process.poll() is not None:
                print(f"Attention! only {frame_idx} out of {node.min_frame_count} frames were processed")
                node.processing_complete = True

            if node.processing_complete:
                print("ROS bag playback has ended. Starting evaluation...\n")
                node.print_averages()
                node.generate_analysis_report()
                node.track_error_progression() ##
                node.print_statistics() ##
                node.compare_segments() ##
                node.generate_summary_report()
                node.plot_metrics()
                break

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
