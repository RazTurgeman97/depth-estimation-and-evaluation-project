import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
import subprocess

class SynchronizedPlaybackNode(Node):
    def __init__(self):
        super().__init__('synchronized_playback_node')
        
        self.bridge = CvBridge()

        # Subscribers with message filters
        self.left_image_sub = message_filters.Subscriber(self, Image, '/camera/camera/infra1/image_rect_raw')
        self.right_image_sub = message_filters.Subscriber(self, Image, '/camera/camera/infra2/image_rect_raw')
        self.d455_image_sub = message_filters.Subscriber(self, Image, '/camera/camera/depth/image_rect_raw')
        self.triangulation_sub = message_filters.Subscriber(self, Image, '/camera_triangulation/raw_depth_map')
        self.hitnet_sub = message_filters.Subscriber(self, Image, '/HITNET/raw_depth')
        self.cre_sub = message_filters.Subscriber(self, Image, '/CRE/raw_depth')

        # Synchronize the topics
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.left_image_sub, self.right_image_sub, self.d455_image_sub, self.triangulation_sub, self.hitnet_sub, self.cre_sub], 
            queue_size=200, 
            slop=0.3
        )
        self.ts.registerCallback(self.synchronized_callback)

        # Command to play the ROS bag
        self.bagfile_path = '/home/student/recordings_proc/indoor_recording/indoor_recording_0.db3'
        self.bag_process = None

        # Publishers
        self.publisher_hitnet_raw_depth = self.create_publisher(Image, '/HITNET/raw_depth_', 1)
        self.publisher_cre_raw_depth = self.create_publisher(Image, '/CRE/raw_depth_', 1)
        self.tri_publisher = self.create_publisher(Image, '/camera_triangulation/raw_depth_map_', 1)
        self.d455_publisher= self.create_publisher(Image, '/camera/camera/depth/image_rect_raw_', 1)
        self.infra1_publisher = self.create_publisher(Image, '/camera/camera/infra1/image_rect_raw_', 1)
        self.infra2_publisher = self.create_publisher(Image, '/camera/camera/infra2/image_rect_raw_', 1)

        # ROS bag record path
        self.record_path = '/home/student/recordings_synchronized/indoor_recording'
        self.record_process = None

        # Timer to check if the playback has finished
        self.timer = self.create_timer(1.0, self.check_playback_status)

        # Start playing the ROS bag
        self.start_rosbag_playback()
        self.start_rosbag_recording()

    def check_playback_status(self):
        # Check if the playback process is still running
        if self.bag_process.poll() is not None:
            self.get_logger().info('ROS bag playback finished.')
            self.stop_and_save_recording()

    def stop_and_save_recording(self):
        if self.record_process:
            self.get_logger().info('Stopping ROS bag recording...')
            self.record_process.terminate()
            self.record_process.wait()  # Wait for the process to terminate
            self.get_logger().info(f'Recording saved to {self.record_path}.')
            self.shutdown()

    def start_rosbag_recording(self):
        try:
            self.get_logger().info(f'Starting ROS bag recording to {self.record_path}...')
            topics_to_record = [
                '/HITNET/raw_depth_', '/CRE/raw_depth_', 
                '/camera_triangulation/raw_depth_map_', '/camera/camera/depth/image_rect_raw_', 
                '/camera/camera/infra1/image_rect_raw_', '/camera/camera/infra2/image_rect_raw_'
            ]
            self.record_process = subprocess.Popen(['ros2', 'bag', 'record', '-o', self.record_path] + topics_to_record)
            self.get_logger().info('ROS bag recording started.')
        except Exception as e:
            self.get_logger().error(f'Failed to start ROS bag recording: {e}')

    def start_rosbag_playback(self):
        try:
            self.get_logger().info('Starting ROS bag playback...')
            self.bag_process = subprocess.Popen(['ros2', 'bag', 'play', self.bagfile_path])
            self.get_logger().info('ROS bag playback started.')
        except Exception as e:
            self.get_logger().error(f'Failed to start ROS bag playback: {e}')

    def synchronized_callback(self, left_msg, right_msg, depth_msg, triangulation_msg, hitnet_msg, cre_msg):
        # Here you can process the synchronized messages
        self.get_logger().info('Synchronized messages received.')
        
        # Convert ROS Image messages to OpenCV images
        left_image = self.bridge.imgmsg_to_cv2(left_msg, 'passthrough')
        right_image = self.bridge.imgmsg_to_cv2(right_msg, 'passthrough')
        d455_image = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
        triangulation_image = self.bridge.imgmsg_to_cv2(triangulation_msg, 'passthrough')
        hitnet_image = self.bridge.imgmsg_to_cv2(hitnet_msg, 'passthrough')
        cre_image = self.bridge.imgmsg_to_cv2(cre_msg, 'passthrough')
        
        # Use the timestamp from the cre_msg
        timestamp = cre_msg.header.stamp

        # Publish the images with the same timestamp
        self.publish_topics(left_image, right_image, d455_image, triangulation_image, hitnet_image, cre_image, timestamp)

    def publish_topics(self, left_image, right_image, d455_image, triangulation_image, hitnet_image, cre_image, timestamp):
        # Convert OpenCV images back to ROS Image messages and publish them
        
        left_msg = self.bridge.cv2_to_imgmsg(left_image, encoding='passthrough')
        left_msg.header.stamp = timestamp
        self.infra1_publisher.publish(left_msg)
        
        right_msg = self.bridge.cv2_to_imgmsg(right_image, encoding='passthrough')
        right_msg.header.stamp = timestamp
        self.infra2_publisher.publish(right_msg)
        
        d455_msg = self.bridge.cv2_to_imgmsg(d455_image, encoding='passthrough')
        d455_msg.header.stamp = timestamp
        self.d455_publisher.publish(d455_msg)
        
        triangulation_msg = self.bridge.cv2_to_imgmsg(triangulation_image, encoding='passthrough')
        triangulation_msg.header.stamp = timestamp
        self.tri_publisher.publish(triangulation_msg)
        
        hitnet_msg = self.bridge.cv2_to_imgmsg(hitnet_image, encoding='passthrough')
        hitnet_msg.header.stamp = timestamp
        self.publisher_hitnet_raw_depth.publish(hitnet_msg)
        
        cre_msg = self.bridge.cv2_to_imgmsg(cre_image, encoding='passthrough')
        cre_msg.header.stamp = timestamp
        self.publisher_cre_raw_depth.publish(cre_msg)

    def shutdown(self):
        if self.bag_process:
            self.bag_process.terminate()
        if self.record_process:
            self.record_process.terminate()
        self.destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SynchronizedPlaybackNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
