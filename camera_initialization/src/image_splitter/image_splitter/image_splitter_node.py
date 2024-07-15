#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


class ImageSplitter(Node):
    def __init__(self):
        super().__init__('image_splitter')
        self.subscription = self.create_subscription(Image,'/usb_cam/image_raw/compressed',self.listener_callback,10)
        self.publisher_left = self.create_publisher(Image, '/left_camera/image_raw', 10)
        self.publisher_right = self.create_publisher(Image, '/right_camera/image_raw', 10)
        self.pub_left_camera_info = self.create_publisher(CameraInfo, '/left_camera/camera_info', 10)
        self.pub_right_camera_info = self.create_publisher(CameraInfo, '/right_camera/camera_info', 10)

        print("Image Split Successfully")

        self.bridge = CvBridge()

    def listener_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        height, width = cv_image.shape[:2]
        right_image = cv_image[:, :width // 2]
        left_image = cv_image[:, width // 2:]

        # Correctly encoding as 'bgr8'
        left_msg = self.bridge.cv2_to_imgmsg(left_image, encoding="bgr8")
        right_msg = self.bridge.cv2_to_imgmsg(right_image, encoding="bgr8")

        # Correct timestamp and frame_id
        now = self.get_clock().now()
        left_msg.header.stamp = now.to_msg()
        right_msg.header.stamp = now.to_msg()
        left_msg.header.frame_id = "left_camera"
        right_msg.header.frame_id = "right_camera"

        self.publisher_left.publish(left_msg)
        self.publisher_right.publish(right_msg)

        self.publish_camera_info(left_msg, left_msg.header, self.pub_left_camera_info)
        self.publish_camera_info(right_msg, right_msg.header, self.pub_right_camera_info)

    def publish_camera_info(self, msg, header, publisher):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        height, width = cv_image.shape[:2]

        camera_info = CameraInfo()
        camera_info.header = header
        # Populate with actual camera info data as necessary
        camera_info.height = height  # Assuming the height is unchanged
        camera_info.width = width // 2  # Width is now half
        # K, D, R, P matrices should be filled in here based on actual camera calibration

        publisher.publish(camera_info)


def main(args=None):
    rclpy.init(args=args)
    image_splitter = ImageSplitter()
    rclpy.spin(image_splitter)
    image_splitter.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()