import rclpy
from rclpy.node import Node
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rosbag2_py import SequentialWriter
from rosidl_runtime_py.utilities import get_message

class BagTrimmer(Node):
    def __init__(self, input_bag, output_bag, num_frames=50):
        super().__init__('bag_trimmer')
        self.input_bag = input_bag
        self.output_bag = output_bag
        self.num_frames = num_frames
        self.sync_topics = [
            '/camera_triangulation/raw_depth_map', 
            '/HITNET/raw_depth', 
            '/CRE/raw_depth', 
            '/camera/camera/depth/image_rect_raw', 
            '/camera/camera/infra1/image_rect_raw', 
            '/camera/camera/infra2/image_rect_raw'
        ]

    def trim_bag(self):
        storage_options = StorageOptions(uri=self.input_bag, storage_id='sqlite3')
        converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        reader = SequentialReader()
        reader.open(storage_options, converter_options)

        writer = SequentialWriter()
        writer.open(StorageOptions(uri=self.output_bag, storage_id='sqlite3'), converter_options)

        # Copy the original bag's metadata
        for topic_metadata in reader.get_all_topics_and_types():
            writer.create_topic(topic_metadata)

        frame_count = 0
        synced_messages = {topic: None for topic in self.sync_topics}

        while reader.has_next() and frame_count < self.num_frames:
            topic, data, t = reader.read_next()

            if topic in self.sync_topics:
                synced_messages[topic] = (data, t)

                # Check if we have a complete set of synchronized messages
                if all(synced_messages[topic] is not None for topic in self.sync_topics):
                    # Write the synchronized messages to the new bag
                    for topic in self.sync_topics:
                        data, timestamp = synced_messages[topic]
                        writer.write(topic, data, timestamp)
                    
                    frame_count += 1
                    synced_messages = {topic: None for topic in self.sync_topics}  # Reset for the next set of messages

        # No need to close the reader explicitly
        self.get_logger().info(f'Trimmed bag saved as {self.output_bag} with {frame_count} synchronized frames.')

def main(args=None):
    rclpy.init(args=args)
    input_bag = '/home/student/ros2_ws/recordings_proc/outdoor_recording/outdoor_recording_0.db3'  # Replace with your input bag path
    output_bag = '/home/student/ros2_ws/recordings_trimmed/outdoor_recording/outdoor_recording_0_trimmed.db3'  # Replace with your output bag path

    bag_trimmer = BagTrimmer(input_bag, output_bag)
    bag_trimmer.trim_bag()

    rclpy.shutdown()

if __name__ == '__main__':
    main()
