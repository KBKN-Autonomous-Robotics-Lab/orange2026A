#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import sensor_msgs.msg as sensor_msgs
import message_filters
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy

class PointCloudMerger(Node):
    def __init__(self):
        super().__init__('pointcloud_merger')
        
        # set QOS
        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth = 1
        )

        # set parameter (launch can change this parameter)
        self.use_sim_time = self.get_parameter('use_sim_time').get_parameter_value().bool_value
        
        # subscriber
        self.pcd_upper_sub = message_filters.Subscriber(self, sensor_msgs.PointCloud2, '/pcd_rotation')
        self.pcd_lower_sub = message_filters.Subscriber(self, sensor_msgs.PointCloud2, '/pcd_rotation_lidar2')
        
        # multi subscriber
        self.ts = message_filters.ApproximateTimeSynchronizer([self.pcd_upper_sub, self.pcd_lower_sub], queue_size=1, slop=0.05)
        self.ts.registerCallback(self.publish_merged)
        
        # Publisher
        self.pcd_multi_pub = self.create_publisher(sensor_msgs.PointCloud2, 'pcd_rotation_merge', qos_profile) #set publish pcd topic name
        
        # pointcloud
        self.point_cloud_1 = None
        self.point_cloud_2 = None

    def publish_merged(self, upper_msg, lower_msg):
        self.point_cloud_1 = upper_msg
        self.point_cloud_2 = lower_msg

        if self.point_cloud_1 is None or self.point_cloud_2 is None:
            return

        if self.point_cloud_1.point_step != self.point_cloud_2.point_step:
            self.get_logger().error("point_step mismatch")
            return

        merged = sensor_msgs.PointCloud2()
        merged.header.frame_id = self.point_cloud_1.header.frame_id
        if self.use_sim_time:
            merged.header.stamp = self.get_clock().now().to_msg()
            self.get_logger().info(f"header stamp = {merged.header.stamp}")
        else:
            merged.header.stamp = self.point_cloud_1.header.stamp
        merged.height = 1
        merged.width = self.point_cloud_1.width + self.point_cloud_2.width
        merged.fields = self.point_cloud_1.fields
        merged.is_bigendian = self.point_cloud_1.is_bigendian
        merged.point_step = self.point_cloud_1.point_step
        merged.row_step = merged.point_step * merged.width
        merged.is_dense = (self.point_cloud_1.is_dense and self.point_cloud_2.is_dense)
        merged.data = (self.point_cloud_1.data + self.point_cloud_2.data)
        self.pcd_multi_pub.publish(merged)

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudMerger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()