#!/usr/bin/env python3
import rclpy
import math
import numpy as np
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Header
import threading
import time
from my_msgs.srv import Avglatlon

class Odom_Combination(Node):
    def __init__(self):
        super().__init__('odom_combination')

        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth = 1
        )
        
        # subscription
        self.odom_sub = self.create_subscription(Odometry, '/odom/wheel_spimu', self.get_odom, qos_profile)
        self.gps_odom_sub = self.create_subscription(Odometry, '/odom/UM982', self.get_gps_odom, qos_profile)
        
        # publisher
        self.odom_pub = self.create_publisher(Odometry, '/odom/combine', qos_profile)
        
        self.position_x = 0.0
        self.position_y = 0.0
        self.theta_z = 0.0
        self.theta = 0.0
        self.initial_xy = None #(0.0, 0.0)
        #self.Position_magnification = 1.0  # 必要なら調整

        self.timer = self.create_timer(0.1, self.combine)
    
    # /odom callback
    def get_odom(self, msg):
        self.position_x = msg.pose.pose.position.x
        self.position_y = msg.pose.pose.position.y
        x, y, z, w = msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w
        roll, pitch, yaw = quaternion_to_euler(x, y, z, w)
        self.theta_z = yaw  # radian
    
    # /odom/UM982 callback
    def get_gps_odom(self, msg):
        if self.initial_xy is None:
            init_x = msg.pose.pose.position.x
            init_y = msg.pose.pose.position.y
            x, y, z, w = msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w
            roll, pitch, yaw = quaternion_to_euler(x, y, z, w)
            self.initial_xy = (init_x, init_y)
            self.init_theta = yaw
            init_degree = yaw * 180.0 / math.pi # for debug
            #self.yaw_offset = self.init_theta - self.theta_z
            self.get_logger().info(f"Initial /odom/UM982 position set to: x={init_x:.3f}, y={init_y:.3f}, degree={init_degree:.3f}")   
    
    def yaw_to_orientation(self, yaw):
        orientation_z = np.sin(yaw / 2.0)
        orientation_w = np.cos(yaw / 2.0)
        return orientation_z, orientation_w
    
    def combine(self):
        if self.initial_xy is None:
            self.get_logger().warn("Waiting for initial /odom/UM982 position...")
            return
            
        pos_x = self.position_x
        pos_y = self.position_y
        #degree_to_radian = math.pi / 180
        #pos_theta = self.theta * degree_to_radian # maybe -self.theta * degree_to_radian
        pos_theta = self.theta_z + self.init_theta

        # rotate init_theta for xy 
        cos_theta = math.cos(self.init_theta)
        sin_theta = math.sin(self.init_theta)

        rotated_x = cos_theta * pos_x - sin_theta * pos_y
        rotated_y = sin_theta * pos_x + cos_theta * pos_y

        # init xy + rotate xy
        combined_x = self.initial_xy[0] + rotated_x
        combined_y = self.initial_xy[1] + rotated_y
                
        # quartanion z,w
        odom_orientation = self.yaw_to_orientation(pos_theta)

        # Odometryメッセージ作成
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "odom"
        odom_msg.pose.pose.position.x = combined_x
        odom_msg.pose.pose.position.y = combined_y
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation.x = 0.0
        odom_msg.pose.pose.orientation.y = 0.0
        odom_msg.pose.pose.orientation.z = float(odom_orientation[0])
        odom_msg.pose.pose.orientation.w = float(odom_orientation[1])

        # publish
        self.odom_pub.publish(odom_msg)        

def quaternion_to_euler(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw
              
def main(args=None):
    rclpy.init(args=args)
    odom_combination = Odom_Combination()
    rclpy.spin(odom_combination)

if __name__ == '__main__':
    main()          

