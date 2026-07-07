# rclpy (ROS 2のpythonクライアント)の機能を使えるようにします。
import rclpy
# rclpy (ROS 2のpythonクライアント)の機能のうちNodeを簡単に使えるようにします。こう書いていない場合、Nodeではなくrclpy.node.Nodeと書く必要があります。
from rclpy.node import Node
import std_msgs.msg as std_msgs
import nav_msgs.msg as nav_msgs
import sensor_msgs.msg as sensor_msgs
import numpy as np
import math
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
import yaml
import os
import time
import geometry_msgs.msg as geometry_msgs

# C++と同じく、Node型を継承します。
class WaypointManager(Node):
    # コンストラクタです、PcdRotationクラスのインスタンスを作成する際に呼び出されます。
    def __init__(self):
        # 継承元のクラスを初期化します。
        super().__init__('waypoint_manager_node')
        
        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth = 10
        )
        
        qos_profile_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth = 10
        )
        
        # Subscriptionを作成。
        self.subscription = self.create_subscription(nav_msgs.Odometry,'/odom_wheel', self.get_odom, qos_profile_sub)
        self.subscription  # 警告を回避するために設置されているだけです。削除しても挙動はかわりません。
        
        # タイマーを0.1秒（100ミリ秒）ごとに呼び出す
        self.timer = self.create_timer(0.1, self.waypoint_manager)
        
        # Publisherを作成
        self.current_waypoint_publisher = self.create_publisher(geometry_msgs.PoseArray, 'current_waypoint', qos_profile) #set publish pcd topic name
        
        #パラメータ
        #waypoint init
        self.current_waypoint = 0
        self.stop_flag = 0
        
        #positon init
        self.position_x = 0.0 #[m]
        self.position_y = 0.0 #[m]
        self.position_z = 0.0 #[m]
        self.theta_x = 0.0 #[deg]
        self.theta_y = 0.0 #[deg]
        self.theta_z = 0.0 #[deg]
        
        # Waypoint YAMLファイルを読み込む
        waypoint_map_yaml_path_name = "kbkn_maps/waypoints/hosei/m2/courtyard_Senior.yaml" # waypoint mapの名前
        py_path = "/home/ubuntu/ros2_ws/src/"#os.path.dirname(os.path.abspath(__file__)) # 実行ファイルのディレクトリ名
        waypoint_map_yaml_file_path = os.path.join(py_path, waypoint_map_yaml_path_name) # パスの連結
        print(f"waypoint_map_yaml_path_name ={waypoint_map_yaml_path_name}")
        print(f"py_path ={py_path}")
        
        with open(waypoint_map_yaml_file_path, 'r') as yaml_file:
            waypoint_map_yaml_data = yaml.safe_load(yaml_file)
        print(f"waypoint_map_yaml_data ={waypoint_map_yaml_data}")
        waypoints = waypoint_map_yaml_data['waypoints']
        x = [point['point']['x'] for point in waypoints]
        y = [point['point']['y'] for point in waypoints]
        z = [point['point']['z'] for point in waypoints]
        self.waypoints = np.array([x, y, z])
        print(f"self.waypoints ={self.waypoints}")
        
    def waypoint_manager(self):
        self.time_stamp = self.get_clock().now().to_msg()
        #self.get_logger().info('waypoint manager cntl')
        position_x=self.position_x; position_y=self.position_y; 
        theta_x=self.theta_x; theta_y=self.theta_y; theta_z=self.theta_z;
        
        #waypoint theta & dist
        relative_point_x = self.waypoints[0,self.current_waypoint] - position_x
        relative_point_y = self.waypoints[1,self.current_waypoint] - position_y
        relative_point = np.vstack((relative_point_x, relative_point_y, self.waypoints[2,self.current_waypoint]))
        relative_point_rot, t_point_rot_matrix = rotation_xyz(relative_point, theta_x, theta_y, -theta_z)
        waypoint_rad = math.atan2(relative_point_rot[1], relative_point_rot[0])
        waypoint_dist = math.sqrt(relative_point_x**2 + relative_point_y**2)
        waypoint_theta = abs(waypoint_rad * (180 / math.pi))
        
        #set judge dist
        if abs(waypoint_theta) > 90:
            determine_dist = 8
        else:
            determine_dist = 3
        #check if the waypoint reached
        if waypoint_dist < determine_dist:
            self.current_waypoint += 1
            if self.current_waypoint > (len(self.waypoints[0,:]) - 1):
                self.stop_flag = 1
                self.get_logger().info('GOAL : stop_flag = %f' % (stop_flag))
        self.get_logger().info('current_waypoint:x = %f, y = %f : waypoint_no = %f' % (self.waypoints[0,self.current_waypoint], self.waypoints[1,self.current_waypoint], self.current_waypoint))
        
        #publish
        pose_array = self.current_waypoint_msg(self.waypoints[:,self.current_waypoint], 'map')
        self.current_waypoint_publisher.publish(pose_array)
        
    def get_odom(self, msg):
        self.position_x = msg.pose.pose.position.x
        self.position_y = msg.pose.pose.position.y
        self.position_z = msg.pose.pose.position.z
        
        flio_q_x = msg.pose.pose.orientation.x
        flio_q_y = msg.pose.pose.orientation.y
        flio_q_z = msg.pose.pose.orientation.z
        flio_q_w = msg.pose.pose.orientation.w
        
        roll, pitch, yaw = quaternion_to_euler(flio_q_x, flio_q_y, flio_q_z, flio_q_w)
        
        self.theta_x = 0 #roll /math.pi*180
        self.theta_y = 0 #pitch /math.pi*180
        self.theta_z = yaw /math.pi*180
        
    def current_waypoint_msg(self, waypoint, set_frame_id):
        pose_array = geometry_msgs.PoseArray()
        pose_array.header.frame_id = set_frame_id
        pose_array.header.stamp = self.time_stamp
        pose = geometry_msgs.Pose()
        pose.position.x = waypoint[0]
        pose.position.y = waypoint[1]
        pose.position.z = waypoint[2]
        
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0
        pose.orientation.w = 1.0
        pose_array.poses.append(pose)
        
        return pose_array

def rotation_xyz(pointcloud, theta_x, theta_y, theta_z):
    theta_x = math.radians(theta_x)
    theta_y = math.radians(theta_y)
    theta_z = math.radians(theta_z)
    rot_x = np.array([[ 1,                 0,                  0],
                      [ 0, math.cos(theta_x), -math.sin(theta_x)],
                      [ 0, math.sin(theta_x),  math.cos(theta_x)]])
    
    rot_y = np.array([[ math.cos(theta_y), 0,  math.sin(theta_y)],
                      [                 0, 1,                  0],
                      [-math.sin(theta_y), 0, math.cos(theta_y)]])
    
    rot_z = np.array([[ math.cos(theta_z), -math.sin(theta_z), 0],
                      [ math.sin(theta_z),  math.cos(theta_z), 0],
                      [                 0,                  0, 1]])
    rot_matrix = rot_z.dot(rot_y.dot(rot_x))
    #print(f"rot_matrix ={rot_matrix}")
    #print(f"pointcloud ={pointcloud.shape}")
    rot_pointcloud = rot_matrix.dot(pointcloud)
    return rot_pointcloud, rot_matrix
    
def quaternion_to_euler(x, y, z, w):
    # クォータニオンから回転行列を計算
    rot_matrix = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x**2 + z**2), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x**2 + y**2)]
    ])

    # 回転行列からオイラー角を抽出
    roll = np.arctan2(rot_matrix[2, 1], rot_matrix[2, 2])
    pitch = np.arctan2(-rot_matrix[2, 0], np.sqrt(rot_matrix[2, 1]**2 + rot_matrix[2, 2]**2))
    yaw = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])
    return roll, pitch, yaw
        
# mainという名前の関数です。C++のmain関数とは異なり、これは処理の開始地点ではありません。
def main(args=None):
    # rclpyの初期化処理です。ノードを立ち上げる前に実装する必要があります。
    rclpy.init(args=args)
    # クラスのインスタンスを作成
    waypoint_manager = WaypointManager()
    # spin処理を実行、spinをしていないとROS 2のノードはデータを入出力することが出来ません。
    rclpy.spin(waypoint_manager)
    # 明示的にノードの終了処理を行います。
    waypoint_manager.destroy_node()
    # rclpyの終了処理、これがないと適切にノードが破棄されないため様々な不具合が起こります。
    rclpy.shutdown()

# 本スクリプト(publish.py)の処理の開始地点です。
if __name__ == '__main__':
    # 関数`main`を実行する。
    main()
