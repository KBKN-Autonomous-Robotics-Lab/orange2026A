# rclpy (ROS 2のpythonクライアント)の機能を使えるようにします。
import rclpy
# rclpy (ROS 2のpythonクライアント)の機能のうちNodeを簡単に使えるようにします。こう書いていない場合、Nodeではなくrclpy.node.Nodeと書く必要があります。
from rclpy.node import Node
# ROS 2の文字列型を使えるようにimport
import std_msgs.msg as std_msgs
import sensor_msgs.msg as sensor_msgs
import nav_msgs.msg as nav_msgs
from livox_ros_driver2.msg import CustomMsg
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import pandas as pd
#import open3d as o3d
from std_msgs.msg import Int8MultiArray
from nav_msgs.msg import OccupancyGrid
import cv2
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
import yaml
import os
import time
import matplotlib.pyplot
import struct
import geometry_msgs.msg as geometry_msgs
from std_msgs.msg import Int32

from scipy import interpolate
from std_msgs.msg import Float32MultiArray
import cv2

from sensor_msgs_py import point_cloud2 as pc2
# import tf
from tf2_ros import Buffer
from tf2_ros import TransformListener
from tf2_ros import TransformException
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

#map save
#ros2 run nav2_map_server map_saver_cli -t /reflect_map_global -f ~/ros2_ws/src/map/test_map --ros-args -p map_subscribe_transient_local:=true -r __ns:=/namespace
#ros2 run nav2_map_server map_saver_cli -t /reflect_map_global --occ 0.10 --free 0.05 -f ~/ros2_ws/src/map/test_map2 --ros-args -p map_subscribe_transient_local:=true -r __ns:=/namespace
#--occ:  occupied_thresh  この閾値よりも大きい占有確率を持つピクセルは、完全に占有されていると見なされます。
#--free: free_thresh	  占有確率がこの閾値未満のピクセルは、完全に占有されていないと見なされます。

# C++と同じく、Node型を継承します。
class PotentialAStar(Node):
    # コンストラクタです、Mid360Subscriberクラスのインスタンスを作成する際に呼び出されます。
    def __init__(self):
        # 継承元のクラスを初期化します。（https://www.python-izm.com/advanced/class_extend/）今回の場合継承するクラスはNodeになります。
        super().__init__('potential_astar_node')
        
        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth = 1
        )
        
        qos_profile_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth = 1
        )
        
        # set parameter (launch can change this parameter)
        self.declare_parameter('odom', '/odom')
        
        # define parameter
        odom_topic = self.get_parameter('odom').get_parameter_value().string_value
        
        # Subscriptionを作成。CustomMsg型,'/livox/lidar'という名前のtopicをsubscribe。
        self.subscription = self.create_subscription(sensor_msgs.PointCloud2, '/pcd_segment_obs', self.potential_astar, qos_profile)
        #self.subscription = self.create_subscription(nav_msgs.Odometry,'/odom/wheel_imu', self.get_odom, qos_profile_sub)
        self.subscription = self.create_subscription(nav_msgs.Odometry, odom_topic, self.get_odom, qos_profile_sub) # /odom/wheel_spimu
        #self.subscription = self.create_subscription(nav_msgs.Odometry,'/odom_fast', self.get_odom, qos_profile_sub)
        #self.subscription = self.create_subscription(nav_msgs.Odometry,'/odom_ekf_match', self.get_odom, qos_profile_sub)
        self.subscription = self.create_subscription(geometry_msgs.PoseArray,'/current_waypoint', self.get_waypoint, qos_profile_sub)
        self.waypoint_number_subscription = self.create_subscription(Int32,'/waypoint_number', self.get_waypoint_number, qos_profile_sub)
        #self.subscription = self.create_subscription(sensor_msgs.PointCloud2, '/map_obs', self.get_map_obs, qos_profile)
        #self.pothole_subscription = self.create_subscription(sensor_msgs.PointCloud2, '/pothole_points', self.get_pot_obs, qos_profile)
        #self.tire_subscription = self.create_subscription(sensor_msgs.PointCloud2, '/tire_points', self.get_tire_obs, qos_profile)
        self.white_subscription = self.create_subscription(sensor_msgs.PointCloud2, '/white_buff', self.get_white_obs, qos_profile)
        self.right_subscription = self.create_subscription(sensor_msgs.PointCloud2, '/line_buff_right', self.get_right_obs, qos_profile)
        self.left_subscription = self.create_subscription(sensor_msgs.PointCloud2, '/line_buff_left', self.get_left_obs, qos_profile)
        self.dot_subscription = self.create_subscription(sensor_msgs.PointCloud2, '/dotted_line', self.get_dot_obs, qos_profile)
        self.subscription  # 警告を回避するために設置されているだけです。削除しても挙動はかわりません。
        #self.timer = self.create_timer(0.05, self.timer_callback)
        
        # Publisherを作成
        self.potential_astar_path_publisher = self.create_publisher(nav_msgs.Path, 'potential_astar_path', qos_profile)
        self.pcd_obs_global_publisher = self.create_publisher(sensor_msgs.PointCloud2, 'pcd_obs_global', qos_profile) 

        # tf
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        #パラメータ
        #mid360 positon init
        self.position_x = 0.0 #[m]
        self.position_y = 0.0 #[m]
        self.position_z = 0.0 #[m]
        self.theta_x = 0.0 #[deg]
        self.theta_y = 0.0 #[deg]
        self.theta_z = 0.0 #[deg]
        
        #mid360 buff
        self.pcd_ground_buff = np.array([[],[],[],[]]);
        
        #potential astar
        #241025パラメータ:self.cg=20; self.lg=20; self.co=20; self.lo=0.55;
        self.cg=20 #ポテンシャルの引力パラメータ
        self.lg=20 #ポテンシャルの引力パラメータ
        self.co=11 #ポテンシャルの斥力パラメータ SICKパラ目：co=11;lo=0.55;
        self.lo=0.45#55 #0.5#0.9#ポテンシャルの斥力パラメータ
        self.est_xy = [0,0]#自己位置仮入力
        self.wp_xy = [10,0]#ウェイポイント仮入力
        self.astar_path = [0,10]#ウェイポイント仮入力
        self.obs_pixel = 1000/50#障害物のグリッドサイズ設定 
        self.obs_range = 10#障害物情報の範囲
        self.obs_judge = 0#obs_judgeより大きい場合障害物ありと判定する
        
        
        self.cg2nd=20 #ポテンシャルの引力パラメータ
        self.lg2nd=20 #ポテンシャルの引力パラメータ
        self.co2nd=11 #ポテンシャルの斥力パラメータ SICKパラ目：co=11;lo=0.55;
        self.lo2nd=0.22#55 #0.5#0.9#ポテンシャルの斥力パラメータ  24/11/29 ok IGVC20250601 0.25 -> 0.22
        #self.lo2nd=0.30#55 #0.5#0.9#ポテンシャルの斥力パラメータ
        
        
        
        #waypoint
        self.waypoint_xy = np.array([[10],[0],[0]])
        self.waypoint = np.array([[10],[0],[0]])
        self.waypoint_number = 0
        self.pothole_number = 0 #functions test
        self.white_number = 0 # front stop:1 lanechange:0 Q1:0 
        self.right_number = 0 # front stop:0 lanechange:1 Q1:0
        self.left_number = 0 # front stop:None lanechange:1 Q1:0
        self.dot_number = 0 # front stop:0 lanechange:0 Q1:0
        
        #map_obs
        self.map_obs_points = np.array([[],[],[]])
        
        #pot_obs
        self.pot_obs_points = np.array([[],[],[]])
        
        #tire_obs
        self.tire_obs_points = np.array([[],[],[]])
        
        #white_obs
        self.white_obs_points = np.array([[],[],[]])
        self.right_obs_points = np.array([[],[],[]])
        self.left_obs_points = np.array([[],[],[]])
        self.dot_obs_points = np.array([[],[],[]])
        
        #DRIVE MODE
        self.functions_test = 0 #autonav:1 selfdrive:0
        
        #obs info for SELF DRIVE
        self.tire_info      = 0
        self.pothole_info   = 1
        self.human_info     = 2
        self.stopsign_info  = 3
        self.whiteline_info = 4
        self.rightline_info = 5 
        self.leftline_info  = 6
        self.dotline_info   = 7
        self.obs_info = [
            #   0       1     2        3         4         5        6       7
            #tire pothole human stopsign whiteline rightline leftline dotline
            [   0,      0,    0,       1,        1,        0,       0,      0], # waypoint  0 front stop
            [   0,      0,    0,       1,        0,        1,       0,      0], # waypoint  1 front r lane
            [   0,      0,    0,       0,        1,        0,       0,      1], # waypoint  2 curve
            [   0,      0,    0,       0,        1,        0,       0,      1], # waypoint  3 front barrel
            [   0,      0,    0,       0,        0,        1,       1,      0], # waypoint  4 next barrel :lanechange
            [   0,      0,    0,       0,        1,        1,       1,      0], # waypoint  5 front barrel
            [   0,      0,    0,       0,        0,        1,       1,      0], # waypoint  6 next barrel :lanechange
            [   0,      0,    0,       2,        1,        0,       0,      1], # waypoint  7 front stop
            [   0,      0,    0,       2,        0,        1,       0,      0], # waypoint  8 intersection
            [   0,      0,    0,       0,        0,        0,       0,      0], # waypoint  9 front stop
            [   0,      0,    1,       0,        0,        1,       0,      0], # waypoint 10 intersection :human
            [   0,      0,    1,       0,        1,        0,       0,      0], # waypoint 11 front r lane
            [   0,      0,    0,       0,        1,        0,       0,      1], # waypoint 12 curve
            [   0,      0,    0,       0,        1,        0,       0,      1], # waypoint 13 front pothole
            [   0,      1,    0,       0,        0,        1,       1,      0], # waypoint 14 next pothole :lanechange
            [   0,      0,    0,       0,        1,        0,       0,      1], # waypoint 15 front tire
            [   1,      0,    0,       0,        0,        1,       1,      0], # waypoint 16 next tire :lanechange
            [   0,      0,    0,       0,        1,        0,       0,      1], # waypoint 17 curve
            [   0,      0,    0,       3,        1,        0,       0,      1], # waypoint 18 front stop
            [   0,      0,    0,       3,        0,        1,       0,      0], # waypoint 19 intersection
            [   0,      0,    0,       0,        1,        0,       0,      0], # waypoint 20 front r lane
            [   0,      0,    0,       0,        1,        0,       0,      0]  # waypoint 21 GOAL!!!!!!
        ]
        
        #tukuba_obs_x =np.linspace(-45,25,860);
        #tukuba_obs_y =np.linspace(97,94,860);
        #tukuba_obs_z =np.linspace(0,0,860);
        #self.tsukuba_obs = np.array([tukuba_obs_x, tukuba_obs_y, tukuba_obs_z]);
        self.tsukuba_obs = np.array([[],[],[]])
               
        
        ################# IGVC SelfDrive Quolification line stop test #20250530# #################
        self.sd_line_stop_test = 0
        ##########################################################################################
        
    def timer_callback(self):
        #
        pass
        
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
        
	#self.est_xy = np.array([self.position_x, self.position_y])
	#self.est_xy = np.array([0, 0]) #test param
	
    def get_waypoint(self, msg):
        #get waypoint
        self.waypoint = np.array([[pose.position.x, pose.position.y, pose.position.z] for pose in msg.poses]).T
        
        ##relative waypoint
        #relative_point_x = self.waypoint[0] - self.position_x
        #relative_point_y = self.waypoint[1] - self.position_y
        #relative_point = np.array((relative_point_x, relative_point_y, self.waypoint[2]))
        #relative_point_rot, t_point_rot_matrix = rotation_xyz(relative_point, self.theta_x, self.theta_y, -self.theta_z)
        
        #self.wp_xy = [relative_point_rot[0], relative_point_rot[1]]
        
    
    def get_waypoint_number(self, msg):
        #get waypoint number
        self.waypoint_number = msg.data
        
    def pointcloud2_to_array(self, cloud_msg):
        available_fields = [field.name for field in cloud_msg.fields]
        
        fields = ("x","y","z")
        use_intensity = "intensity" in available_fields
        
        if use_intensity:
            fields += ("intensity",)

        # PointCloud2 データを取得
        points = pc2.read_points(cloud_msg, field_names=fields, skip_nans=True)

        # データを配列に変換
        x, y, z = [], [], []
        intensity = []

        for pt in points:
            x.append(pt[0])
            y.append(pt[1])
            z.append(pt[2])
            if use_intensity:
                intensity.append(pt[3])

        # numpy 配列に変換
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        if use_intensity:
            intensity = np.array(intensity)
            return np.vstack((x, y, z, intensity))
        else:
            return np.vstack((x, y, z))
        
        """
        # Extract point cloud data
        points = np.frombuffer(cloud_msg.data, dtype=np.uint8).reshape(-1, cloud_msg.point_step)
        x = np.frombuffer(points[:, 0:4].tobytes(), dtype=np.float32)
        y = np.frombuffer(points[:, 4:8].tobytes(), dtype=np.float32)
        z = np.frombuffer(points[:, 8:12].tobytes(), dtype=np.float32)
        intensity = np.frombuffer(points[:, 12:16].tobytes(), dtype=np.float32)

        # Combine into a 4xN matrix
        point_cloud_matrix = np.vstack((x, y, z, intensity))
        
        return point_cloud_matrix
        """
        
    def get_map_obs(self, msg):
        #print stamp message
        t_stamp = msg.header.stamp
        #print(f"t_stamp ={t_stamp}")
        
        #get pcd data
        points = self.pointcloud2_to_array(msg)
        #print(f"points ={points.shape}")
        
        self.map_obs_points = np.vstack((points[0,:], points[1,:], points[2,:]))
        
    def get_pot_obs(self, msg):
        #print("pothole_points received")
        #print stamp message
        t_stamp = msg.header.stamp
        #print(f"t_stamp ={t_stamp}")
        
        #get pcd data
        points = self.pointcloud2_to_array(msg)
        #print(f"points ={points.shape}")
        
        self.pot_obs_points = np.vstack((points[0,:], points[1,:], points[2,:]))
    
    def get_tire_obs(self, msg):
        #print("pothole_points received")
        #print stamp message
        t_stamp = msg.header.stamp
        #print(f"t_stamp ={t_stamp}")
        
        #get pcd data
        points = self.pointcloud2_to_array(msg)
        #print(f"points ={points.shape}")
        
        self.tire_obs_points = np.vstack((points[0,:], points[1,:], points[2,:]))
        
    def get_white_obs(self, msg):
        #print stamp message
        t_stamp = msg.header.stamp
        #print(f"t_stamp ={t_stamp}")
        
        #get pcd data
        points = self.pointcloud2_to_array(msg)
        #print(f"points ={points.shape}")
        
        self.white_obs_points = np.vstack((points[0,:], points[1,:], points[2,:]))

    def get_left_obs(self, msg):
        #print stamp message
        t_stamp = msg.header.stamp
        #print(f"t_stamp ={t_stamp}")
        
        #get pcd data
        points = self.pointcloud2_to_array(msg)
        #print(f"points ={points.shape}")
        
        self.left_obs_points = np.vstack((points[0,:], points[1,:], points[2,:]))        

    def get_right_obs(self, msg):
        #print stamp message
        t_stamp = msg.header.stamp
        #print(f"t_stamp ={t_stamp}")
        
        #get pcd data
        points = self.pointcloud2_to_array(msg)
        #print(f"points ={points.shape}")
        
        self.right_obs_points = np.vstack((points[0,:], points[1,:], points[2,:]))

    def get_dot_obs(self, msg):
        #print stamp message
        t_stamp = msg.header.stamp
        #print(f"t_stamp ={t_stamp}")
        
        #get pcd data
        points = self.pointcloud2_to_array(msg)
        #print(f"points ={points.shape}")
        
        self.dot_obs_points = np.vstack((points[0,:], points[1,:], points[2,:]))
            
    def potential_astar(self, msg):
        
        #print stamp message
        t_stamp = msg.header.stamp
        #print(f"t_stamp ={t_stamp}")
        
        #get pcd data
        try:
            transform = self.tf_buffer.lookup_transform("odom", "livox_frame", rclpy.time.Time.from_msg(msg.header.stamp))
        except TransformException as ex:
            self.get_logger().warn(f"TF lookup failed: {ex}")
            return      
              
        #global_points = do_transform_cloud(msg, transform)

        points = self.pointcloud2_to_array(msg)
        #print(f"points ={points.shape}")
        print(f"self.pot_obs_points.shape = {self.pot_obs_points.shape}")
        
        
        position_x=self.position_x; position_y=self.position_y; 
        theta_x=self.theta_x; theta_y=self.theta_y; theta_z=self.theta_z;
              
        ############ obs info #############
        #pot_obs add(global)
        pothole_local = np.array([[],[],[]])
        if len(self.pot_obs_points[0,:])>0:
            if self.functions_test == 1:
                if self.waypoint_number >= self.pothole_number: # front stop >= 1, lanechange == 0
                    pothole_local = localization_xyz(self.pot_obs_points, position_x, position_y, theta_x, theta_y, theta_z)
            elif self.obs_info[self.waypoint_number][self.pothole_info] == 1:
                pothole_local = localization_xyz(self.pot_obs_points, position_x, position_y, theta_x, theta_y, theta_z)  
        
        #tire_obs add(global)
        tire_local = np.array([[],[],[]])
        if len(self.tire_obs_points[0,:])>0:
            if self.obs_info[self.waypoint_number][self.tire_info] == 1:
                tire_local = localization_xyz(self.tire_obs_points, position_x, position_y, theta_x, theta_y, theta_z)      
        
        #white_obs add(global)
        white_line_local = np.array([[],[],[]])
        if len(self.white_obs_points[0,:])>0: 
            if self.functions_test == 1:
                if self.waypoint_number >= self.white_number: # front stop >= 1, lanechange == 0
                    white_line_local = localization_xyz(self.white_obs_points, position_x, position_y, theta_x, theta_y, theta_z)
            elif self.obs_info[self.waypoint_number][self.whiteline_info] == 1:
                white_line_local = localization_xyz(self.white_obs_points, position_x, position_y, theta_x, theta_y, theta_z)
        
        #right_obs add(global)
        right_line_local = np.array([[],[],[]])
        if len(self.right_obs_points[0,:])>0:
            if self.functions_test == 1:
                if self.waypoint_number >= self.right_number: # front stop >= 1, lanechange == 0
                    right_line_local = localization_xyz(self.right_obs_points, position_x, position_y, theta_x, theta_y, theta_z)
            elif self.obs_info[self.waypoint_number][self.rightline_info] == 1:
                right_line_local = localization_xyz(self.right_obs_points, position_x, position_y, theta_x, theta_y, theta_z)
        
        #left_obs add(global)
        left_line_local = np.array([[],[],[]])
        if len(self.left_obs_points[0,:])>0:
            if self.functions_test == 1:
                if self.waypoint_number >= self.left_number: # front stop == 0, lanechange >= 0
                    left_line_local = localization_xyz(self.left_obs_points, position_x, position_y, theta_x, theta_y, theta_z)
            elif self.obs_info[self.waypoint_number][self.leftline_info] == 1:
                left_line_local = localization_xyz(self.left_obs_points, position_x, position_y, theta_x, theta_y, theta_z)
                
        #dot_obs add(global)
        dot_line_local = np.array([[],[],[]])
        if len(self.dot_obs_points[0,:])>0:
            if self.functions_test == 1:
                if self.waypoint_number >= self.dot_number: # front stop >= 0, lanechange == 0
                    dot_line_local = localization_xyz(self.dot_obs_points, position_x, position_y, theta_x, theta_y, theta_z)
            elif self.obs_info[self.waypoint_number][self.dotline_info] == 1:
                dot_line_local = localization_xyz(self.dot_obs_points, position_x, position_y, theta_x, theta_y, theta_z)
        
        self_radius = 0.8
        angle_min = -60 + 180
        angle_max = 60 + 180
        angles = np.linspace(np.radians(angle_min),np.radians(angle_max),100)
        x_radius = self_radius * np.cos(angles)
        y_radius = self_radius * np.sin(angles)
        z_radius = 0 * np.cos(angles)
        self_radius_points = np.vstack((x_radius, y_radius, z_radius))
        
        """
        #map_obs add
        if len(self.map_obs_points[0,:])>0:
            relative_point_x = self.map_obs_points[0,:] - self.position_x
            relative_point_y = self.map_obs_points[1,:] - self.position_y
            relative_point = np.array((relative_point_x, relative_point_y, self.map_obs_points[2,:]))
            relative_point_rot, t_point_rot_matrix = rotation_xyz(relative_point, self.theta_x, self.theta_y, -self.theta_z)
        else:
            relative_point_rot = np.array([[],[],[]])
        """    
        
        # make map_obs   x1  x2  y1  y2
        obs1 = make_obs(-45, 25, 97, 94) # siyakusyoura minami
        obs2 = make_obs(  5, 21, 100, 101) # siyakusyoura kita
        obs3 = make_obs(-73,-75, 105, 29) # siyakusyo nisi

        self.tsukuba_obs = np.hstack((obs1, obs2, obs3))
        
        #map_obs add
        if len(self.tsukuba_obs[0,:])>0:
            relative_point_x = self.tsukuba_obs[0,:] - self.position_x
            relative_point_y = self.tsukuba_obs[1,:] - self.position_y
            relative_point = np.array((relative_point_x, relative_point_y, self.tsukuba_obs[2,:]))
            relative_point_rot, t_point_rot_matrix = rotation_xyz(relative_point, self.theta_x, self.theta_y, -self.theta_z)
        else:
            relative_point_rot = np.array([[],[],[]])
        ###################################
                
        #obs round&duplicated  :grid_size before:28239 after100:24592 after50:8894 after10:3879
        obs_points = np.vstack((points[0,:], points[1,:], points[2,:]))
        
        
        ################# IGVC SelfDrive Quolification line stop test #20250530# #################
        if self.sd_line_stop_test == 1:
            obs_points = np.array([[],[],[]])
            #white_line_local =  np.array([[],[],[]])
        ##########################################################################################
        
        
        # それぞれが空でなければ追加
        if pothole_local.shape[1] > 0:
            obs_points = np.insert(obs_points, len(obs_points[0,:]), pothole_local.T, axis=1)
            
        if tire_local.shape[1] > 0:
            obs_points = np.insert(obs_points, len(obs_points[0,:]), tire_local.T, axis=1)
   
        if white_line_local.shape[1] > 0:
            obs_points = np.insert(obs_points, len(obs_points[0,:]), white_line_local.T, axis=1)
        
        if right_line_local.shape[1] > 0:
            obs_points = np.insert(obs_points, len(obs_points[0,:]), right_line_local.T, axis=1)
        
        if left_line_local.shape[1] > 0:
            obs_points = np.insert(obs_points, len(obs_points[0,:]), left_line_local.T, axis=1)
        
        if dot_line_local.shape[1] > 0:
            obs_points = np.insert(obs_points, len(obs_points[0,:]), dot_line_local.T, axis=1)
        
        obs_points = np.insert(obs_points, len(obs_points[0,:]), relative_point_rot.T, axis=1)
        obs_points = np.insert(obs_points, len(obs_points[0,:]), self_radius_points.T, axis=1)
        #obs_points = np.insert(obs_points, len(obs_points[0,:]), self.tsukuba_obs.T, axis=1)
        points_round = np.round(obs_points * self.obs_pixel) / self.obs_pixel
        obs_xy_local = points_round[:,~pd.DataFrame({"x":points_round[0,:], "y":points_round[1,:]}).duplicated()]
        obs_xy = np.vstack((obs_xy_local[0,:], obs_xy_local[1,:]))
        
        print(f"obs_points ={obs_points.shape}")
        reflect_set = obs_points[2,~pd.DataFrame({"x":points_round[0,:], "y":points_round[1,:]}).duplicated()]
        #obs global
        #obs_xy_rot, obs_rot_matrix = rotation_xyz(obs_xy_local, self.theta_x, self.theta_y, self.theta_z)
        #obs_x_grobal = obs_xy_rot[0,:] + self.position_x
        #obs_y_grobal = obs_xy_rot[1,:] + self.position_y
        #obs_global = np.vstack((obs_x_grobal, obs_y_grobal, obs_xy_local[2,:], reflect_set) , dtype=np.float32)
        obs_cloud = point_cloud_intensity_msg(np.vstack((obs_xy_local[0], obs_xy_local[1], obs_xy_local[2], reflect_set)).T, t_stamp, 'livox_frame')
        obs_global = do_transform_cloud(obs_cloud, transform)
        print(f"obs_xy ={obs_xy.shape}")
        #print(f"obs_global ={obs_global.shape}")
        #print(f"obs_global ={obs_global.dtype}")
        #set self position
        #self.est_xy = [self.position_x, self.position_y]
        print(f"self.est_xy ={self.est_xy}")
        
        astar_path = self.path_plan(obs_xy)
        print(f"astar_path ={astar_path.shape}")
        #astar_path = np.vstack((astar_path, np.zeros([1,len(astar_path[0,:])]) ))
        print(f"astar_path ={astar_path.shape}")
        astar_path_rot, astar_path_rot_matrix = rotation_xyz(astar_path, self.theta_x, self.theta_y, self.theta_z)
        astar_path_x_grobal = astar_path_rot[0,:] + self.position_x
        astar_path_y_grobal = astar_path_rot[1,:] + self.position_y
        astar_path_grobal = np.vstack((astar_path_x_grobal, astar_path_y_grobal))
        
        #publish for rviz2
        #global map rviz2
        potential_astar_path = path_msg(astar_path_grobal, t_stamp, 'odom')
        self.potential_astar_path_publisher.publish(potential_astar_path)    
        #global obs rviz2
        #obs_global_msg = point_cloud_intensity_msg(obs_global.T, t_stamp, 'odom')
        self.pcd_obs_global_publisher.publish(obs_global) 
		
    def path_plan(self, obs_xy_raw):
        #process: 検索マップ準備
        astar_x =  np.arange(-9.9, 9.9,  0.3) +self.est_xy[0]	#%%Astarのxを定義
        astar_y =  np.arange(9.9, -9.9, -0.3) +self.est_xy[1]	#%%Astarのyを定義
        astar_xy = np.ones([len(astar_x),len(astar_y)])*500     #%%マップを用意 500は暫定？コスト計算のReturnで使ってる？
        astar_xy_find = np.zeros([len(astar_y),len(astar_x)])	#%%一度通過した箇所を記憶
        astar_xn = round(len(astar_x)/2)		#%%探索を行うx座標
        astar_yn = round(len(astar_y)/2)		#%%探索を行うy座標
        astar_xy_find[astar_yn, astar_xn] = 1		#%%初期座標の通過設定
        astar_count = 0							#%%ループ回数チェック用
        astar_2nd = 0
        lg=self.lg
        cg=self.cg
        lo=self.lo
        co=self.co

        #relative waypoint
        relative_wp_x = self.waypoint[0] - self.position_x
        relative_wp_y = self.waypoint[1] - self.position_y
        relative_wp = np.array((relative_wp_x, relative_wp_y, self.waypoint[2]))
        wp_angle = np.arctan2(relative_wp_y, relative_wp_x)  * (180 / np.pi) 
        wp_theta =  -self.theta_z - wp_angle
        relative_wp_rot, t_point_rot_matrix = rotation_xyz(relative_wp, self.theta_x, self.theta_y, wp_theta)
        self.wp_xy = [relative_wp_rot[0], relative_wp_rot[1]]
        print(f"self.waypoint ={self.waypoint}")
        print(f"relative_wp_x ={relative_wp_x}, relative_wp_y ={relative_wp_y}")
        print(f"self.wp_xy ={self.wp_xy}")
        
        #print(f"len(obs_xy[0]) ={len(obs_xy[0])}")
        obs_xy_z =  np.zeros((1, len(obs_xy_raw[0])))
        obs_xy_raw_stack = np.vstack((obs_xy_raw[0], obs_xy_raw[1], obs_xy_z))
        obs_xy, t_point_rot_matrix = rotation_xyz(obs_xy_raw_stack, 0.0, 0.0,  -wp_angle)


        #■  process : A-star search
        astar_path_x=np.array([astar_xn])
        astar_path_y=np.array([astar_yn])
        #現在地より9m先の経路まで生成
        while (astar_xy[astar_xn, astar_yn]>0.1) and ((abs(astar_x[astar_xn]) - self.est_xy[0]) < 9) and ((abs(astar_y[astar_yn]) - self.est_xy[1]) < 9):
            astar_x_search = np.array([astar_x[astar_xn], astar_x[astar_xn-1], astar_x[astar_xn], astar_x[astar_xn+1], astar_x[astar_xn]]) # 探索するx座標[[0, 1, 0],[2, 3, 4],[0, 5, 0]]：0は見ない 1-5の順に十字検索
            astar_y_search = np.array([astar_y[astar_yn-1], astar_y[astar_yn], astar_y[astar_yn], astar_y[astar_yn], astar_y[astar_yn+1]]) # 探索するy座標[[0, 1, 0],[2, 3, 4],[0, 5, 0]]：0は見ない 1-5の順に十字検索
            astar_ug=cg*(1-np.exp(-( (astar_x_search-self.wp_xy[0])**2+(astar_y_search-self.wp_xy[1])**2 )/lg**2)) #引力計算
            #obs_short = ( np.abs(obs_xy[0]-astar_x[astar_xn]) + np.abs(obs_xy[1]-astar_y[astar_yn]) ) < 3 # 2　前は2だったけどとりあえずテスト中は5に
            obs_short = np.sqrt((obs_xy[0]-astar_x[astar_xn])**2 + (obs_xy[1]-astar_y[astar_yn])**2 ) < 2 # 2　前は2だったけどとりあえずテスト中は5に
            obs_short_x = obs_xy[0, np.array(obs_short) ]
            obs_short_y = obs_xy[1, np.array(obs_short) ]
            #astar_uo_x = astar_x_search -  obs_xy[0].reshape(len(obs_xy[0]),1) #x-xo 斥力計算　探索ポイントｘ近場にある障害物を全て行列使って計算
            #astar_uo_y = astar_y_search -  obs_xy[1].reshape(len(obs_xy[1]),1) #y-yo 斥力計算　探索ポイントｘ近場にある障害物を全て行列使って計算
            astar_uo_x = astar_x_search -  obs_short_x.reshape(len(obs_short_x),1) #x-xo 斥力計算　探索ポイントｘ近場にある障害物を全て行列使って計算
            astar_uo_y = astar_y_search -  obs_short_y.reshape(len(obs_short_y),1) #y-yo 斥力計算　探索ポイントｘ近場にある障害物を全て行列使って計算
            astar_uo_x2 = ( astar_uo_x * np.ones([len(obs_short_x),len(astar_x_search)]) ) ** 2 #(x-xo)^2
            astar_uo_y2 = ( astar_uo_y * np.ones([len(obs_short_y),len(astar_y_search)]) ) ** 2 #(y-yo)^2
            astar_uo = sum(co * np.exp(- (astar_uo_x2 + astar_uo_y2) / lo**2 ) ) # Uo計算 sum{co*e(-((x-xo)^2+(y-yo)^2)/lo^2)}
            astar_u=(1/cg*astar_uo+1)*astar_ug #UgとUoでポテンシャル計算
            astar_xy[[astar_xn,astar_xn-1,astar_xn,astar_xn+1,astar_xn],[astar_yn-1,astar_yn,astar_yn,astar_yn,astar_yn+1]] = astar_u #代入
            astar_xymin = astar_xy + (astar_xy_find*500) #一度通過した点を除外
            astar_xymin_ind = np.unravel_index(np.argmin(astar_xymin), astar_xymin.shape) #最もポテンシャルの低い場所のIndexを探す
            astar_xn = astar_xymin_ind[0] #次に探索を行うx座標を指定
            astar_yn = astar_xymin_ind[1] #次に探索を行うy座標を指定
            astar_xy_find[astar_xn, astar_yn] = 1 #探索座標の通過設定
            astar_path_x = np.append(astar_path_x, astar_xn) #x座標の通過記録
            astar_path_y = np.append(astar_path_y, astar_yn) #y座標の通過記録
            
            astar_count = astar_count + 1 #ループカウント
            if astar_xy[astar_yn, astar_xn] <0.2:
                self.get_logger().info(f"Goal: astar_path_x, astar_path_y ={astar_path_x, astar_path_y}")
                break
            if (astar_count > 100) and (astar_2nd==0):
                astar_x =  np.arange(-9.9, 9.9,  0.3) +self.est_xy[0]	#%%Astarのxを定義
                astar_y =  np.arange(9.9, -9.9, -0.3) +self.est_xy[1]	#%%Astarのyを定義
                astar_xy = np.ones([len(astar_x),len(astar_y)])*500     #%%マップを用意 500は暫定？コスト計算のReturnで使ってる？
                astar_xy_find = np.zeros([len(astar_y),len(astar_x)])	#%%一度通過した箇所を記憶
                astar_xn = round(len(astar_x)/2)		#%%探索を行うx座標
                astar_yn = round(len(astar_y)/2)		#%%探索を行うy座標
                astar_xy_find[astar_yn, astar_xn] = 1		#%%初期座標の通過設定
                astar_path_x=np.array([astar_xn])
                astar_path_y=np.array([astar_yn])
                lg=self.lg2nd
                cg=self.cg2nd
                lo=self.lo2nd
                co=self.co2nd
                astar_2nd = 1
            if astar_count >200:
                self.get_logger().info("Count Break")
                break
                
        #■  process : A-star Return
        astar_xy_rev = np.ones([len(astar_y),len(astar_x)])*500		#%%リターンのマップを用意
        astar_xy_find_rev = np.zeros([len(astar_y),len(astar_x)])		#%%リターンの一度通過した箇所を記憶
        astar_xy_find_rev[astar_xn, astar_yn]  = 1		#%%初期座標の通過設定
        astar_path_x_rev = np.array([astar_xn])
        astar_path_y_rev = np.array([astar_yn])
        while not( (astar_xn == round(len(astar_x)/2) ) and (astar_yn == round(len(astar_y)/2)) ):
            astar_x_search_rev = np.array([astar_x[astar_xn-1], astar_x[astar_xn], astar_x[astar_xn+1], astar_x[astar_xn-1], astar_x[astar_xn], astar_x[astar_xn+1], astar_x[astar_xn-1], astar_x[astar_xn], astar_x[astar_xn+1] ]) # 探索するx座標[[1, 2, 3],[4, 5, 6],[7, 8, 9]] 1-9の順に十字検索
            astar_y_search_rev = np.array([astar_y[astar_yn-1],astar_y[astar_yn-1], astar_y[astar_yn-1], astar_y[astar_yn], astar_y[astar_yn], astar_y[astar_yn], astar_y[astar_yn+1], astar_y[astar_yn+1], astar_y[astar_yn+1]]) # 探索するy座標[[1, 2, 3],[4, 5, 6],[7, 8, 9]]： 1-9の順に十字検索
            astar_xy_rev[ [astar_xn-1, astar_xn, astar_xn+1, astar_xn-1, astar_xn, astar_xn+1, astar_xn-1, astar_xn, astar_xn+1 ], [ astar_yn-1, astar_yn-1, astar_yn-1, astar_yn, astar_yn, astar_yn, astar_yn+1, astar_yn+1, astar_yn+1 ] ] = np.sqrt( ( (astar_x_search_rev - self.est_xy[0]) ** 2 ) + ( (astar_y_search_rev - self.est_xy[1]) ** 2 ) ) +500 - 500*astar_xy_find[ [astar_xn-1, astar_xn, astar_xn+1, astar_xn-1, astar_xn, astar_xn+1, astar_xn-1, astar_xn, astar_xn+1 ], [ astar_yn-1, astar_yn-1, astar_yn-1, astar_yn, astar_yn, astar_yn, astar_yn+1, astar_yn+1, astar_yn+1 ] ]#自己位置から探索点までの距離代入
            astar_xymin_rev = ( (astar_xy_rev + 500 * astar_xy_find_rev) )# -500 * astar_xy_find		#一度通過した箇所を除外
            astar_xymin_rev_ind = np.unravel_index(np.argmin(astar_xymin_rev), astar_xymin_rev.shape) #最も開始点に近い場所のIndexを探す
            astar_xn = astar_xymin_rev_ind[0] #次に探索を行うx座標を指定
            astar_yn = astar_xymin_rev_ind[1] #次に探索を行うy座標を指定
            astar_xy_find_rev[astar_xn, astar_yn]  = 1		#%%初期座標の通過設定
            astar_path_x_rev = np.append(astar_path_x_rev, astar_xn) #x座標の通過記録
            astar_path_y_rev = np.append(astar_path_y_rev, astar_yn) #y座標の通過記録
                

        #■  process : A-star path planning 
        astar_path_x_rev2 = astar_path_x_rev[::-1][1:len(astar_path_x_rev)-1] #indexが逆に入っているので元に戻す 自己位置とWaypoint位置は除外
        astar_path_y_rev2 = astar_path_y_rev[::-1][1:len(astar_path_y_rev)-1] #indexが逆に入っているので元に戻す 自己位置とWaypoint位置は除外
        astar_judge_x = astar_x[astar_path_x_rev2] -  obs_xy[0].reshape(len(obs_xy[0]),1) #x-xo 斥力計算　探索ポイントｘ近場にある障害物を全て行列使って計算
        astar_judge_y = astar_y[astar_path_y_rev2] -  obs_xy[1].reshape(len(obs_xy[1]),1) #y-yo 斥力計算　探索ポイントｘ近場にある障害物を全て行列使って計算
        astar_judge_x2 = ( astar_judge_x * np.ones([len(obs_xy[0]),len(astar_x[astar_path_x_rev2])]) ) ** 2 #(x-xo)^2
        astar_judge_y2 = ( astar_judge_y * np.ones([len(obs_xy[1]),len(astar_y[astar_path_y_rev2])]) ) ** 2 #(y-yo)^2
        self.get_logger().info(f"astar_judge_x2 ={len(astar_judge_x2)}")
        self.get_logger().info(f"astar_path_x_rev2 ={len(astar_path_x_rev2)}")
        if len(astar_judge_x2) > 0:
            astar_judge_obs_ind = np.minimum.reduce( np.sqrt(astar_judge_x2 + astar_judge_y2) ) <1.8#1.6
            self.get_logger().info(f"astar_judge_obs_ind ={astar_judge_obs_ind}")
            astar_path_point_x = np.append(np.append(self.est_xy[0],astar_x[astar_path_x_rev2[astar_judge_obs_ind]]),self.wp_xy[0])
            astar_path_point_y = np.append(np.append(self.est_xy[1],astar_y[astar_path_y_rev2[astar_judge_obs_ind]]),self.wp_xy[1])
        else:
            astar_path_point_x = np.append(np.append(self.est_xy[0],astar_x[astar_path_x_rev2[:]]),self.wp_xy[0])
            astar_path_point_y = np.append(np.append(self.est_xy[1],astar_y[astar_path_y_rev2[:]]),self.wp_xy[1])
            
        astar_dist = np.append(0, np.cumsum( np.sqrt(np.diff(astar_path_point_x)**2 + np.diff(astar_path_point_y)**2) ) ) #x/y軸を距離軸で2次元表現 前後のポイント間距離を計算

        astar_interp_x = interpolate.interp1d(astar_dist, astar_path_point_x, kind='linear')
        astar_interp_y = interpolate.interp1d(astar_dist, astar_path_point_y, kind='linear')
        if len(astar_dist) > 0 and not np.isnan(astar_dist[-1]):
            #astar_interp_list = np.linspace(0,astar_dist[len(astar_dist)-1],round(astar_dist[len(astar_dist)-1]/0.5) )
            astar_interp_list = np.linspace(0,astar_dist[len(astar_dist)-1],round(astar_dist[len(astar_dist)-1]/0.25) )
            astar_path_x = astar_interp_x(astar_interp_list)
            astar_path_y = astar_interp_y(astar_interp_list)
        else:
            astar_path_x = [self.est_xy[0]]
            astar_path_y = [self.est_xy[1]]
        
        astar_path_rot = np.append(np.append(astar_path_x,self.wp_xy[0]),np.append(astar_path_y,self.wp_xy[1])).reshape(2,len(astar_path_x)+1)
                
        
        astar_path_rot_z =  np.zeros((1, len(astar_path_rot[0])))
        astar_path_rot_stack = np.vstack((astar_path_rot[0], astar_path_rot[1], astar_path_rot_z))
        astar_path, t_point_rot_matrix = rotation_xyz(astar_path_rot_stack, 0.0, 0.0,  +wp_angle)
        
        return astar_path     
        
        
        

def localization_xyz(pointcloud, position_x, position_y, theta_x, theta_y, theta_z):
    relative_point_x = pointcloud[0,:] - position_x
    relative_point_y = pointcloud[1,:] - position_y
    relative_point = np.array((relative_point_x, relative_point_y, pointcloud[2,:]))
    relative_point_rot, _ = rotation_xyz(relative_point, theta_x, theta_y, -theta_z)
    return relative_point_rot



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
    

def point_cloud_intensity_msg(points, t_stamp, parent_frame):
    # In a PointCloud2 message, the point cloud is stored as an byte 
    # array. In order to unpack it, we also include some parameters 
    # which desribes the size of each individual point.
    ros_dtype = sensor_msgs.PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize # A 32-bit float takes 4 bytes.
    data = points.astype(dtype).tobytes() 

    # The fields specify what the bytes represents. The first 4 bytes 
    # represents the x-coordinate, the next 4 the y-coordinate, etc.
    fields = [
            sensor_msgs.PointField(name='x', offset=0, datatype=ros_dtype, count=1),
            sensor_msgs.PointField(name='y', offset=4, datatype=ros_dtype, count=1),
            sensor_msgs.PointField(name='z', offset=8, datatype=ros_dtype, count=1),
            sensor_msgs.PointField(name='intensity', offset=12, datatype=ros_dtype, count=1),
        ]

    # The PointCloud2 message also has a header which specifies which 
    # coordinate frame it is represented in. 
    header = std_msgs.Header(frame_id=parent_frame, stamp=t_stamp)
    

    return sensor_msgs.PointCloud2(
        header=header,
        height=1, 
        width=points.shape[0],
        is_dense=True,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * 4), # Every point consists of three float32s.
        row_step=(itemsize * 4 * points.shape[0]), 
        data=data
    )

def path_msg(waypoints, stamp, parent_frame):
    wp_msg = nav_msgs.Path()
    wp_msg.header.frame_id = parent_frame
    wp_msg.header.stamp = stamp
        
    # ウェイポイントを追加
    for i in range(waypoints.shape[1]):
        waypoint = geometry_msgs.PoseStamped()
        waypoint.header.frame_id = parent_frame
        waypoint.header.stamp = stamp
        waypoint.pose.position.x = waypoints[0, i]
        waypoint.pose.position.y = waypoints[1, i]
        waypoint.pose.position.z = 0.0
        waypoint.pose.orientation.w = 1.0
        wp_msg.poses.append(waypoint)
    return wp_msg

def make_obs(x1, x2, y1, y2, n=860, z=0):
    x = np.linspace(x1, x2, n)
    y = np.linspace(y1, y2, n)
    z = np.full(n, z)
    return np.array([x, y, z])

# mainという名前の関数です。C++のmain関数とは異なり、これは処理の開始地点ではありません。
def main(args=None):
    # rclpyの初期化処理です。ノードを立ち上げる前に実装する必要があります。
    rclpy.init(args=args)
    # クラスのインスタンスを作成
    potential_astar = PotentialAStar()
    # spin処理を実行、spinをしていないとROS 2のノードはデータを入出力することが出来ません。
    rclpy.spin(potential_astar)
    # 明示的にノードの終了処理を行います。
    potential_astar.destroy_node()
    # rclpyの終了処理、これがないと適切にノードが破棄されないため様々な不具合が起こります。
    rclpy.shutdown()

# 本スクリプト(publish.py)の処理の開始地点です。
if __name__ == '__main__':
    # 関数`main`を実行する。
    main()
