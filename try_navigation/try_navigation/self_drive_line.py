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
from collections import OrderedDict
from collections import deque

from scipy import interpolate
from std_msgs.msg import Float32MultiArray
import cv2
import subprocess
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


#map save
#ros2 run nav2_map_server map_saver_cli -t /reflect_map_global -f ~/ros2_ws/src/map/test_map --ros-args -p map_subscribe_transient_local:=true -r __ns:=/namespace
#ros2 run nav2_map_server map_saver_cli -t /reflect_map_global --occ 0.10 --free 0.05 -f ~/ros2_ws/src/map/test_map2 --ros-args -p map_subscribe_transient_local:=true -r __ns:=/namespace
#--occ:  occupied_thresh  この閾値よりも大きい占有確率を持つピクセルは、完全に占有されていると見なされます。
#--free: free_thresh	  占有確率がこの閾値未満のピクセルは、完全に占有されていないと見なされます。

# C++と同じく、Node型を継承します。
class ReflectionIntensityMap(Node):
    # コンストラクタです、クラスのインスタンスを作成する際に呼び出されます。
    def __init__(self):
        # 継承元のクラスを初期化します。（https://www.python-izm.com/advanced/class_extend/）今回の場合継承するクラスはNodeになります。
        super().__init__('self_drive_line_node')
        
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
        
        map_qos_profile_sub = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth = 1
        )
        # Subscriptionを作成。CustomMsg型,'/livox/lidar'という名前のtopicをsubscribe。
        self.subscription = self.create_subscription(sensor_msgs.PointCloud2, '/pcd_segment_ground', self.reflect_map, qos_profile)
        self.subscription = self.create_subscription(nav_msgs.Odometry,'/odom', self.get_odom, qos_profile_sub)
        self.subscription = self.create_subscription(nav_msgs.Odometry,'/fusion/odom', self.get_ekf_odom, qos_profile_sub)
        #self.subscription = self.create_subscription(nav_msgs.Odometry,'/odom_fast', self.get_odom, qos_profile_sub)
        self.subscription  # 警告を回避するために設置されているだけです。削除しても挙動はかわりません。
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        # Publisherを作成
        self.pcd_ground_global_publisher = self.create_publisher(sensor_msgs.PointCloud2, 'pcd_ground_global', qos_profile) 
        self.reflect_map_local_publisher = self.create_publisher(OccupancyGrid, 'reflect_map_local', map_qos_profile_sub)
        self.reflect_map_global_publisher = self.create_publisher(OccupancyGrid, 'reflect_map_global', map_qos_profile_sub)
        self.white_line = self.create_publisher(PointCloud2, 'white_lines', 10)
        #self.peak_line = self.create_publisher(PointCloud2, 'peak_lines', 10)
        self.peak_right_point = self.create_publisher(PointCloud2, 'right_points', 10)
        self.peak_left_point = self.create_publisher(PointCloud2, 'left_points', 10)
        self.peak_line = self.create_publisher(PointCloud2, 'peak_lines', 10)
       # self.curve_pub = self.create_publisher(PointCloud2, 'curve_lines', 10)
        self.curve_right_pub = self.create_publisher(PointCloud2, "right_curve", 10)
        self.curve_left_pub = self.create_publisher(PointCloud2, "left_curve", 10)
        self.pcd_right_line_buff_publisher = self.create_publisher(PointCloud2, 'line_buff_right', 10)
        self.pcd_left_line_buff_publisher = self.create_publisher(PointCloud2, 'line_buff_left', 10)
        self.white_buff_publisher = self.create_publisher(PointCloud2, 'white_buff', 10)
        self.curve_dotted_pub = self.create_publisher(PointCloud2, "dotted_line", 10)

        self.publisher_edge = self.create_publisher(Image, 'image_edge', 10)
        self.publisher_local = self.create_publisher(Image, 'image_local', 10)
        self.bridge = CvBridge()
        
########################################
        self.angle = 0.0
        self.flag =0
        self.right_angle = 0.0
        self.left_angle = 0.0
        self.right_first = True
        self.right_peak_x = None
        self.left_first = True
        self.left_peak_x = None
        self.right_flag = 0
        self.left_flag = 0
        self.offset = 0
        
        self.right_line_buff =  np.array([[],[],[],[]]);
        self.left_line_buff = np.array([[],[],[],[]]);
        #self.white_buff = np.array([[],[],[],[]]);
        self.white_buff = np.array([[],[],[],[],[]]);
        self.duration = 5.0  # time for buff white_buff
        self.right_x_candidates = [];         
        self.points_right_history = deque(maxlen=5) # this is how many times to stack points which is refered for kinji line arctan
        self.right_combined = [];
       

        self.image_saved = False  # 画像保存フラグ（初回のみ保存する）
        #image_angle
        self.angle_offset = 0
        #パラメータ
        #odom positon init
        self.position_x = 0.0 #[m]
        self.position_y = 0.0 #[m]
        self.position_z = 0.0 #[m]
        self.theta_x = 0.0 #[deg]
        self.theta_y = 0.0 #[deg]
        self.theta_z = 0.0 #[deg]
        #ekf_odom positon init
        self.ekf_position_x = 0.0 #[m]
        self.ekf_position_y = 0.0 #[m]
        self.ekf_position_z = 0.0 #[m]
        self.ekf_theta_x = 0.0 #[deg]
        self.ekf_theta_y = 0.0 #[deg]
        self.ekf_theta_z = 0.0 #[deg]
        
        #mid360 buff
        self.pcd_ground_buff = np.array([[],[],[],[]]);
        #self.solid_array_buff = np.array([[],[],[]]);
        
        #ground 
        self.ground_pixel = 1000/50#障害物のグリッドサイズ設定
        self.MAP_RANGE = 12.0 #[m]auto nav 7m  selfdrive:12
        
        self.MAP_RANGE_GL = 12 #[m] 
        self.MAP_LIM_X_MIN = -12.0 #[m]
        self.MAP_LIM_X_MAX =  12.0 #[m]
        self.MAP_LIM_Y_MIN = -12.0 #[m]
        self.MAP_LIM_Y_MAX =  12.0 #[m]
        
        #map position
        self.map_position_x_buff = 0.0 #[m]
        self.map_position_y_buff = 0.0 #[m]
        self.map_position_z_buff = 0.0 #[m]
        self.map_theta_z_buff = 0.0 #[deg]
        self.map_number = 0 # int
        
        self.map_data = 0
        self.map_data_flag = 0
        self.map_data_gl = 0
        self.map_data_gl_flag = 0
        self.MAKE_GL_MAP_FLAG = 1
        self.save_dir = os.path.expanduser('~/ros2_ws/src/kbkn_maps/maps/tsukuba/whiteline')
        yaml.add_representer(OrderedDict, ordered_dict_representer, Dumper=MyDumper)
        yaml.add_representer(list, list_representer, Dumper=MyDumper)
        
        #sliced_image
        self.num_bands=12 # the number of bands 
        self.band_height=20  #the height for each band
        self.kernel_open = (2, 2)
        self.kernel_close = (2, 2)
        self.map_place_x = -0 #auto nav -0 self drive  range 12  x 0
        self.map_place_y = 24.1 # autona14 self drive   range 12 y 24.1
        self.right_left_side = 140  #120 center 40 = 1(m)  3.5 m = 140
        self.right_right_side = 200     #200 = 5 m
        self.left_left_side = 20    # 0.5 m
        self.left_right_side = 80  # 2m
        
        self.reflect_threshold_param = 0.25 #line detect param

        
    def timer_callback(self):
        if self.map_data_flag > 0:
            self.reflect_map_local_publisher.publish(self.map_data)     
        #gl map
        if self.map_data_gl_flag > 0:
            self.reflect_map_global_publisher.publish(self.map_data_gl) 
        
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
        
    def get_ekf_odom(self, msg):
        self.ekf_position_x = msg.pose.pose.position.x
        self.ekf_position_y = msg.pose.pose.position.y
        self.ekf_position_z = msg.pose.pose.position.z
        
        flio_q_x = msg.pose.pose.orientation.x
        flio_q_y = msg.pose.pose.orientation.y
        flio_q_z = msg.pose.pose.orientation.z
        flio_q_w = msg.pose.pose.orientation.w
        
        roll, pitch, yaw = quaternion_to_euler(flio_q_x, flio_q_y, flio_q_z, flio_q_w)
        
        self.ekf_theta_x = 0 #roll /math.pi*180
        self.ekf_theta_y = 0 #pitch /math.pi*180
        self.ekf_theta_z = yaw /math.pi*180
        
	
    def pointcloud2_to_array(self, cloud_msg):
        # Extract point cloud data
        points = np.frombuffer(cloud_msg.data, dtype=np.uint8).reshape(-1, cloud_msg.point_step)
        x = np.frombuffer(points[:, 0:4].tobytes(), dtype=np.float32)
        y = np.frombuffer(points[:, 4:8].tobytes(), dtype=np.float32)
        z = np.frombuffer(points[:, 8:12].tobytes(), dtype=np.float32)
        intensity = np.frombuffer(points[:, 12:16].tobytes(), dtype=np.float32)

        # Combine into a 4xN matrix
        point_cloud_matrix = np.vstack((x, y, z, intensity))
        
        return point_cloud_matrix
        
    def reflect_map(self, msg):
        
        #print stamp message
        t_stamp = msg.header.stamp
        #print(f"t_stamp ={t_stamp}")
        
        #get pcd data
        points = self.pointcloud2_to_array(msg)
        #print(f"points ={points.shape}")
        
        #position set
        position_x=self.position_x; position_y=self.position_y; position_z=self.position_z;
        position = np.array([position_x, position_y, position_z])
        theta_x=self.theta_x; theta_y=self.theta_y; theta_z=self.theta_z;
        
        ekf_position_x=self.ekf_position_x; ekf_position_y=self.ekf_position_y; ekf_position_z=self.ekf_position_z;
        ekf_position = np.array([ekf_position_x, ekf_position_y, ekf_position_z])
        ekf_theta_x=self.ekf_theta_x; ekf_theta_y=self.ekf_theta_y; ekf_theta_z=self.ekf_theta_z;
        
        #ground global
        ground_rot, ground_rot_matrix = rotation_xyz(points[[0,1,2],:], theta_x, theta_y, theta_z)
        ground_x_grobal = ground_rot[0,:] + position_x
        ground_y_grobal = ground_rot[1,:] + position_y
        ground_global = np.vstack((ground_x_grobal, ground_y_grobal, ground_rot[2,:], points[3,:]) , dtype=np.float32)
        
        #map lim set
        map_lim_x_min = position_x + self.MAP_LIM_X_MIN;
        map_lim_x_max = position_x + self.MAP_LIM_X_MAX;
        map_lim_y_min = position_y + self.MAP_LIM_Y_MIN;
        map_lim_y_max = position_y + self.MAP_LIM_Y_MAX;
        map_lim_ind = self.pcd_serch(self.pcd_ground_buff, map_lim_x_min, map_lim_x_max, map_lim_y_min, map_lim_y_max)
        self.pcd_ground_buff = self.pcd_ground_buff[:,map_lim_ind]
        
        #obs round&duplicated  :grid_size before:28239 after100:24592 after50:8894 after10:3879
        pcd_ground_buff = np.insert(self.pcd_ground_buff, len(self.pcd_ground_buff[0,:]), ground_global.T, axis=1)
        points_round = np.round(pcd_ground_buff * self.ground_pixel) / self.ground_pixel
        self.pcd_ground_buff =points_round[:,~pd.DataFrame({"x":points_round[0,:], "y":points_round[1,:], "z":points_round[2,:]}).duplicated()]
        
        #local reflect map
        ground_reflect_conv = self.pcd_ground_buff[3,:]/255*100.0
        map_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        map_data_set = grid_map_set(self.pcd_ground_buff[1,:], self.pcd_ground_buff[0,:], ground_reflect_conv, position, self.ground_pixel, self.MAP_RANGE)
        
        ##ekf pos local reflect map
        ekf_ground_buff_x = self.pcd_ground_buff[0,:] - position[0]
        ekf_ground_buff_y = self.pcd_ground_buff[1,:] - position[1]
        ekf_ground_buff_z = self.pcd_ground_buff[2,:] - position[2]
        ekf_ground_buff = np.vstack((ekf_ground_buff_x, ekf_ground_buff_y, ekf_ground_buff_z))
        ekf_ground_rot, ekf_ground_rot_matrix = rotation_xyz(ekf_ground_buff, ekf_theta_x-theta_x, ekf_theta_y-theta_y, ekf_theta_z-theta_z)
        ekf_ground_set_x = ekf_ground_rot[0,:] + ekf_position[0]
        ekf_ground_set_y = ekf_ground_rot[1,:] + ekf_position[1]
        ekf_ground_set_z = ekf_ground_rot[2,:] + ekf_position[2]
        ekf_ground_set = np.vstack((ekf_ground_set_x, ekf_ground_set_y, ekf_ground_set_z))
        map_data_set_4save = grid_map_set(ekf_ground_set[1,:], ekf_ground_set[0,:], ground_reflect_conv, ekf_position, self.ground_pixel, self.MAP_RANGE)
        #print(f"map_data_set ={map_data_set.shape}")
	
        #GL reflect map
        #map_data_gl_set = grid_map_set(self.pcd_ground_buff[1,:], self.pcd_ground_buff[0,:], ground_reflect_conv, position, self.ground_pixel, self.MAP_RANGE_GL)
        map_data_gl_set = grid_map_set(ekf_ground_set[1,:], ekf_ground_set[0,:], ground_reflect_conv, ekf_position, self.ground_pixel, self.MAP_RANGE_GL)
        #print(f"map_data_set ={map_data_set.shape}")
	
        
        #publish for rviz2 
        #global ground
        ground_global_msg = point_cloud_intensity_msg(self.pcd_ground_buff.T, t_stamp, 'odom')
        self.pcd_ground_global_publisher.publish(ground_global_msg) 
        #local map
        self.map_data = make_map_msg(map_data_set, self.ground_pixel, position, map_orientation, t_stamp, self.MAP_RANGE, "odom")
        #self.map_data = make_map_msg(map_data_set_4save, self.ground_pixel, ekf_position, map_orientation, t_stamp, self.MAP_RANGE, "odom")
        self.map_data_flag = 1
        #self.reflect_map_local_publisher.publish(self.map_data)     
        #gl map
        self.map_data_gl = make_map_msg(map_data_gl_set, self.ground_pixel, ekf_position, map_orientation, t_stamp, self.MAP_RANGE_GL, "odom")
        #self.reflect_map_global_publisher.publish(self.map_data_gl) 
        self.map_data_gl_flag = 1
        
                 
        
        ##############################  takamori line filter  #################################
        #self.process(map_data_gl_set, position_x, position_y, theta_z, t_stamp)  #IGVC2025 0529 use
        self.process(map_data_gl_set, ekf_position_x, ekf_position_y, ekf_theta_z, t_stamp)  #IGVC2025 0530 change
        #self.process_pgm(map_data_set, position_x, position_y)
        
         
  
        if self.MAKE_GL_MAP_FLAG == 1:
            #self.make_ref_map(position_x, position_y, theta_z)
            #self.make_ref_map(ekf_position_x, ekf_position_y, ekf_theta_z)
            self.make_ref_map(map_data_gl_set, ekf_position_x, ekf_position_y, ekf_theta_z)
            
    def process(self, map_data_set, position_x, position_y, theta_z, t_stamp):
        try:
            self.get_logger().info(f"[1] start process")
            image, image_data = self.ref_to_image(map_data_set)
            if image is None:  
               self.get_logger().error(f"Failed to load 'map_data_set' ")
               return
            #self.log_image_size(image)
            #self.get_logger().info(f"[2] image loaded and converted ")
            '''
            h,w=image.shape[:2]
            ############ rotate image ##################
            
            self.update_angle()
            reflect_map_local = self.rotate_image(image, -self.angle +90)
            #reflect_map_local = self.rotate_image(image, -self.angle +90)
            
            reflect_map_local_cut = self.crop_center(reflect_map_local, w//2, h//2)
            reflect_map_local_set = reflect_map_local_cut.astype(np.uint8)
            #self.get_logger().info(f"[3] image rotated and cropped")
            
            local_image_msg = self.bridge.cv2_to_imgmsg(reflect_map_local_set, encoding='mono8')
            self.publisher_local.publish(local_image_msg)
            
            
            bands, bands_p, sliced_height, sliced_width = list(self.slice_image(reflect_map_local_set, self.band_height, self.num_bands)) 
           # self.get_logger().info(f"[4] band sliced and graph shown")  
            
            self.showgraph(bands)  
            peak_image, peak_r_image, peak_l_image = self.peaks_image(bands, bands_p,sliced_height, sliced_width)  
            #self.get_logger().info(f"[5] peaks detected")
            
            reverse_angle_rad = math.radians(self.angle - 90)
           # reverse_angle_rad = math.radians(self.angle_offset - 90)
            right_point, left_point = self.image_to_pcd_for_peak(peak_r_image, peak_l_image, position_x, position_y, reverse_angle_rad, step=1.0)
            #self.get_logger().info(f"[6] converted peak image to point cloud")
            
            
            #curve_points = self.generate_lines (peak_points, interval = 0.1, extend = 0.5, offset_distance=2.2, direction="right")## line theta koushin
             
            right_line,left_line, dotted_line = self.generate_right_left_curves(right_point, left_point, interval = 0.1, extend = 0.5, parallel_offset = 3)

            #self.get_logger().info(f"[7] right left line gernerated ")
            
            
            #left_line_buff publish
            self.left_line_buff = np.insert(self.left_line_buff, len(self.left_line_buff[0,:]), left_point, axis=1)
            left_line_buff_msg = point_cloud_intensity_msg(self.left_line_buff.T, t_stamp, 'odom')
            self.pcd_left_line_buff_publisher.publish(left_line_buff_msg) 
            #right_line_buff publish
            self.right_line_buff = np.insert(self.right_line_buff, len(self.right_line_buff[0,:]), right_point, axis=1)
            right_line_buff_msg = point_cloud_intensity_msg(self.right_line_buff.T, t_stamp, 'odom')
            self.pcd_right_line_buff_publisher.publish(right_line_buff_msg) 
            #self.get_logger().info(f"[8] right left buffers published")
            
            
           #right_line,left_line = self.generate_lines (right_point, left_point, interval = 0.1, extend = 0.5, offset_distance=2.2, direction="right")## line theta koushin
            print("right_flag", self.right_flag)
            print("角度（deg）self.right_angle:", self.right_angle)
            print("角度（deg）self.offset:", self.offset)
            print("角度（deg）self.angle:", self.angle)
            print("角度（deg）self.theta_z:", self.theta_z)
            #print("角度（deg）:", self.angle_offset)
            self.publish_right_left_lines(right_line, left_line, dotted_line)
            #self.get_logger().info(f"[9] right left line published")
            #self.publish_pcd(right_line)
            
            '''
            binary_image = self.binarize_image(image)
            kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, self.kernel_open)  
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, self.kernel_close) 
            # Open → Close
            opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_open)
            open_close = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
            #Close → Open
            closed_then = cv2.morphologyEx(open_close, cv2.MORPH_CLOSE, kernel_close)
            oc_image = cv2.morphologyEx(closed_then, cv2.MORPH_OPEN, kernel_open)    
             
            edge_image = self.detect_edges(oc_image)
            
            white_point = self.image_to_pcd(oc_image, position_x, position_y, step=1.0)
            
            #white_buff publish
            
            timestamp = time.time()
            timestamps = np.full((white_point.shape[0], 1), timestamp) #N 5
            
            points_with_time = np.hstack((white_point, timestamps))
            
            if self.white_buff.size:
                self.white_buff = np.hstack((self.white_buff, points_with_time.T))#shape 5 N
            else:
                self.white_buff = points_with_time.T
                
            current_time = time.time()
            mask = self.white_buff[4, :] >= current_time - self.duration
            self.white_buff = self.white_buff[:,mask]
            white_buff_msg = point_cloud_intensity_msg(self.white_buff[:4, :].T, t_stamp, 'odom')
            self.white_buff_publisher.publish(white_buff_msg) 
            
            
            
            #self.white_buff = np.insert(self.white_buff, len(self.white_buff[0,:]), white_point, axis=1)
            #white_buff_msg = point_cloud_intensity_msg(self.white_buff.T, t_stamp, 'odom')
           # self.white_buff_publisher.publish(white_buff_msg) 
            
        except Exception as e:
            self.get_logger().error(f"画像処理中にエラーが発生しました: {e}")   
            
    def update_angle(self, max_delta = 15.0, max_history = 5): # max_delta = 15.0
        
        if self.right_flag > 10 and abs(self.right_angle < 10):
            self.offset = self.offset + self.right_angle
            self.angle = self.offset
        else:
            self.angle = self.theta_z
            self.offset = self.theta_z
        
             
        #max_delta (float): 許容される最大変化量 [deg]
        #max_history (int): 平滑化に使う履歴数
        
        #if not hasattr(self, "angle_history"):
        #    self.angle_history = deque(maxlen=max_history)
            #self.angle = self.right_angle  # 初期化
         #   self.angle = self.left_angle  # 初期化
            

     #   if abs(self.left_angle - self.angle) > max_delta:
      ##      self.get_logger().warn(f"[angle] 角度の急激な変化を検出したため無視します: {self.right_angle:.2f}°")
       #     return

     #   self.angle_history.append(self.left_angle)
     #   self.angle = sum(self.angle_history) / len(self.angle_history)
        #self.angle_offset = self.angle_offset + self.angle
     #   self.angle_offset = self.angle
   
    def peaks_image(self, bands, bands_p, height, width):
        # 元画像サイズのマスク画像を作成（初期値0）
        point_mask = np.zeros((height, width), dtype=np.uint8)
        peaks_r_image = np.zeros((height, width), dtype=np.uint8)
        peaks_l_image = np.zeros((height, width), dtype=np.uint8)
        _w = width // 2
        _h = height // 2
        for i, band in enumerate(bands):
            # 反射強度を反転：強い反射ほど大きい値に
            inverted_band = 255 - band
            mean_values = np.mean(inverted_band, axis=0)  # axis=0: 各列の平均
            # 平滑化とピーク検出
            smoothed, peaks = self.smooth_and_find_peaks(mean_values)
            
            y = bands_p[i]  # このバンドのY座標（元画像での高さ位置）
            
            for x in peaks:  # for right line
                if 0 <= x < width and 0 <= y < height:
                    point_mask[y, x] = 255  # 対応する位置に 1 をセット and (height//2) <= y < height 
                    if self.right_flag == 0:
                       if self.right_left_side <= x <= self.right_right_side and 40 <= y < 140:
                            peaks_r_image[y, x] = 255
                            self.right_peak_x = x
                    else:
                        if (self.right_peak_x - 10) <= x <= (self.right_peak_x + 10) and 0 <= y < 160:
                            peaks_r_image[y, x] = 255
                            self.right_peak_x = x
                                
            for x in peaks:  # for left line
                if 0 <= x < width and 0 <= y < height:
                    if self.left_flag == 0:
                        if self.left_left_side <= x <= self.left_right_side and 0 <= y < 160:
                            peaks_l_image[y, x] = 255  # 対応する位置に 1 をセット
                            self.left_peak_x = x
                    else:
                        if (self.left_peak_x - 10) <= x <= (self.left_peak_x + 25) and 0 <= y < 160:
                            peaks_l_image[y, x] = 255
                            self.left_peak_x = x
                                
        return point_mask, peaks_r_image, peaks_l_image


    def generate_right_left_curves(self, points_right, points_left, interval, extend, parallel_offset):
        """
        右と左の点群からそれぞれ近似曲線を生成し、各曲線の方向角を self.right_angle, self.left_angle に保存する。

        Parameters:
            points_right (np.ndarray): shape (N, 4), 右側の点群 [x, y, z, intensity]
            points_left (np.ndarray): shape (N, 4), 左側の点群 [x, y, z, intensity]
            interval (float): 曲線上の点の間隔 [m]
            extend (float): x方向にどれだけ延長するか [m]

        Returns:
            np.ndarray: 右側の曲線点群 (M, 4)
            np.ndarray: 左側の曲線点群 (M, 4)
        """
        def generate_curve(points):
            N = points.shape[0]
            if N < 3:
                print("deg N<3 empty :", np.empty((0, 4), dtype=np.float32), 0.0)
                return np.empty((0, 4), dtype=np.float32), 0.0

            else:
                x_vals = points[:, 0]
                y_vals = points[:, 1]
                coeffs = np.polyfit(x_vals, y_vals, deg=2)
                poly = np.poly1d(coeffs)
                x_mid = 0.5 * (np.min(x_vals) + np.max(x_vals))
                dy_dx = 2 * coeffs[0] * x_mid + coeffs[1]
                pre_angle_rad = np.arctan(dy_dx)
                angle_rad = 0
                if np.degrees(pre_angle_rad) > -10:
                    if np.degrees(pre_angle_rad) < 10:
                        print("deg ok :pre_angle_rad", pre_angle_rad)
                        angle_rad = pre_angle_rad
                else:
                    angle_rad = 0
                    print("deg over :pre_angle_rad", pre_angle_rad)
                #self.right_angle = angle_rad

            x_min = np.min(x_vals)
            x_max = np.max(x_vals) + extend
            x_new = np.arange(x_min, x_max + interval, interval)
            y_new = poly(x_new)

            z_new = np.zeros_like(x_new, dtype=np.float32)
            intensity_new = np.ones_like(x_new, dtype=np.float32)
            curve_points = np.vstack((x_new, y_new, z_new, intensity_new)).T
            
            #print("curve_points.astype(np.float32) ", curve_points.astype(np.float32) )
            #print("np.degrees(angle_rad) ", np.degrees(angle_rad) )

            return curve_points.astype(np.float32), np.degrees(angle_rad)
            
        def generate_parallel_curve(curve_points, offset):
            if curve_points.shape[0] < 2:
                return np.empty((0, 4), dtype=np.float32)

            parallel_points = []
            for i in range(len(curve_points)):
                if i == 0:
                    dx = curve_points[i + 1, 0] - curve_points[i, 0]
                    dy = curve_points[i + 1, 1] - curve_points[i, 1]
                else:
                    dx = curve_points[i, 0] - curve_points[i - 1, 0]
                    dy = curve_points[i, 1] - curve_points[i - 1, 1]

                length = np.hypot(dx, dy)
                if length == 0:
                    offset_x, offset_y = 0.0, 0.0
                else:
                    # 法線方向（右手系）
                    offset_x = -dy / length * offset
                    offset_y = dx / length * offset

                x_new = curve_points[i, 0] + offset_x
                y_new = curve_points[i, 1] + offset_y
                parallel_points.append([x_new, y_new, 0.0, 0.5])  # z=0, intensity=0.5

            return np.array(parallel_points, dtype=np.float32)

        # 安全性を考慮してNoneチェックを追加
        #if points_right is None or not isinstance(points_right, np.ndarray):
        '''
        if points_right is None or points_right.shape[0]<3:
            #self.get_logger().warn("右側点群がNoneまたは無効です。")
            right_curve, right_angle_generate = np.empty((0, 4), dtype=np.float32), 0.0
            self.right_angle = right_angle_generate
            dotted_curve = np.empty((0, 4), dtype=np.float32)
            self.right_flag = 0
        else:
            self.points_right_history.append(points_right)
            right_combined = np.vstack(self.points_right_history)
            right_curve, right_angle_generate = generate_curve(right_combined)
            print("length of point right", len(points_right))
            print("length of combined", len(right_combined))
            #right_curve, right_angle_generate = generate_curve(points_right)
            self.right_angle = right_angle_generate
            
            dotted_curve = generate_parallel_curve(right_curve, parallel_offset)
            #print("right_points",right_points)
            #print("right_conbined",right_combined)
           
            self.right_flag += 1
            '''
        if points_right.shape[0]>2 or self.right_combined.shape[0]>5:
            self.points_right_history.append(points_right)
            self.right_combined = np.vstack(self.points_right_history)
            right_curve, right_angle_generate = generate_curve(self.right_combined)
            print("length of point right", len(points_right))
            print("length of combined", len(self.right_combined))
            #right_curve, right_angle_generate = generate_curve(points_right)
            self.right_angle = right_angle_generate
            
            dotted_curve = generate_parallel_curve(right_curve, parallel_offset)
            #print("right_points",right_points)
            #print("right_conbined",right_combined)
           
            self.right_flag += 1
        else :
            #self.get_logger().warn("右側点群がNoneまたは無効です。")
            right_curve, right_angle_generate = np.empty((0, 4), dtype=np.float32), 0.0
            self.right_angle = right_angle_generate
            dotted_curve = np.empty((0, 4), dtype=np.float32)
            self.right_flag = 0
        

        #if points_left is None or not isinstance(points_left, np.ndarray):
        if points_left is None or points_left.shape[0]<3:
            #self.get_logger().warn("左側点群がNoneまたは無効です。")
            left_curve, self.left_angle = np.empty((0, 4), dtype=np.float32), 0.0
            self.left_flag = 0
        else:
            left_curve, self.left_angle = generate_curve(points_left)
            self.left_flag = 1
        
       # if self.right_flag ==0 and self.left_flag == 0:
       #     self.both_flag ==1
            

        # 右側の線を生成して角度を保存
        # right_curve, right_angle = generate_curve(points_right)
        #self.right_angle = right_angle

        # 左側の線を生成して角度を保存
        #left_curve, left_angle = generate_curve(points_left)
        ##self.left_angle = left_angle

        return right_curve, left_curve, dotted_curve
        
    def rotate_image(self, image, angle): 
        # 画像の中心を計算 
        (h, w) = image.shape[:2] 
        center = (w // 2, h // 2) 
        # 回転行列を生成 
        M = cv2.getRotationMatrix2D(center, angle, 1.0) 
        # 画像を回転 
        rotated_image = cv2.warpAffine(image, M, (w, h)) 
        return rotated_image
        
    def crop_center(self, image, crop_width, crop_height): 
        # 画像の高さと幅を取得 
        height, width = image.shape 
        # 中央の座標を計算 
        center_x, center_y = width // 2, height // 2 
        # 切り抜きの左上と右下の座標を計算 
        x1 = center_x - (crop_width // 2) 
        y1 = center_y - (crop_height // 2) 
        x2 = center_x + (crop_width // 2) 
        y2 = center_y + (crop_height // 2) 
        # 画像を切り抜く 
        cropped_image = image[y1:y2, x1:x2] 
        return cropped_image
        
    def slice_image(self, image,band_height,num_bands):#num_bands = 6
        height, width = image.shape[:2]
        bands = []
        bands_p = []
        centers = np.linspace(band_height//2, height - band_height//2, num_bands, dtype=int)

        for center in centers:
            y_start = max(center - band_height // 2, 0)
            y_end = min(center + band_height // 2, height)
            band = image[y_start:y_end, :]
            bands.append(band)
            band_p = (y_start + y_end)//2
            bands_p.append(band_p)

        return tuple(bands),tuple(bands_p),height,width
        
    def showgraph(self, bands):
        plt.ion()
        plt.clf()
        offset_step = 200 # 200
        for i, band in enumerate(bands):
            # 反射強度を反転：強い反射ほど大きい値に
            inverted_band = 255 - band
            mean_values = np.mean(inverted_band, axis=0)  # axis=0: 各列の平均
            smoothed, peaks = self.smooth_and_find_peaks(mean_values)
            offset = 500-(i * offset_step)
            plt.plot(smoothed + offset,  'b-', linewidth=1.5) 
            plt.plot(mean_values + offset, label=f'Band {i+1}')
            plt.plot(peaks, smoothed[peaks] + offset, 'ro')

        plt.title("Column-wise Mean Reflection Intensity")
        plt.xlabel("Column Index")
        plt.ylabel("Mean Intensity")
        plt.legend()
        plt.pause(0.01)
        #plt.show()

    def smooth_and_find_peaks(self, data, sigma=1, height=50, distance=10, prominence=5):#prominence=10 distance 5
        smoothed = gaussian_filter1d(data, sigma=sigma)
        peaks, _ = find_peaks(smoothed, height=height, distance=distance, prominence=prominence)
        return smoothed, peaks
        
        #これはpeak_image用の角度に対応してるも　下のやつとほぼ一緒

    def image_to_pcd_for_peak(self, image_right, image_left, position_x, position_y, yaw_rad, step=1.0):
        try:
            def convert_peak_to_pointcloud(image, suffix):
                resolution = 1 / self.ground_pixel
                h, w = image.shape[:2]

                origin_x = position_x - (w // 2) * resolution
                origin_y = position_y + (h // 2) * resolution

                obstacle_indices = np.where(image > 128)
                if len(obstacle_indices[0]) == 0:
                    return np.empty((0, 4), dtype=np.float32)

                obs_x = obstacle_indices[1] * resolution + origin_x
                obs_y = -obstacle_indices[0] * resolution + origin_y
                obs_z = np.zeros_like(obs_x, dtype=np.float32)
                obs_intensity = image[obstacle_indices].astype(np.float32)

                obs_matrix = np.vstack((obs_x, obs_y, obs_z, obs_intensity))
                points = obs_matrix.T

                points[:, 0] -= position_x
                points[:, 1] -= position_y

                cos_yaw = np.cos(yaw_rad)
                sin_yaw = np.sin(yaw_rad)
                rot = np.array([
                    [cos_yaw, -sin_yaw, 0],
                    [sin_yaw,  cos_yaw, 0],
                    [0,        0,       1]
                ])
                points[:, :3] = points[:, :3] @ rot.T

                points[:, 0] += position_x
                points[:, 1] += position_y

                header = Header()
                header.stamp = self.get_clock().now().to_msg()
                header.frame_id = "odom"
                fields = [
                    PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
                    PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
                    PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
                    PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
                ]
                pc_msg = pc2.create_cloud(header, fields, points)

                if suffix == 'right':
                    self.peak_right_point.publish(pc_msg)
                elif suffix == 'left':
                    self.peak_left_point.publish(pc_msg)

                return points

            points_right = convert_peak_to_pointcloud(image_right, 'right')
            points_left = convert_peak_to_pointcloud(image_left, 'left')

            return points_right, points_left

        except Exception as e:
            self.get_logger().error(f"[image_to_pcd_for_peak] Error: {e}")

    def publish_right_left_lines(self, right_line, left_line, dotted_line,frame_id="odom"):
        """
        右線・左線の点群をそれぞれ PointCloud2 メッセージとしてパブリッシュする。

        Parameters:
            right_line (np.ndarray): shape (M, 4), 右側の点群 [x, y, z, intensity]
            left_line (np.ndarray): shape (M, 4), 左側の点群 [x, y, z, intensity]
            frame_id (str): PointCloud2 のフレームID（共通）
        """
        def publish_one_line(points, publisher):
            if points.shape[0] == 0:
                return
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = frame_id

            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            ]
            pc_msg = pc2.create_cloud(header, fields, points)
            publisher.publish(pc_msg)

        # 右側をパブリッシュ
        publish_one_line(right_line, self.curve_right_pub)
        # 左側をパブリッシュ
        publish_one_line(left_line, self.curve_left_pub)
        
        publish_one_line(dotted_line, self.curve_dotted_pub)
        
    def image_to_pcd(self, image, position_x, position_y, step=1.0):
        # ---- パラメータ定義 ----
        resolution = 1 / self.ground_pixel  # [m/pixel]
        origin_x = round(position_x - self.MAP_RANGE_GL - self.map_place_x, 1) 
        origin_y = round(position_y - self.MAP_RANGE_GL + self.map_place_y, 1)

        
        obstacle_indices = np.where(image < 128)# binary image <    #edge_image >128
        if len(obstacle_indices[0]) == 0:
            return  # 障害物がなければ処理しない

        # ---- ピクセル座標 → 実空間座標変換 ----
        obs_x = obstacle_indices[1] * resolution + origin_x  # 列方向 → X
        obs_y = -obstacle_indices[0] * resolution + origin_y # 行方向（反転）→ Y
        obs_z = np.zeros_like(obs_x, dtype=np.float32)       # Z軸は平面上なので0
        #obs_intensity = image_norm[obstacle_indices].astype(np.float32)  # 画素値を強度として使う
        obs_intensity = image[obstacle_indices].astype(np.float32)
        # ---- 点群作成 [X, Y, Z, Intensity] ----
        obs_matrix = np.vstack((obs_x, obs_y, obs_z, obs_intensity))
        points = obs_matrix.T  # shape: (N, 4)

        # ---- PointCloud2 メッセージ作成 ----
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "odom"
        
        fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        pc_msg = pc2.create_cloud(header,fields,points)
        self.white_line.publish(pc_msg)   
        return points
         
         
    def ref_to_image(self, map_data_set):
        occ_threshold_param = self.reflect_threshold_param  # 占有空間のしきい値
        free_threshold_param = 0.05 # 自由空間のしきい値

        # 0〜1スケールと仮定し100倍する
        occ_threshold = occ_threshold_param * 100  # = 
        free_threshold = free_threshold_param * 100 # = 13
        
        occupancy_grid_image = np.zeros_like(map_data_set, dtype=np.uint8)
        occupancy_grid_data = np.zeros_like(map_data_set, dtype=np.float32)

        # 処理1: 占有空間（黒: 0）
        occupancy_grid_image[map_data_set >= occ_threshold] = 0
        occupancy_grid_data[map_data_set >= occ_threshold] = 0.0

        # 処理2: 自由空間（白: 255）
        occupancy_grid_image[map_data_set <= free_threshold] = 255
        occupancy_grid_data[map_data_set <= free_threshold] = 1.0

        # 処理3: 未確定領域（グレー階調）
        uncertain_mask = (map_data_set > free_threshold) & (map_data_set < occ_threshold)
        scaled_values = 255 - (map_data_set[uncertain_mask] / occ_threshold * 100)
        occupancy_grid_image[uncertain_mask] = scaled_values.astype(np.uint8)

        # データ用（0〜1のスケーリングを保持）
        occupancy_grid_data[uncertain_mask] = map_data_set[uncertain_mask] / occ_threshold

        return occupancy_grid_image, occupancy_grid_data

           
    def binarize_image(self, image):
        _, binary_image = cv2.threshold(image, 90,255, cv2.THRESH_BINARY)# 90 255
        return binary_image

    def detect_edges(self, image):
        return cv2.Canny(image, 50, 150)#50 150   
    
    def make_ref_map(self, image, position_x, position_y, theta_z):
        map_pos_diff = math.sqrt((position_x - self.map_position_x_buff)**2 + (position_y - self.map_position_y_buff)**2)
        map_theta_diff = abs(theta_z -  self.map_theta_z_buff)
        if ( (map_pos_diff > 10) or ((map_pos_diff > 2) and (map_theta_diff > 40)) ):
            map_number_str = str(self.map_number).zfill(3)
            # 保存ディレクトリの絶対パスを取得
            #save_path = os.path.join(self.save_dir, f'waypoint_map_{map_number_str}')
            pgm_filename = os.path.join(self.save_dir, f'waypoint_map_{map_number_str}' + ".pgm")
            pgm_filename_meta = os.path.join(f'waypoint_map_{map_number_str}' + ".pgm")
            yaml_filename = os.path.join(self.save_dir, f'waypoint_map_{map_number_str}' + ".yaml")
            # ディレクトリが存在するか確認、存在しない場合は作成
            os.makedirs(self.save_dir, exist_ok=True)
            '''
            subprocess.run([
                'ros2', 'run', 'nav2_map_server', 'map_saver_cli',
                '-t', '/reflect_map_global',
                '--occ', '0.13',
                '--free', '0.05',
                '-f', save_path,
                '--ros-args', '-p', 'map_subscribe_transient_local:=true', '-r', '__ns:=/namespace'
            ])
            self.get_logger().info(f'External node executed with argument --arg1 {map_number_str}')
            '''
            # 閾値の設定 
            occ_threshold_param = 0.13 # 占有のしきい値  for save
            occ_threshold = occ_threshold_param * 100 # 占有のしきい値 
            free_threshold_param = 0.05 # 自由空間のしきい値  for save 
            free_threshold = free_threshold_param * 100 # 自由空間のしきい値 
            #image = self.map_data_gl
            #image = np.array(self.map_data_gl.data).reshape((self.map_data_gl.info.height, self.map_data_gl.info.width))
            #print(f"image ={image}")
            # マスクを初期化 
            occupancy_grid = np.zeros_like(image) 
            # 占有空間、自由空間、未確定領域を設定 
            occupancy_grid[image >= occ_threshold] = 255 - 255
            # 占有空間 
            occupancy_grid[image <= free_threshold] = 255 - 0
            # 自由空間 
            occupancy_grid[(image > free_threshold) & (image < occ_threshold)] = 255 - (image[(image > free_threshold) & (image < occ_threshold)])/occ_threshold*100 # 未確定領域は元の値を保持 
            # マップの保存 
            cv2.imwrite(pgm_filename, occupancy_grid)
            
            # メタデータを定義 
            metadata = OrderedDict([ 
                ('image', pgm_filename_meta), 
                ('mode', 'trinary'), 
                ('resolution', 1/self.ground_pixel), 
                ('origin', [round(position_x - self.MAP_RANGE_GL, 1), round(position_y - self.MAP_RANGE_GL, 1), round(0, 1)]), 
                ('negate', 0), ('occupied_thresh', occ_threshold_param), 
                ('free_thresh', free_threshold_param) 
            ])
            
            # YAMLファイルとしてメタデータを保存 
            with open(yaml_filename, 'w') as yaml_file: 
                yaml.dump(metadata, yaml_file, Dumper=MyDumper, default_flow_style=False)
            
            
            self.map_position_x_buff = position_x #[m]
            self.map_position_y_buff = position_y #[m]
            self.map_theta_z_buff = theta_z #[deg]
            self.map_number += 1
        
        
    def pcd_serch(self, pointcloud, x_min, x_max, y_min, y_max):
        pcd_ind = (( (x_min <= pointcloud[0,:]) * (pointcloud[0,:] <= x_max)) * ((y_min <= pointcloud[1,:]) * (pointcloud[1,:]) <= y_max ) )
        return pcd_ind
	

# カスタムDumperの設定を追加 
class MyDumper(yaml.Dumper): 
    def increase_indent(self, flow=False, indentless=False): 
        return super(MyDumper, self).increase_indent(flow=flow, indentless=indentless)
def ordered_dict_representer(dumper, data): 
    return dumper.represent_dict(data.items()) 
def list_representer(dumper, data): 
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True) 
    



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


def make_map_msg(map_data_set, resolution, position, orientation, header_stamp, map_range, frame_id):
    map_data = OccupancyGrid()
    map_data.header.stamp =  header_stamp
    map_data.info.map_load_time = header_stamp
    map_data.header.frame_id = frame_id
    map_data.info.width = map_data_set.shape[0]
    map_data.info.height = map_data_set.shape[1]
    map_data.info.resolution = 1/resolution #50/1000#resolution
    pos_round = np.round(position * resolution) / resolution
    map_data.info.origin.position.x = float(pos_round[0] -map_range) #位置オフセット
    map_data.info.origin.position.y = float(pos_round[1] -map_range)
    map_data.info.origin.position.z = float(0.0) #position[2]
    map_data.info.origin.orientation.w = float(orientation[0])#
    map_data.info.origin.orientation.x = float(orientation[1])
    map_data.info.origin.orientation.y = float(orientation[2])
    map_data.info.origin.orientation.z = float(orientation[3])
    map_data_cv = cv2.flip(map_data_set, 0, dst = None)
    map_data_int8array = [i for row in  map_data_cv.tolist() for i in row]
    map_data.data = Int8MultiArray(data=map_data_int8array).data
    return map_data

'''
フィールド名	内容
image	占有データを含む画像ファイルへのパス。 絶対パス、またはYAMLファイルの場所からの相対パスを設定可能。
resolution	地図の解像度（単位はm/pixel）。
origin	（x、y、yaw）のような地図の左下のピクセルからの2D姿勢で、yawは反時計回りに回転します（yaw = 0は回転しないことを意味します）。現在、システムの多くの部分ではyawを無視しています。
occupied_thresh	この閾値よりも大きい占有確率を持つピクセルは、完全に占有されていると見なされます。
free_thresh	占有確率がこの閾値未満のピクセルは、完全に占有されていないと見なされます。
negate	白/黒について、空き/占有の意味を逆にする必要があるかどうか（閾値の解釈は影響を受けません）
'''

def grid_map_set(map_x, map_y, data, position, map_pixel, map_range):
    map_min_x = (-map_range + position[1] ) * map_pixel
    map_max_x = ( map_range + position[1] ) * map_pixel
    map_min_y = (-map_range + position[0] ) * map_pixel
    map_max_y = ( map_range + position[0] ) * map_pixel
    map_ind_px = np.round(map_x * map_pixel )# index
    map_ind_py = np.round(map_y * map_pixel )
    map_px = np.round(map_x * map_pixel -position[1]*map_pixel )#障害物をグリッドサイズで間引き
    map_py = np.round(map_y * map_pixel -position[0]*map_pixel )
    map_ind = (map_min_x +map_pixel < map_ind_px) * (map_ind_px < map_max_x - (1)) * (map_min_y+map_pixel < map_ind_py) * (map_ind_py < map_max_y - (1))#
    
    #0/1 judge
    #map_xy =  np.zeros([int(map_max_x - map_min_x),int(map_max_y - map_min_y)], np.uint8)
    map_xy =  np.zeros([int(2* map_range * map_pixel),int(2* map_range * map_pixel)], np.uint8)
    map_data = map_xy #reflect to map#np.zeros([int(map_max_x - map_min_x),int(map_max_y - map_min_y),1], np.uint8)
    
    #print(f"map_xy ={map_xy.shape}")
    #print(f"data ={data.shape}")
    #print(f"data(map_ind) ={data[map_ind].shape}")
    
    map_data = map_data.reshape(1,len(map_xy[0,:])*len(map_xy[:,0]))
    map_data[:,:] = 0.0
    map_data_x = (map_px[map_ind] - map_range*map_pixel  ) * len(map_xy[0,:])
    map_data_y =  map_py[map_ind] - map_range*map_pixel
    map_data_xy =  list(map(int, map_data_x + map_data_y ) )
    #print(f"map_data ={map_data.shape}")
    #print(f"map_data_xy ={len(map_data_xy)}")
    #print(f"data[map_ind] ={len(data[map_ind])}")
    
    data_max = np.max(data[map_ind])
   # print(f"data_max ={data_max}")
    map_data_xy_max = np.max(map_data_xy)
    #print(f"map_data_xy_max ={map_data_xy_max}")
    
    
    map_data[0,map_data_xy] = data[map_ind]
    map_data_set = map_data.reshape(len(map_xy[:,0]),len(map_xy[0,:]))
    
    #print(f"map_data_set ={map_data_set.shape}")
    
    #map flipud
    #map_xy = np.flipud(map_xy)
    map_xy = np.flipud(map_data_set)
    
    map_xy_max_ind = np.unravel_index(np.argmax(map_xy), map_xy.shape)
    #print(f"map_xy_max_ind ={map_xy_max_ind}")
    #print(f"map_xy_max ={map_xy[map_xy_max_ind]}")
    
    return map_xy

# mainという名前の関数です。C++のmain関数とは異なり、これは処理の開始地点ではありません。
def main(args=None):
    # rclpyの初期化処理です。ノードを立ち上げる前に実装する必要があります。
    rclpy.init(args=args)
    # クラスのインスタンスを作成
    reflection_intensity_map = ReflectionIntensityMap()
    # spin処理を実行、spinをしていないとROS 2のノードはデータを入出力することが出来ません。
    rclpy.spin(reflection_intensity_map)
    # 明示的にノードの終了処理を行います。
    reflection_intensity_map.destroy_node()
    # rclpyの終了処理、これがないと適切にノードが破棄されないため様々な不具合が起こります。
    rclpy.shutdown()

# 本スクリプト(publish.py)の処理の開始地点です。
if __name__ == '__main__':
    # 関数`main`を実行する。
    main()
