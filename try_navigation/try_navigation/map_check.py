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
import glob
import cv2
from std_msgs.msg import Int8MultiArray
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge
import transforms3d
from geometry_msgs.msg import Quaternion

from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.time import Time

# C++と同じく、Node型を継承します。
class WaypointManagerMaprun(Node):
    # コンストラクタです、PcdRotationクラスのインスタンスを作成する際に呼び出されます。
    def __init__(self):
        # 継承元のクラスを初期化します。
        super().__init__('waypoint_manager_maprun_node')
        
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
        
        map_qos_profile_sub = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth = 10
        )
        # Subscriptionを作成。
        self.subscription = self.create_subscription(nav_msgs.Odometry,'/odom_wheel', self.get_odom, qos_profile_sub)
        #self.subscription = self.create_subscription(nav_msgs.Odometry,'/odom_fast', self.get_odom, qos_profile_sub)
        self.subscription = self.create_subscription(nav_msgs.Odometry,'/fusion/odom', self.get_ekf_odom, qos_profile_sub)
        self.subscription = self.create_subscription(OccupancyGrid,'/reflect_map_local', self.get_reflect_map_local, map_qos_profile_sub)
        self.subscription  # 警告を回避するために設置されているだけです。削除しても挙動はかわりません。
        
        # タイマーを0.1秒（100ミリ秒）ごとに呼び出す
        self.timer = self.create_timer(0.1, self.waypoint_manager)
        
        # Publisherを作成
        self.current_waypoint_publisher = self.create_publisher(geometry_msgs.PoseArray, 'current_waypoint', qos_profile) #set publish pcd topic name
        self.map_match_local_publisher = self.create_publisher(OccupancyGrid, 'reflect_map_match_local', map_qos_profile_sub)
        self.map_match_ref_publisher = self.create_publisher(OccupancyGrid, 'reflect_map_match_ref', map_qos_profile_sub)
        self.map_match_result_publisher = self.create_publisher(OccupancyGrid, 'reflect_map_match_result', map_qos_profile_sub)
        #self.map_match_result_publisher = self.create_publisher(sensor_msgs.Image, 'reflect_map_match_result', map_qos_profile_sub)
        #self.bridge = CvBridge()
        self.odom_ref_slam_publisher = self.create_publisher(nav_msgs.Odometry, 'odom_ref_slam', qos_profile)
        self.waypoint_path_publisher = self.create_publisher(nav_msgs.Path, 'waypoint_path', qos_profile) 
        self.map_obs_publisher = self.create_publisher(sensor_msgs.PointCloud2, 'map_obs', qos_profile) 
        
        self.fused_pub = self.create_publisher(nav_msgs.Odometry, '/odom_ekf_match', 10)
        self.fused_msg = nav_msgs.Odometry()

        self.timer = self.create_timer(0.1, self.publish_fused_value)
        
        
        #パラメータ
        #waypoint init
        self.current_waypoint = 0
        self.stop_flag = 0
        
        #positon init odom
        self.position_x = 0.0 #[m]
        self.position_y = 0.0 #[m]
        self.position_z = 0.0 #[m]
        self.theta_x = 0.0 #[deg]
        self.theta_y = 0.0 #[deg]
        self.theta_z = 0.0 #[deg]
        
        #positon init ekf
        self.ekf_position_x = 0.0 #[m]
        self.ekf_position_y = 0.0 #[m]
        self.ekf_position_z = 0.0 #[m]
        self.ekf_theta_x = 0.0 #[deg]
        self.ekf_theta_y = 0.0 #[deg]
        self.ekf_theta_z = 0.0 #[deg]
        
        
        
        self.odom_x_buff = 0.0
        self.odom_y_buff = 0.0
        self.ref_slam_x_buff = 0.0
        self.ref_slam_y_buff = 0.0
        self.ref_slam_diff = [0.0,0.0,0.0]
        
        #image angle
        self.angle_offset = 0
        
        ## ekf
        self.GTheta = None
        self.GTheta0 = None
        #self.GPSthetayaw0 = 0
        #self.DGPStheta = 0
        self.w = None
        self.Q = None  # Process noise covariance
        self.H = None
        self.R = None
        self.R1 = 0.05**2  # High frequency sensor noise covariance
        self.R2 = 0.05**2  # Low frequency sensor noise covariance
        #self.R3 = 0  # High frequency sensor(heading)
        #self.R4 = 0  # Low frequency sensor(heading)
        self.P = None  # Initial covariance
        self.XX = None
        self.prev_time = None
        self.prev_pos = None
        self.Speed = 0
        self.SmpTime = 0.1
        self.GpsXY = None
        #self.GPS_conut = 0
        #self.GOffset = 0
        #self.offsetyaw = 0
        #self.combineyaw = 0
        self.robot_yaw = 0
        self.combyaw = 0
        self.robot_orientationz = 0
        self.robot_orientationw = 0
        #self.Number_of_satellites = 0
        self.kalf_speed_param = 1.2
        
        
        
        # Waypoint YAMLファイルを読み込む
        #Global reflect map load: PGM、JPEG、YAMLファイルを行列形式で読み込み
        #map_base_name = "nakaniwa_test" # 対象画像の名前
        folder_path = os.path.expanduser('~/ros2_ws/src/map/nakaniwa_1129')
        map_base_name = "waypoint_map" # 対象画像の名前
        #folder_path = os.path.expanduser('~/ros2_ws/src/map/tukuba_kakunin')
        #folder_path = os.path.expanduser('~/ros2_ws/src/map/tukuba_all')
        # フォルダ内のpgmファイルを検索
        pgm_files = glob.glob(os.path.join(folder_path, '*.pgm'))
        pgm_file_count = len(pgm_files)
        map_file_path = os.path.join(folder_path, map_base_name) # パスの連結
        
        global_reflect_map_pgms = []
        map_resolution = []; map_origin=[]; map_occupied_thresh=[];map_free_thresh=[];
        reflect_map_obs_matrices = []
        for map_number in range(pgm_file_count):
            map_number_str = str(map_number).zfill(3)
            #load pgm
            pgm_filename = os.path.join(folder_path, f'{map_file_path}_{map_number_str}' + ".pgm")
            print(f"pgm_filename: {pgm_filename}")
            global_reflect_map_pgm = cv2.imread(pgm_filename, cv2.IMREAD_GRAYSCALE)
            global_reflect_map_pgms.append(global_reflect_map_pgm)
            #load yaml
            yaml_filename = os.path.join(folder_path, f'{map_file_path}_{map_number_str}' + ".yaml")
            with open(yaml_filename, 'r') as yaml_file:
                map_yaml_data = yaml.safe_load(yaml_file)
            map_resolution.append(map_yaml_data['resolution'])
            map_origin.append(map_yaml_data['origin'])
            map_occupied_thresh.append(map_yaml_data['occupied_thresh'])
            map_free_thresh.append(map_yaml_data['free_thresh'])
            #load jpeg
            jpeg_filename = os.path.join(folder_path, f'{map_file_path}_{map_number_str}' + ".jpeg")
            reflect_map_obs = cv2.imread(jpeg_filename)
            red_judge1 = reflect_map_obs[:,:,0] > 200
            red_judge2 = reflect_map_obs[:,:,2] < 100
            reflect_map_obs_data = red_judge1 * red_judge2 * 100
            reflect_map_obs_matrices.append(reflect_map_obs_data)
            resolution = map_yaml_data['resolution']
            map_len = reflect_map_obs_data.shape[0]/2
            print(f"resolution: {resolution}")
            print(f"map_len: {map_len}")
            map_obs_index = np.where(reflect_map_obs_data>0)
            print(f"map_obs_index: {map_obs_index}")
            if len(map_obs_index[0]) >0:
                map_obs_x = ( map_obs_index[1] - map_len) * resolution
                map_obs_y = (-map_obs_index[0] + map_len) * resolution
                print(f"map_obs_x map_obs_y: {map_obs_x, map_obs_y}")
            
        self.global_reflect_map_pgm = np.stack(global_reflect_map_pgms)
        self.global_reflect_map_resolution = np.stack(map_resolution)
        self.global_reflect_map_origin = np.stack(map_origin)
        self.global_reflect_map_occupied_thresh = np.stack(map_occupied_thresh)
        self.global_reflect_map_free_thresh = np.stack(map_free_thresh)
        self.reflect_map_obs_matrices = np.stack(reflect_map_obs_matrices)
        
        #print(f"self.global_reflect_map_pgm: {self.global_reflect_map_pgm.shape}")
        #print(f"self.global_reflect_map_pgm: {self.global_reflect_map_pgm.shape[0]}")
        #print(f"self.global_reflect_map_pgm[0]: {self.global_reflect_map_pgm[0]}")
        #print(f"self.global_reflect_map_resolution[0]: {self.global_reflect_map_resolution[0]}")
        #print(f"self.global_reflect_map_origin[0]: {self.global_reflect_map_origin[0]}")
        #print(f"self.reflect_map_obs_matrices[0]: {self.reflect_map_obs_matrices[0]}")
        
        wp_x = []; wp_y=[]; wp_z=[];
        for wp_number in range(self.global_reflect_map_pgm.shape[0]-1):
            x_offset = (len(self.global_reflect_map_pgm[wp_number][0]) * self.global_reflect_map_resolution[0])/2
            y_offset = (len(self.global_reflect_map_pgm[wp_number][1]) * self.global_reflect_map_resolution[0])/2
            x = self.global_reflect_map_origin[wp_number][0] + x_offset #pgm end + (pgm len * grid)/2
            y = self.global_reflect_map_origin[wp_number][1] + y_offset #pgm end + (pgm len * grid)/2
            z = self.global_reflect_map_origin[wp_number][2]
            wp_x.append(x)
            wp_y.append(y)
            wp_z.append(z)
        self.waypoints = np.array([wp_x,wp_y,wp_z])
        print(f"self.waypoints ={self.waypoints}")
        print(f"self.waypoints ={self.waypoints.shape}")
        #############################
        '''
        waypoint_map_yaml_path_name = "kbkn_maps/waypoints/hosei/m2/courtyard_Senior.yaml" # waypoint mapの名前
        py_path = "/home/ubuntu/ros2_ws/src/"#os.path.dirname(os.path.abspath(__file__)) # 実行ファイルのディレクトリ名
        waypoint_map_yaml_file_path = os.path.join(py_path, waypoint_map_yaml_path_name) # パスの連結
        print(f"waypoint_map_yaml_path_name ={waypoint_map_yaml_path_name}")
        print(f"py_path ={py_path}")
        
        with open(waypoint_map_yaml_file_path, 'r') as yaml_file:
            waypoint_map_yaml_data = yaml.safe_load(yaml_file)
        #print(f"waypoint_map_yaml_data ={waypoint_map_yaml_data}")
        waypoints = waypoint_map_yaml_data['waypoints']
        x = [point['point']['x'] for point in waypoints]
        y = [point['point']['y'] for point in waypoints]
        z = [point['point']['z'] for point in waypoints]
        self.waypoints = np.array([x, y, z])
        print(f"self.waypoints ={self.waypoints}")
        print(f"self.waypoints ={self.waypoints.shape}")
        '''
    def waypoint_manager(self):
        self.time_stamp = self.get_clock().now().to_msg()
        #self.get_logger().info('waypoint manager cntl')
        ###### odom position ######
        #position_x=self.ekf_position_x; position_y=self.ekf_position_y; 
        theta_x=self.theta_x; theta_y=self.theta_y; theta_z=self.theta_z-self.angle_offset;
        ###### ekf position ######
        position_x=self.ekf_position_x; position_y=self.ekf_position_y; 
        #theta_x=self.ekf_theta_x; theta_y=self.ekf_theta_y; theta_z=self.ekf_theta_z-self.angle_offset;
        ekf_theta_z=self.ekf_theta_z;
        ###### map ekf position ###
        position_x= self.fused_msg.pose.pose.position.x
        position_y= self.fused_msg.pose.pose.position.y
        theta_z = self.robot_yaw
        
        
        #waypoint theta & dist
        #set_waypoint = [self.waypoints[0,self.current_waypoint] - (position_x - odom_position_x), self.waypoints[1,self.current_waypoint] - (position_y - odom_position_y), self.waypoints[2,self.current_waypoint]]
        #set_waypoint = self.waypoints[:,self.current_waypoint] - self.ref_slam_diff
        set_waypoint = self.waypoints[:,self.current_waypoint]
        
        relative_point_x = set_waypoint[0] - position_x
        relative_point_y = set_waypoint[1] - position_y
        relative_point = np.vstack((relative_point_x, relative_point_y, self.waypoints[2,self.current_waypoint]))
        relative_point_rot, t_point_rot_matrix = rotation_xyz(relative_point, theta_x, theta_y, -theta_z)
        waypoint_rad = math.atan2(relative_point_rot[1], relative_point_rot[0])
        waypoint_dist = math.sqrt(relative_point_x**2 + relative_point_y**2)
        waypoint_theta = abs(waypoint_rad * (180 / math.pi))
        
        #set judge dist
        if abs(waypoint_theta) > 90:
            determine_dist = 6
        else:
            determine_dist = 3
        #check if the waypoint reached
        if waypoint_dist < determine_dist:
            self.current_waypoint += 1
            if self.current_waypoint > (len(self.waypoints[0,:]) - 1):
                self.stop_flag = 1
                self.get_logger().info('GOAL : stop_flag = %f' % (self.stop_flag))
        #self.get_logger().info('current_waypoint:x = %f, y = %f : waypoint_no = %f' % (self.waypoints[0,self.current_waypoint], self.waypoints[1,self.current_waypoint], self.current_waypoint))
        
        
        #publish
        pose_array = self.current_waypoint_msg(set_waypoint, 'odom')
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
        
        
        ########### ekf ###########
        # current_time = self.get_clock().now().to_msg()
        diff_time_stamp = Clock(clock_type=ClockType.ROS_TIME).now()
        current_time = diff_time_stamp.nanoseconds / 1000000000
        if self.prev_time is not None:
            self.SmpTime = current_time - self.prev_time
        else:
            self.SmpTime = 0.1
        self.prev_time = current_time
                
        current_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        ])
        if self.prev_pos is not None:
            distance = np.linalg.norm(current_pos - self.prev_pos)
            self.Speed = distance / self.SmpTime
        else:
            self.Speed = 0
        self.prev_pos = current_pos
        
        
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
        
    def get_reflect_map_local(self, msg):
        ############ data get ##################
        t_stamp = msg.header.stamp
        width = msg.info.width
        height = msg.info.height
        data = np.array(msg.data).reshape((height, width))
        # 範囲外の値を修正
        data[data == -1] = 255  # 未探索セルを白に設定
        data[data == 100] = 0   # 障害物セルを黒に設定
        data[data == 0] = 200   # 空きセルをグレーに設定
        # 占有確率が20以上のセルを100に置き換え 
        occupied_thresh = self.global_reflect_map_occupied_thresh[self.current_waypoint] * 100
        data[data >= occupied_thresh] = 100 # 占有確率occupied_thresh%以上を障害物として設定
        flipped_data = cv2.flip(data, 0, dst=None)
        # NumPy配列をJPEGフォーマットにエンコード
        _, buffer = cv2.imencode('.jpg', flipped_data)
        # バッファをデコードして画像データとして読み込む
        reflect_map_local_raw = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
        
        ############ position set ##################
        ##odom position
        position_x=self.position_x; position_y=self.position_y; position_z=self.position_z;
        position = np.array([position_x, position_y, position_z])
        theta_x=self.theta_x; theta_y=self.theta_y; theta_z=self.theta_z;
        ##ekf position
        #position_x=self.ekf_position_x; position_y=self.ekf_position_y; position_z=self.ekf_position_z;
        #position = np.array([position_x, position_y, position_z])
        #theta_x=self.ekf_theta_x; theta_y=self.ekf_theta_y; theta_z=self.ekf_theta_z;
        ekf_theta_z=self.ekf_theta_z;
        ###map ekf
        ekf_match_position_x= self.fused_msg.pose.pose.position.x
        ekf_match_position_y= self.fused_msg.pose.pose.position.y
        
        
        map_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        MAP_RANGE = 10.0
        ground_pixel = 1000/50
        
        ############ rotate image ##################
        reflect_map_local = self.rotate_image(reflect_map_local_raw, self.angle_offset)
        reflect_map_local_cut = crop_center(reflect_map_local, 400, 400)
        reflect_map_local_set = reflect_map_local_cut.astype(np.uint8)
        #otsu_thresh, reflect_map_local_image = cv2.threshold(reflect_map_local_set, 0, 100, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        ############ global map set ##################
        map_global_data_set = np.array(100-(self.global_reflect_map_pgm[self.current_waypoint])/255*100, dtype='i8')
        x_offset = (len(self.global_reflect_map_pgm[self.current_waypoint][0]) * self.global_reflect_map_resolution[0])/2
        y_offset = (len(self.global_reflect_map_pgm[self.current_waypoint][1]) * self.global_reflect_map_resolution[0])/2
        map_x = self.global_reflect_map_origin[self.current_waypoint][0] + x_offset #pgm end + (pgm len * grid)/2
        map_y = self.global_reflect_map_origin[self.current_waypoint][1] + y_offset #pgm end + (pgm len * grid)/2
        position_map = np.array([map_x, map_y, 0.0])
        MAP_RANGE_GL = x_offset
        map_ground_pixel = float(1.0/self.global_reflect_map_resolution[self.current_waypoint])
        self.map_data_global = make_map_msg(map_global_data_set, map_ground_pixel, position_map, map_orientation, t_stamp, MAP_RANGE_GL, "odom")
        self.map_match_ref_publisher.publish(self.map_data_global)
        
        ############ matching ##################
        ##### otu map set #####
        temp_global_map = map_global_data_set.astype(np.uint8)
        otsu_thresh, reflect_map_global_otsu = cv2.threshold(temp_global_map, 0, 100, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        reflect_map_local_otu = cv2.threshold(reflect_map_local_set, otsu_thresh, 100, cv2.THRESH_BINARY)[1]
        temp_result_image_set, ref_slam_xyz, match_percentage = self.map_template_match(reflect_map_global_otsu, reflect_map_local_otu, map_ground_pixel, MAP_RANGE_GL, MAP_RANGE, position_map, 30, 3.0)
        print(f"|| otu ||| match_percentage:{match_percentage} ")
        
        ##### Ratio-based Thresholding #####
        temp_global_map_ratio = cv2.normalize(map_global_data_set, None, 0, 255, cv2.NORM_MINMAX)
        print(f"np.min(temp_global_map_ratio): {np.min(temp_global_map_ratio)}")
        # 画像のヒストグラムを計算 
        hist = cv2.calcHist([temp_global_map], [0], None, [256], [0, 256]).flatten() 
        # 適切なしきい値を見つける 
        cumsum = np.cumsum(hist) 
        #print(f"cumsum: {cumsum}")
        print(f"len(cumsum): {len(cumsum)}")
        diff_values = np.hstack([cumsum[0],np.diff(cumsum)])
        #print(f"diff_values: {diff_values}")
        print(f"len(diff_values): {len(diff_values)}")
        diff_values_list= np.array(range(len(diff_values)))
        diff_values_ind = diff_values>0
        values_list = diff_values_list[diff_values_ind]
        print(f"values_list: {values_list}")
        print(f"diff_values(diff_values_ind): {diff_values[diff_values_ind]}")
        hist_threshold = sum(diff_values[diff_values_ind])*0.05
        hist_sum = values_list[(len(values_list)-1)]
        threshold_ind = 1
        threshold_value = values_list[(len(values_list)-1)]
        for threshold_ind in range(len(values_list)):
            #print(f"diff_values[values_list[(len(values_list)-1) - threshold_ind]]: {diff_values[values_list[(len(values_list)-1) - threshold_ind]]}")
            if (hist_sum + diff_values[values_list[(len(values_list)-1) - threshold_ind]]) < hist_threshold:
                hist_sum += diff_values[values_list[(len(values_list)-1) - threshold_ind]]
                threshold_value = values_list[(len(values_list)-1) - threshold_ind]
                #print(f"hist_sum: {hist_sum}")
                #print(f"threshold_value: {threshold_value}")
            else:
                break
        
        #if (len(values_list) >threshold_ind+1) and :
        #    threshold_value = values_list[(len(values_list)-1) -  threshold_ind]
        #else :
        #    threshold_value = diff_values_list[(len(diff_values_ind)-1)]
        print(f"threshold_value: {threshold_value}")
        _, reflect_map_global_ratio = cv2.threshold(temp_global_map, threshold_value, 100, cv2.THRESH_BINARY)
        reflect_map_local_ratio_set = reflect_map_local_set.copy()
        _, reflect_map_local_ratio = cv2.threshold(reflect_map_local_ratio_set, threshold_value, 100, cv2.THRESH_BINARY)
        
        if ref_slam_xyz[0] is None :
            print(f"!!xxx noto otu match xxx!!")
            temp_result_image_set, ref_slam_xyz, match_percentage = self.map_template_match(reflect_map_global_ratio, reflect_map_local_ratio, map_ground_pixel, MAP_RANGE_GL, MAP_RANGE, position_map, 27, 3.5)
            print(f"|| ratio ||| match_percentage:{match_percentage} ")
        
        if ref_slam_xyz[0] is None :
            #rotation_angle = np.array([3,-3,6.5,-6.5,10,-10,13,-13])# /180*math.pi
            #rotation_angle = np.array([4,-4,9,-9,13,-13])# /180*math.pi
            rotation_angle = np.array([4,-4,8,-8, 12,-12])# /180*math.pi
            for set_angle in rotation_angle:
                set_angle_offset = self.angle_offset + set_angle
                reflect_map_local = self.rotate_image(reflect_map_local_raw, set_angle_offset)
                reflect_map_local_cut = crop_center(reflect_map_local, 400, 400)
                reflect_map_local_set_angle = reflect_map_local_cut.astype(np.uint8)
                reflect_map_local_ratio = cv2.threshold(reflect_map_local_set_angle, threshold_value, 100, cv2.THRESH_BINARY)[1]
                temp_result_image_set, ref_slam_xyz, match_percentage = self.map_template_match(reflect_map_global_ratio, reflect_map_local_ratio, map_ground_pixel, MAP_RANGE_GL, MAP_RANGE, position_map, 27, 4.5)
                #print(f" +++++++ set_angle: {set_angle} ++++++++")
                if ref_slam_xyz[0] is not None :
                    reflect_map_local_set = reflect_map_local_set_angle
                    self.angle_offset = self.angle_offset + set_angle
                    print(f"!!!!!!!!!!local map rotation angle: {set_angle}!!!!!!!!!!!")
                    print(f"|| rotation ||| match_percentage:{match_percentage} ")
                    break
        
        
        ### ref_slam set ###
        if ref_slam_xyz[0] is not None :
            ref_slam_x = ref_slam_xyz[0]
            ref_slam_y = ref_slam_xyz[1]
        else:
            #not_match_x = position[0] - self.odom_x_buff
            #not_match_y = position[1] - self.odom_y_buff
            #not_match_z = 0
            #not_match_theta = self.theta_z-self.angle_offset - ekf_theta_z
            #not_match_diff = np.vstack((not_match_x, not_match_y, not_match_z))
            #not_match_diff_rot, t_point_rot_matrix = rotation_xyz(not_match_diff, 0, 0, not_match_theta)
            
            #ref_slam_x = float(self.ref_slam_x_buff + position[0] - self.odom_x_buff)
            #ref_slam_y = float(self.ref_slam_y_buff + position[1] - self.odom_y_buff)
            
            ##### dead rec
            dist_diff = math.sqrt((position[0] - self.odom_x_buff)**2 + (position[1] - self.odom_y_buff)**2)
            dist_diff_mat = np.vstack((dist_diff, 0.0, 0.0))
            dist_theta = theta_z + self.angle_offset
            dist_diff_rot, t_point_rot_matrix = rotation_xyz(dist_diff_mat, 0, 0, dist_theta) 
            ref_slam_x = float(self.ref_slam_x_buff + dist_diff_rot[0])
            ref_slam_y = float(self.ref_slam_y_buff + dist_diff_rot[1])
            
            print(f"XXXXX NOT Match location XXXXX")
            ref_slam_z = float(0.0)
            ref_slam_xyz = np.array([ref_slam_x, ref_slam_y, ref_slam_z])
                
        ###### buff set ########    
        self.odom_x_buff = position[0]
        self.odom_y_buff = position[1]
        self.ref_slam_x_buff = ref_slam_x
        self.ref_slam_y_buff = ref_slam_y
        #self.ref_slam_x_buff = position[0]
        #self.ref_slam_y_buff = position[1]
        #self.ref_slam_x_buff = self.ekf_position_x
        #self.ref_slam_y_buff = self.ekf_position_y
        
        
        ##### diff match rotation #########
        match_diff_x = position[0] - self.odom_x_buff
        match_diff_y = position[1] - self.odom_y_buff
        match_diff_z = 0
        match_diff = np.vstack((match_diff_x, match_diff_y, match_diff_z))
        match_diff_rot, t_point_rot_matrix = rotation_xyz(match_diff, 0, 0, self.angle_offset)
        #self.ref_slam_diff = [ref_slam_x - position[0], ref_slam_y - position[1], 0] ## odom map set
        self.ref_slam_diff = [ref_slam_x - ekf_match_position_x, ref_slam_y - ekf_match_position_y, 0] ## ekf_match set
        ##### diff local set #########
        #self.ref_slam_diff = [match_diff_x, match_diff_y, 0]
        
        
        ########## map obs set ############
        resolution = self.global_reflect_map_resolution[self.current_waypoint]
        map_obs = self.reflect_map_obs_matrices[self.current_waypoint]
        #rotate image
        map_obs = cv2.normalize(map_obs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        #map_obs = self.rotate_image(map_obs, -self.angle_offset)
        map_obs_index = np.where(map_obs>0)
        if len(map_obs_index[0]) >0:
            map_obs_x =  map_obs_index[1] * resolution - MAP_RANGE_GL + position_map[0] #- self.ref_slam_diff[0]
            map_obs_y = -map_obs_index[0] * resolution + MAP_RANGE_GL + position_map[1] #- self.ref_slam_diff[1]
            map_obs_z = np.zeros([1,len(map_obs_x)]) 
            map_obs_intensity = np.zeros([1,len(map_obs_x)]) 
            map_obs_matrix = np.vstack((map_obs_x, map_obs_y, map_obs_z, map_obs_intensity))
            map_obs_msg = point_cloud_intensity_msg(map_obs_matrix.T, t_stamp, 'odom')
            self.map_obs_publisher.publish(map_obs_msg) 
        
        ########## publish ################
        self.map_data_local = make_map_msg(reflect_map_local_set, ground_pixel, position, map_orientation, t_stamp, MAP_RANGE, "odom")
        self.map_match_local_publisher.publish(self.map_data_local)
        temp_result_image = make_map_msg(temp_result_image_set, map_ground_pixel, ref_slam_xyz, map_orientation, t_stamp, MAP_RANGE, "odom")
        self.map_match_result_publisher.publish(temp_result_image)   
        odom_ref_slam_msg = odometry_msg(ref_slam_x, ref_slam_y, position_z, theta_x, theta_y, theta_z+self.angle_offset, t_stamp, 'odom')
        self.odom_ref_slam_publisher.publish(odom_ref_slam_msg)
        waypoint_path = path_msg(self.waypoints, t_stamp, 'odom')
        self.waypoint_path_publisher.publish(waypoint_path) 
        
        ########## ekf ################
        self.GpsXY =  np.array([ref_slam_x, ref_slam_y])
        self.robot_yaw = (theta_z+self.angle_offset) /180*math.pi
        
            
        '''
        # グレースケール画像の型をCV_8Uに変換
        img1 = cv2.normalize(map_global_data_set, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img2 = cv2.normalize(reflect_map_local, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        angle, tx, ty, good_matches, keypoints1, keypoints2 = self.match_images(img1, img2)
        if angle is not None and tx is not None and ty is not None:
            print(f"Rotation angle: {angle:.2f} degrees")
            print(f"Translation: x = {tx:.2f}, y = {ty:.2f}")
            # マッチング結果の画像をパブリッシュ
            matches_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            ros_image = self.bridge.cv2_to_imgmsg(matches_img, encoding="bgr8")
            self.map_match_result_publisher.publish(ros_image)
            #self.map_match_result_publisher.publish(self.map_data_local)
        '''
        
    def map_template_match(self, global_map, local_map, pixel, global_map_range, local_map_range, position_map, match_threshold, dist_threshold):
        temp_result = cv2.matchTemplate(global_map.astype(np.uint8), local_map, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(temp_result)
        match_percentage = max_val * 100
        max_loc_x = max_loc[0]/pixel
        max_loc_y = max_loc[1]/pixel
        max_loc_position_x = position_map[0] - global_map_range + max_loc_x + local_map_range
        max_loc_position_y = position_map[1] + global_map_range - max_loc_y - local_map_range
        ref_slam_diff_x = max_loc_position_x - self.ref_slam_x_buff
        ref_slam_diff_y = max_loc_position_y - self.ref_slam_y_buff
        ref_slam_diff = math.sqrt(ref_slam_diff_x**2 + ref_slam_diff_y**2 )
        
        gps_slam_diff_x = max_loc_position_x - self.fused_msg.pose.pose.position.x
        gps_slam_diff_y = max_loc_position_y - self.fused_msg.pose.pose.position.y
        gps_slam_diff = math.sqrt(gps_slam_diff_x**2 + gps_slam_diff_y**2 )
        
        if (match_percentage > match_threshold) and ( (ref_slam_diff < dist_threshold) or (gps_slam_diff < dist_threshold)):
            ref_slam_x = float(max_loc_position_x)
            ref_slam_y = float(max_loc_position_y)
            ref_slam_z = float(0.0)
            ref_slam_xyz = np.array([ref_slam_x, ref_slam_y, ref_slam_z])
            temp_result_image_set = global_map[max_loc[1]:max_loc[1]+len(local_map[:,0]), max_loc[0]:max_loc[0]+len(local_map[0,:])]
            
            ###### angle match SAD ######
            angle_range = 5 # 探索する角度の範囲（度） 
            step = 0.1 # 探索する角度のステップ（度） # 最適な回転角度を見つける 
            img1 = cv2.normalize(temp_result_image_set, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            img2 = cv2.normalize(local_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            best_angle, best_sad = self.find_best_rotation_angle(img1, img2, angle_range, step)
            print(f"match_percentage: { match_percentage}, match angle: { best_angle:.2f}")
        else:
            ref_slam_x = None
            ref_slam_y = None
            ref_slam_z = None
            ref_slam_xyz = [ref_slam_x, ref_slam_y, ref_slam_z]
            temp_result_image_set = np.zeros([len(local_map[:,0]),len(local_map[0,:])], dtype='i8')
        return temp_result_image_set, ref_slam_xyz, match_percentage
        
    def match_images(self, img1, img2, min_matches=5, match_threshold=20):
        
        # ORB特徴点検出器の生成
        orb = cv2.ORB_create()

        # 特徴点とディスクリプタの検出
        keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

        # BFMatcherを使用して特徴点をマッチング
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)
        
         # マッチング結果が少ない場合は処理を中止
        if len(matches) < min_matches:
            self.get_logger().warn(f"Not enough matches ({len(matches)}/{min_matches}). Skipping.")
            return None, None, None, None, None, None

        # 有効なマッチングのみを抽出
        good_matches = [m for m in matches if m.distance < match_threshold]

        # 有効なマッチングが少ない場合は処理を中止
        if len(good_matches) < min_matches:
            self.get_logger().warn(f"Not enough good matches ({len(good_matches)}/{min_matches}). Skipping.")
            return None, None, None, None, None, None
        
        # マッチングされた特徴点を抽出
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        # ホモグラフィ行列の計算
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # 回転角度の計算
        angle = -np.degrees(np.arctan2(M[1, 0], M[0, 0]))

        # 平行移動量の計算
        tx = M[0, 2]
        ty = M[1, 2]
        
        return angle, tx, ty, good_matches, keypoints1, keypoints2
    def sad_score(self, img1, img2): 
        return np.sum(np.abs(img1.astype(np.int16) - img2.astype(np.int16)))
    
    def find_best_rotation_angle(self, img1, img2, angle_range, step): 
        best_angle = 0 
        min_sad = float('inf') 
        total_pixels = img1.shape[0] * img1.shape[1]
        angle_set = np.arange(-angle_range, angle_range+step, step)
        #print(f"angle_set: {angle_set}")
        for angle in angle_set: 
            M = cv2.getRotationMatrix2D((img1.shape[1] // 2, img1.shape[0] // 2), angle, 1.0) 
            rotated_img = cv2.warpAffine(img1, M, (img1.shape[1], img1.shape[0])) 
            sad = self.sad_score(rotated_img, img2) 
            match_rate = (total_pixels - sad) / total_pixels * 100 # 絶対差の逆数として計算
            #print(f"angle:match_rate: {angle, match_rate}")
            if sad < min_sad: 
                min_sad = sad 
                best_angle = angle 
        self.angle_offset = self.angle_offset - best_angle
        print(f"self.angle_offset: {self.angle_offset}")
        return best_angle, min_sad
    
    
    def calculate_match_rate(self, sad, total_pixels): 
        # SADスコアを基にマッチ率を計算 
        match_rate = 100 * (1 - sad / (total_pixels * 255)) 
        return match_rate 
    def find_best_match(self, img1, img2): 
        # 大きい方のサイズに合わせて画像をリサイズする 
        h1, w1 = img1.shape 
        h2, w2 = img2.shape 
        h = max(h1, h2) 
        w = max(w1, w2) 
        # テンプレートマッチングの初期化 
        best_sad = float('inf') 
        best_location = (0, 0) 
        for y in range(h2 - h1 + 1): 
            for x in range(w2 - w1 + 1): 
                # サブ画像を取り出してSADスコアを計算 
                sub_img = img2[y:y+h1, x:x+w1] 
                sad = self.sad_score(img1, sub_img) 
                if sad < best_sad: 
                    best_sad = sad 
                    best_location = (x, y) 
        # マッチ率を計算 
        match_rate = self.calculate_match_rate(best_sad, resized_img1.size) 
        return best_location, best_sad, match_rate
    
    
    def rotate_image(self, image, angle): 
        # 画像の中心を計算 
        (h, w) = image.shape[:2] 
        center = (w // 2, h // 2) 
        # 回転行列を生成 
        M = cv2.getRotationMatrix2D(center, angle, 1.0) 
        # 画像を回転 
        rotated_image = cv2.warpAffine(image, M, (w, h)) 
        return rotated_image
    
    
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
        
    ############ ekf #############
    def orientation_to_yaw(self, z, w):
        yaw = np.arctan2(2.0 * (w * z), 1.0 - 2.0 * (z ** 2))
        return yaw

    def yaw_to_orientation(self, yaw):
        orientation_z = np.sin(yaw / 2.0)
        orientation_w = np.cos(yaw / 2.0)
        return orientation_z, orientation_w
        
    def initializeGPS(self, GpsXY, GTheta, SmpTime):
        self.GTheta0 = GTheta
        self.XX = np.array(
            [GpsXY[0], GpsXY[1], np.cos(GTheta), np.sin(GTheta)])
        self.w = np.array([(1.379e-3)**2, (0.03 * np.pi / 180 * SmpTime)**2])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = np.array(
            [[(1.379e-3)**2, 0], [0, (0.03 * np.pi / 180 * SmpTime)**2]])
        G0 = np.array([[1, 0], [0, 0], [0, 0], [0, 1]])
        self.P = G0 @ self.Q @ G0.T
        
    def KalfGPSXY(self, Speed, SmpTime, GTheta, GpsXY, R1, R2):
        if self.H is None:
            self.initializeGPS(GpsXY, GTheta, SmpTime)

        self.R = np.array([[R1, 0], [0, R2]])

        DTheta = GTheta - self.GTheta0
        self.GTheta0 = GTheta

        # equation of state F G
        F = np.array([
            [1, 0, Speed * SmpTime *
                np.cos(DTheta), -Speed * SmpTime * np.sin(DTheta)],
            [0, 1, Speed * SmpTime *
                np.sin(DTheta), Speed * SmpTime * np.cos(DTheta)],
            [0, 0, np.cos(DTheta), -np.sin(DTheta)],
            [0, 0, np.sin(DTheta), np.cos(DTheta)]
        ])

        G = np.array([
            [np.cos(GTheta), -Speed * SmpTime * np.sin(GTheta)],
            [np.sin(GTheta), Speed * SmpTime * np.cos(GTheta)],
            [0, -np.sin(GTheta)],
            [0, np.cos(GTheta)]
        ])

        Y = np.array([GpsXY[0], GpsXY[1]])

        self.XX = F @ self.XX  # filter equation
        self.P = F @ self.P @ F.T + G @ self.Q @ G.T  # Prior Error Covariance
        # kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(self.H @
                                              self.P @ self.H.T + self.R)
        self.XX = self.XX + K @ (Y - self.H @ self.XX)  # estimated value
        self.P = self.P - K @ self.H @ self.P  # Posterior Error Covariance

        return self.XX[:2]
        
    def publish_fused_value(self):
        #if self.Speed is not None and self.SmpTime is not None and self.GTheta is not None:
        if self.Speed is not None and self.SmpTime is not None :
            if self.GpsXY is not None :
                kalf_speed = self.Speed * self.kalf_speed_param
                fused_value = self.KalfGPSXY(
                    kalf_speed, self.SmpTime, self.robot_yaw, self.GpsXY, self.R1, self.R2)    
                
                robot_orientation = self.yaw_to_orientation(self.robot_yaw)
                self.robot_orientationz = robot_orientation[0]
                self.robot_orientationw = robot_orientation[1]

                self.fused_msg.pose.pose.position.x = float(fused_value[0])
                self.fused_msg.pose.pose.position.y = float(fused_value[1])
                self.fused_msg.pose.pose.orientation.z = float(
                    self.robot_orientationz)
                self.fused_msg.pose.pose.orientation.w = float(
                    self.robot_orientationw)
                
                self.fused_msg.header.stamp = self.get_clock().now().to_msg()
                self.fused_msg.header.frame_id = "odom"
                self.fused_pub.publish(self.fused_msg)
                print(f"fused_value: {fused_value}")

def crop_center(image, crop_width, crop_height): 
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

def odometry_msg(pos_x, pos_y, pos_z, theta_x, theta_y, theta_z, stamp, frame_id):
    odom_msg = nav_msgs.Odometry()
    odom_msg.header.stamp = stamp
    odom_msg.header.frame_id = frame_id
    
    # 位置情報を設定
    odom_msg.pose.pose.position.x = pos_x 
    odom_msg.pose.pose.position.y = pos_y
    odom_msg.pose.pose.position.z = pos_z
    
    # YawをQuaternionに変換
    roll = theta_x /180*math.pi
    pitch = theta_y /180*math.pi
    yaw = theta_z /180*math.pi
    quat = transforms3d.euler.euler2quat(roll, pitch, yaw)
    odom_msg.pose.pose.orientation = Quaternion(x=quat[1], y=quat[2], z=quat[3], w=quat[0])
    
    return odom_msg


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
        
# mainという名前の関数です。C++のmain関数とは異なり、これは処理の開始地点ではありません。
def main(args=None):
    # rclpyの初期化処理です。ノードを立ち上げる前に実装する必要があります。
    rclpy.init(args=args)
    # クラスのインスタンスを作成
    waypoint_manager_maprun = WaypointManagerMaprun()
    # spin処理を実行、spinをしていないとROS 2のノードはデータを入出力することが出来ません。
    rclpy.spin(waypoint_manager_maprun)
    # 明示的にノードの終了処理を行います。
    waypoint_manager_maprun.destroy_node()
    # rclpyの終了処理、これがないと適切にノードが破棄されないため様々な不具合が起こります。
    rclpy.shutdown()

# 本スクリプト(publish.py)の処理の開始地点です。
if __name__ == '__main__':
    # 関数`main`を実行する。
    main()
