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
import time
import geometry_msgs.msg as geometry_msgs
from rclpy.action import ActionServer ####
from my_msgs.action import StopFlag ####
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32
from std_msgs.msg import String
import tkinter as tk
from threading import Thread
from my_msgs.msg import RoadsideInfo

navigation_status = "Initializing..."

# GUI
def start_gui():
    global navigation_status
    root = tk.Tk()
    root.title("Navigation Status")
    label = tk.Label(root, text=navigation_status, font=("Helvetica", 32))
    label.pack(padx=20, pady=20)

    def update_label():
        label.config(text=navigation_status)
        root.after(200, update_label)

    update_label()
    root.mainloop()
    
# C++と同じく、Node型を継承します。
class PathFollower(Node):
    # コンストラクタです、PcdRotationクラスのインスタンスを作成する際に呼び出されます。
    def __init__(self):
        # 継承元のクラスを初期化します。
        super().__init__('path_follower_node')
        
        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth = 1
        ) # BEST_EFFORT
        
        qos_profile_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth = 1
        )

        # set parameter (launch can change this parameter)
        self.declare_parameter('odom', '/fusion/odom')
        
        # define parameter
        odom_topic = self.get_parameter('odom').get_parameter_value().string_value

        # actionサーバーの生成(tuika)
        self.server = ActionServer(self,
            StopFlag, "stop_flag", self.listener_callback)
        
        # Subscriptionを作成。
        self.subscription = self.create_subscription(nav_msgs.Path, '/potential_astar_path', self.get_path, qos_profile) #set subscribe pcd topic name
        #self.subscription = self.create_subscription(nav_msgs.Odometry,'/odom/wheel_imu', self.get_odom, qos_profile_sub) # /odom/wheel_spimu
        self.subscription = self.create_subscription(nav_msgs.Odometry, odom_topic, self.get_odom, qos_profile_sub)
        #self.subscription = self.create_subscription(nav_msgs.Odometry,'/odom_ekf_match', self.get_odom, qos_profile_sub)
        #self.subscription = self.create_subscription(nav_msgs.Odometry,'/odom_ref_slam', self.get_odom_ref, qos_profile_sub)
        self.subscription = self.create_subscription(nav_msgs.Odometry, odom_topic, self.get_odom_ref, qos_profile_sub) #/fusion/odom
        self.subscription = self.create_subscription(sensor_msgs.PointCloud2, '/pcd_segment_obs', self.obs_steer, qos_profile)
        self.step_sub = self.create_subscription(sensor_msgs.PointCloud2, '/pcd_segment_low_step', self.low_obs_steer, qos_profile)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_pose_callback, qos_profile)
        self.stop_sub = self.create_subscription(String, '/stop_sign_status', self.stop_sign_callback, 10)
        self.human_sub = self.create_subscription(String, '/human_status', self.human_callback, 10)
        self.waypoint_number_sub = self.create_subscription(Int32,'/waypoint_number', self.get_waypoint_number, qos_profile_sub)
        self.roadside_sub = self.create_subscription(RoadsideInfo, "/roadside_info", self.roadside_callback, 10)
        self.subscription  # 警告を回避するために設置されているだけです。削除しても挙動はかわりません。
        
        # タイマーを0.05秒（50ミリ秒）ごとに呼び出す
        self.timer = self.create_timer(0.05, self.robot_ctrl)
        
        
        # Publisherを作成
        self.cmd_vel_publisher = self.create_publisher(geometry_msgs.Twist, 'cmd_vel', qos_profile) #set publish pcd topic name
        self.pcd_test_publisher = self.create_publisher(sensor_msgs.PointCloud2, 'pcd_test_global', qos_profile) 
        self.pcd_jam_publisher = self.create_publisher(sensor_msgs.PointCloud2, 'pcd_jam', qos_profile) 
        #self.marker_pub = self.create_publisher(MarkerArray, 'wall_follow_markers', 10)

        #パラメータ init
        self.path_plan = np.array([[0],[0],[0]])
        
        #positon init
        self.position_x = 0.0 #[m]
        self.position_y = 0.0 #[m]
        self.position_z = 0.0 #[m]
        self.theta_x = 0.0 #[deg]
        self.theta_y = 0.0 #[deg]
        self.theta_z = 0.0 #[deg]
        
        #positon init
        self.ref_position_x = 0.0 #[m]
        self.ref_position_y = 0.0 #[m]
        self.ref_position_z = 0.0 #[m]
        self.ref_theta_x = 0.0 #[deg]
        self.ref_theta_y = 0.0 #[deg]
        self.ref_theta_z = 0.0 #[deg]
        
        #path follow
        self.target_dist = 1.2
        self.target_dist_near = 0.4
        self.stop_flag = 1
        
        #pd init
        self.e_n = 0;
        self.e_n1 = 0;
        self.k_p = 0.6;
        self.k_d = 0.3;

        self.max_acc = 0.1
        self.max_dec = 0.3
        self.last_speed = 0.0
        
        self.stop_xy_test = [8, 10, -10, 10]
        self.stop_xy_test_flag = 1
        
        self.stop_xy = np.array([ #xmin,xmax,ymin,ymax, flag
            [ 64.2,  65.2,  19.0,  39.0, 1.0], #shiyakusyo
            [100.0, 101.0,  25.0,  45.0, 1.0], #dourotan1
            [177.7, 178.7,  25.0,  45.0, 1.0], #dourotan2
            [257.5, 277.5, -60.0, -59.0, 1.0], #singoumaeteisisen1
            [257.5, 277.5, -66.5, -65.5, 1.0], #singoumae1
            [256.5, 276.5, -86.2, -85.2, 1.0], #singoumaeteisisen2
            [270.3, 271.3, -99.0, -79.5, 1.0], #singoumae2
            [405.0, 425.0, -80.2, -79.2, 1.0], #ekimae oudanhodou1 y-1
            [545.0, 568.0, -85.0, -60.0, 0.0], #ekimae not stop
            [410.0, 417.0, -71.5, -70.5, 1.0], #ekimae oudanhodou2 y-2
            [290.6, 291.6, -98.0, -78.0, 1.0], #singoumaeteisisen3
            [284.3, 285.3, -98.0, -78.0, 1.0], #singoumae3
            [259.5, 279.5, -83.7, -82.7, 1.0], #singoumaeteisisen4 12
            [259.5, 279.5, -80.3, -79.3, 1.0], #singoumae4 13
            [185.0, 186.0,  25.0,  45.0, 1.0], #dourotan3
            [107.5, 108.5,  25.0,  45.0, 1.0], #dourotan4
            [ 64.0, 104.0, -30.0, -25.0, 1.0], #GOAL!!!!
            [  999,   999,   999,   999, 0.0] ]) #
        self.stop_num = 0;
        
        #obs
        self.obs_points = np.array([[],[],[],[]])
        self.low_step_obs_points = np.array([[],[],[],[]])
        self.rh_obs = 0
        self.lh_obs = 0
        self.ch_obs = 0
        self.t_stamp = 0

        # angle megin for gps heading 
        self.angle_diff = 30
        
        # jam process
        self.jam_timer = self.get_clock().now()
        self.none_jam_timer = self.get_clock().now()
        self.jam_active = False
        self.jam_last_print = -1
        self.c_jam_obs = 0
        self.waypoint_number = 0

        self.roadside_detected = False
        self.boundary_distance = 0.0
        self.boundary_angle = 0.0
    
    def roadside_callback(self, msg):
        self.roadside_detected = msg.detected
        self.boundary_distance = msg.boundary_distance
        self.boundary_angle = msg.boundary_angle

    # actionリクエストの受信時に呼ばれる(tuika)
    def listener_callback(self, goal_handle):
        global navigation_status
        self.get_logger().info(f"Received goal with a: {goal_handle.request.a}, b: {goal_handle.request.b}")
        
        # クライアントから送られたaをstop_flagに代入
        self.stop_flag = goal_handle.request.a
        print(f"stop_flag set to: {self.stop_flag}")
        navigation_status = "GO"
        
        # フィードバックの返信
        for i in range(1):
            feedback = StopFlag.Feedback()
            feedback.rate = i * 0.1
            goal_handle.publish_feedback(feedback)
            #time.sleep(0.5)

        # レスポンスの返信
        goal_handle.succeed()
        result = StopFlag.Result()
        result.sum = goal_handle.request.a + goal_handle.request.b  # 結果の計算
        return result         
        
    def get_path(self, msg):
        #self.get_logger().info('Received path with %d waypoints' % len(msg.poses))
        path_x=[];path_y=[];path_z=[];  
        for pose in msg.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            z = pose.pose.position.z
            path_x = np.append(path_x,x)
            path_y = np.append(path_y,y)
            path_z = np.append(path_z,z)
        
        self.path_plan = np.vstack((path_x, path_y, path_z))
        
    def goal_pose_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        
        xyz = np.vstack((x,y,z))
        
        self.get_logger().info(f"Received goal: x={x:.3f}, y={y:.3f}")    
    
    # stop_sign_status トピックのコールバック
    def stop_sign_callback(self, msg):
        # 現在の状態が "Stop" になったら stop_flag を True にする
        self.previous_status = msg.data   
    
    # human_status トピックのコールバック
    def human_callback(self, msg):
        # 現在の状態が "Stop" になったら stop_flag を True にする
        self.human_status = msg.data       
   
    def get_waypoint_number(self, msg):
        #get waypoint number
        self.waypoint_number = msg.data
        
    def robot_ctrl(self):
        #self.get_logger().info('0.05秒ごとに車両制御を実行')
        
        path = self.path_plan;
        position_x=self.position_x; position_y=self.position_y; 
        theta_x=self.theta_x; theta_y=self.theta_y; theta_z=self.theta_z;
        #set target_rad
        path_x_diff = path[0,:] - position_x
        path_y_diff = path[1,:] - position_y
        path_diff = np.sqrt(path_x_diff**2 + path_y_diff**2)
        path_diff_target_dist = np.abs(path_diff - self.target_dist)
        path_target_ind_sort = np.argsort(path_diff_target_dist)[:4] #check 4point
        target_ind = np.max(path_target_ind_sort)
        target_point = path[:,target_ind]
        reverse_theta_z = (theta_z + 180) % 360
        relative_point_x = target_point[0] - position_x
        relative_point_y = target_point[1] - position_y
        relative_point = np.vstack((relative_point_x, relative_point_y, target_point[2]))
        relative_point_rot, t_point_rot_matrix = rotation_xyz(relative_point, theta_x, theta_y, -theta_z)
        #relative_point_rot, t_point_rot_matrix = rotation_xyz(relative_point, theta_x, theta_y, -reverse_theta_z)
        target_rad = math.atan2(relative_point_rot[1], relative_point_rot[0])
        target_theta = (target_rad) * (180 / math.pi)
        
        ################### Straight Waypoint ############################
        if 191 <= self.waypoint_number <= 192:# or 0 <= self.waypoint_number <= 5:
            target_waypoint = path[:,-1]
            relative_point_x = target_waypoint[0] - position_x
            relative_point_y = target_waypoint[1] - position_y
            relative_point = np.vstack((relative_point_x, relative_point_y, target_waypoint[2]))
            relative_point_rot, t_point_rot_matrix = rotation_xyz(relative_point, theta_x, theta_y, -theta_z)
            #relative_point_rot, t_point_rot_matrix = rotation_xyz(relative_point, theta_x, theta_y, -reverse_theta_z)
            target_rad = math.atan2(relative_point_rot[1], relative_point_rot[0])
            target_theta = (target_rad) * (180 / math.pi)
        ##################################################################
        
        
        #set speed
        
        speed_set = 0.55#55 AutoNav 1.10
        speed = speed_set

        #############################################################
        # Roadside Tracking
        #############################################################
        safe_dist = 0.20      # この距離より近付いたら回避開始
        recover_dist = 0.55   # この距離まで戻したい
        approaching = self.boundary_angle < -10*np.pi/180
        if self.roadside_detected:
            if self.boundary_distance < safe_dist or approaching:
                # 境界まで近すぎる
                error = recover_dist - self.boundary_distance
                print("--- Roadside --- Befor target_theta[deg]:",target_theta)
                target_rad = 1.0 * error
                target_theta = (target_rad) * (180 / math.pi)
                print("--- Roadside --- After target_theta[deg]:",target_theta)
        
        #points = self.obs_points
        points = np.concatenate([self.obs_points, self.low_step_obs_points], axis=1)
        obs_theta = np.arctan2(points[1,:],points[0,:]) * 180/math.pi #arctan2(y,x)
        obs_dist = np.sqrt(points[0,:]**2 + points[1,:]**2)
        
        r_obs = (-120<obs_theta) * (obs_theta< -60) * (obs_dist<0.9)
        l_obs = (  60<obs_theta) * (obs_theta< 120) * (obs_dist<0.9)
        c_obs = ( -60<obs_theta) * (obs_theta<  60) * (obs_dist<1.2) #1.2
        c_obs_near = ( -50<obs_theta) * (obs_theta<  50) * (obs_dist<0.5)
        c_obs_back = ( -50<obs_theta) * (obs_theta<  50) * (obs_dist<0.4)

        
        if np.any(r_obs) and np.any(l_obs) and ~np.any(c_obs) :
            speed = 0.25
            #target_theta = 0
            #target_rad = target_theta/180*math.pi
            target_rad, target_theta = self.set_target_rad(path, position_x, position_y, self.target_dist_near, theta_x, theta_y, theta_z)
            #self.get_logger().info('||||||||||| center |||||||||||||||||')
        elif np.any(r_obs) and ~np.any(c_obs) :
            speed = 0.10 # autonav
            target_rad, target_theta = self.set_target_rad(path, position_x, position_y, self.target_dist_near, theta_x, theta_y, theta_z)
            #if target_theta < 0:
            #    target_theta = 0.0#lim_steer
            #    target_rad = target_theta/180*math.pi
            #self.get_logger().info('lllllllllll go left lllllllllllllllll')
        elif np.any(l_obs) and ~np.any(c_obs) :
            speed = 0.10 # autonav
            target_rad, target_theta = self.set_target_rad(path, position_x, position_y, self.target_dist_near, theta_x, theta_y, theta_z)
            #if 0 < target_theta:
            #    target_theta = 0.0#-lim_steer
            #    target_rad = target_theta/180*math.pi
            #self.get_logger().info('rrrrrrrrrr go right rrrrrrrrrrrrrrrrr')
        elif np.any(r_obs) and ~np.any(l_obs) and np.any(c_obs_near) :
            speed = 0.0
            target_rad, target_theta = self.set_target_rad(path, position_x, position_y, self.target_dist_near, theta_x, theta_y, theta_z)
            #if abs(target_theta) < 3: #250530 off
            #    speed = -0.10         #250530 off
            #    #target_theta = lim_steer
            #    #target_rad = lim_steer/180*math.pi
            #self.get_logger().info('ccccccccccccccccc c_obs_near lllllllllllllll')
        elif ~np.any(r_obs) and np.any(l_obs) and np.any(c_obs_near) :
            speed = 0.0
            target_rad, target_theta = self.set_target_rad(path, position_x, position_y, self.target_dist_near, theta_x, theta_y, theta_z)
            #if abs(target_theta) < 3: #250530 off
            #    speed = -0.10          #250530 off
                #target_theta = -lim_steer
                #target_rad = -lim_steer/180*math.pi
            #self.get_logger().info('ccccccccccccccccc c_obs_near rrrrrrrrrrrrrrr')
        elif np.any(c_obs_near) :
            speed = -0.10
            target_rad, target_theta = self.set_target_rad(path, position_x, position_y, self.target_dist_near, theta_x, theta_y, theta_z)
            #self.get_logger().info('ccccccccccccccccc c_obs_near ccccccccccccccc')
        elif np.any(c_obs) :
            speed = 0.20 # autonav 0.30
            target_rad, target_theta = self.set_target_rad(path, position_x, position_y, self.target_dist_near, theta_x, theta_y, theta_z)
            #self.get_logger().info('dddddddddd speed down ddddddddddd')
        
        '''
        if self.rh_obs and ~self.lh_obs:
            speed = 0.15
            target_rad = lim_steer/180*math.pi
        elif ~self.rh_obs and self.lh_obs:
            speed = 0.15
            target_rad = -lim_steer/180*math.pi
        '''
        
        rh_obs = self.rh_obs
        lh_obs = self.lh_obs
        ch_obs = self.ch_obs
        c_jam_obs = self.c_jam_obs
        #c_obs_near = ( -50<obs_theta) * (obs_theta<  50) * (obs_dist<0.5)
        #c_obs_back = ( -50<obs_theta) * (obs_theta<  50) * (obs_dist<0.4)
        cf = 1.15        

        #-----------------------jam process-----------------
        now = self.get_clock().now()
        if np.any(c_jam_obs):
            if (73 <= self.waypoint_number <= 74) or (self.waypoint_number == 45) or (136 <= self.waypoint_number <= 137) or (190 <= self.waypoint_number <= 192) or (176 <= self.waypoint_number <= 176):
                # jam条件 active でないこと none_jam_timer から 5s 経っていること）
                if not self.jam_active:
                    none_elapsed = (now - self.none_jam_timer).nanoseconds / 1e9
                    print(f"none jam timer :{none_elapsed}")
                    if none_elapsed > 4.0:
                        self.jam_active = True
                        self.jam_timer = now
                        self.jam_last_print = -1
                        print("Front obstacle detected. Timer started.")
                        print(f"none jam timer :{none_elapsed}")

                if self.jam_active:
                    elapsed = (now - self.jam_timer).nanoseconds / 1e9  # 秒（float）

                    #just showing passed time per sec
                    sec = int(elapsed)
                    if sec != self.jam_last_print:
                        self.jam_last_print = sec
                        print(f"time: {sec} sec")

                    # stop before 20 sec passed
                    if elapsed <= 50.0:
                        speed = 0.0
                        target_rad = 0.0
                        return
                    else: #20 sec passed 
                        self.jam_active = False
                        self.none_jam_timer = now
                        print("ch obs persistence exceeded 20 sec -> FORCING EXIT FROM JAM PROCESS!!")
        else:
            # ch_obs is none   reset 
            if self.jam_active:
                print("Front obstacle cleared. Timer reset.")
                self.none_jam_timer = now
            self.jam_active = False

        #---------------------------        
        
        if ~np.any(ch_obs) :
            if np.any(lh_obs) and np.any(rh_obs) :  #真ん中　
                target_theta = (target_rad) * (180 / math.pi)
                print("--- Center --- Befor target_theta[deg]:",target_theta)
                speed = 0.25
                lh_obs_close = min(lh_obs[1,:]) # y0 lh min
                rh_obs_close = max(rh_obs[1,:]) # y0 rh min
                cy = cf
                cx = (lh_obs_close + rh_obs_close) / 2
                #c_point = (cx, cy)
                target_rad = math.atan2(cx, cf)
                target_theta = (target_rad) * (180 / math.pi)
                print("--- Center --- After target_theta[deg]:",target_theta)
                            
            #or ((0 <= self.waypoint_number <= 3) and (0 - self.angle_diff <= theta_z <= 0 + self.angle_diff)) \
            #((25 <= self.waypoint_number <= 28) and (-60 - self.angle_diff <= theta_z <= -60 + self.angle_diff)) \
            elif ~np.any(lh_obs) and np.any(rh_obs):   #右寄り　
                speed = 0.25
                if (
                    (18 <= self.waypoint_number <= 20 and ((-150 - self.angle_diff <= theta_z <= -150) or (150  <= theta_z <= 150 + self.angle_diff)))
                    or (139 <= self.waypoint_number <= 143 and 0 - self.angle_diff <= theta_z <= 0 + self.angle_diff)
                    or (146 <= self.waypoint_number <= 151 and 0 - self.angle_diff <= theta_z <= 0 + self.angle_diff)
                    or (179 <= self.waypoint_number <= 188 and ((-150 - self.angle_diff <= theta_z <= -150) or (150  <= theta_z <= 150 + self.angle_diff)))
                    or (198 <= self.waypoint_number <= 208 and 90 - self.angle_diff <= theta_z <= 90 + self.angle_diff)
                    or (33 <= self.waypoint_number <= 35 and -20 - self.angle_diff <= theta_z <= -20 + self.angle_diff)
                ): #1 3 5 6 7 8
                    target_theta = (target_rad) * (180 / math.pi)
                    print("!!!RH!!!! Befor target_theta[deg]:",target_theta)
                    rh_obs_close = max(rh_obs[1,:]) # y0 rh min
                    rh_dist = 0.65 # 0.65
                    cy = cf
                    cx = rh_obs_close + rh_dist
                    target_rad = math.atan2(cx, cf)
                    target_theta = (target_rad) * (180 / math.pi)
                    print("!!!RH!!!! After target_theta[deg]:",target_theta)
                    
            elif np.any(lh_obs) and ~np.any(rh_obs):  #左寄り 11 <= self.waypoint_number <= 12 \ 
                speed = 0.25
                if (
                    (65 <= self.waypoint_number <= 72 and -90 - self.angle_diff <= theta_z <= -90 + self.angle_diff)
                    or (162 <= self.waypoint_number <= 168 and ((-150 - self.angle_diff <= theta_z <= -150) or (150  <= theta_z <= 150 + self.angle_diff)))
                    or (172 <= self.waypoint_number <= 175 and ((-150 - self.angle_diff <= theta_z <= -150) or (150  <= theta_z <= 150 + self.angle_diff)))
                    ):    #2 4
                    target_theta = (target_rad) * (180 / math.pi)
                    print("!!!LH!!!! Befor target_theta[deg]:",target_theta)
                    lh_obs_close = min(lh_obs[1,:]) # y0 lh min
                    lh_dist = -0.65 # 0.65
                    cy = cf
                    cx = lh_obs_close + lh_dist
                    target_rad = math.atan2(cx, cf)
                    target_theta = (target_rad) * (180 / math.pi)
                    print("!!!LH!!!! After target_theta[deg]:",target_theta)
            
            #elif ~np.any(rh_obs) and ~np.any(lh_obs):
            #    speed = 0.25
                           
        lim_steer = 20
        #lim_steer = 30 24/11/29 ok
        #if abs(target_theta) < 10 and 0.0 < speed and speed < 0.5:
        if abs(target_theta) < 10:
            if 0.0 < speed:
                if speed < 0.5:
                    speed = 0.5 # autonav 0.8
        #elif target_theta  < -lim_steer:
        if target_theta  < -lim_steer:
            speed = 0.10
            target_rad = -lim_steer/180*math.pi
            #print("over write1")
        elif lim_steer < target_theta:
            speed = 0.10
            target_rad = lim_steer/180*math.pi
            #print("over write1")
        if abs(target_theta)  > 90:
            speed = -0.10
        if np.any(c_obs_back) :
            speed = 0.10     
        
        #elif abs(target_theta)  > 90:
        #    speed = 0.2
        #else:
        #    speed = 0.2
        #self.get_logger().info('speed = %f' % (speed))
        target_theta = target_theta +90     ######/180*math.pi
        target_rad_pd = self.sensim0(target_rad)
        #target_rad_pd = target_rad
        last_speed = self.last_speed
        acc = self.max_acc * 0.05
        dec = self.max_dec * 0.05

        if speed > last_speed:
            last_speed = min(last_speed + acc, speed)
        else:
            last_speed = max(last_speed - dec, speed)
        
        self.last_speed = last_speed

        #make msg
        twist_msg = geometry_msgs.Twist()
        #check stop flag
        if self.stop_flag == 0:
            #twist_msg.linear.x = speed #0.3  # 前進速度 (m/s)
            twist_msg.linear.x = last_speed #0.3  # 前進速度 (m/s)
            twist_msg.angular.z = target_rad_pd  # 角速度 (rad/s)
            #twist_msg.linear.x = -speed #0.3  # 前進速度 (m/s)
            #twist_msg.angular.z = -target_rad_pd # 角速度 (rad/s) back left to left
        else:
            twist_msg.linear.x = 0.0  # 前進速度 (m/s)
            twist_msg.angular.z = 0.0  # 角速度 (rad/s)
        
        self.cmd_vel_publisher.publish(twist_msg)
        #self.get_logger().info('Publishing cmd_vel: linear.x = %f, angular.z = %f : %f deg' % (twist_msg.linear.x, twist_msg.angular.z, math.degrees(twist_msg.angular.z)))
        
    def set_target_rad(self, path, position_x, position_y, target_dist, theta_x, theta_y, theta_z):
        path_x_diff = path[0,:] - position_x
        path_y_diff = path[1,:] - position_y
        path_diff = np.sqrt(path_x_diff**2 + path_y_diff**2)
        path_diff_target_dist = np.abs(path_diff - target_dist)
        path_target_ind_sort = np.argsort(path_diff_target_dist)[:4] #check 4point
        target_ind = np.max(path_target_ind_sort)
        target_point = path[:,target_ind]
        relative_point_x = target_point[0] - position_x
        relative_point_y = target_point[1] - position_y
        relative_point = np.vstack((relative_point_x, relative_point_y, target_point[2]))
        relative_point_rot, t_point_rot_matrix = rotation_xyz(relative_point, theta_x, theta_y, -theta_z)
        target_rad = math.atan2(relative_point_rot[1], relative_point_rot[0])
        target_theta = (target_rad) * (180 / math.pi)
        return target_rad, target_theta
        
        
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
        
    def sensim0(self, steering):
        self.e_n = steering
        steering = (self.k_p * self.e_n + self.k_d*(self.e_n - self.e_n1))
        self.e_n1 = self.e_n
        return steering
    
    def get_odom_ref(self, msg):
        global navigation_status
        self.ref_position_x = msg.pose.pose.position.x
        self.ref_position_y = msg.pose.pose.position.y
        self.ref_position_z = msg.pose.pose.position.z
        
        flio_q_x = msg.pose.pose.orientation.x
        flio_q_y = msg.pose.pose.orientation.y
        flio_q_z = msg.pose.pose.orientation.z
        flio_q_w = msg.pose.pose.orientation.w
        
        roll, pitch, yaw = quaternion_to_euler(flio_q_x, flio_q_y, flio_q_z, flio_q_w)
        
        self.ref_theta_x = 0 #roll /math.pi*180
        self.ref_theta_y = 0 #pitch /math.pi*180
        self.ref_theta_z = yaw /math.pi*180
        
        if self.waypoint_number >= 225: # after dourotan4
            if self.stop_num <= 15:
                self.stop_num = 16
        elif self.waypoint_number >= 198: # after singou
            if self.stop_num <= 13:
                self.stop_num = 14
        elif self.waypoint_number >= 178: # after ekimae oudanhodou2
            if self.stop_num <= 9:
                self.stop_num = 10
        elif self.waypoint_number >= 138: # after ekimae oudanhodou1
            if self.stop_num <= 7:
                self.stop_num = 8
        elif self.waypoint_number >= 81: # after singou
            if self.stop_num <= 6:
                self.stop_num = 7
        elif self.waypoint_number >= 57: # after dourotan2
            if self.stop_num <= 2:
                self.stop_num = 3
        
        if ((self.stop_xy[self.stop_num,0] < self.ref_position_x) and (self.ref_position_x < self.stop_xy[self.stop_num,1]) and (self.stop_xy[self.stop_num,2] < self.ref_position_y) and (self.ref_position_y < self.stop_xy[self.stop_num,3]) ) or ((self.stop_xy[self.stop_num,0] < self.position_x) and (self.position_x < self.stop_xy[self.stop_num,1]) and (self.stop_xy[self.stop_num,2] < self.position_y) and (self.position_y < self.stop_xy[self.stop_num,3]) ):
            if self.stop_xy[self.stop_num,4] > 0:
                self.get_logger().info('####### stop flag on %f #######' % (self.stop_num))
                self.stop_flag = 1;
                navigation_status = "STOP"
                #print(self.stop_num)
            else:
                self.get_logger().info('####### through flag on %f #######' % (self.stop_num))
            self.stop_num = self.stop_num + 1;
        
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
        
    def obs_steer(self, msg):
        
        #print stamp message
        t_stamp = msg.header.stamp
        #print(f"t_stamp ={t_stamp}")
        self.t_stamp = t_stamp
        
        #get pcd data
        points = self.pointcloud2_to_array(msg)
        #print(f"points ={points.shape}")
        
        #map_obs
        self.obs_points = points
        
        rh_obs = self.pcd_serch(points, 1.0,1.3,-1.0,0)#1.0,1.3,-0.7,0
        if len(rh_obs[0,:]) > 10:
            self.rh_obs = 1
        else:
            self.rh_obs = 0
        lh_obs = self.pcd_serch(points, 1.0,1.3,0,1.0)#1.0,1.3,0,0.7
        if len(lh_obs[0,:]) > 10:
            self.lh_obs = 1
        else:
            self.lh_obs = 0
        
        ch_obs = self.pcd_serch(points, 1.0,1.3,-0.4,0.4)

        c_jam_obs = self.pcd_serch(points, 0.0,2.0,-0.6,0.6,0.6,1.0) #1.0,1.3,-0.4,0.4 #1.0,2.0,-0.8,0.8
            
        self.rh_obs = rh_obs
        self.lh_obs = lh_obs
        self.ch_obs = ch_obs
        self.c_jam_obs = c_jam_obs
        
        #global test obs rviz2 kesu
        obs_jam_msg = point_cloud_intensity_msg(c_jam_obs.T, t_stamp, 'odom')
        obs_test_msg = point_cloud_intensity_msg(ch_obs.T, t_stamp, 'odom')
        self.pcd_test_publisher.publish(obs_test_msg) 
        self.pcd_test_publisher.publish(obs_jam_msg)
    
    def low_obs_steer(self, msg):
        
        #print stamp message
        #t_stamp = msg.header.stamp
        #print(f"t_stamp ={t_stamp}")
        #self.t_stamp = t_stamp
        
        #get pcd data
        points = self.pointcloud2_to_array(msg)
        #print(f"points ={points.shape}")
        
        #map_obs
        if (
            (self.waypoint_number == 45)
            or (self.waypoint_number == 137)
            or (self.waypoint_number == 177)
            or (76 <= self.waypoint_number <= 78)
            or (148 <= self.waypoint_number <= 153)
            or (162 <= self.waypoint_number <= 167)
            or (195 <= self.waypoint_number <= 198)
            or (227 <= self.waypoint_number <= 232)
        ): # nakaniwa 14~20
            self.low_step_obs_points = points
            #self.get_logger().info(f"#######Add Low Step#######: {self.low_step_obs_points}")
        else:
            self.low_step_obs_points = np.array([[],[],[],[]])
            #self.get_logger().info(f"#######No Low Step#######: {self.low_step_obs_points}")

    # pcd_serch を z も考慮する形で置換（点群抽出用ユーティリティ）
    def pcd_serch(self, pointcloud, x_min, x_max, y_min, y_max, z_min=None, z_max=None):
        """
        pointcloud: 4 x N array (x,y,z,intensity)
        returns: filtered 4 x M array
        """
        mask = ((pointcloud[0, :] >= x_min) & (pointcloud[0, :] <= x_max) &
                (pointcloud[1, :] >= y_min) & (pointcloud[1, :] <= y_max))
        if z_min is not None and z_max is not None:
            mask = mask & ((pointcloud[2, :] >= z_min) & (pointcloud[2, :] <= z_max))
        return pointcloud[:, mask]

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
        
# mainという名前の関数です。C++のmain関数とは異なり、これは処理の開始地点ではありません。
def main(args=None):
    # rclpyの初期化処理です。ノードを立ち上げる前に実装する必要があります。
    rclpy.init(args=args)
    # クラスのインスタンスを作成
    path_follower = PathFollower()
    # GUI
    gui_thread = Thread(target=start_gui, daemon=True)
    gui_thread.start()
    # spin処理を実行、spinをしていないとROS 2のノードはデータを入出力することが出来ません。
    rclpy.spin(path_follower)
    # 明示的にノードの終了処理を行います。
    path_follower.destroy_node()
    # rclpyの終了処理、これがないと適切にノードが破棄されないため様々な不具合が起こります。
    rclpy.shutdown()

# 本スクリプト(publish.py)の処理の開始地点です。
if __name__ == '__main__':
    # 関数`main`を実行する。
    main()
