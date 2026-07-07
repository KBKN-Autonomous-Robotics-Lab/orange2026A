# rclpy (ROS 2のpythonクライアント)の機能を使えるようにします。
import rclpy
# rclpy (ROS 2のpythonクライアント)の機能のうちNodeを簡単に使えるようにします。こう書いていない場合、Nodeではなくrclpy.node.Nodeと書く必要があります。
from rclpy.node import Node
# ROS 2の文字列型を使えるようにimport
import std_msgs.msg as std_msgs
import sensor_msgs.msg as sensor_msgs
import nav_msgs.msg as nav_msgs
import numpy as np
import math
#import open3d as o3d
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
import os
import time
import transforms3d
from geometry_msgs.msg import Quaternion

# C++と同じく、Node型を継承します。
class OdomWheel(Node):
    # コンストラクタです、クラスのインスタンスを作成する際に呼び出されます。
    def __init__(self):
        # 継承元のクラスを初期化します。（https://www.python-izm.com/advanced/class_extend/）今回の場合継承するクラスはNodeになります。
        super().__init__('odom_wheel_node')
        
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
        self.subscription = self.create_subscription(nav_msgs.Odometry,'/odom', self.get_wheel_odom, qos_profile_sub)
        self.subscription  # 警告を回避するために設置されているだけです。削除しても挙動はかわりません。
        
        # Publisherを作成
        self.odom_wheel_publisher = self.create_publisher(nav_msgs.Odometry, 'odom_wheel', qos_profile)       
        
        
        
        #パラメータ
        
        #positon init
        self.position_x = 0.0 #[m]
        self.position_y = 0.0 #[m]
        self.position_z = 0.0 #[m]
        self.theta_x = 0.0 #[m]
        self.theta_y = 0.0 #[m]
        self.theta_z = 0.0 #[m]
        
        
        #buff init
        self.imu_sec_buff = 0
        self.imu_nanosec_buff = 0
        self.position_x_buff = 0
        self.position_y_buff = 0
        self.position_z_buff = 0
        self.roll_buff = 0
        self.pitch_buff = 0
        self.yaw_buff = 0
        
        self.odom_fast_buff = 0
        self.odom_wheel_buff = 0
            
        self.wheel_speed = 0
        self.wheel_ang_yaw = 0
        
        self.est_pos_x = 0
        self.est_pos_y = 0
        self.est_pos_z = 0
        self.est_theta = 0
        self.t_stamp = 0
        self.t_stamp_flag = 0
        
                
    def get_wheel_odom(self, msg):
        #print(f"!get fastlio odom!")
        t_stamp = msg.header.stamp
        t_sec = msg.header.stamp.sec
        #print(f"t_stamp_odom_sec ={t_stamp_imu_sec}")
        t_nanosec = msg.header.stamp.nanosec
        #print(f"t_stamp_odom_nanosec ={t_stamp_imu_nanosec}")
        
        imu_time = (t_sec + t_nanosec/1000000000)
        imu_buff_time = (self.imu_sec_buff + self.imu_nanosec_buff/1000000000)
        
        if self.imu_sec_buff == 0:
            imu_dt = 0
        else :
            imu_dt = imu_time - imu_buff_time
        
        #buff update
        self.imu_sec_buff = t_sec
        self.imu_nanosec_buff = t_nanosec
        
        #get position
        pos_x = msg.pose.pose.position.x
        pos_y = msg.pose.pose.position.y
        pos_z = msg.pose.pose.position.z
        
        #get orientation
        flio_q_x = msg.pose.pose.orientation.x
        flio_q_y = msg.pose.pose.orientation.y
        flio_q_z = msg.pose.pose.orientation.z
        flio_q_w = msg.pose.pose.orientation.w
        
        roll, pitch, yaw = quaternion_to_euler(flio_q_x, flio_q_y, flio_q_z, flio_q_w)
        
        theta_x = roll *180/math.pi
        theta_y = pitch *180/math.pi
        #theta_z = (yaw + self.yaw_buff) /2 *180/math.pi
        ##yaw1 =  (yaw*180/math.pi) % 360
        ##yaw2 =  (self.yaw_buff*180/math.pi) %360
        ##theta_z = (yaw1 + yaw2) /2
        ##if theta_z > 180:
        ##    theta_z = theta_z - 360
        
        yaw1 = yaw *180/math.pi
        yaw2 = self.yaw_buff *180/math.pi
        theta_z = average_yaw(yaw1, yaw2)
        
        diff_roll = roll - self.roll_buff
        diff_pitch = pitch - self.pitch_buff
        diff_yaw = yaw - self.yaw_buff
        
        
        #publish
        odom_msg = odometry_msg(pos_x, pos_y, pos_z, theta_x, theta_y, theta_z, t_stamp, 'odom')
        self.odom_wheel_publisher.publish(odom_msg)
        
        self.position_x_buff = pos_x
        self.position_y_buff = pos_y
        self.position_z_buff = pos_z
        self.roll_buff = roll
        self.pitch_buff = pitch
        self.yaw_buff = yaw
        self.t_stamp = t_stamp
        self.t_stamp_flag = 1


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

def imu_msg(linear_acceleration, angular_velocity, orientation, stamp, frame_id):
    imu_msg = sensor_msgs.Imu()
    imu_msg.header.stamp = stamp
    imu_msg.header.frame_id = frame_id
    
    #make imu msg
    imu_msg.linear_acceleration.x = linear_acceleration[0]
    imu_msg.linear_acceleration.y = linear_acceleration[1]
    imu_msg.linear_acceleration.z = linear_acceleration[2]
    imu_msg.angular_velocity.x = angular_velocity[0]
    imu_msg.angular_velocity.y = angular_velocity[1]
    imu_msg.angular_velocity.z = angular_velocity[2]
    imu_msg.orientation.x = orientation[0]
    imu_msg.orientation.y = orientation[1]
    imu_msg.orientation.z = orientation[2]
    imu_msg.orientation.w = orientation[3]
    
    return imu_msg

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
    
def average_yaw(yaw1, yaw2): 
    # 角度をラジアンに変換 
    yaw1_rad = np.deg2rad(yaw1) 
    yaw2_rad = np.deg2rad(yaw2) 
    # sinとcosの値を計算 
    sin1, cos1 = np.sin(yaw1_rad), np.cos(yaw1_rad) 
    sin2, cos2 = np.sin(yaw2_rad), np.cos(yaw2_rad) 
    # sinとcosの平均を計算 
    sin_avg = (sin1 + sin2) / 2 
    cos_avg = (cos1 + cos2) / 2 
    # atan2関数を使用して平均角度を算出 
    avg_yaw_rad = np.arctan2(sin_avg, cos_avg) 
    # 平均角度を度に変換 
    avg_yaw_deg = np.rad2deg(avg_yaw_rad) 
    
    return avg_yaw_deg
    
# mainという名前の関数です。C++のmain関数とは異なり、これは処理の開始地点ではありません。
def main(args=None):
    # rclpyの初期化処理です。ノードを立ち上げる前に実装する必要があります。
    rclpy.init(args=args)
    # Mid360Subscriberクラスのインスタンスを作成
    odom_wheel = OdomWheel()
    # spin処理を実行、spinをしていないとROS 2のノードはデータを入出力することが出来ません。
    rclpy.spin(odom_wheel)
    # 明示的にノードの終了処理を行います。
    odom_wheel.destroy_node()
    # rclpyの終了処理、これがないと適切にノードが破棄されないため様々な不具合が起こります。
    rclpy.shutdown()

# 本スクリプト(publish.py)の処理の開始地点です。
if __name__ == '__main__':
    # 関数`main`を実行する。
    main()
