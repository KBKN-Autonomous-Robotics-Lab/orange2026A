#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# rclpy (ROS 2のpythonクライアント)の機能を使えるようにします。
import rclpy
# rclpy (ROS 2のpythonクライアント)の機能のうちNodeを簡単に使えるようにします。こう書いていない場合、Nodeではなくrclpy.node.Nodeと書く必要があります。
from rclpy.node import Node
import std_msgs.msg as std_msgs
import sensor_msgs.msg as sensor_msgs
import numpy as np
import math
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
import time


# C++と同じく、Node型を継承します。
class PcdRotation(Node):
    # コンストラクタです、PcdRotationクラスのインスタンスを作成する際に呼び出されます。
    def __init__(self):
        # 継承元のクラスを初期化します。
        super().__init__('pcd_rotation_lidar2_node')
        
        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth = 1
        )
        
        qos_profile_sub = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth = 10
        )
        
        # Subscriptionを作成。 <-- ここを最小限変更して /livox/lidar2 を購読するようにしました
        self.subscription = self.create_subscription(sensor_msgs.PointCloud2, '/converted_pointcloud2_lidar2', self.pcd_rotation, qos_profile_sub) #set subscribe pcd topic name
        self.subscription  # 警告を回避するために設置されているだけです。削除しても挙動はかわりません。
        
        # Publisherを作成
        self.pcd_rotation_publisher = self.create_publisher(sensor_msgs.PointCloud2, 'pcd_rotation_lidar2', qos_profile) #set publish pcd topic name
        
        #パラメータ
        #set LiDAR position
        self.MID360_HIGHT = 750/1000; #self.MID360_HIGHT =1.7483812830163088
            
        #上下反転  LiDAR init
        self.THETA_INIT_X = 0 #[deg]-180
        self.THETA_INIT_Y = 0.05435636478316808 #self.THETA_INIT_Y =0.05435636478316808#[deg]
        self.THETA_INIT_Z = 0 #[deg]  
        
        #initialize calibration
        self.initialize_calibration = 1
        self.pcd_buff = np.array([[],[],[],[]]);
        self.x1_init_point = 1.5
        self.x2_init_point = 2.5
        self.y_init_point = 0.0
        self.x_range = 0.1
        self.y_range = 0.3
        
        
        
    def pointcloud2_to_array(self, cloud_msg):
        # Extract point cloud data
        points = np.frombuffer(cloud_msg.data, dtype=np.uint8).reshape(-1, cloud_msg.point_step)
        x = np.frombuffer(points[:, 0:4].tobytes(), dtype=np.float32)
        y = np.frombuffer(points[:, 4:8].tobytes(), dtype=np.float32)
        z = np.frombuffer(points[:, 8:12].tobytes(), dtype=np.float32)
        intensity = np.frombuffer(points[:, 12:16].tobytes(), dtype=np.float32)

        # Combine into a 4xN matrix
        point_cloud_matrix = np.vstack((x, y, z, intensity))
        #print(point_cloud_matrix)
        #print(f"point_cloud_matrix ={point_cloud_matrix.shape}")
        #print(f"x ={x.dtype, x.shape}")
        
        return point_cloud_matrix
        
    def pcd_rotation(self, msg):
        
        #print stamp message
        t_stamp = msg.header.stamp
        #print(f"t_stamp ={t_stamp}")
        
        #get pcd data
        points = self.pointcloud2_to_array(msg)
        #print(f"points ={points.shape}")
        #for pcd rotation
        xyz_point = np.vstack([points[0,:],points[1,:],points[2,:]])
        #print(f"xyz_point ={xyz_point.shape}")
        pointcloud, rot_matrix = rotation_xyz(xyz_point, self.THETA_INIT_X, self.THETA_INIT_Y, self.THETA_INIT_Z)
        #add intensity
        pointcloud_intensity = np.insert(pointcloud, 3, points[3,:], axis=0)
        
        if self.initialize_calibration == 0:
            pcd_buff_len = 3* 2*10*4
            self.pcd_buff = np.insert(self.pcd_buff, len(self.pcd_buff[0,:]), pointcloud_intensity.T, axis=1)
            print(f"len(self.pcd_buff), {self.pcd_buff.shape}")
            if (len(self.pcd_buff[0,:]) > pcd_buff_len):
                #self.pcd_buff = self.pcd_buff[:,(len(self.pcd_buff[0,:])-pcd_buff_len):len(self.pcd_buff[0,:])]
                x1_min = self.x1_init_point - self.x_range;
                x1_max = self.x1_init_point + self.x_range;
                x2_min = self.x2_init_point - self.x_range;
                x2_max = self.x2_init_point + self.x_range;
                y_min = self.y_init_point  - self.y_range;
                y_max = self.y_init_point  + self.y_range;
                #print(f"x1_min,x1_max,x2_min,x2_max,y_min,y_max ={ x1_min, x1_max, x2_min, x2_max, y_min, y_max}")
                pcd_x1_ind = self.pcd_serch(self.pcd_buff, x1_min, x1_max, y_min, y_max)
                pcd_x2_ind = self.pcd_serch(self.pcd_buff, x2_min, x2_max, y_min, y_max)
                #print(f"len(self.pcd_buff[0,pcd_x1_ind]), {len(self.pcd_buff[0,pcd_x1_ind])}")
                #print(f"len(self.pcd_buff[0,pcd_x2_ind]), {len(self.pcd_buff[0,pcd_x2_ind])}")
                if len(self.pcd_buff[0,pcd_x1_ind]) & len(self.pcd_buff[0,pcd_x2_ind]) > 0:
                    pcd_x1 = self.pcd_buff[:,pcd_x1_ind]
                    pcd_x2 = self.pcd_buff[:,pcd_x2_ind]
                    #pcd_x1_closest_ind = np.abs(pcd_x1[0,:] - self.x1_init_point).argmin()
                    z1_median = np.median(pcd_x1[2,:])
                    pcd_x1_closest_ind = np.abs(pcd_x1[2,:]-z1_median).argmin()
                    #pcd_x2_closest_ind = np.abs(pcd_x2[0,:] - self.x2_init_point).argmin()
                    z2_median = np.median(pcd_x2[2,:])
                    pcd_x2_closest_ind = np.abs(pcd_x2[2,:]-z2_median).argmin()
                    #z= {slope}x + intercept
                    slope = (pcd_x2[2,pcd_x2_closest_ind] - pcd_x1[2,pcd_x1_closest_ind]) / (pcd_x2[0,pcd_x2_closest_ind] - pcd_x1[0,pcd_x1_closest_ind])
                    intercept = pcd_x1[2,pcd_x1_closest_ind] - slope*pcd_x1[0,pcd_x1_closest_ind]
                    
                    
                    print(f"pcd_x1 x,y,z ={pcd_x1[0,pcd_x1_closest_ind], pcd_x1[1,pcd_x1_closest_ind], pcd_x1[2,pcd_x1_closest_ind]}")
                    print(f"pcd_x2 x,y,z ={pcd_x2[0,pcd_x2_closest_ind], pcd_x2[1,pcd_x2_closest_ind], pcd_x2[2,pcd_x2_closest_ind]}")
                    delta_x = pcd_x2[0,pcd_x2_closest_ind] - 0
                    delta_z = - (intercept - pcd_x2[2,pcd_x2_closest_ind]) 
                    theta = np.arctan2(delta_z, delta_x) /math.pi*180
                    print(f"delta_x ={delta_x}")
                    print(f"delta_z ={delta_z}")
                    print(f"slope ={slope}")
                    print(f"intercept ={intercept}")
                    print(f"theta ={theta}")
                    self.MID360_HIGHT = - intercept; #hight position[m]
                    self.THETA_INIT_Y = self.THETA_INIT_Y + theta #[deg]
                    self.initialize_calibration = 1
                    #print(f"self.MID360_HIGHT ={self.MID360_HIGHT}")
                    #print(f"self.THETA_INIT_Y ={self.THETA_INIT_Y}")
                    
        
        #add mid height position
        pointcloud_intensity[2,:] += self.MID360_HIGHT
        #print(f"pointcloud_intensity ={pointcloud_intensity.shape}")
        
        #publish for rviz2
        if self.initialize_calibration == 1:
            self.pcd_rotation = point_cloud_intensity_msg(pointcloud_intensity.T, t_stamp, 'livox_frame')
            self.pcd_rotation_publisher.publish(self.pcd_rotation ) 
        
    def pcd_serch(self, pointcloud, x_min, x_max, y_min, y_max):
        pcd_ind_x = ((x_min <= pointcloud[0,:]) * (pointcloud[0,:] <= x_max )) 
        pcd_ind_y = ((y_min <= pointcloud[1,:]) * (pointcloud[1,:] <= y_max ))
        pcd_ind = pcd_ind_x * pcd_ind_y
        return pcd_ind
        
def rotation_xyz(pointcloud, theta_x, theta_y, theta_z):
    rad_x = math.radians(theta_x)
    rad_y = math.radians(theta_y)
    rad_z = math.radians(theta_z)
    rot_x = np.array([[ 1,               0,                0],
                      [ 0, math.cos(rad_x), -math.sin(rad_x)],
                      [ 0, math.sin(rad_x),  math.cos(rad_x)]])
    
    rot_y = np.array([[ math.cos(rad_y), 0,  math.sin(rad_y)],
                      [               0, 1,                0],
                      [-math.sin(rad_y), 0,  math.cos(rad_y)]])
    
    rot_z = np.array([[ math.cos(rad_z), -math.sin(rad_z), 0],
                      [ math.sin(rad_z),  math.cos(rad_z), 0],
                      [               0,                0, 1]])
    rot_matrix = rot_z.dot(rot_y.dot(rot_x))
    #print(f"rot_matrix ={rot_matrix}")
    #print(f"pointcloud ={pointcloud.shape}")
    rot_pointcloud = rot_matrix.dot(pointcloud)
    return rot_pointcloud, rot_matrix

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
    pcd_rotation = PcdRotation()
    # spin処理を実行、spinをしていないとROS 2のノードはデータを入出力することが出来ません。
    rclpy.spin(pcd_rotation)
    # 明示的にノードの終了処理を行います。
    pcd_rotation.destroy_node()
    # rclpyの終了処理、これがないと適切にノードが破棄されないため様々な不具合が起こります。
    rclpy.shutdown()

# 本スクリプト(publish.py)の処理の開始地点です。
if __name__ == '__main__':
    # 関数`main`を実行する。
    main()


