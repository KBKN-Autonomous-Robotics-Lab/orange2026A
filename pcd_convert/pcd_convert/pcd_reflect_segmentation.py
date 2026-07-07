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
class PcdReflectSegmentation(Node):
    # コンストラクタです、PcdReflectSegmentationクラスのインスタンスを作成する際に呼び出されます。
    def __init__(self):
        # 継承元のクラスを初期化します。
        super().__init__('pcd_reflect_segmentation_node')
        
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
        
        # Subscriptionを作成。
        self.subscription = self.create_subscription(sensor_msgs.PointCloud2, '/pcd_rotation', self.pcd_reflect_segmentation, qos_profile) #set subscribe pcd topic name
        self.subscription  # 警告を回避するために設置されているだけです。削除しても挙動はかわりません。
        
        # Publisherを作成
        #self.pcd_segment_paper_publisher = self.create_publisher(sensor_msgs.PointCloud2, 'pcd_segment_paper', qos_profile) #set publish pcd topic name
        #self.pcd_segment_cardboard_publisher = self.create_publisher(sensor_msgs.PointCloud2, 'pcd_segment_cardboard', qos_profile) #set publish pcd topic name
        #self.pcd_segment_asphalt_publisher = self.create_publisher(sensor_msgs.PointCloud2, 'pcd_segment_asphalt', qos_profile) #set publish pcd topic name
        #self.pcd_segment_whiteline_publisher = self.create_publisher(sensor_msgs.PointCloud2, 'pcd_segment_whiteline', qos_profile) #set publish pcd topic name
        #self.pcd_segment_gravel_publisher = self.create_publisher(sensor_msgs.PointCloud2, 'pcd_segment_gravel', qos_profile) #set publish pcd topic name
        #self.pcd_segment_grass_publisher = self.create_publisher(sensor_msgs.PointCloud2, 'pcd_segment_grass', qos_profile) #set publish pcd topic name
        #self.pcd_segment_high_ref_publisher = self.create_publisher(sensor_msgs.PointCloud2, 'pcd_segment_high_ref', qos_profile) #set publish pcd topic name
        #self.pcd_segment_low_ref_publisher = self.create_publisher(sensor_msgs.PointCloud2, 'pcd_segment_low_ref', qos_profile) #set publish pcd topic name
        self.pcd_segment_shibafu_publisher = self.create_publisher(sensor_msgs.PointCloud2, 'pcd_segment_shibafu', qos_profile) #set publish pcd topic name
        #self.pcd_segment_otiba_publisher = self.create_publisher(sensor_msgs.PointCloud2, 'pcd_segment_otiba', qos_profile) #set publish pcd topic name
        
        #パラメータ
        #set obs range
        self.OBS_MASK_X_MIN = -550/1000; #x mask range[m]
        self.OBS_MASK_X_MAX =  200/1000; #x mask range[m]
        self.OBS_MASK_Y_MIN = -350/1000; #y mask range[m]
        self.OBS_MASK_Y_MAX =  350/1000; #y mask range[m]
        
        self.obj_reflect =[
            #     0       1
            # median  sigma
            [  47.00,  6.26], # 0:paper ポットホール（紙）
            [  26.00,  2.33], # 1:cardboard 段ボール
            [  10.00,  0.92], # 2:asphalt アスファルト
            [ 155.00, 25.33], # 3:whiteline 白線
            [  15.00,  3.29], # 4:gravel 砂利
            [  64.00,  8.03], # 5:grass 
            [ 119.05, 10.32], # 6:high_ref 
            [   3.62,  1.21], # 7:low_ref 
            [  52.00,  5.25], # 8:shibafu
            [  72.00, 16.81], # 9:otiba 
            [   0,      0]    # 
        ]
        self.sigma = 2
        
        
    def pointcloud2_to_array(self, cloud_msg):
        # Extract point cloud data
        points = np.frombuffer(cloud_msg.data, dtype=np.uint8).reshape(-1, cloud_msg.point_step)
        x = np.frombuffer(points[:, 0:4].tobytes(), dtype=np.float32)
        y = np.frombuffer(points[:, 4:8].tobytes(), dtype=np.float32)
        z = np.frombuffer(points[:, 8:12].tobytes(), dtype=np.float32)
        intensity = np.frombuffer(points[:, 12:16].tobytes(), dtype=np.float32)

        # Combine into a 4xN matrix
        point_cloud_matrix = np.vstack((x, y, z, intensity))
        print(point_cloud_matrix)
        print(f"point_cloud_matrix ={point_cloud_matrix.shape}")
        print(f"x ={x.dtype, x.shape}")
        
        return point_cloud_matrix
        
    def pcd_reflect_segmentation(self, msg):
        
        #print stamp message
        t_stamp = msg.header.stamp
        print(f"t_stamp ={t_stamp}")
        
        #get pcd data
        points = self.pointcloud2_to_array(msg)
        print(f"points ={points.shape}")
        
        #obs segment
        pcd_mask_cut = self.pcd_mask(points, self.OBS_MASK_X_MIN, self.OBS_MASK_X_MAX, self.OBS_MASK_Y_MIN, self.OBS_MASK_Y_MAX)
        print(f"pcd_mask_cut ={pcd_mask_cut.shape}")
        print(f"self.obj_reflect ={len(self.obj_reflect)}")
        
        
        pcd_obj_reflect_ind_list = []  # 列数に基づいて初期化
        for obj_ind in range(len(self.obj_reflect)-1):
             if self.obj_reflect[obj_ind][0] < 151:
                 obj_reflect_min = self.obj_reflect[obj_ind][0] - self.sigma * self.obj_reflect[obj_ind][1]
                 obj_reflect_max = self.obj_reflect[obj_ind][0] + self.sigma * self.obj_reflect[obj_ind][1]
                 if obj_reflect_min < 0:
                     obj_reflect_min = 0
                 if obj_reflect_max > 150:
                     obj_reflect_max = 150
             else:
                 obj_reflect_min = 151
                 obj_reflect_max = 255
             pcd_obj_reflect_ind =  np.array(self.reflect_segment(pcd_mask_cut, obj_reflect_min, obj_reflect_max))
             pcd_obj_reflect_ind_list.append(pcd_obj_reflect_ind) 
        print(f"pcd_obj_reflect_ind_list ={pcd_obj_reflect_ind_list[0]}")
        
        #publish for rviz2
        pcd_segment_paper = pcd_mask_cut[:, pcd_obj_reflect_ind_list[0]]
        
        #self.pcd_segment_paper     = point_cloud_intensity_msg(pcd_mask_cut[:, pcd_obj_reflect_ind_list[0]].T, t_stamp, 'odom')
        #self.pcd_segment_paper_publisher.publish(self.pcd_segment_paper ) 
        #self.pcd_segment_cardboard = point_cloud_intensity_msg(pcd_mask_cut[:, pcd_obj_reflect_ind_list[1]].T, t_stamp, 'odom')
        #self.pcd_segment_cardboard_publisher.publish(self.pcd_segment_cardboard ) 
        #self.pcd_segment_asphalt   = point_cloud_intensity_msg(pcd_mask_cut[:, pcd_obj_reflect_ind_list[2]].T, t_stamp, 'odom')
        #self.pcd_segment_asphalt_publisher.publish(self.pcd_segment_asphalt ) 
        #self.pcd_segment_whiteline = point_cloud_intensity_msg(pcd_mask_cut[:, pcd_obj_reflect_ind_list[3]].T, t_stamp, 'odom')
        #self.pcd_segment_whiteline_publisher.publish(self.pcd_segment_whiteline ) 
        #self.pcd_segment_gravel    = point_cloud_intensity_msg(pcd_mask_cut[:, pcd_obj_reflect_ind_list[4]].T, t_stamp, 'odom')
        #self.pcd_segment_gravel_publisher.publish(self.pcd_segment_gravel ) 
        #self.pcd_segment_grass    = point_cloud_intensity_msg(pcd_mask_cut[:, pcd_obj_reflect_ind_list[5]].T, t_stamp, 'odom')
        #self.pcd_segment_grass_publisher.publish(self.pcd_segment_grass ) 
        #self.pcd_segment_high_ref    = point_cloud_intensity_msg(pcd_mask_cut[:, pcd_obj_reflect_ind_list[6]].T, t_stamp, 'odom')
        #self.pcd_segment_high_ref_publisher.publish(self.pcd_segment_high_ref ) 
        #self.pcd_segment_low_ref    = point_cloud_intensity_msg(pcd_mask_cut[:, pcd_obj_reflect_ind_list[7]].T, t_stamp, 'odom')
        #self.pcd_segment_low_ref_publisher.publish(self.pcd_segment_low_ref ) 
        
        self.pcd_segment_shibafu    = point_cloud_intensity_msg(pcd_mask_cut[:, pcd_obj_reflect_ind_list[8]].T, t_stamp, 'odom')
        self.pcd_segment_shibafu_publisher.publish(self.pcd_segment_shibafu ) 
        #self.pcd_segment_otiba    = point_cloud_intensity_msg(pcd_mask_cut[:, pcd_obj_reflect_ind_list[9]].T, t_stamp, 'odom')
        #self.pcd_segment_otiba_publisher.publish(self.pcd_segment_otiba ) 
        
        
        
        
        
    def height_segment(self, pointcloud, height_min, height_max):
        pcd_ind = ((height_min <= pointcloud[2,:]) * (pointcloud[2,:] <= height_max ))
        pcd_segment = pointcloud[:, pcd_ind]
        return pcd_segment
        
    def reflect_segment(self, pointcloud, reflect_min, reflect_max):
        pcd_ind = ((reflect_min <= pointcloud[3,:]) * (pointcloud[3,:] <= reflect_max ))
        return pcd_ind
        
    def pcd_mask(self, pointcloud, x_min, x_max, y_min, y_max):
        pcd_ind = (( (x_min <= pointcloud[0,:]) * (pointcloud[0,:] <= x_max)) * ((y_min <= pointcloud[1,:]) * (pointcloud[1,:]) <= y_max ) )
        pcd_mask = pointcloud[:, ~pcd_ind]
        return pcd_mask
        
    def pcd_serch(self, pointcloud, x_min, x_max, y_min, y_max):
        pcd_ind = (( (x_min <= pointcloud[0,:]) * (pointcloud[0,:] <= x_max)) * ((y_min <= pointcloud[1,:]) * (pointcloud[1,:]) <= y_max ) )
        pcd_mask = pointcloud[:, pcd_ind]
        return pcd_mask
        

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
    pcd_reflect_segmentation = PcdReflectSegmentation()
    # spin処理を実行、spinをしていないとROS 2のノードはデータを入出力することが出来ません。
    rclpy.spin(pcd_reflect_segmentation)
    # 明示的にノードの終了処理を行います。
    pcd_reflect_segmentation.destroy_node()
    # rclpyの終了処理、これがないと適切にノードが破棄されないため様々な不具合が起こります。
    rclpy.shutdown()

# 本スクリプト(publish.py)の処理の開始地点です。
if __name__ == '__main__':
    # 関数`main`を実行する。
    main()
