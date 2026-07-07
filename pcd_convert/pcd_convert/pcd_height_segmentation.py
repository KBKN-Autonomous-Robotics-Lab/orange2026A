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
class PcdHeightSegmentation(Node):
    # コンストラクタです、PcdHeightSegmentationクラスのインスタンスを作成する際に呼び出されます。
    def __init__(self):
        # 継承元のクラスを初期化します。
        super().__init__('pcd_heigth_segmentation_node')
        
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
        self.subscription = self.create_subscription(sensor_msgs.PointCloud2, '/pcd_rotation_merge', self.pcd_heigth_segmentation, qos_profile) #set subscribe pcd topic name /pcd_rotation_merge
        self.subscription  # 警告を回避するために設置されているだけです。削除しても挙動はかわりません。
        
        # Publisherを作成
        self.pcd_segment_obs_publisher = self.create_publisher(sensor_msgs.PointCloud2, 'pcd_segment_obs', qos_profile) #set publish pcd topic name
        self.pcd_segment_ground_publisher = self.create_publisher(sensor_msgs.PointCloud2, 'pcd_segment_ground', qos_profile) #set publish pcd topic name
        self.pcd_segment_middle_publisher = self.create_publisher(sensor_msgs.PointCloud2, 'pcd_segment_middle', qos_profile) #set publish pcd topic name
        self.pcd_segment_high_publisher = self.create_publisher(sensor_msgs.PointCloud2, 'pcd_segment_high', qos_profile) #set publish pcd topic name
        self.pcd_segment_step_publisher = self.create_publisher(sensor_msgs.PointCloud2, 'pcd_segment_step', qos_profile) #set publish pcd topic name
        self.pcd_segment_low_step_publisher = self.create_publisher(sensor_msgs.PointCloud2, 'pcd_segment_low_step', qos_profile) #set publish pcd topic name
        
        #パラメータ
        #set obs range
        self.OBS_HIGHT_MIN =   150/1000; #hight range[m]
        self.OBS_HIGHT_MAX =  1000/1000; #hight range[m]
        self.OBS_MASK_X_MIN = -550/1000; #x mask range[m]
        self.OBS_MASK_X_MAX =  400/1000; #x mask range[m]
        self.OBS_MASK_Y_MIN = -350/1000; #y mask range[m]
        self.OBS_MASK_Y_MAX =  350/1000; #y mask range[m]
        #set ground range
        self.GROUND_HIGHT_MIN = -150/1000; #hight range[m] # IGVC20250601 -150 -> -10
        self.GROUND_HIGHT_MAX =  120/1000; #hight range[m]
        #set middle range
        self.MIDDLE_HIGHT_MIN = 500/1000; #hight range[m] # IGVC20250601 -150 -> -10
        self.MIDDLE_HIGHT_MAX = 1000/1000; #hight range[m]
        #set high range
        self.HIGH_HIGHT_MIN = 2000/1000; #hight range[m] # IGVC20250601 -150 -> -10
        self.HIGH_HIGHT_MAX = 4000/1000; #hight range[m]
        #set step range
        self.STEP_HIGHT_MIN =   140/1000; #hight range[m]
        self.STEP_HIGHT_MAX =   self.OBS_HIGHT_MIN# 200/1000; #hight range[m]
        self.STEP_X_MIN     = -1500/1000; #x mask range[m]
        self.STEP_X_MAX     =  1500/1000; #x mask range[rosbag2_2024_10_26-03_14_14_20241026_kakunin_bag1m]
        self.STEP_Y_MIN     = -5000/1000; #y mask range[m]
        self.STEP_Y_MAX     =  5000/1000; #y mask range[m]
        #set low step range
        self.LOW_STEP_HIGHT_MIN =   70/1000; #hight range[m]
        self.LOW_STEP_HIGHT_MAX =   self.OBS_HIGHT_MIN# 200/1000; #hight range[m]
        self.LOW_STEP_X_MIN     = -1500/1000; #x mask range[m]
        self.LOW_STEP_X_MAX     =  1500/1000; #x mask range[rosbag2_2024_10_26-03_14_14_20241026_kakunin_bag1m]
        self.LOW_STEP_Y_MIN     = -5000/1000; #y mask range[m]
        self.LOW_STEP_Y_MAX     =  5000/1000; #y mask range[m]
        
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
        
    def pcd_heigth_segmentation(self, msg):
        
        #print stamp message
        t_stamp = msg.header.stamp
        print(f"t_stamp ={t_stamp}")
        
        #get pcd data
        points = self.pointcloud2_to_array(msg)
        print(f"points ={points.shape}")
        
        #obs segment
        pcd_obs_height = self.height_segment(points, self.OBS_HIGHT_MIN, self.OBS_HIGHT_MAX)
        pcd_obs = self.pcd_mask(pcd_obs_height, self.OBS_MASK_X_MIN, self.OBS_MASK_X_MAX, self.OBS_MASK_Y_MIN, self.OBS_MASK_Y_MAX)
        print(f"pcd_obs_height ={pcd_obs_height.shape}")
        print(f"pcd_obs ={pcd_obs.shape}")
        
        #ground segment
        pcd_ground = self.height_segment(points, self.GROUND_HIGHT_MIN, self.GROUND_HIGHT_MAX)
        print(f"pcd_ground ={pcd_ground.shape}")

        #middle segment
        pcd_middle = self.height_segment(points, self.MIDDLE_HIGHT_MIN, self.MIDDLE_HIGHT_MAX)
        print(f"pcd_middle ={pcd_middle.shape}")

        #high segment
        pcd_high = self.height_segment(points, self.HIGH_HIGHT_MIN, self.HIGH_HIGHT_MAX)
        print(f"pcd_high ={pcd_high.shape}")
        
        #step segment
        pcd_step = self.height_segment(points, self.STEP_HIGHT_MIN, self.STEP_HIGHT_MAX)
        #pcd_step_height_under = self.height_segment(points, -self.STEP_HIGHT_MAX, -self.STEP_HIGHT_MIN)
        #pcd_step = np.insert(pcd_step, len(pcd_step[0,:]), pcd_step_height_under.T, axis=1)
        pcd_step = self.pcd_serch(pcd_step, self.STEP_X_MIN, self.STEP_X_MAX, self.STEP_Y_MIN, self.STEP_Y_MAX)
        pcd_obs = np.insert(pcd_obs, len(pcd_obs[0,:]), pcd_step.T, axis=1)
        
        #low step segment
        pcd_low_step = self.height_segment(points, self.LOW_STEP_HIGHT_MIN, self.LOW_STEP_HIGHT_MAX)
        #pcd_step_height_under = self.height_segment(points, -self.STEP_HIGHT_MAX, -self.STEP_HIGHT_MIN)
        #pcd_step = np.insert(pcd_step, len(pcd_step[0,:]), pcd_step_height_under.T, axis=1)
        pcd_low_step = self.pcd_serch(pcd_low_step, self.LOW_STEP_X_MIN, self.LOW_STEP_X_MAX, self.LOW_STEP_Y_MIN, self.LOW_STEP_Y_MAX)
        
        #publish for rviz2
        self.pcd_segment_obs = point_cloud_intensity_msg(pcd_obs.T, t_stamp, 'livox_frame')
        self.pcd_segment_obs_publisher.publish(self.pcd_segment_obs ) 
        self.pcd_segment_ground = point_cloud_intensity_msg(pcd_ground.T, t_stamp, 'livox_frame')
        self.pcd_segment_ground_publisher.publish(self.pcd_segment_ground ) 
        self.pcd_segment_middle = point_cloud_intensity_msg(pcd_middle.T, t_stamp, 'livox_frame')
        self.pcd_segment_middle_publisher.publish(self.pcd_segment_middle ) 
        self.pcd_segment_high = point_cloud_intensity_msg(pcd_high.T, t_stamp, 'livox_frame')
        self.pcd_segment_high_publisher.publish(self.pcd_segment_high ) 
        self.pcd_segment_step = point_cloud_intensity_msg(pcd_step.T, t_stamp, 'livox_frame')
        self.pcd_segment_step_publisher.publish(self.pcd_segment_step ) 
        self.pcd_segment_low_step = point_cloud_intensity_msg(pcd_low_step.T, t_stamp, 'livox_frame')
        self.pcd_segment_low_step_publisher.publish(self.pcd_segment_low_step ) 
        
    def height_segment(self, pointcloud, height_min, height_max):
        pcd_ind = ((height_min <= pointcloud[2,:]) * (pointcloud[2,:] <= height_max ))
        pcd_segment = pointcloud[:, pcd_ind]
        return pcd_segment
        
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
    pcd_height_segmentation = PcdHeightSegmentation()
    # spin処理を実行、spinをしていないとROS 2のノードはデータを入出力することが出来ません。
    rclpy.spin(pcd_height_segmentation)
    # 明示的にノードの終了処理を行います。
    pcd_height_segmentation.destroy_node()
    # rclpyの終了処理、これがないと適切にノードが破棄されないため様々な不具合が起こります。
    rclpy.shutdown()

# 本スクリプト(publish.py)の処理の開始地点です。
if __name__ == '__main__':
    # 関数`main`を実行する。
    main()
