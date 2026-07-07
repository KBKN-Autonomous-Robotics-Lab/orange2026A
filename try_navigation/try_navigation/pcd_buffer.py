#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# rclpy (ROS 2のpythonクライアント)の機能を使えるようにします。
import rclpy
# rclpy の Node を使いやすくインポート
from rclpy.node import Node
import std_msgs.msg as std_msgs
import sensor_msgs.msg as sensor_msgs
import numpy as np
import math
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
import time
import pandas as pd
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

# sensor_msgs_py の point_cloud2 が使えれば利用（環境依存）
try:
    from sensor_msgs_py import point_cloud2 as pc2
except Exception:
    pc2 = None


class PcdBufferNode(Node):
    """
    指定の PointCloud2 トピックを購読して点群をバッファ（グリッド丸め -> 重複削除 -> MAX_POINTSに切り詰め）し、
    新しいトピックに publish するノード。
    """

    def __init__(self):
        super().__init__('pcd_buffer')

        # QoS 設定
        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        qos_profile_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        # パラメータ
        # 入出力トピック
        self.INPUT_TOPIC   = '/pcd_obs_global'          
        self.OUTPUT_TOPIC  = 'pcd_obs_global_buff'  

        # 丸め精度
        self.GROUND_PIXEL  = 1000.0 / 50.0      

        self.MAX_POINTS    = 50000
        self.FRAME_ID      = 'odom'
        # 内部バッファ（4 x N: x, y, z, intensity）
        self.pcd_obs_global_buff = np.array([[], [], [], []], dtype=np.float32)

        # Subscriptionを作成
        self.subscription = self.create_subscription(
            sensor_msgs.PointCloud2,
            self.INPUT_TOPIC,
            self.pcd_buffer_callback,
            qos_profile_sub
        )
        self.subscription  # 警告回避用

        # Publisherを作成
        self.pcd_obs_global_buff_publisher = self.create_publisher(sensor_msgs.PointCloud2, self.OUTPUT_TOPIC, qos_profile)

        # ログ
        self.get_logger().info(f'PcdBufferNode ready. sub:{self.INPUT_TOPIC} pub:{self.OUTPUT_TOPIC} ground_pixel:{self.GROUND_PIXEL} MAX_POINTS:{self.MAX_POINTS}')

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

    def pcd_buffer_callback(self, msg):
        """
        PointCloud2 を受け取りバッファに追加 -> グリッド丸め -> 重複削除 -> 切り詰め -> publish
        """
        #print stamp message
        t_stamp = msg.header.stamp
        print(f"t_stamp ={t_stamp}")

        # PointCloud2 -> numpy 
        points = self.pointcloud2_to_array(msg)  
        print(f"points ={points.shape}")

        # add buffer----------------------------------------------------------------
        if hasattr(self, 'pcd_obs_global_buff') and self.pcd_obs_global_buff is not None and self.pcd_obs_global_buff.size != 0:
            try:
                self.pcd_obs_global_buff = np.hstack((self.pcd_obs_global_buff, points))
            except Exception as e:
                self.get_logger().warning(f'hstack failed: {e}')
                # フォールバック（ concatenate ）
                self.pcd_obs_global_buff = np.concatenate((self.pcd_obs_global_buff, points), axis=1)
        else:
            self.pcd_obs_global_buff = points.copy()

        # 3) グリッド丸め（位置を quantize して重複を削除）
        points_xyz = self.pcd_obs_global_buff[:3, :]  # (3, M)
        points_i = self.pcd_obs_global_buff[3, :]    # (M,)
        points_round = np.round(points_xyz * self.GROUND_PIXEL) / self.GROUND_PIXEL

        # pandas を使って重複削除（丸め後の x,y,z が同じものを 1 つにする）
        df = pd.DataFrame({
            'x': points_round[0, :],
            'y': points_round[1, :],
            'z': points_round[2, :],
        })
        mask_unique = ~df.duplicated()
        unique_idx = np.where(mask_unique)[0]

        if unique_idx.size == 0:
            # 重複で何も残らない場合は空バッファ
            self.pcd_obs_global_buff = np.array([[], [], [], []], dtype=np.float32)
        else:
            new_xyz = points_round[:, unique_idx]
            new_i = points_i[unique_idx]
            self.pcd_obs_global_buff = np.vstack((new_xyz, new_i))

        # 4) 最大点数を超える場合は末尾（最新）を残す
        if self.pcd_obs_global_buff.shape[1] > self.MAX_POINTS:
            self.pcd_obs_global_buff = self.pcd_obs_global_buff[:, -self.MAX_POINTS:].copy()

        # 5) Publish（PointCloud2 へ変換して publish）
        try:
            n_by_4 = self.pcd_obs_global_buff.T  # (N,4)
            cloud_msg = point_cloud_intensity_msg(n_by_4, t_stamp, self.FRAME_ID)
            self.pcd_obs_global_buff_publisher.publish(cloud_msg)
        except Exception as e:
            self.get_logger().error(f'publish failed: {e}')


def point_cloud_intensity_msg(points, t_stamp, parent_frame):
    """
    points: (N,4) numpy array (x,y,z,intensity)
    t_stamp: builtin_interfaces.msg.Time (そのまま header.stamp に代入)
    parent_frame: frame id string
    """
    if points is None or len(points) == 0:
        header = std_msgs.Header(frame_id=parent_frame, stamp=t_stamp)
        return sensor_msgs.PointCloud2(header=header, height=0, width=0, is_dense=True, is_bigendian=False, fields=[], point_step=0, row_step=0, data=b'')

    ros_dtype = sensor_msgs.PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize  # 4 bytes
    arr = np.asarray(points, dtype=dtype)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] == 3:
        zeros = np.zeros((arr.shape[0], 1), dtype=dtype)
        arr = np.hstack((arr, zeros))
    elif arr.shape[1] > 4:
        arr = arr[:, :4]

    data = arr.astype(dtype).tobytes()

    fields = [
        sensor_msgs.PointField(name='x', offset=0, datatype=ros_dtype, count=1),
        sensor_msgs.PointField(name='y', offset=4, datatype=ros_dtype, count=1),
        sensor_msgs.PointField(name='z', offset=8, datatype=ros_dtype, count=1),
        sensor_msgs.PointField(name='intensity', offset=12, datatype=ros_dtype, count=1),
    ]

    header = std_msgs.Header(frame_id=parent_frame, stamp=t_stamp)

    cloud = sensor_msgs.PointCloud2(
        header=header,
        height=1,
        width=arr.shape[0],
        is_dense=True,
        is_bigendian=False,
        fields=fields,
        point_step=itemsize * 4,
        row_step=itemsize * 4 * arr.shape[0],
        data=data
    )
    return cloud


# main 関数
def main(args=None):
    rclpy.init(args=args)
    node = PcdBufferNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
