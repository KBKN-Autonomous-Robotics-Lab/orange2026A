#!/usr/bin/env python3
"""反射強度の閾値で路側帯領域を抽出し、追従目標を出力する。

  sub : /pcd_segment_ground
  pub : /roadside/detected, /roadside/target_angle, /roadside/target_point

デバッグ用に各段階の点群も出力する。
  /roadside/accum_points   蓄積直後（強度フィルタ済み）
  /roadside/roi_points     ROIとdedup通過後
  /roadside/all_clusters   DBSCAN後の点数上位クラスタ
"""

import math
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSDurabilityPolicy,
                       QoSHistoryPolicy, QoSReliabilityPolicy)

import sensor_msgs.msg as sensor_msgs
from geometry_msgs.msg import Point, PointStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32, Header
from visualization_msgs.msg import Marker

from sklearn.cluster import DBSCAN


class RoadsideFollow(Node):

    def __init__(self):
        super().__init__('roadside_follow')

        self.declare_parameter('accumulate_frames', 10)      # 重ねるフレーム数
        self.declare_parameter('grid_size', 0.05)            # 間引きのマス目 [m]
        self.declare_parameter('intensity_threshold', 24.0)  # 路側帯とみなす強度の下限
        self.declare_parameter('intensity_max', 60.0)        # 同上限。標識等の高反射を除外
        self.declare_parameter('roi_r_min', 0.5)             # 探索の最短距離 [m]
        self.declare_parameter('roi_r_max', 5.0)             # 探索の最遠距離 [m]
        self.declare_parameter('roi_half_angle_deg', 100.0)  # 前方扇形の片側 [deg]
        self.declare_parameter('dbscan_eps', 0.25)           # 同じ塊とみなす点間距離 [m]
        self.declare_parameter('dbscan_min_samples', 8)      # 塊の核とみなす近傍点数
        self.declare_parameter('min_cluster_points', 60)     # 採用する塊の最小点数
        self.declare_parameter('max_cluster_offset', 3.0)    # 塊の重心の横方向許容 [m]
        self.declare_parameter('lookahead', 1.2)             # 目標点までの距離 [m]
        self.declare_parameter('ring_band', 0.30)            # 先読み円の帯幅（片側）[m]
        self.declare_parameter('min_ring_points', 10)        # 帯に必要な点数
        self.declare_parameter('max_target_angle_deg', 70.0) # 目標角の上限 [deg]
        self.declare_parameter('auto_enable', True)          # 起動時から検出するか
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('verbose', True)              # 検出結果をログに出すか
        self.declare_parameter('debug_cloud', True)          # 段階別の点群を出すか
        self.declare_parameter('debug_top_n', 3)             # 表示するクラスタ数

        qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST,
                         reliability=QoSReliabilityPolicy.RELIABLE,
                         durability=QoSDurabilityPolicy.VOLATILE, depth=1)

        self.enabled = self.get_parameter('auto_enable').value
        self.frame_buf = deque()      # 世界座標に直した点群の履歴

        self.odom_ok = False
        self.px = self.py = 0.0       # 自己位置 [m]
        self.yaw = 0.0                # 方位 [rad]

        self.create_subscription(sensor_msgs.PointCloud2, '/pcd_segment_ground',
                                 self.cb_ground, qos)
        self.create_subscription(Odometry, self.get_parameter('odom_topic').value,
                                 self.cb_odom, qos)
        self.create_subscription(Bool, '/roadside_follow/enable', self.cb_enable, qos)

        self.pub_detected = self.create_publisher(Bool, '/roadside/detected', qos)
        self.pub_target = self.create_publisher(PointStamped, '/roadside/target_point', qos)
        self.pub_angle = self.create_publisher(Float32, '/roadside/target_angle', qos)
        self.pub_cluster = self.create_publisher(sensor_msgs.PointCloud2,
                                                 '/roadside/cluster_points', qos)
        self.pub_ring = self.create_publisher(sensor_msgs.PointCloud2,
                                              '/roadside/ring_points', qos)
        self.pub_marker = self.create_publisher(Marker, '/roadside/marker', qos)

        self.pub_accum = self.create_publisher(sensor_msgs.PointCloud2,
                                               '/roadside/accum_points', qos)
        self.pub_roi = self.create_publisher(sensor_msgs.PointCloud2,
                                             '/roadside/roi_points', qos)
        self.pub_all = self.create_publisher(sensor_msgs.PointCloud2,
                                             '/roadside/all_clusters', qos)

        self.get_logger().info('roadside_follow started. enabled=%s' % self.enabled)

    def cb_odom(self, msg):
        p = msg.pose.pose.position
        self.px, self.py = p.x, p.y
        self.yaw = quat_to_yaw(msg.pose.pose.orientation)
        self.odom_ok = True

    def cb_enable(self, msg):
        if msg.data != self.enabled:
            self.get_logger().info('detection %s' % ('ON' if msg.data else 'OFF'))
            self.frame_buf.clear()
        self.enabled = msg.data

    def cb_ground(self, msg):
        if not self.enabled:
            return
        if not self.odom_ok:
            self.get_logger().warn('waiting for odom', throttle_duration_sec=2.0)
            return

        pts = pointcloud2_to_array(msg)
        if pts.shape[1] == 0:
            return

        gp = self.get_parameter
        lo = gp('intensity_threshold').value
        hi = gp('intensity_max').value
        pts = pts[:, (pts[3, :] >= lo) & (pts[3, :] <= hi)]
        if pts.shape[1] == 0:
            self.no_detection(msg, 'no points in intensity range')
            return

        # 世界座標に直してから溜める。走行中でも同じ路面が重なるようにする
        R = rot2d(self.yaw)
        xy_g = R @ pts[0:2, :] + np.array([[self.px], [self.py]])
        pts_g = np.vstack((xy_g, pts[2:4, :]))

        n_frames = int(gp('accumulate_frames').value)
        self.frame_buf.append(pts_g)
        while len(self.frame_buf) > n_frames:
            self.frame_buf.popleft()

        buf = np.hstack(list(self.frame_buf))

        # 現在の自己位置基準に戻す
        Rt = rot2d(-self.yaw)
        xy_l = Rt @ (buf[0:2, :] - np.array([[self.px], [self.py]]))
        local = np.vstack((xy_l, buf[2:4, :]))

        # 段階1: 蓄積直後
        if gp('debug_cloud').value:
            self.publish_cloud(self.pub_accum, local, msg, 50.0)

        self.detect(local, msg)

    def detect(self, local, msg):
        gp = self.get_parameter

        # 前方の扇形だけ残す。後方は通り過ぎた路側帯が溜まっている
        r = np.hypot(local[0, :], local[1, :])
        ang = np.arctan2(local[1, :], local[0, :])
        half = math.radians(gp('roi_half_angle_deg').value)
        m = ((r >= gp('roi_r_min').value) & (r <= gp('roi_r_max').value)
             & (np.abs(ang) <= half))
        roi = local[:, m]
        if roi.shape[1] < gp('dbscan_min_samples').value:
            self.no_detection(msg, 'roi too few')
            return

        # 1マスにつき1点まで。蓄積で重なった点を落として密度を揃える
        g = gp('grid_size').value
        key = np.round(roi[0:2, :] / g).astype(np.int64)
        _, uniq = np.unique(key, axis=1, return_index=True)
        roi = roi[:, uniq]

        # 段階2: ROIとdedup通過後
        if gp('debug_cloud').value:
            self.publish_cloud(self.pub_roi, roi, msg, 100.0)

        labels = DBSCAN(eps=gp('dbscan_eps').value,
                        min_samples=int(gp('dbscan_min_samples').value)
                        ).fit_predict(roi[0:2, :].T)

        # 段階3: DBSCAN後の上位クラスタ（採用可否に関わらず出す）
        if gp('debug_cloud').value:
            self.publish_all_clusters(roi, labels, msg)

        best = self.select_cluster(roi, labels)
        if best is None:
            self.no_detection(msg, 'no cluster')
            return
        cl, n_ring = best

        target = self.ring_target(cl)
        if target is None:
            self.no_detection(msg, 'ring failed')
            return
        tx, ty, t_ang, ring = target

        self.pub_detected.publish(Bool(data=True))

        ps = PointStamped()
        ps.header.stamp = msg.header.stamp
        ps.header.frame_id = msg.header.frame_id
        ps.point.x, ps.point.y, ps.point.z = float(tx), float(ty), 0.0
        self.pub_target.publish(ps)
        self.pub_angle.publish(Float32(data=float(t_ang)))

        self.publish_cloud(self.pub_cluster, cl, msg, 100.0)
        self.publish_cloud(self.pub_ring, ring, msg, 200.0)
        self.publish_marker(tx, ty, msg)

        if gp('verbose').value:
            self.get_logger().info(
                'frames=%d roi=%d cluster=%d ring=%d | target (%.2f, %.2f) %.1f deg'
                % (len(self.frame_buf), roi.shape[1], cl.shape[1], n_ring,
                   tx, ty, math.degrees(t_ang)),
                throttle_duration_sec=1.0)

    def select_cluster(self, roi, labels):
        """先読み円と最も多く交差するクラスタを選ぶ。

        最大クラスタを選ぶと道路外の芝生や壁を掴む場合があるため、
        点数ではなく先読み円との交差数を条件にする。
        """
        gp = self.get_parameter
        R = gp('lookahead').value
        band = gp('ring_band').value
        min_pts = int(gp('min_cluster_points').value)
        max_off = gp('max_cluster_offset').value

        best = None
        best_ring = -1

        for lb in set(labels):
            if lb < 0:
                continue
            cl = roi[:, labels == lb]
            if cl.shape[1] < min_pts:
                continue
            if abs(float(cl[1, :].mean())) > max_off:
                continue

            r = np.hypot(cl[0, :], cl[1, :])
            n_ring = int(np.count_nonzero(np.abs(r - R) <= band))
            if n_ring > best_ring:
                best_ring = n_ring
                best = cl

        if best is None or best_ring < int(gp('min_ring_points').value):
            return None
        return best, best_ring

    def ring_target(self, cl):
        """半径 lookahead の帯にある点の角度中央値を目標にする。

        重心は路側帯が細長いため前後方向に引かれて不安定になる。
        """
        gp = self.get_parameter
        R = gp('lookahead').value
        band = gp('ring_band').value

        r = np.hypot(cl[0, :], cl[1, :])
        m = np.abs(r - R) <= band
        if np.count_nonzero(m) < int(gp('min_ring_points').value):
            return None

        ring = cl[:, m]
        t_ang = circular_median(np.arctan2(ring[1, :], ring[0, :]))
        return R * math.cos(t_ang), R * math.sin(t_ang), t_ang, ring

    def publish_all_clusters(self, roi, labels, msg):
        """点数上位のクラスタを intensity で色分けして出す。

        採用されなかったものも含めるため、なぜ落ちたかを目視で追える。
        1位=50, 2位=100, 3位=150 ...
        """
        gp = self.get_parameter
        top_n = int(gp('debug_top_n').value)
        R = gp('lookahead').value
        band = gp('ring_band').value

        groups = [roi[:, labels == lb] for lb in set(labels) if lb >= 0]
        groups.sort(key=lambda c: c.shape[1], reverse=True)
        groups = groups[:top_n]

        out = []
        for i, cl in enumerate(groups):
            c = cl.copy()
            c[3, :] = 50.0 * (i + 1)
            out.append(c)

        arr = np.hstack(out) if out else np.zeros((4, 0), np.float32)
        self.pub_all.publish(
            point_cloud_intensity_msg(arr.T, msg.header.stamp, msg.header.frame_id))

        if gp('verbose').value and groups:
            info = '  '.join(
                'n=%d ctr(%.2f,%.2f) ring=%d' % (
                    cl.shape[1], cl[0, :].mean(), cl[1, :].mean(),
                    int(np.count_nonzero(
                        np.abs(np.hypot(cl[0, :], cl[1, :]) - R) <= band)))
                for cl in groups)
            self.get_logger().info('clusters: %s' % info,
                                   throttle_duration_sec=1.0)

    def publish_cloud(self, pub, pts, msg, value):
        """4xN の点群を単色にして publish する"""
        out = pts.copy()
        if out.shape[1] > 0:
            out[3, :] = value
        pub.publish(
            point_cloud_intensity_msg(out.T, msg.header.stamp, msg.header.frame_id))

    def no_detection(self, msg, reason=''):
        self.pub_detected.publish(Bool(data=False))
        self.clear_cloud(self.pub_cluster, msg)
        self.clear_cloud(self.pub_ring, msg)
        self.clear_marker()
        if reason and self.get_parameter('verbose').value:
            self.get_logger().info('not detected: %s' % reason,
                                   throttle_duration_sec=1.0)

    def clear_cloud(self, pub, msg):
        empty = np.zeros((0, 4), dtype=np.float32)
        pub.publish(point_cloud_intensity_msg(
            empty, msg.header.stamp, msg.header.frame_id))

    def clear_marker(self):
        mk = Marker()
        mk.ns = 'roadside'
        mk.id = 0
        mk.action = Marker.DELETE
        self.pub_marker.publish(mk)

    def publish_marker(self, tx, ty, msg):
        R = self.get_parameter('lookahead').value

        pts = []
        n = 48
        for i in range(n):
            a0 = 2.0 * math.pi * i / n
            a1 = 2.0 * math.pi * (i + 1) / n
            pts.append(Point(x=R * math.cos(a0), y=R * math.sin(a0), z=0.0))
            pts.append(Point(x=R * math.cos(a1), y=R * math.sin(a1), z=0.0))
        pts.append(Point(x=0.0, y=0.0, z=0.0))
        pts.append(Point(x=float(tx), y=float(ty), z=0.0))

        mk = Marker()
        mk.header.frame_id = msg.header.frame_id
        mk.header.stamp = msg.header.stamp
        mk.ns = 'roadside'
        mk.id = 0
        mk.type = Marker.LINE_LIST
        mk.action = Marker.ADD
        mk.scale.x = 0.04
        mk.color.r, mk.color.g, mk.color.b, mk.color.a = 0.1, 0.9, 0.5, 1.0
        mk.pose.orientation.w = 1.0
        mk.points = pts
        self.pub_marker.publish(mk)


def pointcloud2_to_array(cloud_msg):
    """PointCloud2 -> 4xN [x, y, z, intensity]"""
    pts = np.frombuffer(cloud_msg.data, dtype=np.uint8).reshape(-1, cloud_msg.point_step)
    x = np.frombuffer(pts[:, 0:4].tobytes(),   dtype=np.float32)
    y = np.frombuffer(pts[:, 4:8].tobytes(),   dtype=np.float32)
    z = np.frombuffer(pts[:, 8:12].tobytes(),  dtype=np.float32)
    i = np.frombuffer(pts[:, 12:16].tobytes(), dtype=np.float32)
    return np.vstack((x, y, z, i))


def point_cloud_intensity_msg(points, t_stamp, parent_frame):
    """Nx4 -> PointCloud2"""
    ros_dtype = sensor_msgs.PointField.FLOAT32
    itemsize = np.dtype(np.float32).itemsize
    fields = [
        sensor_msgs.PointField(name='x',         offset=0,  datatype=ros_dtype, count=1),
        sensor_msgs.PointField(name='y',         offset=4,  datatype=ros_dtype, count=1),
        sensor_msgs.PointField(name='z',         offset=8,  datatype=ros_dtype, count=1),
        sensor_msgs.PointField(name='intensity', offset=12, datatype=ros_dtype, count=1),
    ]
    return sensor_msgs.PointCloud2(
        header=Header(frame_id=parent_frame, stamp=t_stamp),
        height=1, width=points.shape[0],
        is_dense=True, is_bigendian=False, fields=fields,
        point_step=itemsize * 4, row_step=itemsize * 4 * points.shape[0],
        data=points.astype(np.float32).tobytes(),
    )


def quat_to_yaw(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def rot2d(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def circular_median(angles):
    """角度の中央値。±pi をまたぐ場合に備えて平均方向を基準に畳む"""
    mean_dir = math.atan2(float(np.mean(np.sin(angles))),
                          float(np.mean(np.cos(angles))))
    wrapped = np.arctan2(np.sin(angles - mean_dir), np.cos(angles - mean_dir))
    return mean_dir + float(np.median(wrapped))


def main(args=None):
    rclpy.init(args=args)
    node = RoadsideFollow()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
