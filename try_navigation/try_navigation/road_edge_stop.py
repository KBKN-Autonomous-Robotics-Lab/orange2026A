#!/usr/bin/env python3
"""
road_edge_stop.py
横断歩道の縞（反射強度）から道路端を推定し、停止目標を出力するノード

パイプライン:
  /pcd_segment_ground → odom補正つき短期蓄積(Nフレーム)
    → 強度フィルタ → ROI → grid dedup → DBSCAN
    → 形状ゲート(長さ/幅/向き) → PCA で長軸 v̂
    → 縞の手前端から基準線 L → 停止目標距離を算出
    → N回連続一致で停止フラグ

状態:
  /road_edge_stop/enable (Bool) が True の間だけ蓄積・判定を行う。
  停止フラグが立ったら enable が False に戻るまで判定を無効化する。

可視化は viz_frame（既定 livox_frame）のローカル座標で publish する。
蓄積時のみ odom を経由するが、判定・表示は現在姿勢基準に戻した座標で行う。
"""

import math
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSDurabilityPolicy,
                       QoSHistoryPolicy, QoSReliabilityPolicy)

import sensor_msgs.msg as sensor_msgs
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32, Header
from visualization_msgs.msg import Marker

from sklearn.cluster import DBSCAN


# ============================================================
#  ユーティリティ
# ============================================================

def pointcloud2_to_array(cloud_msg):
    """sensor_msgs/PointCloud2 → np.ndarray [4, N] (x, y, z, intensity)"""
    pts = np.frombuffer(cloud_msg.data, dtype=np.uint8).reshape(-1, cloud_msg.point_step)
    x = np.frombuffer(pts[:, 0:4].tobytes(),   dtype=np.float32)
    y = np.frombuffer(pts[:, 4:8].tobytes(),   dtype=np.float32)
    z = np.frombuffer(pts[:, 8:12].tobytes(),  dtype=np.float32)
    i = np.frombuffer(pts[:, 12:16].tobytes(), dtype=np.float32)
    return np.vstack((x, y, z, i))


def point_cloud_intensity_msg(points, t_stamp, parent_frame):
    """np.ndarray [N, 4] → sensor_msgs/PointCloud2"""
    ros_dtype = sensor_msgs.PointField.FLOAT32
    itemsize = np.dtype(np.float32).itemsize
    fields = [
        sensor_msgs.PointField(name='x',         offset=0,  datatype=ros_dtype, count=1),
        sensor_msgs.PointField(name='y',         offset=4,  datatype=ros_dtype, count=1),
        sensor_msgs.PointField(name='z',         offset=8,  datatype=ros_dtype, count=1),
        sensor_msgs.PointField(name='intensity', offset=12, datatype=ros_dtype, count=1),
    ]
    header = Header(frame_id=parent_frame, stamp=t_stamp)
    return sensor_msgs.PointCloud2(
        header=header, height=1, width=points.shape[0],
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


# ============================================================
#  メインノード
# ============================================================

class RoadEdgeStop(Node):

    def __init__(self):
        super().__init__('road_edge_stop')

        # ---------- パラメータ ----------
        # 蓄積
        self.declare_parameter('accumulate_frames', 5)      # 蓄積フレーム数
        self.declare_parameter('grid_size', 0.05)           # 重複除去グリッド [m]
        # 強度・ROI
        self.declare_parameter('intensity_threshold', 30.0)
        self.declare_parameter('roi_x_min', 1.0)            # [m] 足元の反射物を除外
        self.declare_parameter('roi_x_max', 8.0)
        self.declare_parameter('roi_y_abs', 3.0)
        # DBSCAN
        self.declare_parameter('dbscan_eps', 0.25)
        self.declare_parameter('dbscan_min_samples', 8)
        # 形状ゲート
        self.declare_parameter('min_long_length', 1.5)      # 長軸長さ下限 [m]
        self.declare_parameter('min_short_width', 0.20)     # 短軸幅 下限 [m]
        self.declare_parameter('max_short_width', 0.80)     # 短軸幅 上限 [m]
        self.declare_parameter('max_angle_dev_deg', 30.0)   # 進行方向との直交からのズレ許容 [deg]
        # 停止幾何
        self.declare_parameter('offset_d', 0.30)            # 縞手前端→道路端 [m] 地点ごとに実測
        self.declare_parameter('safety_margin', 0.50)       # 道路端からの余裕 [m]
        self.declare_parameter('robot_front_x', 0.45)       # base原点→前端 [m]
        self.declare_parameter('near_edge_percentile', 5.0)
        # 判定
        self.declare_parameter('confirm_count', 3)          # N回連続一致で確定
        self.declare_parameter('hold_after_stop', True)
        self.declare_parameter('auto_enable', False)        # ベンチ試験用
        self.declare_parameter('verbose', True)
        # 可視化
        self.declare_parameter('viz_frame', 'livox_frame')

        # ---------- QoS ----------
        qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST,
                         reliability=QoSReliabilityPolicy.RELIABLE,
                         durability=QoSDurabilityPolicy.VOLATILE, depth=1)

        # ---------- 状態 ----------
        self.enabled = self.get_parameter('auto_enable').value
        self.viz_frame = self.get_parameter('viz_frame').value
        self.stop_latched = False
        self.frame_buf = deque()          # 各要素: np.ndarray [4, N] (global xy, z, intensity)
        self.hit_count = 0

        self.odom_ok = False
        self.px = self.py = self.pz = 0.0
        self.yaw = 0.0

        # ---------- Sub ----------
        self.create_subscription(sensor_msgs.PointCloud2, '/pcd_segment_ground',
                                 self.cb_ground, qos)
        self.create_subscription(Odometry, '/odom/wheel_spimu', self.cb_odom, qos)
        self.create_subscription(Bool, '/road_edge_stop/enable', self.cb_enable, qos)

        # ---------- Pub ----------
        self.pub_stop = self.create_publisher(Bool, '/road_edge/stop', qos)
        self.pub_dist = self.create_publisher(Float32, '/road_edge/distance', qos)
        self.pub_cluster = self.create_publisher(sensor_msgs.PointCloud2,
                                                 '/road_edge/cluster_points', qos)
        self.pub_marker = self.create_publisher(Marker, '/road_edge/line_marker', qos)

        self.get_logger().info('road_edge_stop started. enabled=%s viz_frame=%s'
                               % (self.enabled, self.viz_frame))

    # ------------------------------------------------------------
    #  コールバック
    # ------------------------------------------------------------

    def cb_odom(self, msg):
        p = msg.pose.pose.position
        self.px, self.py, self.pz = p.x, p.y, p.z
        self.yaw = quat_to_yaw(msg.pose.pose.orientation)
        self.odom_ok = True

    def cb_enable(self, msg):
        if msg.data and not self.enabled:
            self.get_logger().info('enabled: buffer cleared, detection ON')
        if (not msg.data) and self.enabled:
            self.get_logger().info('disabled: detection OFF')
        if msg.data != self.enabled:
            self.frame_buf.clear()
            self.hit_count = 0
            self.stop_latched = False
        self.enabled = msg.data

    def cb_ground(self, msg):
        if not self.enabled:
            return
        if not self.odom_ok:
            self.get_logger().warn('waiting for /odom/wheel_spimu', throttle_duration_sec=2.0)
            return
        if self.stop_latched and self.get_parameter('hold_after_stop').value:
            self.pub_stop.publish(Bool(data=True))
            return

        pts = pointcloud2_to_array(msg)          # [4, N] local
        if pts.shape[1] == 0:
            return

        # --- 強度フィルタ（蓄積前に落として軽量化） ---
        th = self.get_parameter('intensity_threshold').value
        pts = pts[:, pts[3, :] >= th]
        if pts.shape[1] == 0:
            self._no_detection(msg)
            return

        # --- local → global（蓄積のためだけに一度グローバルへ） ---
        xy_g = rot2d(self.yaw) @ pts[0:2, :] + np.array([[self.px], [self.py]])
        pts_g = np.vstack((xy_g, pts[2:4, :]))

        # --- 蓄積 ---
        n_frames = int(self.get_parameter('accumulate_frames').value)
        self.frame_buf.append(pts_g)
        while len(self.frame_buf) > n_frames:
            self.frame_buf.popleft()

        buf = np.hstack(list(self.frame_buf))

        # --- global → local（現在姿勢基準に戻す。以降すべてローカル座標） ---
        xy_l = rot2d(-self.yaw) @ (buf[0:2, :] - np.array([[self.px], [self.py]]))
        local = np.vstack((xy_l, buf[2:4, :]))   # [4, M]

        self.detect(local, msg)

    # ------------------------------------------------------------
    #  検出本体
    # ------------------------------------------------------------

    def detect(self, local, msg):
        gp = self.get_parameter

        # --- ROI ---
        x, y = local[0, :], local[1, :]
        m = (x >= gp('roi_x_min').value) & (x <= gp('roi_x_max').value) \
            & (np.abs(y) <= gp('roi_y_abs').value)
        roi = local[:, m]
        if roi.shape[1] < gp('dbscan_min_samples').value:
            self._no_detection(msg)
            return

        # --- grid dedup ---
        g = gp('grid_size').value
        key = np.round(roi[0:2, :] / g).astype(np.int64)
        _, uniq = np.unique(key, axis=1, return_index=True)
        roi = roi[:, uniq]

        # --- DBSCAN ---
        labels = DBSCAN(eps=gp('dbscan_eps').value,
                        min_samples=int(gp('dbscan_min_samples').value)
                        ).fit_predict(roi[0:2, :].T)

        best = None          # (near_s, v_hat, n_hat, cluster_pts)
        passed = []          # 形状ゲートを通過したクラスタ [4, n] のリスト
        n_cluster = 0

        for lb in set(labels):
            if lb < 0:
                continue
            n_cluster += 1
            cl = roi[:, labels == lb]
            if cl.shape[1] < 3:
                continue

            stats = self.cluster_shape(cl)
            if stats is None:
                continue
            length, width, v_hat, ang_perp_dev, near_s, n_hat = stats

            ok = (length >= gp('min_long_length').value
                  and gp('min_short_width').value <= width <= gp('max_short_width').value
                  and ang_perp_dev <= gp('max_angle_dev_deg').value)

            if gp('verbose').value:
                self.get_logger().info(
                    '  cl%-2d n=%-4d len=%.2f w=%.2f dev=%.1f near=%.2f %s'
                    % (lb, cl.shape[1], length, width, ang_perp_dev, near_s,
                       'PASS' if ok else 'reject'))

            if not ok:
                continue
            passed.append(cl)
            if best is None or near_s < best[0]:
                best = (near_s, v_hat, n_hat, cl)

        # --- 通過クラスタを全部 publish（intensity = 通し番号）---
        self.publish_clusters(passed, msg)

        if best is None:
            if gp('verbose').value:
                self.get_logger().info('frames=%d pts=%d clusters=%d pass=0'
                                       % (len(self.frame_buf), roi.shape[1], n_cluster))
            self._no_detection(msg, keep_cluster=True)
            return

        near_s, v_hat, n_hat, cl = best

        # --- 停止目標距離 ---
        # 基準線 L : { p | p·n̂ = near_s }。ロボット前方軸との交点 x
        nx = n_hat[0]
        if abs(nx) < 1e-3:
            self._no_detection(msg, keep_cluster=True)
            return
        x_edge_line = near_s / nx                       # L までの前方距離
        x_road_edge = x_edge_line - gp('offset_d').value
        remaining = (x_road_edge
                     - gp('safety_margin').value
                     - gp('robot_front_x').value)

        # --- 確定判定 ---
        if remaining <= 0.0:
            self.hit_count += 1
        else:
            self.hit_count = 0

        stop = self.hit_count >= int(gp('confirm_count').value)
        if stop and not self.stop_latched:
            self.stop_latched = True
            self.get_logger().warn('STOP: road edge at %.2f m (L=%.2f, d=%.2f)'
                                   % (x_road_edge, x_edge_line, gp('offset_d').value))

        if gp('verbose').value:
            self.get_logger().info(
                'frames=%d pts=%d clusters=%d pass=%d | L=%.2f edge=%.2f rem=%.2f hit=%d'
                % (len(self.frame_buf), roi.shape[1], n_cluster, len(passed),
                   x_edge_line, x_road_edge, remaining, self.hit_count))

        self.pub_dist.publish(Float32(data=float(remaining)))
        self.pub_stop.publish(Bool(data=bool(stop)))
        self.publish_marker(near_s, v_hat, n_hat, msg)

    # ------------------------------------------------------------
    #  クラスタ形状（PCA）
    # ------------------------------------------------------------

    def cluster_shape(self, cl):
        """戻り値: (長軸長さ, 短軸幅, v̂, 直交からのズレ[deg], 手前端 s, n̂)"""
        xy = cl[0:2, :]
        mean = xy.mean(axis=1, keepdims=True)
        c = xy - mean
        cov = (c @ c.T) / max(c.shape[1] - 1, 1)
        w, V = np.linalg.eigh(cov)          # 昇順
        v_hat = V[:, 1]                     # 第1主成分 = 長軸
        u_hat = V[:, 0]                     # 短軸

        pl = v_hat @ c
        ps = u_hat @ c
        length = float(pl.max() - pl.min())
        width = float(ps.max() - ps.min())
        if length < 1e-6:
            return None

        # 進行方向(x軸)と長軸のなす角。90°に近いほど良い
        ang = math.degrees(math.acos(min(1.0, abs(float(v_hat[0])))))
        dev = abs(90.0 - ang)

        # 手前端: 前方を向く法線 n̂ への射影の下側パーセンタイル
        n_hat = u_hat if u_hat[0] > 0 else -u_hat
        s = n_hat @ xy
        near_s = float(np.percentile(s, self.get_parameter('near_edge_percentile').value))

        return length, width, v_hat, dev, near_s, n_hat

    # ------------------------------------------------------------
    #  可視化（すべてローカル座標）
    # ------------------------------------------------------------

    def publish_clusters(self, passed, msg):
        """形状ゲート通過クラスタを全部 publish。intensity をクラスタ通し番号に置換"""
        if not passed:
            empty = np.zeros((0, 4), dtype=np.float32)
            self.pub_cluster.publish(
                point_cloud_intensity_msg(empty, msg.header.stamp, self.viz_frame))
            return

        out = []
        for k, cl in enumerate(passed):
            idx = np.full(cl.shape[1], float(k), dtype=np.float32)
            out.append(np.vstack((cl[0:3, :], idx)))
        pts = np.hstack(out).T                       # [M, 4]
        self.pub_cluster.publish(
            point_cloud_intensity_msg(pts, msg.header.stamp, self.viz_frame))

    def publish_marker(self, near_s, v_hat, n_hat, msg):
        """L・垂直線・停止目標線・停止許容帯を1つの LINE_LIST で表示"""
        gp = self.get_parameter

        x_L = near_s / n_hat[0]
        x_edge = x_L - gp('offset_d').value
        x_stop = x_edge - gp('safety_margin').value
        x_zone = x_edge - 1.5                      # 規約の許容帯 手前1.5m
        half = 2.0                                 # 線分の半長 [m]

        def at(xf, t):
            """ロボット前方距離 xf の位置で、v̂ 方向に t ずれた点"""
            return n_hat * (xf * n_hat[0]) + v_hat * t

        seg = []
        seg += [at(x_L, -half), at(x_L, half)]                  # L（縞の手前端）
        seg += [at(x_edge, -half), at(x_edge, half)]            # 道路端の推定
        seg += [at(x_stop, -half), at(x_stop, half)]            # 停止目標線
        seg += [at(x_zone, -half), at(x_zone, half)]            # 許容帯の手前側
        seg += [at(x_zone, -half), at(x_edge, -half)]           # 許容帯 側辺
        seg += [at(x_zone,  half), at(x_edge,  half)]
        seg += [at(x_L, 0.0), at(x_stop, 0.0)]                  # 中央の垂直線

        mk = Marker()
        mk.header.frame_id = self.viz_frame
        mk.header.stamp = msg.header.stamp
        mk.ns = 'road_edge'
        mk.id = 0
        mk.type = Marker.LINE_LIST
        mk.action = Marker.ADD
        mk.scale.x = 0.05
        mk.color.r, mk.color.g, mk.color.b, mk.color.a = 0.1, 0.9, 0.5, 1.0
        mk.pose.orientation.w = 1.0
        mk.points = [Point(x=float(p[0]), y=float(p[1]), z=0.0) for p in seg]
        self.pub_marker.publish(mk)

    def clear_marker(self, msg):
        mk = Marker()
        mk.header.frame_id = self.viz_frame
        mk.header.stamp = msg.header.stamp
        mk.ns = 'road_edge'
        mk.id = 0
        mk.action = Marker.DELETE
        self.pub_marker.publish(mk)

    # ------------------------------------------------------------
    #  補助
    # ------------------------------------------------------------

    def _no_detection(self, msg, keep_cluster=False):
        self.hit_count = 0
        self.pub_stop.publish(Bool(data=False))
        self.pub_dist.publish(Float32(data=float('nan')))
        self.clear_marker(msg)
        if not keep_cluster:
            empty = np.zeros((0, 4), dtype=np.float32)
            self.pub_cluster.publish(
                point_cloud_intensity_msg(empty, msg.header.stamp, self.viz_frame))


def main(args=None):
    rclpy.init(args=args)
    node = RoadEdgeStop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()