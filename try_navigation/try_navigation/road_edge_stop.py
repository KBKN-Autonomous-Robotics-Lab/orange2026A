#!/usr/bin/env python3
"""
road_edge_stop.py
横断歩道の縞（反射強度）から道路端を推定し、停止目標を出力するノード

パイプライン:
  /pcd_segment_ground → odom補正つき短期蓄積(Nフレーム)
    → 強度フィルタ → 円形ROI → grid dedup → DBSCAN
    → 形状ゲート(長さ/幅) → PCA で長軸 v̂
    → 連続性の検証（等間隔性 ＋ 縞どうしの平行性）
    → 縞の手前端から基準線 L → 停止目標距離を算出
    → N回連続一致で停止フラグ

方位について:
  絶対的な向き（進行方向との角度）は問わない。
  代わりに「縞どうしが互いに平行であること」を条件にするため、
  横断歩道を真横から見ても検出できる。
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


def angle_between_axes(a, b):
    """2つの軸方向のなす角[deg]。符号の違いは無視して 0〜90 に畳む"""
    d = abs(float(np.dot(a, b)))
    return math.degrees(math.acos(min(1.0, d)))


# ============================================================
#  メインノード
# ============================================================

class RoadEdgeStop(Node):

    def __init__(self):
        super().__init__('road_edge_stop')

        # ---------- パラメータ ----------
        # 蓄積
        self.declare_parameter('accumulate_frames', 10)
        self.declare_parameter('grid_size', 0.05)
        # 強度・ROI（円形）
        self.declare_parameter('intensity_threshold', 30.0)
        self.declare_parameter('roi_r_min', 0.5)          # 足元を除外 [m]
        self.declare_parameter('roi_r_max', 5.0)          # 強度が持つ範囲 [m]
        # DBSCAN
        self.declare_parameter('dbscan_eps', 0.25)
        self.declare_parameter('dbscan_min_samples', 8)
        # 形状ゲート（向きは問わない）
        self.declare_parameter('min_long_length', 1.5)
        self.declare_parameter('min_short_width', 0.20)
        self.declare_parameter('max_short_width', 0.80)
        # 連続性ゲート
        self.declare_parameter('min_stripe_count', 2)     # 必要な縞の本数
        self.declare_parameter('pitch_min', 0.50)         # 縞間隔の下限 [m]
        self.declare_parameter('pitch_max', 1.50)         # 縞間隔の上限 [m]
        self.declare_parameter('pitch_tolerance', 0.15)   # 間隔のばらつき許容 [m]
        self.declare_parameter('max_parallel_dev_deg', 15.0)  # 縞どうしの平行性 [deg]
        # 停止幾何
        self.declare_parameter('offset_d', 0.30)
        self.declare_parameter('safety_margin', 0.50)
        self.declare_parameter('robot_front_x', 0.45)
        self.declare_parameter('near_edge_percentile', 5.0)
        self.declare_parameter('min_normal_x', 0.20)      # 前方成分の下限（距離換算の暴走防止）
        # 判定
        self.declare_parameter('confirm_count', 3)
        self.declare_parameter('hold_after_stop', False)
        self.declare_parameter('auto_enable', True)
        self.declare_parameter('odom_topic', '/odom/wheel_spimu')
        self.declare_parameter('verbose', True)

        # ---------- QoS ----------
        qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST,
                         reliability=QoSReliabilityPolicy.RELIABLE,
                         durability=QoSDurabilityPolicy.VOLATILE, depth=1)

        # ---------- 状態 ----------
        self.enabled = self.get_parameter('auto_enable').value
        self.stop_latched = False
        self.frame_buf = deque()
        self.hit_count = 0

        self.odom_ok = False
        self.px = self.py = self.pz = 0.0
        self.yaw = 0.0

        # ---------- Sub ----------
        self.create_subscription(sensor_msgs.PointCloud2, '/pcd_segment_ground',
                                 self.cb_ground, qos)
        self.create_subscription(Odometry, self.get_parameter('odom_topic').value,
                                 self.cb_odom, qos)
        self.create_subscription(Bool, '/road_edge_stop/enable', self.cb_enable, qos)

        # ---------- Pub ----------
        self.pub_stop = self.create_publisher(Bool, '/road_edge/stop', qos)
        self.pub_dist = self.create_publisher(Float32, '/road_edge/distance', qos)
        self.pub_cluster = self.create_publisher(sensor_msgs.PointCloud2,
                                                 '/road_edge/cluster_points', qos)
        self.pub_marker = self.create_publisher(Marker, '/road_edge/line_marker', qos)

        self.get_logger().info('road_edge_stop started. enabled=%s' % self.enabled)

    # ------------------------------------------------------------
    #  コールバック
    # ------------------------------------------------------------

    def cb_odom(self, msg):
        p = msg.pose.pose.position
        self.px, self.py, self.pz = p.x, p.y, p.z
        self.yaw = quat_to_yaw(msg.pose.pose.orientation)
        self.odom_ok = True

    def cb_enable(self, msg):
        if msg.data != self.enabled:
            self.get_logger().info('detection %s' % ('ON' if msg.data else 'OFF'))
            self.frame_buf.clear()
            self.hit_count = 0
            self.stop_latched = False
        self.enabled = msg.data

    def cb_ground(self, msg):
        if not self.enabled:
            return
        if not self.odom_ok:
            self.get_logger().warn('waiting for odom', throttle_duration_sec=2.0)
            return
        if self.stop_latched and self.get_parameter('hold_after_stop').value:
            self.pub_stop.publish(Bool(data=True))
            return

        pts = pointcloud2_to_array(msg)
        if pts.shape[1] == 0:
            return

        th = self.get_parameter('intensity_threshold').value
        pts = pts[:, pts[3, :] >= th]
        if pts.shape[1] == 0:
            self._no_detection()
            return

        # local → global
        R = rot2d(self.yaw)
        xy_g = R @ pts[0:2, :] + np.array([[self.px], [self.py]])
        pts_g = np.vstack((xy_g, pts[2:4, :]))

        # 蓄積
        n_frames = int(self.get_parameter('accumulate_frames').value)
        self.frame_buf.append(pts_g)
        while len(self.frame_buf) > n_frames:
            self.frame_buf.popleft()

        buf = np.hstack(list(self.frame_buf))

        # global → local（現在姿勢基準）
        Rt = rot2d(-self.yaw)
        xy_l = Rt @ (buf[0:2, :] - np.array([[self.px], [self.py]]))
        local = np.vstack((xy_l, buf[2:4, :]))

        self.detect(local, msg)

    # ------------------------------------------------------------
    #  検出本体
    # ------------------------------------------------------------

    def detect(self, local, msg):
        gp = self.get_parameter

        # --- 円形 ROI ---
        r = np.hypot(local[0, :], local[1, :])
        m = (r >= gp('roi_r_min').value) & (r <= gp('roi_r_max').value)
        roi = local[:, m]
        if roi.shape[1] < gp('dbscan_min_samples').value:
            self._no_detection()
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

        cands = []
        n_cluster = 0

        for lb in set(labels):
            if lb < 0:
                continue
            n_cluster += 1
            cl = roi[:, labels == lb]
            if cl.shape[1] < 3:
                continue

            st = self.cluster_shape(cl)
            if st is None:
                continue
            length, width, v_hat, dev, near_s, n_hat = st

            # 向きは問わず、寸法のみで足切り
            ok = (length >= gp('min_long_length').value
                  and gp('min_short_width').value <= width <= gp('max_short_width').value)

            if gp('verbose').value:
                self.get_logger().info(
                    '  cl%-2d n=%-4d len=%.2f w=%.2f dev=%.1f near=%.2f %s'
                    % (lb, cl.shape[1], length, width, dev, near_s,
                       'PASS' if ok else 'reject'))
            if ok:
                cands.append(dict(pts=cl, v=v_hat, n=n_hat, near=near_s,
                                  cnt=cl.shape[1],
                                  ctr=cl[0:2, :].mean(axis=1)))

        # --- 連続性（平行性 ＋ 等間隔性）の検証 ---
        chain = self.check_periodicity(cands)

        if not chain:
            if gp('verbose').value:
                self.get_logger().info('frames=%d pts=%d clusters=%d pass=%d chain=0'
                                       % (len(self.frame_buf), roi.shape[1],
                                          n_cluster, len(cands)))
            self._no_detection()
            return

        head = chain[0]                       # 最も手前の縞
        near_s, v_hat, n_hat = head['near'], head['v'], head['n']

        # --- 停止目標距離 ---
        nx = n_hat[0]
        if abs(nx) < gp('min_normal_x').value:
            # 縞が真横を向いている等、前方距離に換算できない配置
            if gp('verbose').value:
                self.get_logger().info('skip: normal_x=%.2f too small' % nx)
            self._no_detection()
            return  # 検出はできているが前方距離に換算できない配置
        # 引き算は法線方向（縞に垂直）で行い、最後に前方距離へ換算する
        s_edge = near_s - gp('offset_d').value
        s_stop = s_edge - gp('safety_margin').value
        x_L = near_s / nx
        x_edge = s_edge / nx
        remaining = s_stop / nx - gp('robot_front_x').value

        # --- 確定判定 ---
        if remaining <= 0.0:
            self.hit_count += 1
        else:
            self.hit_count = 0

        stop = self.hit_count >= int(gp('confirm_count').value)
        if stop and not self.stop_latched:
            self.stop_latched = True
            self.get_logger().warn('STOP: road edge at %.2f m (L=%.2f)' % (x_edge, x_L))

        if gp('verbose').value:
            gaps = [chain[i + 1]['s'] - chain[i]['s'] for i in range(len(chain) - 1)]
            gs = ' '.join('%.2f' % v for v in gaps) if gaps else '-'
            self.get_logger().info(
                'frames=%d pts=%d clusters=%d pass=%d chain=%d gaps=[%s] | L=%.2f edge=%.2f rem=%.2f hit=%d'
                % (len(self.frame_buf), roi.shape[1], n_cluster, len(cands),
                   len(chain), gs, x_L, x_edge, remaining, self.hit_count))

        self.pub_dist.publish(Float32(data=float(remaining)))
        self.pub_stop.publish(Bool(data=bool(stop)))

        # --- 採用した縞群を単色で publish ---
        allpts = np.hstack([c['pts'] for c in chain]).copy()
        allpts[3, :] = 100.0
        self.pub_cluster.publish(
            point_cloud_intensity_msg(allpts.T, msg.header.stamp, msg.header.frame_id))

        t_c = float(v_hat @ head['ctr'])
        self.publish_marker(near_s, v_hat, n_hat, t_c, msg)

    # ------------------------------------------------------------
    #  連続性の検証（平行性 ＋ 等間隔性）
    # ------------------------------------------------------------

    def check_periodicity(self, cands):
        """互いに平行で等間隔に並ぶ縞の連なりを探し、手前から順に返す"""
        gp = self.get_parameter
        need = int(gp('min_stripe_count').value)
        if len(cands) == 0:
            return []

        # 点数が最も多いクラスタを基準にする
        base = max(cands, key=lambda c: c['cnt'])
        ref_v = base['v']
        ref_n = base['n']

        # 基準と平行なものだけ残す
        pdev = gp('max_parallel_dev_deg').value
        para = [c for c in cands if angle_between_axes(c['v'], ref_v) <= pdev]
        if len(para) == 0:
            return []

        # 共通の法線で重心を投影して並べる
        for c in para:
            c['s'] = float(ref_n @ c['pts'][0:2, :].mean(axis=1))
        para = sorted(para, key=lambda c: c['s'])

        if need <= 1:
            return [para[0]]
        if len(para) < need:
            return []

        pmin = gp('pitch_min').value
        pmax = gp('pitch_max').value
        tol = gp('pitch_tolerance').value

        best = []
        for i in range(len(para)):
            chain = [para[i]]
            gaps = []
            for j in range(i + 1, len(para)):
                gap = para[j]['s'] - chain[-1]['s']
                if gap < pmin:
                    continue
                if gap > pmax:
                    break
                if gaps and abs(gap - float(np.mean(gaps))) > tol:
                    break
                chain.append(para[j])
                gaps.append(gap)
            if len(chain) > len(best):
                best = chain

        return best if len(best) >= need else []

    # ------------------------------------------------------------
    #  クラスタ形状（PCA）
    # ------------------------------------------------------------

    def cluster_shape(self, cl):
        xy = cl[0:2, :]
        mean = xy.mean(axis=1, keepdims=True)
        c = xy - mean
        cov = (c @ c.T) / max(c.shape[1] - 1, 1)
        w, V = np.linalg.eigh(cov)
        v_hat = V[:, 1]
        u_hat = V[:, 0]

        pl = v_hat @ c
        ps = u_hat @ c
        length = float(pl.max() - pl.min())
        width = float(ps.max() - ps.min())
        if length < 1e-6:
            return None

        ang = math.degrees(math.acos(min(1.0, abs(float(v_hat[0])))))
        dev = abs(90.0 - ang)          # ログ表示用（判定には使わない）

        n_hat = u_hat if u_hat[0] > 0 else -u_hat
        s = n_hat @ xy
        near_s = float(np.percentile(s, self.get_parameter('near_edge_percentile').value))

        return length, width, v_hat, dev, near_s, n_hat

    # ------------------------------------------------------------
    #  補助
    # ------------------------------------------------------------

    def _no_detection(self):
        self.hit_count = 0
        self.pub_stop.publish(Bool(data=False))
        self.pub_dist.publish(Float32(data=float('nan')))
        self.clear_marker()
        self.clear_cluster()

    def clear_cluster(self):
        empty = np.zeros((0, 4), dtype=np.float32)
        msg = point_cloud_intensity_msg(empty, self.get_clock().now().to_msg(), 'odom')
        self.pub_cluster.publish(msg)

    def clear_marker(self):
        mk = Marker()
        mk.ns = 'road_edge'
        mk.id = 0
        mk.action = Marker.DELETE
        self.pub_marker.publish(mk)

    def publish_marker(self, near_s, v_hat, n_hat, t_c, msg):
        """L・道路端・停止目標・許容帯を1つの LINE_LIST で表示
        すべて法線方向の距離で扱うため、姿勢によらず間隔は一定"""
        gp = self.get_parameter
        s_edge = near_s - gp('offset_d').value
        s_stop = s_edge - gp('safety_margin').value
        s_zone = s_edge - 1.5
        half = 2.0

        def at(s_val, t):
            p = n_hat * s_val + v_hat * (t_c + t)
            return Point(x=float(p[0]), y=float(p[1]), z=0.0)

        pts = []
        pts += [at(near_s, -half), at(near_s, half)]
        pts += [at(s_edge, -half), at(s_edge, half)]
        pts += [at(s_stop, -half), at(s_stop, half)]
        pts += [at(s_zone, -half), at(s_zone, half)]
        pts += [at(s_zone, -half), at(s_edge, -half)]
        pts += [at(s_zone,  half), at(s_edge,  half)]
        pts += [at(near_s, 0.0), at(s_stop, 0.0)]

        mk = Marker()
        mk.header.frame_id = msg.header.frame_id
        mk.header.stamp = msg.header.stamp
        mk.ns = 'road_edge'
        mk.id = 0
        mk.type = Marker.LINE_LIST
        mk.action = Marker.ADD
        mk.scale.x = 0.05
        mk.color.r, mk.color.g, mk.color.b, mk.color.a = 0.1, 0.9, 0.5, 1.0
        mk.pose.orientation.w = 1.0
        mk.points = pts
        self.pub_marker.publish(mk)


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