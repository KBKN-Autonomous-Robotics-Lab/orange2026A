#!/usr/bin/env python3
"""地図座標系の自己位置 EKF（マッチング＋GNSS）: ekf_gnss_match.

旧 ekf_myself_gps.py（/fusion/odom を出す EKF）と同じ構成・関数名で書いてある。
旧から変えたところには「変更点:」と書いた。

旧 → 新 の対応
  __init__             パラメータを YAML から読む（旧は直書き）。購読は 2本 → 7本
  sensor_a_callback    /odom/combine の姿勢を保存するだけ（旧はここで Speed を計算）
  sensor_b_callback    /odom/UM982。σ²（cov[7]）と pos_type も保存。同じ座標の再送は捨てる
  sensor_c_callback    （新）/odom_ekf_match（マッチング）
  determination_of_R   GNSS 無効区間（gps_rr_flag）と R1, R2（GNSS）、R_match（マッチング）
  initialize_odomA     XX と P の初期化
  KalfXY               予測だけ。毎周期呼び、P も伝播する
  KalfGPSXY            GNSS の更新だけ（χ²検定つき）
  KalfMatchXY          （新）マッチングの更新
  update_yaw_offset    GNSS 方位とのずれ（旧は publish_fused_value の中にあった）
  publish_fused_value  0.1 秒タイマー: R → 方位のずれ → 予測 → 更新 → 配信
  削除                 initialize, initializeGPS, sensim0, combine_yaw, calculate_offset,
                       rotation_xyz（旧でも使われていない）

入力
  /odom/combine    予測（移動量）
  /odom_ekf_match  マッチング（地図座標系の位置。常に使う）
  /odom/UM982      GNSS（cov[0]=衛星数, cov[7]=σ² [m²]。0 なら σ なし）
  /gps_raw         pos_type（BESTNAVA、無ければ GGA の quality）
  /odom_ref_slam   マッチング成功の時刻（match_r_inflate 用）
  /waypoint_number GNSS 無効区間（ウェイポイント番号）用
  /occlusion_rate  遮蔽率（r_mode=occ のとき）
出力
  /fusion/odom_gnss_match         ナビ用（旧の /fusion/odom はそのまま）
  /fusion/odom_gnss_match/debug   Float64MultiArray（項目名は layout.dim[0].label）
パラメータは config/ekf_gnss_match.yaml（インストール先）を起動時に自動で読む。-p や launch の指定が優先。
  ros2 run try_navigation ekf_gnss_match --ros-args -p use_sim_time:=true
"""
from collections import deque
import math
import os
import re

from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Float32, Float64MultiArray, Int32, MultiArrayDimension, String
import tf2_ros
import yaml

# ======================================================================
# パラメータの既定値（YAML に無い項目はこの値）。self.<名前> に入る
# ======================================================================
PARAM_DEFAULTS = {
    # ---- トピック ----
    'odom_topic': '/odom/combine',
    'match_topic': '/odom_ekf_match',
    'ref_slam_topic': '/odom_ref_slam',
    'gnss_topic': '/odom/UM982',
    'gnss_raw_topic': '/gps_raw',
    'occ_topic': '/occlusion_rate',
    'waypoint_topic': '/waypoint_number',
    'output_topic': '/fusion/odom_gnss_match',
    'frame_id': 'odom',
    'child_frame_id': 'base_footprint',
    'ekf_publish_TF': False,          # 旧と同じ名前（旧の既定は True。二重にならないよう False）
    'use_pos_type': True,
    'status_log_sec': 5.0,
    # ---- 使う観測（研究の比較もここで切り替え）----
    'use_gnss': True,
    'use_slam': True,
    'use_gnss_heading': True,
    'use_bias': False,                # GNSS のずれ [bx, by] も推定する
    'r_mode': 'cov7',                 # fixed / cov7 / occ
    # ---- 予測（/odom/combine）----
    'kalf_speed_param': 1.0,          # 移動量の倍率（旧と同じ名前。旧は GPS あり 1.05 / なし 1.00）
    'odom_jump_thresh': 1.0,          # 1 周期の移動量がこれを超えたら予測しない [m]
    'q_dist_ratio': 0.05,             # 移動量に比例するσ [-]
    'q_dist_rw': 0.05,                # 走行距離に対するランダムウォーク [m/sqrt(m)]
    'q_pos_time': 0.01,               # 時間に対するランダムウォーク [m/sqrt(s)]
    'q_yaw_ratio': 0.02,              # 旋回角に比例するσ [-]
    'q_yaw_time': 0.001,              # 方位のランダムウォーク [rad/sqrt(s)]
    'q_bias_dist': 0.01,              # バイアスの変化（距離）[m/sqrt(m)]
    'q_bias_time': 0.001,             # バイアスの変化（時間）[m/sqrt(s)]
    'init_pos_sigma': 1.0,
    'init_yaw_sigma_deg': 1.0,
    'init_bias_sigma': 1.0,
    # ---- マッチング ----
    'match_sigma': 0.2,
    'match_gate_chi2': 13.8,          # 自由度2・99.9%。0 以下で検定なし
    'match_reaccept_count': 30,
    'match_r_inflate': False,         # 最後の /odom_ref_slam からの経過時間で R を大きくする
    'match_sigma_growth': 0.05,       # [m/s]
    'match_sigma_max': 2.0,
    'match_inflate_grace': 1.0,       # [s]
    # ---- GNSS ----
    'gnss_sigma_min': 0.05,
    'gnss_sigma_fixed': 0.17,         # r_mode=fixed のσ（旧EKFの良好区間の値）
    'gnss_sigma_fix': 0.05,           # 種類ごとのσの下限（σ が無いときはこの値）
    'gnss_sigma_float': 0.5,
    'gnss_sigma_ppp': 0.2,
    'gnss_sigma_unknown': 0.17,       # 種類も σ も分からないとき
    'gnss_accept_types': ['FIX', 'FLOAT', 'PPP', 'UNKNOWN'],   # 種類か pos_type 名
    'gnss_raw_timeout': 0.5,
    'gnss_gate_chi2': 13.8,
    'gnss_reaccept_count': 30,
    'gnss_reaccept_sigma_max': 0.1,
    'gnss_antenna_x': 0.0,            # アンテナ位置（ロボット座標、前方+）[m]
    'gnss_antenna_y': 0.0,            # （左+）[m]
    'occ_a': 0.05,                    # r_mode=occ: σ = clamp(a + b*occ)（仮値・要較正）
    'occ_b': 1.0,
    'occ_sigma_min': 0.05,
    'occ_sigma_max': 5.0,
    'occ_timeout': 1.0,
    # ---- 受け入れ直し（共通）----
    'reaccept_spread': 0.3,           # 除外したイノベーションのばらつきの上限 [m]
    'reaccept_block_sec': 2.0,        # GNSS を使っている間はマッチングの受け入れ直しをしない
    # ---- 方位のずれ（GNSS 方位 − combine 方位）----
    'heading_accept_types': ['FIX'],
    'heading_sigma_max': 0.05,
    'set_yaw_satellites_no': 4,       # 旧と同じ名前（pos_type が分からないときの条件）
    'set_yaw_count': 100,             # 旧と同じ名前（この数たまったら offsetyaw を更新）
    'theta_diff_thresh': 40.0,        # 旧と同じ名前（これ以上曲がったらサンプルを捨てる）[deg]
    'heading_reset_sec': 2.0,         # 有効なサンプルがこの秒数来なければ捨てる（旧は 20 周期）
    # ---- GNSS 無効区間（運用の保険。既定はなし）----
    'gnss_off_rects': [],             # [xmin, xmax, ymin, ymax, ...]
    'gnss_off_waypoints': [],         # [start, end, ...]
}

# 観測ごとの処理結果（debug トピックに出す）
STATUS_USED = 1          # 使った
STATUS_REACCEPTED = 2    # 連続で除外したあと受け入れ直した
STATUS_GATED = -1        # χ²検定で除外
STATUS_ZONE = -2         # GNSS 無効区間
STATUS_TYPE = -3         # pos_type が対象外
STATUS_DUP = -4          # 同じエポック（同じ座標）の再送
STATUS_DISABLED = -5     # use_gnss が false
STATUS_NOT_INIT = -6     # 初期化前

CATEGORY_CODE = {
    'UNKNOWN': 0, 'FIX': 1, 'FLOAT': 2, 'PPP': 3, 'PPP_CONV': 4,
    'DGPS': 5, 'SINGLE': 6, 'NONE': 7,
}

DEBUG_FIELDS = [
    'gnss_status', 'gnss_d2', 'gnss_innov_x', 'gnss_innov_y', 'gnss_sigma',
    'gnss_category', 'gnss_reject_count',
    'match_status', 'match_d2', 'match_innov_x', 'match_innov_y', 'match_sigma',
    'match_reject_count',
    'bias_x', 'bias_y', 'bias_sigma', 'pos_sigma',
    'yaw_offset', 'ekf_yaw', 'heading_samples', 'heading_updates',
    'in_gnss_off_zone', 'waypoint', 'time_since_ref_slam', 'distance',
    'gnss_dup_count', 'odom_jump_count',
]

TIMER_PERIOD = 0.1   # publish_fused_value の周期 [s]（旧と同じ）


class ExtendedKalmanFilter(Node):
    def __init__(self):
        # 変更点: ノード名は 'ekf_gnss_match'（旧の 'sensor_fusion' と同時に動かすため）
        super().__init__('ekf_gnss_match')

        # ---------------- パラメータ ----------------
        # 変更点: 旧は値をコードに直書き → YAML から読む。
        # PARAM_DEFAULTS と同じ名前で self に入る（例: self.match_sigma, self.kalf_speed_param）。
        # 既定値 = YAML の値（無い項目は PARAM_DEFAULTS）。-p / launch の指定はさらに優先される
        file_params = load_default_params(self.get_logger())
        for name, default in PARAM_DEFAULTS.items():
            init = coerce_param(default, file_params.get(name, default))
            self.declare_parameter(name, init, ParameterDescriptor(dynamic_typing=True))
            setattr(self, name, coerce_param(default, self.get_parameter(name).value))
        unknown = sorted(set(file_params) - set(PARAM_DEFAULTS))
        if unknown:
            self.get_logger().warn('YAML に知らない項目がある（名前の変更漏れ？）: %s'
                                   % ', '.join(unknown))
        if self.r_mode not in ('fixed', 'cov7', 'occ'):
            self.get_logger().warn('r_mode %r は不明 → cov7 として扱う' % self.r_mode)
            self.r_mode = 'cov7'
        if len(self.gnss_off_rects) % 4:
            self.get_logger().warn('gnss_off_rects は 4 個ずつ（xmin, xmax, ymin, ymax）。端数は無視')
        if len(self.gnss_off_waypoints) % 2:
            self.get_logger().warn('gnss_off_waypoints は 2 個ずつ（start, end）。端数は無視')

        # ---------------- EKF ----------------
        self.n_state = 6 if (self.use_bias and self.use_slam) else 4
        self.XX = None   # 状態 [x, y, cosθ, sinθ]（旧と同じ。use_bias のときは [.., bx, by]）
        self.P = None    # 誤差共分散
        self.Q = None    # プロセスノイズ（変更点: 旧は 2x2 [速度, 方位] → 3x3 [前後, 左右, 方位]）
        self.H = None    # 観測行列
        self.R = None    # 観測ノイズ（2x2）
        self.R1 = 0.0    # GNSS の x の分散（旧と同じ）
        self.R2 = 0.0    # GNSS の y の分散（旧と同じ）
        self.R_match = 0.0   # （新）マッチングの分散
        self.is_initialized = False

        # ---------------- /odom/combine（sensor_a）----------------
        self.GTheta = None       # combine の方位
        self.GTheta0 = None      # 前回の予測で使った robot_yaw（DTheta の計算用。旧と同じ）
        self.SmpTime = TIMER_PERIOD
        self.odom_xy = None      # （新）combine の最新の位置
        self.odom_stamp = None
        self.odom_twist = None
        self.odom_new = False    # 新しい combine が届いたか（旧: Speed が None でないか）
        self.prev_odom = None    # 前回の予測で使った combine（時刻, x, y, 方位）
        self.distance = 0.0

        # ---------------- /odom/UM982（sensor_b）----------------
        self.GpsXY = None        # 新しく届いた GNSS の位置（使ったら None に戻す）
        self.GPSYaw = 0.0
        self.Number_of_satellites = 0
        self.GpsVar = 0.0        # （新）cov[7] の σ²
        self.GpsInfo = None      # （新）同じエポックの /gps_raw（区分, pos_type, σ）
        self.gps_time = None
        self.last_gps_xy = None  # （新）同じ座標の再送の判定用

        # ---------------- /odom_ekf_match（sensor_c）ほか ----------------
        self.MatchXY = None      # 新しく届いたマッチングの位置（使ったら None に戻す）
        self.raw_info = None     # 最新の /gps_raw（区分, pos_type, σ）
        self.raw_time = None
        self.occ_value = None
        self.occ_time = None
        self.waypoint = None
        self.last_ref_slam_t = None
        self.start_time = None

        # ---------------- 方位のずれ（旧と同じ名前）----------------
        self.offsetyaw = 0.0            # GNSS 方位 − combine 方位（中央値）
        self.robot_yaw = 0.0            # 出力する方位 = combine の方位 + offsetyaw
        self.diff_yaw_buff = []         # ずれのサンプル
        self.GPS_angle_conut = 0        # サンプルの数
        self.GPS_angle_reset_count = 0  # 有効なサンプルが来なかった周期の数
        self.theta_buff = None          # 直進判定の基準の方位
        self.yaw_update_count = 0       # （新）offsetyaw を更新した回数

        # ---------------- GNSS 無効区間 ----------------
        self.gps_rr_flag = 1            # 1: GNSS を使う, 0: 無効区間（旧と同じ意味）
        self.off_rects = parse_rects(self.gnss_off_rects)
        self.off_waypoints = parse_ranges(self.gnss_off_waypoints)

        # ---------------- χ²検定・受け入れ直し（新）----------------
        self.reaccept_need = {'gnss': self.gnss_reaccept_count, 'match': self.match_reaccept_count}
        self.reject_count = {'gnss': 0, 'match': 0}
        self.reject_buff = {name: deque(maxlen=max(1, need))
                            for name, need in self.reaccept_need.items()}
        self.last_accept = {'gnss': None, 'match': None}

        self.debug = dict.fromkeys(DEBUG_FIELDS, 0.0)
        self.last_status_log = None

        # ---------------- 購読 ----------------
        # 変更点: 旧は /odom/combine と /odom/UM982 の 2 本。
        # BEST_EFFORT にすると、RELIABLE / BEST_EFFORT どちらの配信側ともつながる
        qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=10,
                         reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         durability=QoSDurabilityPolicy.VOLATILE)
        self.sub_a = self.create_subscription(
            Odometry, self.odom_topic, self.sensor_a_callback, qos)
        if self.use_gnss or self.use_gnss_heading:
            self.sub_b = self.create_subscription(
                Odometry, self.gnss_topic, self.sensor_b_callback, qos)
            if self.use_pos_type:
                self.sub_raw = self.create_subscription(
                    String, self.gnss_raw_topic, self.gps_raw_callback, qos)
        if self.use_slam:
            self.sub_c = self.create_subscription(
                Odometry, self.match_topic, self.sensor_c_callback, qos)
            if self.ref_slam_topic and self.ref_slam_topic != self.match_topic:
                self.sub_ref = self.create_subscription(
                    Odometry, self.ref_slam_topic, self.ref_slam_callback, qos)
        if self.r_mode == 'occ':
            self.sub_occ = self.create_subscription(
                Float32, self.occ_topic, self.occ_callback, qos)
        self.sub_wp = self.create_subscription(
            Int32, self.waypoint_topic, self.waypoint_callback, qos)

        # ---------------- 配信 ----------------
        self.t = TransformStamped()
        self.br = tf2_ros.TransformBroadcaster(self)
        self.fused_pub = self.create_publisher(Odometry, self.output_topic, 10)
        self.fused_msg = Odometry()
        self.debug_pub = self.create_publisher(Float64MultiArray, self.output_topic + '/debug', 10)

        self.timer = self.create_timer(TIMER_PERIOD, self.publish_fused_value)

        self.get_logger().info(
            f'ekf_gnss_match: out={self.output_topic} use_gnss={self.use_gnss} '
            f'use_slam={self.use_slam} r_mode={self.r_mode} use_bias={self.n_state == 6} '
            f'match={self.match_topic} R_match={self.match_sigma:.2f}m '
            f'gnss_off: rects={len(self.off_rects)} wp={len(self.off_waypoints)} '
            f'tf={self.ekf_publish_TF}')

    def reset_tf_buffer(self):
        # キャッシュのクリアとして、Bufferのインスタンスを再作成（旧と同じ）
        self.br = tf2_ros.TransformBroadcaster(self)
        self.get_logger().info('TransformBroadcaster has been reset')

    def orientation_to_yaw(self, z, w):
        yaw = np.arctan2(2.0 * (w * z), 1.0 - 2.0 * (z ** 2))
        return yaw

    def yaw_to_orientation(self, yaw):
        orientation_z = np.sin(yaw / 2.0)
        orientation_w = np.cos(yaw / 2.0)
        return orientation_z, orientation_w

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ==================================================================
    # 購読
    # ==================================================================
    def sensor_a_callback(self, data):
        """/odom/combine: 最新の位置・方位を保存する.

        変更点: 旧はここで Speed = |位置の差| / SmpTime（大きさだけ）を計算していた。
        新は姿勢を保存するだけで、移動量は publish_fused_value（odom_motion）で、
        前回の予測からの差として計算する（後退を正しく扱える。2回分届いても取りこぼさない）。
        """
        self.odom_xy = (data.pose.pose.position.x, data.pose.pose.position.y)
        self.GTheta = float(self.orientation_to_yaw(
            data.pose.pose.orientation.z, data.pose.pose.orientation.w))
        stamp = stamp_sec(data.header.stamp)
        self.odom_stamp = stamp if stamp > 0.0 else self.now_sec()
        self.odom_twist = data.twist
        self.odom_new = True

        # initialize odom（旧と同じ: 最初の combine の位置で初期化）
        if not self.is_initialized:
            self.initialize_odomA(self.odom_xy[0], self.odom_xy[1], self.GTheta)

    def sensor_b_callback(self, data):
        """/odom/UM982: GNSS の位置・方位・衛星数を保存する（旧と同じ）.

        変更点: σ²（cov[7]）と、同じエポックの /gps_raw（pos_type）も一緒に保存する。
        同じ座標の再送（同じエポック）は捨てる。
        """
        xy = (data.pose.pose.position.x, data.pose.pose.position.y)
        if not self.is_initialized:
            self.debug['gnss_status'] = STATUS_NOT_INIT
        if xy == self.last_gps_xy:
            self.debug['gnss_status'] = STATUS_DUP
            self.debug['gnss_dup_count'] += 1
            return
        self.last_gps_xy = xy
        self.gps_time = self.now_sec()
        self.GpsXY = np.array(xy)
        self.GPSYaw = float(self.orientation_to_yaw(
            data.pose.pose.orientation.z, data.pose.pose.orientation.w))
        self.Number_of_satellites = data.pose.covariance[0]
        self.GpsVar = data.pose.covariance[7]
        fresh = (self.raw_time is not None
                 and self.gps_time - self.raw_time <= self.gnss_raw_timeout)
        self.GpsInfo = self.raw_info if fresh else None

    def sensor_c_callback(self, data):
        """（新）/odom_ekf_match: マッチングの位置を保存する（使ったら None に戻す）."""
        self.MatchXY = np.array([data.pose.pose.position.x, data.pose.pose.position.y])
        if not self.is_initialized:
            self.debug['match_status'] = STATUS_NOT_INIT
        if self.match_topic == self.ref_slam_topic:
            self.last_ref_slam_t = self.now_sec()

    def gps_raw_callback(self, msg):
        """（新）/gps_raw: 測位の種類（pos_type）と σ を読む."""
        info = parse_gnss_raw(msg.data)
        if info is not None:
            self.raw_info = info
            self.raw_time = self.now_sec()

    def ref_slam_callback(self, msg):
        """（新）/odom_ref_slam: マッチングが成功した時刻（match_r_inflate 用）."""
        self.last_ref_slam_t = self.now_sec()

    def waypoint_callback(self, msg):
        """（新）/waypoint_number: GNSS 無効区間（ウェイポイント番号）用."""
        self.waypoint = int(msg.data)

    def occ_callback(self, msg):
        """（新）/occlusion_rate: 遮蔽率（r_mode=occ 用）."""
        self.occ_value = float(msg.data)
        self.occ_time = self.now_sec()

    # ==================================================================
    # R の決定
    # ==================================================================
    def determination_of_R(self, now):
        """GNSS 無効区間（gps_rr_flag）と、観測ノイズ R1, R2（GNSS）、R_match（マッチング）を決める.

        変更点:
          - 無効区間: 旧は GpsXY の矩形をコードに直書き → YAML の gnss_off_rects（矩形）と
            gnss_off_waypoints（ウェイポイント番号）。判定に使う位置は EKF の推定位置。
          - GNSS の R: 旧は区間の内外で 0.17² / 1.0² の 2 択 → r_mode（fixed / cov7 / occ）で決める。
          - 旧の RR_count_bad（区間を出た直後 5 秒は R=1.0²）は、χ²検定（kalman_update）に置き換え。
        """
        # ---- GNSS 無効区間（1: 使う, 0: 使わない）----
        in_rect = bool(self.off_rects) and in_rects(self.XX[0], self.XX[1], self.off_rects)
        in_wp = (self.waypoint is not None and bool(self.off_waypoints)
                 and in_ranges(self.waypoint, self.off_waypoints))
        self.gps_rr_flag = 0 if (in_rect or in_wp) else 1

        # ---- GNSS の R ----
        category, _, sigma_rx = self.gnss_quality()
        if self.r_mode == 'fixed':
            sigma = self.gnss_sigma_fixed
        elif (self.r_mode == 'occ' and self.occ_value is not None
              and now - self.occ_time <= self.occ_timeout):
            sigma = min(max(self.occ_a + self.occ_b * self.occ_value, self.occ_sigma_min),
                        self.occ_sigma_max)
        else:
            # cov7（occ が来ていないときもここ）: R = max(σ², 下限²)。下限は種類ごと
            # （Float の σ は楽観的なので、RTK が外れたらマッチングが主役になるようにする）
            floor = {'FIX': self.gnss_sigma_fix, 'FLOAT': self.gnss_sigma_float,
                     'PPP': self.gnss_sigma_ppp}.get(category)
            if sigma_rx is not None:
                sigma = sigma_rx if floor is None else max(sigma_rx, floor)
            else:
                sigma = self.gnss_sigma_unknown if floor is None else floor
            sigma = max(sigma, self.gnss_sigma_min)
        self.R1 = sigma ** 2   # GNSS の x
        self.R2 = sigma ** 2   # GNSS の y

        # ---- マッチングの R（match_r_inflate: マッチング成功から時間が経つほど大きくする）----
        sigma_m = self.match_sigma
        if self.match_r_inflate:
            ref = self.last_ref_slam_t if self.last_ref_slam_t is not None else self.start_time
            if ref is not None:
                elapsed = max(0.0, now - ref - self.match_inflate_grace)
                sigma_m = min(sigma_m + self.match_sigma_growth * elapsed,
                              max(self.match_sigma_max, sigma_m))
        self.R_match = sigma_m ** 2

        R = np.array([self.R1, self.R2, self.R_match])
        return R

    def gnss_quality(self):
        """（新）GNSS の区分（FIX など）・pos_type・σ。σ は cov[7] → BESTNAVA の順で取る."""
        info = self.GpsInfo
        category = info[0] if info is not None else 'UNKNOWN'
        pos_type = info[1] if info is not None else ''
        sigma_rx = None
        if self.GpsVar > 0.0:
            sigma_rx = math.sqrt(self.GpsVar)
        elif info is not None and info[2] is not None:
            sigma_rx = info[2]
        return category, pos_type, sigma_rx

    # ==================================================================
    # EKF
    # ==================================================================
    def initialize_odomA(self, x, y, GTheta):
        """最初に届いた combine の位置で XX と P を初期化する（旧と同じ名前）.

        変更点: P の初期値を現実的な値に（旧は Q と同じ極小値）。use_bias のときはバイアスも持つ。
        """
        n = self.n_state
        yaw = wrap_angle(GTheta + self.offsetyaw)
        self.GTheta0 = yaw
        self.XX = np.zeros(n)
        self.XX[0:4] = [x, y, math.cos(yaw), math.sin(yaw)]
        self.P = np.zeros((n, n))
        self.P[0, 0] = self.P[1, 1] = self.init_pos_sigma ** 2
        j = np.array([-math.sin(yaw), math.cos(yaw)])
        self.P[2:4, 2:4] = np.outer(j, j) * math.radians(self.init_yaw_sigma_deg) ** 2
        if n == 6:
            self.P[4, 4] = self.P[5, 5] = self.init_bias_sigma ** 2
        self.prev_odom = (self.odom_stamp, x, y, GTheta)
        self.start_time = self.now_sec()
        self.is_initialized = True
        self.get_logger().info(f'EKF initialized with OdomA at: ({x:.3f}, {y:.3f})')

    def odom_motion(self):
        """前回の予測から今までの combine の移動量（前後 DistF・左右 DistL）と経過時間 SmpTime.

        変更点: 旧の Speed = |位置の差| / 時間（大きさだけ。後退も前進扱い）
        → 前回の向きを基準にした符号付きの移動量。combine が飛んだ回（odom_jump_thresh 超え）は予測しない。
        """
        t0, x0, y0, yaw0 = self.prev_odom
        t, (x, y), yaw = self.odom_stamp, self.odom_xy, self.GTheta
        self.prev_odom = (t, x, y, yaw)
        SmpTime = min(max(t - t0, 0.0), 1.0)
        dx, dy = x - x0, y - y0
        if math.hypot(dx, dy) > self.odom_jump_thresh:
            self.debug['odom_jump_count'] += 1
            return None, None, SmpTime
        c0, s0 = math.cos(yaw0), math.sin(yaw0)
        DistF = (c0 * dx + s0 * dy) * self.kalf_speed_param    # 前後（後退は負）
        DistL = (-s0 * dx + c0 * dy) * self.kalf_speed_param   # 左右
        self.distance += math.hypot(DistF, DistL)
        return DistF, DistL, SmpTime

    def KalfXY(self, DistF, DistL, SmpTime, GTheta):
        """予測（旧と同じ名前）. combine の移動量で XX と P を進める.

        GTheta には robot_yaw（combine の方位 + offsetyaw）を渡す。旧と同じく
        DTheta = GTheta - GTheta0 で方位を回す（offsetyaw が変わったときも、その分回る）。
        変更点:
          - 旧は GPS が無いときだけ呼ばれ、P を伝播していなかった → 毎周期呼び、P も伝播する。
          - 移動量: 旧は Speed*SmpTime（大きさだけ）を新しい方位の向きに進めていた
            → combine の実際の移動量（前回の向きを基準にした前後 DistF・左右 DistL、符号付き）。
          - 旧の XX = F@XX + G@w（分散 w を状態に足していた）は削除。
          - Q: 旧は 1 ステップ σ=1.4mm の極小値（オドメトリをほぼ完全に信じる）
            → 移動量と時間に応じた値（χ²検定で正しい観測を捨てないため）。
        """
        n = self.n_state
        DTheta = wrap_angle(GTheta - self.GTheta0)
        self.GTheta0 = GTheta
        c, s = self.XX[2], self.XX[3]
        cd, sd = math.cos(DTheta), math.sin(DTheta)

        # equation of state F（旧と同じ形。旧の Speed*SmpTime*cos(DTheta), Speed*SmpTime*sin(DTheta)
        # の所に、前後・左右の移動量が入る）
        F = np.eye(n)
        F[0, 2], F[0, 3] = DistF, -DistL
        F[1, 2], F[1, 3] = DistL, DistF
        F[2, 2], F[2, 3] = cd, -sd
        F[3, 2], F[3, 3] = sd, cd

        # G: ノイズ [前後, 左右, 方位] → 状態（変更点: 旧は [速度, 方位] の 2 列）
        c1, s1 = cd * c - sd * s, sd * c + cd * s   # 回したあとの方位
        G = np.zeros((n, 3))
        G[0, 0], G[1, 0] = c, s       # 前後
        G[0, 1], G[1, 1] = -s, c      # 左右
        G[2, 2], G[3, 2] = -s1, c1    # 方位
        dist = math.hypot(DistF, DistL)
        q_pos = ((self.q_dist_ratio * dist) ** 2 + self.q_dist_rw ** 2 * dist
                 + self.q_pos_time ** 2 * SmpTime)
        q_yaw = (self.q_yaw_ratio * abs(DTheta)) ** 2 + self.q_yaw_time ** 2 * SmpTime
        self.Q = np.diag([q_pos, q_pos, q_yaw])

        self.XX = F @ self.XX  # filter equation
        self.P = F @ self.P @ F.T + G @ self.Q @ G.T  # Prior Error Covariance
        if n == 6:   # バイアスはゆっくり変わる
            q_bias = self.q_bias_dist ** 2 * dist + self.q_bias_time ** 2 * SmpTime
            self.P[4, 4] += q_bias
            self.P[5, 5] += q_bias
        self.P = 0.5 * (self.P + self.P.T)
        return self.XX[:2]

    def KalfGPSXY(self, GpsXY, R1, R2, now):
        """GNSS の観測更新（旧と同じ名前）.

        変更点:
          - 予測は KalfXY に分けた。ここは更新だけ。
          - 観測の式にアンテナ位置（gnss_antenna_x, y）とバイアス（use_bias）を入れる（既定は 0 / オフ）。
          - χ²検定で外れ値を捨てる。σ が小さい（RTK Fix 相当）ときは、連続で捨てたあと受け入れ直す。
        """
        n = self.n_state
        c, s = self.XX[2], self.XX[3]
        lx, ly = self.gnss_antenna_x, self.gnss_antenna_y
        self.R = np.array([[R1, 0.0], [0.0, R2]])
        self.H = np.zeros((2, n))
        self.H[0, 0] = self.H[1, 1] = 1.0
        self.H[0, 2], self.H[0, 3] = lx, -ly
        self.H[1, 2], self.H[1, 3] = ly, lx
        hX = np.array([self.XX[0] + c * lx - s * ly, self.XX[1] + s * lx + c * ly])
        if n == 6:
            self.H[0, 4] = self.H[1, 5] = 1.0
            hX = hX + self.XX[4:6]

        Y = np.array([GpsXY[0], GpsXY[1]])
        allow = math.sqrt(R1) <= self.gnss_reaccept_sigma_max
        return self.kalman_update('gnss', Y, hX, self.gnss_gate_chi2, allow, now)

    def KalfMatchXY(self, MatchXY, R_match, now):
        """（新）マッチングの観測更新. KalfGPSXY と同じ形（観測の式は位置そのもの）.

        GNSS を使っている間（最後に使ってから reaccept_block_sec 秒以内）は、受け入れ直しをしない
        （地図と RTK が食い違うときは RTK Fix を優先）。
        """
        n = self.n_state
        self.R = np.array([[R_match, 0.0], [0.0, R_match]])
        self.H = np.zeros((2, n))
        self.H[0, 0] = self.H[1, 1] = 1.0
        hX = self.XX[0:2].copy()

        Y = np.array([MatchXY[0], MatchXY[1]])
        last = self.last_accept['gnss']
        allow = n == 6 or last is None or now - last > self.reaccept_block_sec
        return self.kalman_update('match', Y, hX, self.match_gate_chi2, allow, now)

    def kalman_update(self, name, Y, hX, gate, allow_reaccept, now):
        """観測更新の共通部分（旧 KalfGPSXY の後半と同じ式）. self.H と self.R を使う.

        変更点:
          - χ²検定: d2 = νᵀ S⁻¹ ν が gate を超えたら捨てる（外れ値）。連続で必要な回数だけ捨て、
            その間のずれのばらつきが reaccept_spread 以下なら、P を広げて受け入れ直す
            （観測ではなく EKF の側がずれていた、とみなす）。
          - P の更新は Joseph 形式（旧の P - K@H@P は数値的に崩れやすい）。
        """
        nu = Y - hX   # イノベーション（旧: Y - H@XX）
        S = self.H @ self.P @ self.H.T + self.R
        S = 0.5 * (S + S.T)
        try:
            d2 = float(nu @ np.linalg.solve(S, nu))
        except np.linalg.LinAlgError:
            d2 = math.inf
        status = STATUS_USED
        if gate > 0.0 and d2 > gate:
            self.reject_count[name] += 1
            self.reject_buff[name].append(nu.copy())
            if not (allow_reaccept and self.ready_to_reaccept(name)):
                return STATUS_GATED, d2, nu
            # 受け入れ直し: ずれの分だけ P を広げる（use_bias の GNSS はバイアスの側を広げる）
            if name == 'gnss' and self.n_state == 6:
                self.P[4:6, 4:6] += np.outer(nu, nu)
            else:
                self.P[0:2, 0:2] += np.outer(nu, nu)
            S = self.H @ self.P @ self.H.T + self.R
            S = 0.5 * (S + S.T)
            status = STATUS_REACCEPTED

        # kalman gain（旧と同じ式）
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.XX = self.XX + K @ nu  # estimated value
        I_KH = np.eye(self.n_state) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T  # Posterior Error Covariance（Joseph 形式）
        self.P = 0.5 * (self.P + self.P.T)
        self.normalize_heading()

        self.reject_count[name] = 0
        self.reject_buff[name].clear()
        self.last_accept[name] = now
        return status, d2, nu

    def ready_to_reaccept(self, name):
        """（新）連続で必要な回数だけ捨てていて、ずれのばらつきが小さければ True."""
        need = self.reaccept_need[name]
        buff = self.reject_buff[name]
        if need <= 0 or self.reject_count[name] < need or not buff:
            return False
        arr = np.array(buff)
        spread = float(np.max(np.linalg.norm(arr - arr.mean(axis=0), axis=1)))
        return spread <= self.reaccept_spread

    def normalize_heading(self):
        """（新）更新後に [cosθ, sinθ] の長さを 1 に戻す（P も合わせて射影する）."""
        v = self.XX[2:4].copy()
        norm = float(np.linalg.norm(v))
        if norm < 1e-9:
            return
        u = v / norm
        T = np.eye(self.n_state)
        T[2:4, 2:4] = (np.eye(2) - np.outer(u, u)) / norm
        self.XX[2:4] = u
        self.P = T @ self.P @ T.T
        self.P = 0.5 * (self.P + self.P.T)

    # ==================================================================
    # 方位のずれ
    # ==================================================================
    def update_yaw_offset(self, now):
        """GNSS 方位と combine 方位のずれ（offsetyaw）を、直進中のサンプルの中央値で更新する.

        旧は publish_fused_value の中にあった処理。考え方（直進中にためて、set_yaw_count 個ごとに
        中央値を offsetyaw にする。theta_diff_thresh 以上曲がったら捨てる）は旧と同じ。
        変更点:
          - サンプルを足す条件: 旧は「衛星数 >= set_yaw_satellites_no かつ 区間外」
            → 「RTK Fix（heading_accept_types）かつ σ <= heading_sigma_max かつ 区間外」。
            pos_type が分からないときは旧の衛星数の条件を使う。
          - サンプルは新しい GNSS が届いたときだけ足す（旧は毎周期、同じ GPS 方位を足していた）。
          - 中央値は円周上で取る（旧は 0〜2π の差の中央値で、0° 付近で少しずれていた）。
          - 旧は GPS_angle_reset_count が 20 を超えても diff_yaw_buff を消していなかった → 一緒に消す。
        """
        if self.theta_buff is None:
            self.theta_buff = self.GTheta
        # 直進を判定（基準の方位から theta_diff_thresh 以上曲がったら、ためたサンプルを捨てる）
        if abs(wrap_angle(self.GTheta - self.theta_buff)) > math.radians(self.theta_diff_thresh):
            self.reset_yaw_buff()

        valid = False
        if self.GpsXY is not None and self.use_gnss_heading and self.gps_rr_flag == 1:
            category, pos_type, sigma_rx = self.gnss_quality()
            valid = self.heading_quality_ok(category, pos_type, sigma_rx)
        if valid:
            self.GPS_angle_reset_count = 0
            self.diff_yaw_buff.append(wrap_angle(self.GPSYaw - self.GTheta))
            self.GPS_angle_conut = len(self.diff_yaw_buff)
        else:
            # 有効なサンプルが heading_reset_sec 秒来なければ、ためたサンプルを捨てる（旧: 20 周期）
            self.GPS_angle_reset_count += 1
            if self.GPS_angle_reset_count > self.heading_reset_sec / TIMER_PERIOD:
                self.reset_yaw_buff()

        # set_yaw_count 個たまったら、中央値を offsetyaw にする（旧と同じ）
        if self.GPS_angle_conut >= self.set_yaw_count:
            self.offsetyaw = circular_median(self.diff_yaw_buff)
            self.yaw_update_count += 1
            self.reset_yaw_buff()
            self.get_logger().info(f'offsetyaw: {math.degrees(self.offsetyaw):.2f} deg')

    def reset_yaw_buff(self):
        self.diff_yaw_buff = []
        self.GPS_angle_conut = 0
        self.GPS_angle_reset_count = 0
        self.theta_buff = self.GTheta

    def heading_quality_ok(self, category, pos_type, sigma_rx):
        """（新）方位のサンプルに使ってよい GNSS か."""
        if sigma_rx is not None and sigma_rx > self.heading_sigma_max:
            return False
        if category != 'UNKNOWN':
            return category in self.heading_accept_types or pos_type in self.heading_accept_types
        return self.Number_of_satellites >= self.set_yaw_satellites_no   # 旧の条件

    # ==================================================================
    # 本体（0.1 秒タイマー）
    # ==================================================================
    def publish_fused_value(self):
        """0.1 秒ごとの本体（旧と同じ名前・同じ役割）.

        流れ: R の決定 → 方位のずれ → 予測（KalfXY）→ GNSS の更新（KalfGPSXY）
              → マッチングの更新（KalfMatchXY）→ 配信
        変更点:
          - 旧は「GPS あり → KalfGPSXY（予測＋更新）／なし → KalfXY（予測だけ）」の 2 分岐
            → 予測は毎周期、更新は観測が届いたときだけ（1 本の流れ）。
          - 使った観測は None に戻す（旧は「gps reset」がコメントアウトされていて、同じ GPS を毎周期使っていた）。
        """
        # 新しい combine が届いたときだけ動く（旧: Speed, SmpTime, GTheta が揃ったら）
        if not self.is_initialized or not self.odom_new:
            return
        self.odom_new = False
        now = self.now_sec()

        # ---- 観測ノイズ R と GNSS 無効区間 ----
        self.determination_of_R(now)
        self.debug['in_gnss_off_zone'] = float(self.gps_rr_flag == 0)

        # ---- 方位のずれ → 出力する方位（旧と同じ: robot_yaw = combine の方位 + offsetyaw）----
        self.update_yaw_offset(now)
        self.robot_yaw = wrap_angle(self.GTheta + self.offsetyaw)

        # ---- 予測 ----
        DistF, DistL, self.SmpTime = self.odom_motion()
        if DistF is not None:
            self.KalfXY(DistF, DistL, self.SmpTime, self.robot_yaw)

        # ---- GNSS の更新 ----
        if self.GpsXY is not None:
            category, pos_type, _ = self.gnss_quality()
            self.debug['gnss_category'] = CATEGORY_CODE.get(category, 0)
            if not self.use_gnss:
                status = STATUS_DISABLED
            elif self.gps_rr_flag == 0:   # 無効区間（旧と同じ）
                status = STATUS_ZONE
            elif (category not in self.gnss_accept_types
                  and pos_type not in self.gnss_accept_types):
                status = STATUS_TYPE
            else:
                status, d2, nu = self.KalfGPSXY(self.GpsXY, self.R1, self.R2, now)
                self.debug.update(gnss_d2=d2, gnss_innov_x=nu[0], gnss_innov_y=nu[1],
                                  gnss_sigma=math.sqrt(self.R1))
            self.debug['gnss_status'] = status
            self.debug['gnss_reject_count'] = self.reject_count['gnss']
            self.GpsXY = None   # 変更点: 使った GNSS は消す（gps reset）

        # ---- マッチングの更新 ----
        if self.MatchXY is not None:
            status, d2, nu = self.KalfMatchXY(self.MatchXY, self.R_match, now)
            self.debug.update(match_status=status, match_d2=d2, match_innov_x=nu[0],
                              match_innov_y=nu[1], match_sigma=math.sqrt(self.R_match),
                              match_reject_count=self.reject_count['match'])
            self.MatchXY = None

        # ---- 配信（旧と同じ）----
        robot_orientation = self.yaw_to_orientation(self.robot_yaw)
        self.fused_msg.header.stamp = self.get_clock().now().to_msg()
        self.fused_msg.header.frame_id = self.frame_id
        self.fused_msg.child_frame_id = self.child_frame_id
        self.fused_msg.pose.pose.position.x = float(self.XX[0])
        self.fused_msg.pose.pose.position.y = float(self.XX[1])
        self.fused_msg.pose.pose.orientation.z = float(robot_orientation[0])
        self.fused_msg.pose.pose.orientation.w = float(robot_orientation[1])
        # 変更点: 共分散も出す（位置 2x2 と方位）
        cov = [0.0] * 36
        cov[0], cov[1] = float(self.P[0, 0]), float(self.P[0, 1])
        cov[6], cov[7] = float(self.P[1, 0]), float(self.P[1, 1])
        j = np.array([-self.XX[3], self.XX[2]])
        cov[35] = float(j @ self.P[2:4, 2:4] @ j)
        self.fused_msg.pose.covariance = cov
        if self.odom_twist is not None:
            self.fused_msg.twist = self.odom_twist
        self.fused_pub.publish(self.fused_msg)
        self.publish_debug(now)

        if self.ekf_publish_TF:
            self.t.header.stamp = self.fused_msg.header.stamp
            self.t.header.frame_id = self.frame_id
            self.t.child_frame_id = self.child_frame_id
            self.t.transform.translation.x = float(self.XX[0])
            self.t.transform.translation.y = float(self.XX[1])
            self.t.transform.translation.z = 0.0
            self.t.transform.rotation.x = 0.0
            self.t.transform.rotation.y = 0.0
            self.t.transform.rotation.z = float(robot_orientation[0])
            self.t.transform.rotation.w = float(robot_orientation[1])
            self.br.sendTransform(self.t)

        self.log_status(now)

    # ==================================================================
    # デバッグ（新）
    # ==================================================================
    def publish_debug(self, now):
        d, P = self.debug, self.P
        d['pos_sigma'] = math.sqrt(max(float(np.max(np.linalg.eigvalsh(P[0:2, 0:2]))), 0.0))
        if self.n_state == 6:
            d['bias_x'], d['bias_y'] = float(self.XX[4]), float(self.XX[5])
            d['bias_sigma'] = math.sqrt(max(0.5 * (P[4, 4] + P[5, 5]), 0.0))
        d['yaw_offset'] = self.offsetyaw
        d['ekf_yaw'] = math.atan2(self.XX[3], self.XX[2])
        d['heading_samples'] = len(self.diff_yaw_buff)
        d['heading_updates'] = self.yaw_update_count
        d['waypoint'] = -1 if self.waypoint is None else self.waypoint
        ref = self.last_ref_slam_t
        d['time_since_ref_slam'] = -1.0 if ref is None else now - ref
        d['distance'] = self.distance
        msg = Float64MultiArray()
        msg.layout.dim = [MultiArrayDimension(label=','.join(DEBUG_FIELDS),
                                              size=len(DEBUG_FIELDS), stride=len(DEBUG_FIELDS))]
        msg.data = [float(d[k]) for k in DEBUG_FIELDS]
        self.debug_pub.publish(msg)

    def log_status(self, now):
        if self.status_log_sec <= 0.0 or (self.last_status_log is not None
                                          and now - self.last_status_log < self.status_log_sec):
            return
        self.last_status_log = now
        d = self.debug
        self.get_logger().info(
            f"pos_sigma={d['pos_sigma']:.3f}m gnss={int(d['gnss_status'])}"
            f"(rej {int(d['gnss_reject_count'])}, sigma {d['gnss_sigma']:.3f}) "
            f"match={int(d['match_status'])}(rej {int(d['match_reject_count'])}) "
            f"yaw_offset={math.degrees(d['yaw_offset']):.2f}deg "
            f"bias=({d['bias_x']:.2f},{d['bias_y']:.2f}) zone={int(d['in_gnss_off_zone'])}")


# ======================================================================
# 補助の関数（旧の rotation_xyz と同じく、クラスの外に置く）
# ======================================================================
def stamp_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def wrap_angle(a):
    """[-pi, pi) に折り返す."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def circular_median(angles):
    """±180°をまたいでも壊れない中央値（平均方向のまわりで折り返してから取る）."""
    ref = math.atan2(sum(math.sin(a) for a in angles), sum(math.cos(a) for a in angles))
    rel = sorted(wrap_angle(a - ref) for a in angles)
    return wrap_angle(ref + rel[len(rel) // 2])


# BESTNAVA の pos_type → 区分
POS_TYPE_CATEGORY = {
    'NARROW_INT': 'FIX', 'WIDE_INT': 'FIX', 'L1_INT': 'FIX', 'INS_RTKFIXED': 'FIX',
    'NARROW_FLOAT': 'FLOAT', 'L1_FLOAT': 'FLOAT', 'IONOFREE_FLOAT': 'FLOAT',
    'INS_RTKFLOAT': 'FLOAT',
    'PPP': 'PPP', 'INS_PPP': 'PPP', 'PPP_BASIC': 'PPP', 'INS_PPP_BASIC': 'PPP',
    'PPP_CONVERGING': 'PPP_CONV', 'INS_PPP_CONVERGING': 'PPP_CONV',
    'PPP_BASIC_CONVERGING': 'PPP_CONV', 'INS_PPP_BASIC_CONVERGING': 'PPP_CONV',
    'PSRDIFF': 'DGPS', 'SBAS': 'DGPS', 'INS_PSRDIFF': 'DGPS', 'INS_SBAS': 'DGPS',
    'SINGLE': 'SINGLE', 'INS_PSRSP': 'SINGLE',
    'NONE': 'NONE', 'FIXEDPOS': 'NONE', 'FIXEDHEIGHT': 'NONE', 'DOPPLER_VELOCITY': 'NONE',
}

# GGA の quality → 区分（BESTNAVA が無いときの代わり）
GGA_QUALITY_CATEGORY = {0: 'NONE', 1: 'SINGLE', 2: 'DGPS', 4: 'FIX', 5: 'FLOAT', 6: 'NONE'}

_RE_BESTNAV = re.compile(r'#BESTNAV[A-Z]*,[^;]*;([^*\r\n]*)')
_RE_GGA = re.compile(r'\$[A-Z]{2}GGA,([^*\r\n]*)')
_RE_TOKEN = re.compile(r'[^A-Z0-9_]+')


def parse_gnss_raw(text):
    """/gps_raw の文字列から（区分, pos_type, σ）を取り出す. 読めなければ None.

    1. #BESTNAVA の本体（';' の後）の 2 番目 = pos_type、8・9 番目 = lat_σ・lon_σ
    2. 'BEST' の後に出てくる最初の pos_type の語（加工済みの形式向け）
    3. $xxGGA の quality（4=Fix, 5=Float, ...）
    """
    if not text:
        return None
    m = _RE_BESTNAV.search(text)
    if m:
        f = [v.strip() for v in m.group(1).split(',')]
        if len(f) > 1 and f[1] in POS_TYPE_CATEGORY:
            sigma = None
            if len(f) > 8:
                try:
                    lat_sd, lon_sd = float(f[7]), float(f[8])
                    if lat_sd > 0.0 and lon_sd > 0.0:
                        sigma = max(lat_sd, lon_sd)
                except ValueError:
                    pass
            return (POS_TYPE_CATEGORY[f[1]], f[1], sigma)
    i = text.find('BEST')
    if i >= 0:
        for tok in _RE_TOKEN.split(text[i:]):
            if tok in POS_TYPE_CATEGORY:
                return (POS_TYPE_CATEGORY[tok], tok, None)
    m = _RE_GGA.search(text)
    if m:
        f = m.group(1).split(',')
        if len(f) > 5 and f[5].strip().isdigit():
            q = int(f[5])
            return (GGA_QUALITY_CATEGORY.get(q, 'NONE'), 'GGA%d' % q, None)
    return None


def parse_rects(flat):
    """[xmin, xmax, ymin, ymax, ...] → [(xmin, xmax, ymin, ymax), ...]（端数は無視）."""
    vals = [float(v) for v in (flat or [])]
    rects = []
    for i in range(0, len(vals) - len(vals) % 4, 4):
        x0, x1, y0, y1 = vals[i:i + 4]
        rects.append((min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)))
    return rects


def parse_ranges(flat):
    """[start, end, ...] → [(start, end), ...]（両端を含む。端数は無視）."""
    vals = [int(v) for v in (flat or [])]
    return [(min(vals[i], vals[i + 1]), max(vals[i], vals[i + 1]))
            for i in range(0, len(vals) - len(vals) % 2, 2)]


def in_rects(x, y, rects):
    return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, x1, y0, y1 in rects)


def in_ranges(n, ranges):
    return any(a <= n <= b for a, b in ranges)


DEFAULT_PARAMS_FILE = ('try_navigation', 'config', 'ekf_gnss_match.yaml')


def load_default_params(logger):
    """インストール先の ekf_gnss_match.yaml を読む（無ければ空 → PARAM_DEFAULTS）."""
    try:
        pkg, sub, name = DEFAULT_PARAMS_FILE
        path = os.path.join(get_package_share_directory(pkg), sub, name)
        with open(path, encoding='utf-8') as fp:
            data = yaml.safe_load(fp) or {}
    except (PackageNotFoundError, OSError, yaml.YAMLError) as e:
        logger.warn('パラメータファイルを読めない → コード内の既定値で動く: %s' % e)
        return {}
    params = {}
    for key in ('/**', 'ekf_gnss_match', '/ekf_gnss_match'):
        block = data.get(key)
        if isinstance(block, dict):
            params.update(block.get('ros__parameters') or {})
    logger.info('パラメータファイル: %s（%d 項目）' % (path, len(params)))
    return params


def coerce_param(default, value):
    """Launch / YAML の書き方の揺れ（1 と 1.0、'true' など）を既定値の型に合わせる."""
    if value is None:
        return default
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in ('true', '1', 'yes', 'on')
        return bool(value)
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, str):
        return str(value)
    if isinstance(default, list):
        if isinstance(value, (bytes, bytearray)):
            return []
        return list(value)
    return value


def main(args=None):
    rclpy.init(args=args)
    ekf = ExtendedKalmanFilter()
    ekf.reset_tf_buffer()
    try:
        rclpy.spin(ekf)
    except KeyboardInterrupt:
        pass
    ekf.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()