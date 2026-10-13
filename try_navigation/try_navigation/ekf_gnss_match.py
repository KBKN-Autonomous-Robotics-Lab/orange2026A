#!/usr/bin/env python3
"""地図座標系の自己位置 EKF（マッチング＋GNSS）.

入力
  /odom/combine    予測（符号付きの移動量。ロボット座標に直してから使う）
  /odom_ekf_match  マッチング（地図座標系の位置。常に使う）
  /odom/UM982      GNSS（cov[0]=衛星数, cov[7]=σ² [m²]。0 なら σ なし）
  /gps_raw         pos_type（BESTNAVA、無ければ GGA の quality）
  /odom_ref_slam   マッチング成功の時刻（match_r_inflate 用）
  /waypoint_number GNSS 無効区間（ウェイポイント番号）用
出力
  /fusion/odom_gnss_match         ナビ用（既存の /fusion/odom はそのまま）
  /fusion/odom_gnss_match/debug   Float64MultiArray（項目名は layout.dim[0].label）
状態: [x, y, cos(yaw), sin(yaw)]（use_bias のときは GNSS のずれ [bx, by] を追加）
出力の方位は従来どおり「combine の方位 + GNSS 方位とのずれ（中央値）」。
TF は既定で出さない（publish_tf）。
"""
from collections import deque
from dataclasses import dataclass, field, fields
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
# EKF の計算部分
# ======================================================================
IX, IY, IC, IS, IBX, IBY = range(6)

# 観測ごとの処理結果（debug トピックに出す）
STATUS_NONE = 0
STATUS_USED = 1          # 使った
STATUS_REACCEPTED = 2    # 連続で除外したあと受け入れ直した
STATUS_GATED = -1        # χ²検定で除外
STATUS_ZONE = -2         # GNSS 無効区間
STATUS_TYPE = -3         # pos_type が対象外
STATUS_DUP = -4          # 同じエポック（同じ座標）の再送
STATUS_DISABLED = -5     # use_gnss / use_slam が false
STATUS_NOT_INIT = -6     # 初期化前

CATEGORY_CODE = {
    'UNKNOWN': 0, 'FIX': 1, 'FLOAT': 2, 'PPP': 3, 'PPP_CONV': 4,
    'DGPS': 5, 'SINGLE': 6, 'NONE': 7,
}

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

_RE_BESTNAV = re.compile(r'#BESTNAV[A-Z]*,[^;]*;([^*\r\n]*)')
_RE_GGA = re.compile(r'\$[A-Z]{2}GGA,([^*\r\n]*)')
_RE_TOKEN = re.compile(r'[^A-Z0-9_]+')


def wrap_angle(a):
    """[-pi, pi) に折り返す."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quaternion(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def circular_mean(angles):
    return math.atan2(sum(math.sin(a) for a in angles), sum(math.cos(a) for a in angles))


def circular_median(angles):
    """±180°をまたいでも壊れない中央値（平均方向のまわりで折り返してから取る）."""
    ref = circular_mean(angles)
    rel = sorted(wrap_angle(a - ref) for a in angles)
    return wrap_angle(ref + rel[len(rel) // 2])


@dataclass
class GnssRawInfo:
    category: str
    pos_type: str
    sigma: float = None   # BESTNAVA の水平σ [m]（max(lat_σ, lon_σ)）。無ければ None
    source: str = ''


def parse_gnss_raw(text):
    """/gps_raw の文字列から測位の種類（と σ）を取り出す. 読めなければ None.

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
            return GnssRawInfo(POS_TYPE_CATEGORY[f[1]], f[1], sigma, 'BESTNAV')
    i = text.find('BEST')
    if i >= 0:
        for tok in _RE_TOKEN.split(text[i:]):
            if tok in POS_TYPE_CATEGORY:
                return GnssRawInfo(POS_TYPE_CATEGORY[tok], tok, None, 'BEST')
    m = _RE_GGA.search(text)
    if m:
        f = m.group(1).split(',')
        if len(f) > 5 and f[5].strip().isdigit():
            q = int(f[5])
            return GnssRawInfo(GGA_QUALITY_CATEGORY.get(q, 'NONE'), 'GGA%d' % q, None, 'GGA')
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


@dataclass
class FusionConfig:
    # 使う観測
    use_gnss: bool = True
    use_slam: bool = True
    use_gnss_heading: bool = True
    use_bias: bool = False
    r_mode: str = 'cov7'                 # fixed / cov7 / occ

    # 予測（/odom/combine）
    odom_scale: float = 1.0
    odom_jump_thresh: float = 1.0        # 1回の移動量がこれを超えたら予測しない [m]
    q_dist_ratio: float = 0.05           # 1ステップの移動量に比例するσ [-]
    q_dist_rw: float = 0.05              # 走行距離に対するランダムウォーク [m/sqrt(m)]
    q_pos_time: float = 0.01             # 時間に対するランダムウォーク [m/sqrt(s)]
    q_yaw_ratio: float = 0.02            # 旋回角に比例するσ [-]
    q_yaw_time: float = 0.001            # 方位のランダムウォーク [rad/sqrt(s)]
    q_bias_dist: float = 0.01            # バイアスの変化（距離）[m/sqrt(m)]
    q_bias_time: float = 0.001           # バイアスの変化（時間）[m/sqrt(s)]

    # 初期値
    init_pos_sigma: float = 1.0
    init_yaw_sigma_deg: float = 1.0
    init_bias_sigma: float = 1.0

    # マッチング
    match_sigma: float = 0.2
    match_gate_chi2: float = 13.8        # 自由度2・99.9%。0以下で検定なし
    match_reaccept_count: int = 30
    match_r_inflate: bool = False        # 最後の /odom_ref_slam からの経過時間で R を大きくする
    match_sigma_growth: float = 0.05     # [m/s]
    match_sigma_max: float = 2.0
    match_inflate_grace: float = 1.0     # [s]

    # GNSS
    gnss_sigma_min: float = 0.05
    gnss_sigma_fixed: float = 0.17       # r_mode=fixed のσ（旧EKFの良好区間の値）
    gnss_sigma_fix: float = 0.05         # 種類ごとのσの下限（σ が無いときはこの値）
    gnss_sigma_float: float = 0.5
    gnss_sigma_ppp: float = 0.2
    gnss_sigma_unknown: float = 0.17     # 種類も σ も分からないとき
    # 種類（FIX/FLOAT/PPP/PPP_CONV/DGPS/SINGLE/NONE/UNKNOWN）か pos_type 名（NARROW_INT など）
    gnss_accept_types: list = field(default_factory=lambda: ['FIX', 'FLOAT', 'PPP', 'UNKNOWN'])
    gnss_raw_timeout: float = 0.5
    gnss_gate_chi2: float = 13.8
    gnss_reaccept_count: int = 30
    gnss_reaccept_sigma_max: float = 0.1
    gnss_antenna_x: float = 0.0          # アンテナ位置（ロボット座標、前方+）[m]
    gnss_antenna_y: float = 0.0          # （左+）[m]
    occ_a: float = 0.05                  # r_mode=occ: σ = clamp(a + b*occ)（仮値・要較正）
    occ_b: float = 1.0
    occ_sigma_min: float = 0.05
    occ_sigma_max: float = 5.0
    occ_timeout: float = 1.0

    # 受け入れ直し（共通）
    reaccept_spread: float = 0.3         # 除外したイノベーションのばらつきの上限 [m]
    reaccept_block_sec: float = 2.0      # GNSS を使っている間はマッチングの受け入れ直しをしない

    # 方位のずれ（GNSS 方位 − combine 方位）
    heading_accept_types: list = field(default_factory=lambda: ['FIX'])
    heading_sigma_max: float = 0.05
    heading_min_sats: int = 4            # pos_type が分からないときの代わりの条件
    heading_samples: int = 100
    heading_straight_deg: float = 40.0
    heading_reset_sec: float = 2.0

    # GNSS 無効区間（運用の保険。既定はなし）
    gnss_off_rects: list = field(default_factory=list)       # [xmin, xmax, ymin, ymax, ...]
    gnss_off_waypoints: list = field(default_factory=list)   # [start, end, ...]


class GnssMatchEKF:
    """位置 EKF. 方位は cos/sin で持つ（旧 ekf_myself_gps と同じ形）."""

    def __init__(self, use_bias=False):
        self.use_bias = bool(use_bias)
        self.n = 6 if self.use_bias else 4
        self.x = None
        self.P = None

    @property
    def initialized(self):
        return self.x is not None

    def initialize(self, x, y, yaw, pos_sigma, yaw_sigma, bias_sigma=1.0):
        self.x = np.zeros(self.n)
        self.x[IX], self.x[IY] = x, y
        self.x[IC], self.x[IS] = math.cos(yaw), math.sin(yaw)
        self.P = np.zeros((self.n, self.n))
        self.P[IX, IX] = self.P[IY, IY] = pos_sigma ** 2
        j = np.array([-math.sin(yaw), math.cos(yaw)])
        self.P[IC:IS + 1, IC:IS + 1] = np.outer(j, j) * yaw_sigma ** 2
        if self.use_bias:
            self.P[IBX, IBX] = self.P[IBY, IBY] = bias_sigma ** 2

    @property
    def yaw(self):
        return math.atan2(self.x[IS], self.x[IC])

    def yaw_var(self):
        j = np.array([-self.x[IS], self.x[IC]])
        return float(j @ self.P[IC:IS + 1, IC:IS + 1] @ j)

    def predict(self, d_forward, d_left, dpsi, dt, q):
        """ロボット座標の移動量 (d_forward, d_left) と方位の変化 dpsi で予測する."""
        c, s = self.x[IC], self.x[IS]
        cd, sd = math.cos(dpsi), math.sin(dpsi)
        x_new = self.x.copy()
        x_new[IX] += c * d_forward - s * d_left
        x_new[IY] += s * d_forward + c * d_left
        x_new[IC] = cd * c - sd * s
        x_new[IS] = sd * c + cd * s

        F = np.eye(self.n)
        F[IX, IC], F[IX, IS] = d_forward, -d_left
        F[IY, IC], F[IY, IS] = d_left, d_forward
        F[IC, IC], F[IC, IS] = cd, -sd
        F[IS, IC], F[IS, IS] = sd, cd

        dist = math.hypot(d_forward, d_left)
        dt = max(dt, 0.0)
        Q = np.zeros((self.n, self.n))
        q_pos = ((q.q_dist_ratio * dist) ** 2 + q.q_dist_rw ** 2 * dist
                 + q.q_pos_time ** 2 * dt)
        Q[IX, IX] = Q[IY, IY] = q_pos
        q_yaw = (q.q_yaw_ratio * abs(dpsi)) ** 2 + q.q_yaw_time ** 2 * dt
        j = np.array([-x_new[IS], x_new[IC]])
        Q[IC:IS + 1, IC:IS + 1] = np.outer(j, j) * q_yaw
        if self.use_bias:
            Q[IBX, IBX] = Q[IBY, IBY] = q.q_bias_dist ** 2 * dist + q.q_bias_time ** 2 * dt

        self.x = x_new
        self.P = F @ self.P @ F.T + Q
        self._symmetrize()

    def observe_position(self):
        """マッチング: z = [x, y]."""
        H = np.zeros((2, self.n))
        H[0, IX] = H[1, IY] = 1.0
        return self.x[[IX, IY]].copy(), H

    def observe_gnss(self, lever_x=0.0, lever_y=0.0):
        """GNSS: z = [x, y] + R(yaw) * lever (+ b)."""
        c, s = self.x[IC], self.x[IS]
        h = np.array([self.x[IX] + c * lever_x - s * lever_y,
                      self.x[IY] + s * lever_x + c * lever_y])
        H = np.zeros((2, self.n))
        H[0, IX] = H[1, IY] = 1.0
        H[0, IC], H[0, IS] = lever_x, -lever_y
        H[1, IC], H[1, IS] = lever_y, lever_x
        if self.use_bias:
            h += self.x[[IBX, IBY]]
            H[0, IBX] = H[1, IBY] = 1.0
        return h, H

    def innovation(self, z, h, H, R):
        nu = np.asarray(z, dtype=float) - h
        S = H @ self.P @ H.T + R
        S = 0.5 * (S + S.T)
        try:
            d2 = float(nu @ np.linalg.solve(S, nu))
        except np.linalg.LinAlgError:
            d2 = math.inf
        return nu, S, d2

    def apply_update(self, nu, S, H, R):
        K = np.linalg.solve(S, H @ self.P).T          # = P H^T S^-1（S, P は対称）
        self.x = self.x + K @ nu
        I_KH = np.eye(self.n) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T  # Joseph 形式
        self._symmetrize()
        self._normalize_heading()

    def inflate_position(self, nu):
        """位置の不確かさをイノベーションの方向に広げる（受け入れ直し用）."""
        self.P[IX:IY + 1, IX:IY + 1] += np.outer(nu, nu)

    def inflate_bias(self, nu):
        self.P[IBX:IBY + 1, IBX:IBY + 1] += np.outer(nu, nu)

    def rotate_heading(self, delta):
        """方位のずれの更新分だけ、状態の方位を回す."""
        cd, sd = math.cos(delta), math.sin(delta)
        T = np.eye(self.n)
        T[IC:IS + 1, IC:IS + 1] = [[cd, -sd], [sd, cd]]
        self.x = T @ self.x
        self.P = T @ self.P @ T.T
        self._symmetrize()

    def _normalize_heading(self):
        v = self.x[IC:IS + 1].copy()
        norm = float(np.linalg.norm(v))
        if norm < 1e-9:
            return
        u = v / norm
        T = np.eye(self.n)
        T[IC:IS + 1, IC:IS + 1] = (np.eye(2) - np.outer(u, u)) / norm
        self.x[IC:IS + 1] = u
        self.P = T @ self.P @ T.T
        self._symmetrize()

    def _symmetrize(self):
        self.P = 0.5 * (self.P + self.P.T)


class HeadingOffsetEstimator:
    """GNSS 方位と combine 方位の差を、直進中にためて中央値をとる（旧方式を維持）."""

    def __init__(self, n_samples=100, straight_deg=40.0, reset_sec=2.0):
        self.n_samples = max(1, int(n_samples))
        self.straight = math.radians(straight_deg)
        self.reset_sec = reset_sec
        self.offset = 0.0
        self.samples = []
        self.window_yaw0 = None
        self.last_valid_t = None
        self.n_updates = 0

    def reset(self, odom_yaw=None):
        self.samples = []
        self.window_yaw0 = odom_yaw

    def add(self, t, gnss_yaw, odom_yaw, valid):
        """1 サンプル追加. ずれを更新したら変化量 [rad] を返す（更新なしは None）."""
        if (self.window_yaw0 is None
                or abs(wrap_angle(odom_yaw - self.window_yaw0)) > self.straight):
            self.reset(odom_yaw)              # 曲がったら最初から（直進中だけ集める）
        if not valid:
            if self.last_valid_t is None or t - self.last_valid_t > self.reset_sec:
                self.reset(odom_yaw)
            return None
        self.last_valid_t = t
        self.samples.append(wrap_angle(gnss_yaw - odom_yaw))
        if len(self.samples) < self.n_samples:
            return None
        new = circular_median(self.samples)
        delta = wrap_angle(new - self.offset)
        self.offset = new
        self.n_updates += 1
        self.reset(odom_yaw)
        return delta


class _SourceState:
    """観測ごとの「連続で除外した回数」と、その間のイノベーション."""

    def __init__(self, need):
        self.need = int(need)
        self.count = 0
        self.buf = deque(maxlen=max(1, self.need))
        self.last_accept = None

    def reject(self, nu):
        self.count += 1
        self.buf.append(np.array(nu, dtype=float))

    def accept(self, now):
        self.count = 0
        self.buf.clear()
        self.last_accept = now

    def ready(self, spread_max):
        if self.need <= 0 or self.count < self.need or not self.buf:
            return False
        arr = np.array(self.buf)
        spread = float(np.max(np.linalg.norm(arr - arr.mean(axis=0), axis=1)))
        return spread <= spread_max


class GnssMatchFusion:
    """メッセージ処理（ROS ノードからは値だけを渡す）."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.ekf = GnssMatchEKF(use_bias=cfg.use_bias and cfg.use_slam)
        self.heading = HeadingOffsetEstimator(
            cfg.heading_samples, cfg.heading_straight_deg, cfg.heading_reset_sec)
        self.rects = parse_rects(cfg.gnss_off_rects)
        self.wp_ranges = parse_ranges(cfg.gnss_off_waypoints)
        self.accept_types = set(cfg.gnss_accept_types)
        self.heading_types = set(cfg.heading_accept_types)
        self.src = {'gnss': _SourceState(cfg.gnss_reaccept_count),
                    'match': _SourceState(cfg.match_reaccept_count)}
        self.prev_odom = None          # (t, x, y, yaw)
        self.odom_yaw = None
        self.last_gnss_xy = None
        self.raw = None                # (受信時刻, GnssRawInfo)
        self.occ = None                # (受信時刻, 遮蔽率)
        self.waypoint = None
        self.last_ref_slam_t = None
        self.start_t = None
        self.distance = 0.0
        self.debug = dict.fromkeys(DEBUG_FIELDS, 0.0)

    # ---------- 入力 ----------
    def on_odom(self, now, stamp, x, y, yaw):
        """/odom/combine. 予測を 1 回行う. 出力してよければ True."""
        cfg = self.cfg
        if self.start_t is None:
            self.start_t = now
        t = stamp if stamp > 0.0 else now
        self.odom_yaw = yaw
        if not self.ekf.initialized:
            self.ekf.initialize(x, y, wrap_angle(yaw + self.heading.offset),
                                cfg.init_pos_sigma, math.radians(cfg.init_yaw_sigma_deg),
                                cfg.init_bias_sigma)
            self.prev_odom = (t, x, y, yaw)
            return True
        t0, x0, y0, yaw0 = self.prev_odom
        self.prev_odom = (t, x, y, yaw)
        dt = min(max(t - t0, 0.0), 1.0)
        dx, dy = x - x0, y - y0
        if math.hypot(dx, dy) > cfg.odom_jump_thresh:
            self.debug['odom_jump_count'] += 1     # combine が飛んだ → この回は予測しない
            return True
        c0, s0 = math.cos(yaw0), math.sin(yaw0)
        d_forward = (c0 * dx + s0 * dy) * cfg.odom_scale     # 後退は負になる
        d_left = (-s0 * dx + c0 * dy) * cfg.odom_scale
        self.ekf.predict(d_forward, d_left, wrap_angle(yaw - yaw0), dt, cfg)
        self.distance += math.hypot(d_forward, d_left)
        return True

    def on_ref_slam(self, now):
        self.last_ref_slam_t = now

    def on_waypoint(self, n):
        self.waypoint = int(n)

    def on_occ(self, now, value):
        self.occ = (now, float(value))

    def on_gnss_raw(self, now, text):
        info = parse_gnss_raw(text)
        if info is not None:
            self.raw = (now, info)
        return info

    def on_match(self, now, x, y, is_ref_slam=False):
        d = self.debug
        if is_ref_slam:
            self.on_ref_slam(now)
        if not self.ekf.initialized:
            d['match_status'] = STATUS_NOT_INIT
            return STATUS_NOT_INIT
        if not self.cfg.use_slam:
            d['match_status'] = STATUS_DISABLED
            return STATUS_DISABLED
        sigma = self._match_sigma(now)
        h, H = self.ekf.observe_position()
        gnss = self.src['gnss']
        allow = (self.ekf.use_bias or gnss.last_accept is None
                 or now - gnss.last_accept > self.cfg.reaccept_block_sec)
        status, d2, nu = self._update('match', now, (x, y), h, H, sigma,
                                      self.cfg.match_gate_chi2, allow, self.ekf.inflate_position)
        d.update(match_status=status, match_d2=d2, match_innov_x=nu[0], match_innov_y=nu[1],
                 match_sigma=sigma, match_reject_count=self.src['match'].count)
        return status

    def on_gnss(self, now, x, y, yaw, sats, var7):
        """/odom/UM982. cov[0]=衛星数, cov[7]=σ² [m²]（0 以下なら無し）."""
        cfg, d = self.cfg, self.debug
        if not self.ekf.initialized:
            d['gnss_status'] = STATUS_NOT_INIT
            return STATUS_NOT_INIT
        if self.last_gnss_xy == (x, y):
            d['gnss_status'] = STATUS_DUP
            d['gnss_dup_count'] += 1
            return STATUS_DUP
        self.last_gnss_xy = (x, y)

        info = self._fresh_raw(now)
        category = info.category if info is not None else 'UNKNOWN'
        pos_type = info.pos_type if info is not None else ''
        sigma_rx = None
        if var7 is not None and var7 > 0.0:
            sigma_rx = math.sqrt(var7)
        elif info is not None and info.sigma is not None:
            sigma_rx = info.sigma
        in_zone = self._in_gnss_off_zone()
        d.update(gnss_category=CATEGORY_CODE.get(category, 0), in_gnss_off_zone=float(in_zone))

        if cfg.use_gnss_heading and self.odom_yaw is not None:
            ok = (not in_zone) and self._heading_quality_ok(category, pos_type, sigma_rx, sats)
            delta = self.heading.add(now, yaw, self.odom_yaw, ok)
            if delta is not None:
                self.ekf.rotate_heading(delta)

        if not cfg.use_gnss:
            status = STATUS_DISABLED
        elif in_zone:
            status = STATUS_ZONE
        elif category not in self.accept_types and pos_type not in self.accept_types:
            status = STATUS_TYPE
        else:
            sigma = self._gnss_sigma(now, category, sigma_rx)
            h, H = self.ekf.observe_gnss(cfg.gnss_antenna_x, cfg.gnss_antenna_y)
            inflate = self.ekf.inflate_bias if self.ekf.use_bias else self.ekf.inflate_position
            allow = sigma <= cfg.gnss_reaccept_sigma_max
            status, d2, nu = self._update('gnss', now, (x, y), h, H, sigma,
                                          cfg.gnss_gate_chi2, allow, inflate)
            d.update(gnss_d2=d2, gnss_innov_x=nu[0], gnss_innov_y=nu[1], gnss_sigma=sigma)
        d.update(gnss_status=status, gnss_reject_count=self.src['gnss'].count)
        return status

    # ---------- 出力 ----------
    def robot_yaw(self):
        if self.odom_yaw is None:
            return self.ekf.yaw
        return wrap_angle(self.odom_yaw + self.heading.offset)

    def output(self):
        """(x, y, yaw, 位置の共分散 2x2, 方位の分散)."""
        P = self.ekf.P
        return (float(self.ekf.x[IX]), float(self.ekf.x[IY]), self.robot_yaw(),
                P[IX:IY + 1, IX:IY + 1].copy(), self.ekf.yaw_var())

    def debug_vector(self, now):
        d, P = self.debug, self.ekf.P
        d['pos_sigma'] = math.sqrt(max(float(np.max(np.linalg.eigvalsh(P[:2, :2]))), 0.0))
        if self.ekf.use_bias:
            d['bias_x'], d['bias_y'] = float(self.ekf.x[IBX]), float(self.ekf.x[IBY])
            d['bias_sigma'] = math.sqrt(max(0.5 * (P[IBX, IBX] + P[IBY, IBY]), 0.0))
        d['yaw_offset'] = self.heading.offset
        d['ekf_yaw'] = self.ekf.yaw
        d['heading_samples'] = len(self.heading.samples)
        d['heading_updates'] = self.heading.n_updates
        d['waypoint'] = -1 if self.waypoint is None else self.waypoint
        ref = self.last_ref_slam_t
        d['time_since_ref_slam'] = -1.0 if ref is None else now - ref
        d['distance'] = self.distance
        return [float(d[k]) for k in DEBUG_FIELDS]

    # ---------- 内部 ----------
    def _update(self, name, now, z, h, H, sigma, gate, allow_reaccept, inflate):
        """χ²検定つきの更新. 連続で除外して条件がそろえば、不確かさを広げて受け入れ直す."""
        st = self.src[name]
        R = np.eye(2) * sigma ** 2
        nu, S, d2 = self.ekf.innovation(z, h, H, R)
        status = STATUS_USED
        if gate > 0.0 and d2 > gate:
            st.reject(nu)
            if not (allow_reaccept and st.ready(self.cfg.reaccept_spread)):
                return STATUS_GATED, d2, nu
            inflate(nu)
            nu, S, _ = self.ekf.innovation(z, h, H, R)
            status = STATUS_REACCEPTED
        self.ekf.apply_update(nu, S, H, R)
        st.accept(now)
        return status, d2, nu

    def _match_sigma(self, now):
        cfg = self.cfg
        sigma = cfg.match_sigma
        if cfg.match_r_inflate:
            ref = self.last_ref_slam_t if self.last_ref_slam_t is not None else self.start_t
            if ref is not None:
                elapsed = max(0.0, now - ref - cfg.match_inflate_grace)
                sigma = min(sigma + cfg.match_sigma_growth * elapsed,
                            max(cfg.match_sigma_max, sigma))
        return sigma

    def _gnss_sigma(self, now, category, sigma_rx):
        cfg = self.cfg
        if cfg.r_mode == 'fixed':
            return cfg.gnss_sigma_fixed
        if cfg.r_mode == 'occ' and self.occ is not None and now - self.occ[0] <= cfg.occ_timeout:
            s = cfg.occ_a + cfg.occ_b * self.occ[1]
            return min(max(s, cfg.occ_sigma_min), cfg.occ_sigma_max)
        # cov7（occ が来ていないときもここ）: R = max(σ², 下限²). 下限は種類ごと
        # （Float の σ は楽観的なので、RTK が外れたらマッチングが主役になるようにする）
        floor = {'FIX': cfg.gnss_sigma_fix, 'FLOAT': cfg.gnss_sigma_float,
                 'PPP': cfg.gnss_sigma_ppp}.get(category)
        if sigma_rx is not None:
            s = sigma_rx if floor is None else max(sigma_rx, floor)
        else:
            s = cfg.gnss_sigma_unknown if floor is None else floor
        return max(s, cfg.gnss_sigma_min)

    def _heading_quality_ok(self, category, pos_type, sigma_rx, sats):
        cfg = self.cfg
        if sigma_rx is not None and sigma_rx > cfg.heading_sigma_max:
            return False
        if category != 'UNKNOWN':
            return category in self.heading_types or pos_type in self.heading_types
        return sats >= cfg.heading_min_sats            # pos_type が無いときは旧条件

    def _fresh_raw(self, now):
        if self.raw is None or now - self.raw[0] > self.cfg.gnss_raw_timeout:
            return None
        return self.raw[1]

    def _in_gnss_off_zone(self):
        if self.rects and in_rects(self.ekf.x[IX], self.ekf.x[IY], self.rects):
            return True
        return (self.waypoint is not None and bool(self.wp_ranges)
                and in_ranges(self.waypoint, self.wp_ranges))


# ======================================================================
# ROS2 ノード
# ======================================================================
NODE_PARAMS = {
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
    'publish_tf': False,
    'use_pos_type': True,
    'status_log_sec': 5.0,
}


def _stamp_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9

DEFAULT_PARAMS_FILE = ('try_navigation', 'config', 'ekf_gnss_match.yaml')


def load_default_params(logger):
    """インストール先の ekf_gnss_match.yaml を読む（無ければ空 → コード内の既定値）."""
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

def _coerce(default, value):
    """Launch / YAML の書き方の揺れ（1 と 1.0、'true' など）を吸収する."""
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


class EkfGnssMatch(Node):
    def __init__(self):
        super().__init__('ekf_gnss_match')
        # 既定値 = YAML の値（無い項目はコード内の値）。-p / launch の指定はさらに優先される
        file_params = load_default_params(self.get_logger())
        cfg = FusionConfig()
        for f in fields(FusionConfig):
            default = getattr(cfg, f.name)
            init = _coerce(default, file_params.get(f.name, default))
            self.declare_parameter(f.name, init, ParameterDescriptor(dynamic_typing=True))
            setattr(cfg, f.name, _coerce(default, self.get_parameter(f.name).value))
        p = {}
        for name, default in NODE_PARAMS.items():
            init = _coerce(default, file_params.get(name, default))
            self.declare_parameter(name, init, ParameterDescriptor(dynamic_typing=True))
            p[name] = _coerce(default, self.get_parameter(name).value)
        self.p = p
        if cfg.r_mode not in ('fixed', 'cov7', 'occ'):
            self.get_logger().warn('r_mode %r は不明 → cov7 として扱う' % cfg.r_mode)
            cfg.r_mode = 'cov7'
        if len(cfg.gnss_off_rects) % 4:
            self.get_logger().warn('gnss_off_rects は 4 個ずつ（xmin, xmax, ymin, ymax）。端数は無視')
        if len(cfg.gnss_off_waypoints) % 2:
            self.get_logger().warn('gnss_off_waypoints は 2 個ずつ（start, end）。端数は無視')

        self.cfg = cfg
        self.fusion = GnssMatchFusion(cfg)
        self.last_status_log = None

        # BEST_EFFORT は RELIABLE / BEST_EFFORT どちらの配信側ともつながる（QoS 不一致で届かないのを防ぐ）
        qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=10,
                         reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         durability=QoSDurabilityPolicy.VOLATILE)
        self.create_subscription(Odometry, p['odom_topic'], self.cb_odom, qos)
        if cfg.use_slam:
            self.create_subscription(Odometry, p['match_topic'], self.cb_match, qos)
            if p['ref_slam_topic'] and p['ref_slam_topic'] != p['match_topic']:
                self.create_subscription(Odometry, p['ref_slam_topic'], self.cb_ref_slam, qos)
        if cfg.use_gnss or cfg.use_gnss_heading:
            self.create_subscription(Odometry, p['gnss_topic'], self.cb_gnss, qos)
            if p['use_pos_type']:
                self.create_subscription(String, p['gnss_raw_topic'], self.cb_gnss_raw, qos)
        if cfg.r_mode == 'occ':
            self.create_subscription(Float32, p['occ_topic'], self.cb_occ, qos)
        self.create_subscription(Int32, p['waypoint_topic'], self.cb_waypoint, qos)

        self.pub = self.create_publisher(Odometry, p['output_topic'], 10)
        self.pub_debug = self.create_publisher(Float64MultiArray, p['output_topic'] + '/debug', 10)
        self.br = tf2_ros.TransformBroadcaster(self) if p['publish_tf'] else None

        self.get_logger().info(
            f"ekf_gnss_match: out={p['output_topic']} use_gnss={cfg.use_gnss} "
            f'use_slam={cfg.use_slam} r_mode={cfg.r_mode} use_bias={self.fusion.ekf.use_bias} '
            f"match={p['match_topic']} R_match={cfg.match_sigma:.2f}m "
            f'gnss_off: rects={len(self.fusion.rects)} wp={len(self.fusion.wp_ranges)} '
            f"tf={p['publish_tf']}")

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ---------- callbacks ----------
    def cb_odom(self, msg):
        q = msg.pose.pose.orientation
        pos = msg.pose.pose.position
        now = self._now()
        if self.fusion.on_odom(now, _stamp_sec(msg.header.stamp), pos.x, pos.y,
                               yaw_from_quaternion(q.x, q.y, q.z, q.w)):
            self._publish(msg, now)

    def cb_match(self, msg):
        pos = msg.pose.pose.position
        self.fusion.on_match(self._now(), pos.x, pos.y,
                             is_ref_slam=(self.p['match_topic'] == self.p['ref_slam_topic']))

    def cb_ref_slam(self, msg):
        self.fusion.on_ref_slam(self._now())

    def cb_gnss(self, msg):
        q = msg.pose.pose.orientation
        pos = msg.pose.pose.position
        cov = msg.pose.covariance
        self.fusion.on_gnss(self._now(), pos.x, pos.y, yaw_from_quaternion(q.x, q.y, q.z, q.w),
                            sats=cov[0], var7=cov[7])

    def cb_gnss_raw(self, msg):
        self.fusion.on_gnss_raw(self._now(), msg.data)

    def cb_occ(self, msg):
        self.fusion.on_occ(self._now(), msg.data)

    def cb_waypoint(self, msg):
        self.fusion.on_waypoint(msg.data)

    # ---------- output ----------
    def _publish(self, msg_in, now):
        x, y, yaw, p_pos, yaw_var = self.fusion.output()
        out = Odometry()
        if _stamp_sec(msg_in.header.stamp) > 0.0:
            out.header.stamp = msg_in.header.stamp
        else:
            out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.p['frame_id']
        out.child_frame_id = self.p['child_frame_id']
        out.pose.pose.position.x = x
        out.pose.pose.position.y = y
        out.pose.pose.orientation.z = math.sin(yaw / 2.0)
        out.pose.pose.orientation.w = math.cos(yaw / 2.0)
        cov = [0.0] * 36
        cov[0], cov[1] = float(p_pos[0, 0]), float(p_pos[0, 1])
        cov[6], cov[7] = float(p_pos[1, 0]), float(p_pos[1, 1])
        cov[35] = float(yaw_var)
        out.pose.covariance = cov
        out.twist = msg_in.twist
        self.pub.publish(out)

        dbg = Float64MultiArray()
        dbg.layout.dim = [MultiArrayDimension(label=','.join(DEBUG_FIELDS),
                                              size=len(DEBUG_FIELDS), stride=len(DEBUG_FIELDS))]
        dbg.data = self.fusion.debug_vector(now)
        self.pub_debug.publish(dbg)

        if self.br is not None:
            t = TransformStamped()
            t.header.stamp = out.header.stamp
            t.header.frame_id = self.p['frame_id']
            t.child_frame_id = self.p['child_frame_id']
            t.transform.translation.x = x
            t.transform.translation.y = y
            t.transform.rotation.z = out.pose.pose.orientation.z
            t.transform.rotation.w = out.pose.pose.orientation.w
            self.br.sendTransform(t)

        self._log_status(now)

    def _log_status(self, now):
        period = self.p['status_log_sec']
        if period <= 0.0 or (self.last_status_log is not None
                             and now - self.last_status_log < period):
            return
        self.last_status_log = now
        d = self.fusion.debug
        self.get_logger().info(
            f"pos_sigma={d['pos_sigma']:.3f}m gnss={int(d['gnss_status'])}"
            f"(rej {int(d['gnss_reject_count'])}, sigma {d['gnss_sigma']:.3f}) "
            f"match={int(d['match_status'])}(rej {int(d['match_reject_count'])}) "
            f"yaw_offset={math.degrees(d['yaw_offset']):.2f}deg "
            f"bias=({d['bias_x']:.2f},{d['bias_y']:.2f}) zone={int(d['in_gnss_off_zone'])}")


def main(args=None):
    rclpy.init(args=args)
    node = EkfGnssMatch()
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
