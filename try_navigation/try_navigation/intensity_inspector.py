#!/usr/bin/env python3
"""intensity_inspector.py

/pcd_segment_ground の反射強度を鳥瞰マップと横断プロファイルで確認する。

蓄積は odom 補正つき（road_edge_stop と同じ方式）。走行中でも点がぶれない。
鳥瞰マップ上の緑の破線は road_edge_stop の円形 ROI（roi_r_min / roi_r_max）。

キー:
  [ ]      蓄積フレーム数 -1 / +1
  a d      プロファイル位置 -0.5 / +0.5 m
  r        カラースケール自動フィット (p2-p98)
  f        カラースケール 0-255
  s        現在の蓄積点群を npz 保存
  q        終了
"""

import math
import os
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Circle


_DTYPE_MAP = {
    1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
    5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64,
}


def pointcloud2_to_xyzi(msg):
    """PointCloud2 -> (N,4) float32 [x, y, z, intensity]"""
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
    cols = {}
    for f in msg.fields:
        if f.name not in ('x', 'y', 'z', 'intensity'):
            continue
        dt = np.dtype(_DTYPE_MAP[f.datatype])
        seg = raw[:, f.offset:f.offset + dt.itemsize]
        cols[f.name] = np.frombuffer(seg.tobytes(), dtype=dt).astype(np.float32)

    n = raw.shape[0]
    if 'intensity' not in cols:
        cols['intensity'] = np.zeros(n, dtype=np.float32)
    return np.stack([cols['x'], cols['y'], cols['z'], cols['intensity']], axis=1)


def quat_to_yaw(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def rot2d(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


class IntensityInspector(Node):

    def __init__(self):
        super().__init__('intensity_inspector')

        self.declare_parameter('topic', '/pcd_segment_ground')
        self.declare_parameter('odom_topic', '/odom/wheel_spimu')
        self.declare_parameter('map_range', 10.0)
        self.declare_parameter('grid_res', 0.10)
        self.declare_parameter('profile_x', 3.0)
        self.declare_parameter('profile_band', 0.25)
        self.declare_parameter('profile_y_range', 4.0)
        self.declare_parameter('profile_bin', 0.05)
        self.declare_parameter('edge_smooth', 3)
        self.declare_parameter('accum', 10)
        self.declare_parameter('accum_max', 30)
        self.declare_parameter('draw_rate', 2.0)
        self.declare_parameter('save_dir', '~/ros2_ws/maps/intensity_log')
        self.declare_parameter('roi_r_min', 0.5)
        self.declare_parameter('roi_r_max', 5.0)
        self.declare_parameter('roi_half_angle_deg', 100.0)

        g = lambda k: self.get_parameter(k).value
        self.topic = g('topic')
        self.odom_topic = g('odom_topic')
        self.map_range = float(g('map_range'))
        self.grid_res = float(g('grid_res'))
        self.profile_x = float(g('profile_x'))
        self.profile_band = float(g('profile_band'))
        self.profile_y_range = float(g('profile_y_range'))
        self.profile_bin = float(g('profile_bin'))
        self.edge_smooth = int(g('edge_smooth'))
        self.accum_max = int(g('accum_max'))
        self.accum = max(1, min(int(g('accum')), self.accum_max))
        self.draw_rate = float(g('draw_rate'))
        self.save_dir = os.path.expanduser(g('save_dir'))
        self.roi_r_min = float(g('roi_r_min'))
        self.roi_r_max = float(g('roi_r_max'))
        self.roi_half_angle_deg = float(g('roi_half_angle_deg'))
        os.makedirs(self.save_dir, exist_ok=True)

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(PointCloud2, self.topic, self.cb, qos)
        self.create_subscription(Odometry, self.odom_topic, self.cb_odom, qos)

        self._lock = threading.Lock()
        self._buf = deque(maxlen=self.accum_max)

        self.odom_ok = False
        self.px = 0.0
        self.py = 0.0
        self.yaw = 0.0

        self.get_logger().info(
            f'subscribing: {self.topic} / {self.odom_topic}  accum={self.accum}')

    # ------------------------------------------------------------

    def cb_odom(self, msg):
        p = msg.pose.pose.position
        with self._lock:
            self.px, self.py = p.x, p.y
            self.yaw = quat_to_yaw(msg.pose.pose.orientation)
            self.odom_ok = True

    def cb(self, msg):
        """受信した点群を odom でグローバル座標に直してから貯める"""
        pts = pointcloud2_to_xyzi(msg)
        if pts.shape[0] == 0:
            return
        with self._lock:
            px, py, yaw = self.px, self.py, self.yaw
        xy = rot2d(yaw) @ pts[:, 0:2].T + np.array([[px], [py]])
        g = np.column_stack([xy.T, pts[:, 2], pts[:, 3]]).astype(np.float32)
        with self._lock:
            self._buf.append(g)

    def get_accumulated(self):
        """直近 accum フレームを連結し、現在姿勢基準のローカル座標に戻す"""
        with self._lock:
            if not self._buf:
                return None, 0
            frames = list(self._buf)[-self.accum:]
            px, py, yaw = self.px, self.py, self.yaw
        buf = np.concatenate(frames, axis=0)
        xy = rot2d(-yaw) @ (buf[:, 0:2].T - np.array([[px], [py]]))
        local = np.column_stack([xy.T, buf[:, 2], buf[:, 3]]).astype(np.float32)
        return local, len(frames)

    def set_accum(self, n):
        self.accum = max(1, min(int(n), self.accum_max))
        
    def set_half_angle(self, deg):
        self.roi_half_angle_deg = max(10.0, min(float(deg), 180.0))


def make_birdseye(pts, map_range, res):
    """鳥瞰 intensity マップ（セル平均）。データなしセルは NaN"""
    n = int(round(2 * map_range / res))
    x, y, inten = pts[:, 0], pts[:, 1], pts[:, 3]

    m = (np.abs(x) < map_range) & (np.abs(y) < map_range)
    x, y, inten = x[m], y[m], inten[m]
    if x.size == 0:
        return np.full((n, n), np.nan, dtype=np.float32)

    ix = ((x + map_range) / res).astype(np.int32)
    iy = ((y + map_range) / res).astype(np.int32)
    np.clip(ix, 0, n - 1, out=ix)
    np.clip(iy, 0, n - 1, out=iy)
    flat = ix * n + iy

    s = np.bincount(flat, weights=inten, minlength=n * n)
    c = np.bincount(flat, minlength=n * n)
    with np.errstate(invalid='ignore', divide='ignore'):
        img = (s / c).reshape(n, n).astype(np.float32)
    img[c.reshape(n, n) == 0] = np.nan
    return img


def make_profile(pts, px, band, y_range, y_bin):
    """前方 px[m] の帯を左右方向にビン平均"""
    k = int(round(2 * y_range / y_bin))
    y_edges = np.linspace(-y_range, y_range, k + 1)
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    x, y, inten = pts[:, 0], pts[:, 1], pts[:, 3]
    m = (np.abs(x - px) < band) & (np.abs(y) < y_range)
    if not np.any(m):
        return y_centers, np.full(k, np.nan, dtype=np.float32)

    idx = np.clip(((y[m] + y_range) / y_bin).astype(np.int32), 0, k - 1)
    s = np.bincount(idx, weights=inten[m], minlength=k)
    c = np.bincount(idx, minlength=k)
    with np.errstate(invalid='ignore', divide='ignore'):
        prof = (s / c).astype(np.float32)
    prof[c == 0] = np.nan
    return y_centers, prof


def detect_band_edges(y_centers, prof, smooth=3):
    """微分の正/負ピークから帯の両端を1本ずつ検出"""
    v = prof.copy()
    ok = ~np.isnan(v)
    if ok.sum() < 2 * smooth + 5:
        return None
    v = np.interp(np.arange(v.size), np.flatnonzero(ok), v[ok])

    kern = np.ones(smooth) / smooth
    v = np.convolve(v, kern, mode='same')

    d = np.diff(v)
    d[:smooth] = 0.0
    d[-smooth:] = 0.0
    if d.size < 3:
        return None

    i_rise = int(np.argmax(d))
    i_fall = int(np.argmin(d))
    if i_rise == i_fall:
        return None

    i_a, i_b = sorted((i_rise, i_fall))
    if i_b - i_a < 2:
        return None

    inside = prof[i_a + 1:i_b + 1]
    outside = np.concatenate([prof[:i_a + 1], prof[i_b + 1:]])
    inside = inside[~np.isnan(inside)]
    outside = outside[~np.isnan(outside)]
    if inside.size == 0 or outside.size == 0:
        return None

    y_lo = float(y_centers[i_a])
    y_hi = float(y_centers[i_b])
    return {
        'y_lo': y_lo,
        'y_hi': y_hi,
        'width': abs(y_hi - y_lo),
        'step_rise': float(d[i_rise]),
        'step_fall': float(-d[i_fall]),
        'mean_inside': float(inside.mean()),
        'mean_outside': float(outside.mean()),
        'contrast': float(inside.mean() - outside.mean()),
    }


class Viewer:

    HELP = '[ ]=accum  z/x=fan angle  a/d=profile x  r=autoscale  f=0-255  s=save  q=quit'
    def __init__(self, node):
        self.node = node
        self.running = True
        self.last_pts = None
        self.last_img = None
        self._slider_busy = False

        plt.rcParams['font.size'] = 9
        self.fig = plt.figure(figsize=(12.5, 6.0))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0],
                                   left=0.06, right=0.97,
                                   top=0.92, bottom=0.20,
                                   wspace=0.24)
        self.ax_map = self.fig.add_subplot(gs[0, 0])
        self.ax_prof = self.fig.add_subplot(gs[0, 1])

        mr = node.map_range
        n = int(round(2 * mr / node.grid_res))
        self.im = self.ax_map.imshow(
            np.full((n, n), np.nan), origin='lower',
            extent=[-mr, mr, -mr, mr],
            cmap='viridis', vmin=0, vmax=255, interpolation='nearest')
        self.ax_map.set_xlim(mr, -mr)
        self.ax_map.set_xlabel('y  left [m]')
        self.ax_map.set_ylabel('x  forward [m]')
        self.ax_map.grid(True, lw=0.3, alpha=0.4)
        self.cbar = self.fig.colorbar(self.im, ax=self.ax_map,
                                      fraction=0.046, pad=0.03)
        self.cbar.set_label('intensity')

        # --- 円形 ROI（road_edge_stop と同じ範囲）---
        for _r in (node.roi_r_min, node.roi_r_max):
            self.ax_map.add_patch(Circle((0, 0), _r, fill=False,
                                         ec='#1D9E75', lw=1.2, ls='--'))
        self.ax_map.text(mr - 0.3, node.roi_r_max + 0.2,
                         f'ROI {node.roi_r_min:.1f}-{node.roi_r_max:.1f} m',
                         color='#1D9E75', fontsize=8, ha='left')
                
        # --- 前方扇形（roi_half_angle_deg）---
        self.ln_fan, = self.ax_map.plot([], [], '-', lw=1.3,
                                        color='#D85A30', alpha=0.9)
        self.txt_fan = self.ax_map.text(
            0.02, 0.02, '', transform=self.ax_map.transAxes,
            color='#D85A30', fontsize=9, family='monospace')
        self.update_fan()       
                         

        self.band_lo = self.ax_map.axhline(0, color='r', lw=0.8, ls='--')
        self.band_hi = self.ax_map.axhline(0, color='r', lw=0.8, ls='--')
        self._install_hover()

        # --- 操作説明を画面下部に常時表示 ---
        self.fig.text(0.5, 0.012, self.HELP, ha='center',
                      fontsize=8.5, family='monospace', color='#555555')

        ax_vmin = self.fig.add_axes([0.08, 0.085, 0.36, 0.025])
        ax_vmax = self.fig.add_axes([0.08, 0.040, 0.36, 0.025])
        self.s_vmin = Slider(ax_vmin, 'vmin', 0, 255, valinit=0, valstep=1,
                             color='#85B7EB')
        self.s_vmax = Slider(ax_vmax, 'vmax', 0, 255, valinit=255, valstep=1,
                             color='#85B7EB')
        self.s_vmin.on_changed(self._on_slider)
        self.s_vmax.on_changed(self._on_slider)

        yr = node.profile_y_range
        self.ln_prof, = self.ax_prof.plot([], [], '-', lw=1.4, color='#185FA5')
        self.edge_lo = self.ax_prof.axvline(0, color='#D85A30', lw=1.1,
                                            ls='--', visible=False)
        self.edge_hi = self.ax_prof.axvline(0, color='#D85A30', lw=1.1,
                                            ls='--', visible=False)
        self.edge_mid = self.ax_prof.axvline(0, color='#1D9E75', lw=1.1,
                                             ls=':', visible=False)
        self.ax_prof.set_xlim(yr, -yr)
        self.ax_prof.set_ylim(0, 255)
        self.ax_prof.set_xlabel('y  left [m]')
        self.ax_prof.set_ylabel('intensity')
        self.ax_prof.grid(True, lw=0.3, alpha=0.5)
        self.txt_prof = self.ax_prof.text(
            0.02, 0.96, '', transform=self.ax_prof.transAxes,
            va='top', ha='left', fontsize=8.5, family='monospace')

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('close_event', lambda e: self.stop())
        plt.show(block=False)

    def stop(self):
        self.running = False

    def _on_slider(self, _val):
        if self._slider_busy:
            return
        lo, hi = self.s_vmin.val, self.s_vmax.val
        if hi <= lo:
            hi = lo + 1
        self.im.set_clim(lo, hi)
        self.ax_prof.set_ylim(lo, hi)

    def _set_clim(self, lo, hi):
        self._slider_busy = True
        self.s_vmin.set_val(lo)
        self.s_vmax.set_val(hi)
        self._slider_busy = False
        self._on_slider(None)

    def autoscale(self):
        if self.last_img is None:
            return
        v = self.last_img[~np.isnan(self.last_img)]
        if v.size < 50:
            return
        lo, hi = np.percentile(v, [2, 98])
        if hi - lo < 2:
            lo, hi = lo - 1, hi + 1
        self._set_clim(float(np.floor(lo)), float(np.ceil(hi)))
        
    def update_fan(self):
        """前方扇形の外形線を描く（原点→円弧→原点）"""
        node = self.node
        half = math.radians(node.roi_half_angle_deg)
        r = node.roi_r_max
        a = np.linspace(-half, half, 60)

        # 描画は (横軸=y_left, 縦軸=x_forward)
        px = np.concatenate([[0.0], r * np.sin(a), [0.0]])
        py = np.concatenate([[0.0], r * np.cos(a), [0.0]])
        self.ln_fan.set_data(px, py)
        self.txt_fan.set_text(f'fan +-{node.roi_half_angle_deg:.0f} deg')

    def _install_hover(self):
        node = self.node

        def fmt(xv, yv):
            img = self.im.get_array()
            n = img.shape[0]
            iy = int((xv + node.map_range) / node.grid_res)
            ix = int((yv + node.map_range) / node.grid_res)
            if 0 <= ix < n and 0 <= iy < n:
                v = img[ix, iy]
                if not np.ma.is_masked(v) and not np.isnan(v):
                    return f'y={xv:+.2f}  x={yv:+.2f}  intensity={float(v):.1f}'
            return f'y={xv:+.2f}  x={yv:+.2f}  intensity=--'

        self.ax_map.format_coord = fmt

    def on_key(self, ev):
        node = self.node
        if ev.key == 'q':
            self.stop()
        elif ev.key == 's':
            self.save()
        elif ev.key == 'a':
            node.profile_x = max(0.5, node.profile_x - 0.5)
        elif ev.key == 'd':
            node.profile_x = min(node.map_range - 0.5, node.profile_x + 0.5)
        elif ev.key == '[':
            node.set_accum(node.accum - 1)
        elif ev.key == ']':
            node.set_accum(node.accum + 1)
        elif ev.key == 'z':
            node.set_half_angle(node.roi_half_angle_deg - 5)
            self.update_fan()
        elif ev.key == 'x':
            node.set_half_angle(node.roi_half_angle_deg + 5)
            self.update_fan()
        elif ev.key == 'r':
            self.autoscale()
        elif ev.key == 'f':
            self._set_clim(0, 255)

    def save(self):
        if self.last_pts is None:
            return
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(self.node.save_dir, f'ground_{ts}.npz')
        np.savez_compressed(path, points=self.last_pts,
                            profile_x=self.node.profile_x,
                            accum=self.node.accum)
        print(f'[save] {path}  ({self.last_pts.shape[0]} points)')

    def update(self):
        node = self.node
        pts, nframe = node.get_accumulated()
        if pts is None:
            return
        self.last_pts = pts

        img = make_birdseye(pts, node.map_range, node.grid_res)
        self.last_img = img
        self.im.set_data(img)
        odom_tag = 'odom OK' if node.odom_ok else 'odom --'
        self.ax_map.set_title(
            f'birds-eye intensity   accum {nframe}/{node.accum}   '
            f'{pts.shape[0]} pts   {odom_tag}')

        yc, prof = make_profile(pts, node.profile_x, node.profile_band,
                                node.profile_y_range, node.profile_bin)
        self.ln_prof.set_data(yc, prof)

        e = detect_band_edges(yc, prof, node.edge_smooth)
        if e is not None:
            mid = 0.5 * (e['y_lo'] + e['y_hi'])
            self.edge_lo.set_xdata([e['y_lo'], e['y_lo']])
            self.edge_hi.set_xdata([e['y_hi'], e['y_hi']])
            self.edge_mid.set_xdata([mid, mid])
            for a in (self.edge_lo, self.edge_hi, self.edge_mid):
                a.set_visible(True)
            info = (f"x = {node.profile_x:.1f} m\n"
                    f"edges  {e['y_lo']:+.2f} / {e['y_hi']:+.2f} m\n"
                    f"width  {e['width']:.2f} m   mid {mid:+.2f} m\n"
                    f"step   rise {e['step_rise']:.1f}  fall {e['step_fall']:.1f}\n"
                    f"inside {e['mean_inside']:.1f}  outside {e['mean_outside']:.1f}\n"
                    f"contrast {e['contrast']:+.1f}")
        else:
            for a in (self.edge_lo, self.edge_hi, self.edge_mid):
                a.set_visible(False)
            valid = prof[~np.isnan(prof)]
            if valid.size:
                info = (f'x = {node.profile_x:.1f} m\n'
                        f'no clear edge\n'
                        f'mean {valid.mean():5.1f}  sd {valid.std():4.1f}')
            else:
                info = f'x = {node.profile_x:.1f} m\nno data'
        self.txt_prof.set_text(info)
        self.ax_prof.set_title(f'cross-section  (x = {node.profile_x:.1f} m)')

        lo = node.profile_x - node.profile_band
        hi = node.profile_x + node.profile_band
        self.band_lo.set_ydata([lo, lo])
        self.band_hi.set_ydata([hi, hi])

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


def main():
    rclpy.init()
    node = IntensityInspector()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    viewer = Viewer(node)
    print(Viewer.HELP)

    period = 1.0 / max(0.2, node.draw_rate)
    try:
        while viewer.running and rclpy.ok():
            t0 = time.time()
            viewer.update()
            dt = time.time() - t0
            plt.pause(max(0.01, period - dt))
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
