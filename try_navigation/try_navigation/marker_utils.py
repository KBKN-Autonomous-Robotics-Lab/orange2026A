
import rclpy
from rclpy.node import Node
import numpy as np

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from nav_msgs.msg import Odometry

class MarkerVisualizer(Node):

    def __init__(self):
        super().__init__('marker_visualizer')

        # ---------- publisher ----------
        self.marker_pub = self.create_publisher(MarkerArray, '/stop_zones_marker_array', 10)

        #-----------subscriber-----------
        self.oodm_sub = self.create_subscription(Odometry, '/odom', self.odom_callback,10)
        self.oodm_sub = self.create_subscription(Odometry, '/fusion/odom', self.fusion_odom_callback,10)


        self.pub_rate_hz = 1.0
        self.timer = self.create_timer(1.0 / self.pub_rate_hz, self.timer_callback)

        self.frame_id = 'odom'
        self.stop_xy = np.array([
            [ 64.2,  65.2,  19.0,  39.0, 1.0], #shiyakusyo
            [100.0, 101.0,  25.0,  45.0, 1.0], #dourotan1
            [177.7, 178.7,  25.0,  45.0, 1.0], #dourotan2
            [257.5, 277.5, -60.0, -59.0, 1.0], #singoumaeteisisen1
            [257.5, 277.5, -66.5, -65.5, 1.0], #singoumae1
            [256.5, 276.5, -86.2, -85.2, 1.0], #singoumaeteisisen2
            [270.3, 271.3, -99.0, -79.5, 1.0], #singoumae2
            [405.0, 425.0, -80.2, -79.2, 1.0], #ekimae oudanhodou1 y-1
            [545.0, 568.0, -85.0, -60.0, 0.0], #ekimae not stop
            [410.0, 417.0, -71.5, -70.5, 1.0], #ekimae oudanhodou2 y-2
            [290.6, 291.6, -98.0, -78.0, 1.0], #singoumaeteisisen3
            [284.3, 285.3, -98.0, -78.0, 1.0], #singoumae3
            [259.5, 279.5, -83.7, -82.7, 1.0], #singoumaeteisisen4 12
            [259.5, 279.5, -80.3, -79.3, 1.0], #singoumae4 13
            [185.0, 186.0,  25.0,  45.0, 1.0], #dourotan3
            [107.5, 108.5,  25.0,  45.0, 1.0], #dourotan4
            [ 64.0, 104.0, -30.0, -25.0, 1.0], #GOAL!!!!
            [ 999,  999,  999,  999, 0.0]
        ])
        self.gps_dr_xy_tsukuba = np.array([
            [ -60.0,  60.0,  35.0,  110.0, 0],
            [ 476.0, 600.0,  -84.0,  -25.0, 1.0],
            [ 999,  999,  999,  999, 0.0]
        ])
        self.gps_dr_xy_nakaniwa = np.array([
            [ -41.0,  7.0,  -52.0,  71.0, 0],
            [ 999,  999,  999,  999, 0.0]
        ])

        self.ekf_noGPS_area = np.array([
            [-60.0,  60.0, 200.0, 210.0, 1.0], #shiyakusyo
            [260.0, 275.0, -66.0, -40.0, 1.0], #nakaniwa
            [539.0, 600.0, -84.0, -25.0, 1.0] ]) #

        #------stop marker state------ 
        self.stop_markers = []
        self.next_stop_id = 1000
        self.prev_moving = True
        self.current_pose = None
        self.current_twist = None
        self.fusion_current_pose = None

        #------ threshold for stop place-------
        self.linear_vel_threshold = 0.10  #m/s
        self.angular_vel_threshold = 0.10 #rad / s

        self.get_logger().info('MarkerVisualizer started')

    def timer_callback(self):
        now = self.get_clock().now().to_msg()
        ma_stop = create_zone_markers(self.stop_xy, self.frame_id, now, ns='stop_zone', start_id=0)
        ma_dr = create_zone_markers(self.ekf_noGPS_area, self.frame_id, now, ns='ma_dr', start_id=1)

        ma = MarkerArray()
        ma.markers.extend(ma_stop.markers)
        ma.markers.extend(ma_dr.markers)
        # 永続的な停止マーカーを追加（最新 timestamp に更新）
        for m in self.stop_markers:
            m.header.stamp = now
            ma.markers.append(m)

        if ma.markers:
            self.marker_pub.publish(ma)
    
    def fusion_odom_callback(self, msg:Odometry):
        self.fusion_current_pose = msg.pose.pose

    def odom_callback(self, msg: Odometry):
        #self.fusion_current_pose = msg.pose.pose
        self.current_twist = msg.twist.twist

        lin = self.current_twist.linear
        ang = self.current_twist.angular
        #lin_speed = (lin.x**2 + lin.y**2 + lin.z**2) ** 0.5
        #ang_speed = (ang.x**2 + ang.y**2 + ang.z**2) ** 0.5

        #is_moving = not (lin_speed < self.linear_vel_threshold and ang_speed < self.angular_vel_threshold)
        is_moving = not (abs(lin.x) < self.linear_vel_threshold and abs(ang.z) < self.angular_vel_threshold)

        if self.fusion_current_pose is None:
            return
        if not is_moving:
            print(f"lin : {lin.x:.5f}, ang : {ang.z:.5f}")

        # 動いていた -> 停止へ遷移した瞬間を検出してマーカー追加
        if self.prev_moving and (not is_moving) and (self.fusion_current_pose is not None):
            # 停止地点マーカーを作成して保持
            now = self.get_clock().now().to_msg()
            m = create_stop_marker(self.fusion_current_pose, self.frame_id, now, self.next_stop_id)
            self.next_stop_id += 1
            self.stop_markers.append(m)
            self.get_logger().info(f'Stop detected, marker added id={m.id} x={m.pose.position.x:.2f} y={m.pose.position.y:.2f}')

        self.prev_moving = is_moving


def create_zone_markers(range_xy, frame_id, now, ns , start_id = 0) -> MarkerArray:
    ma = MarkerArray()

    for idx, row in enumerate(range_xy):
        xmin, xmax, ymin, ymax, flag = row.tolist()

        # sentinel 行 を skip
        if xmin >= 999 or ymin >= 999:
            continue

        base_id = start_id + idx * 10
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0

        # outline 
        outline = Marker()
        outline.header.frame_id = frame_id
        outline.header.stamp = now
        outline.ns = ns
        outline.id = base_id + 0
        outline.type = Marker.LINE_STRIP
        outline.action = Marker.ADD
        outline.scale.x = 0.1
        if flag == 1.0:
            outline.color.r, outline.color.g, outline.color.b, outline.color.a = 1.0, 0.0, 0.0, 1.0
        else:
            outline.color.r, outline.color.g, outline.color.b, outline.color.a = 1.0, 1.0, 0.0, 1.0
        p1 = Point(x=xmin, y=ymin, z=0.0)
        p2 = Point(x=xmax, y=ymin, z=0.0)
        p3 = Point(x=xmax, y=ymax, z=0.0)
        p4 = Point(x=xmin, y=ymax, z=0.0)
        outline.points = [p1, p2, p3, p4, p1]
        ma.markers.append(outline)

        # fill 
        fill = Marker()
        fill.header.frame_id = frame_id
        fill.header.stamp = now
        fill.ns = ns
        fill.id = base_id + 1
        fill.type = Marker.CUBE
        fill.action = Marker.ADD
        fill.pose.position.x = float(cx)
        fill.pose.position.y = float(cy)
        fill.pose.position.z = 0.1
        fill.pose.orientation.w = 1.0
        fill.scale.x = float(max(0.001, (xmax - xmin)))
        fill.scale.y = float(max(0.001, (ymax - ymin)))
        fill.scale.z = 0.2
        if flag == 1.0:
            fill.color.r, fill.color.g, fill.color.b, fill.color.a = 1.0, 0.0, 0.0, 0.15
        else:
            fill.color.r, fill.color.g, fill.color.b, fill.color.a = 0.0, 1.0, 0.0, 0.15
        #ma.markers.append(fill)

        # 3) label
        text = Marker()
        text.header.frame_id = frame_id
        text.header.stamp = now
        text.ns = ns
        text.id = base_id + 2
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = float(cx)
        text.pose.position.y = float(cy)
        text.pose.position.z = 0.4
        text.pose.orientation.w = 1.0
        text.scale.z = 0.4
        text.text = f'zone_{idx} flag={int(flag)}'
        text.color.r, text.color.g, text.color.b, text.color.a = 1.0, 1.0, 1.0, 1.0
        ma.markers.append(text)
    return ma

def create_stop_marker(pose, frame_id, now, mid) -> Marker:
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = now
    m.ns = 'stop_points'
    m.id = mid
    m.type = Marker.SPHERE
    m.action = Marker.ADD
    m.pose = pose

    # 少し浮かせる
    m.pose.position.z = 0.2
    m.scale.x = 0.4
    m.scale.y = 0.4
    m.scale.z = 0.4

    # 色（半透明赤）
    m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.0, 0.0, 0.8
    return m

def main(args=None):
    rclpy.init(args=args)
    node = MarkerVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

