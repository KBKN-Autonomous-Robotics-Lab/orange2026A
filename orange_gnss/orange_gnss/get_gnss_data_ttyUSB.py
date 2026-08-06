#!/usr/bin/env python3
from orange_msgs.msg import PppNav
import math
import rclpy
import serial
import tkinter as tk
from rclpy.node import Node
from sensor_msgs.msg import Imu, NavSatFix, NavSatStatus
from nav_msgs.msg import Odometry
from std_msgs.msg import Header, String
from geometry_msgs.msg import Quaternion, Pose, Point, Twist, Vector3
import threading
import time
from my_msgs.srv import Avglatlon
import csv
from datetime import datetime
import os

class GPSData(Node):
    def __init__(self):
        super().__init__('gps_data_acquisition')

        self.declare_parameter('port', '/dev/sensors/GNSS_UM982')
        self.declare_parameter('baud', 115200)
        self.declare_parameter('country_id', 0)
        self.declare_parameter('heading', 0.0)
        self.declare_parameter('start_lat', 35.425952230280004) # tsukuba start point right 36.04974095972727, 140.04593633886364 , left 36.04976195993636, 140.04593755179093/nakaniwa 35.4257898377487,139.313807281254 /35.425952230280004, 139.31380123427
        self.declare_parameter('start_lon', 139.31380123427)

        self.dev_name = self.get_parameter('port').get_parameter_value().string_value
        self.serial_baud = self.get_parameter('baud').get_parameter_value().integer_value
        self.country_id = self.get_parameter('country_id').get_parameter_value().integer_value
        #self.theta = self.get_parameter('heading').get_parameter_value().double_value
        self.tsukuba_theta= self.get_parameter('heading').get_parameter_value().double_value # nakaniwa 180 tsukuba 93
        self.theta = self.tsukuba_theta

        self.initial_coordinate = None
        self.start_lat = self.get_parameter('start_lat').get_parameter_value().double_value
        self.start_lon = self.get_parameter('start_lon').get_parameter_value().double_value
        self.start_GPS_coordinate = [self.start_lat, self.start_lon]
        self.fix_data = None
        self.count = 0
        
        self.initialized = False  # 平均初期座標が取得できたかどうか

        # Publishers
        self.raw_latlon_pub = self.create_publisher(String, '/gps_raw_latlon', 1)
        self.raw_latlon_msg = String()
        self.raw_heading_pub = self.create_publisher(String, '/gps_raw_heading', 1)
        self.raw_heading_msg = String()
        self.raw_gps_pub = self.create_publisher(String, '/gps_raw', 1)
        self.raw_gps_msg = String()
        self.ppp_pub = self.create_publisher(PppNav, '/gnss/ppp_status', 10)
        
        # service client
        self.client = self.create_client(Avglatlon, 'send_avg_gps')
        #while not self.client.wait_for_service(timeout_sec=1.0):
        #    self.get_logger().info("service not available...")

        # serial port (open once, keep open)
        self.serial_port = None

        self.declare_parameter('cache_max_age', 3.0)
        self.cache_max_age = self.get_parameter('cache_max_age').get_parameter_value().double_value

        self.cache_lock = threading.Lock()
        self.latest_gga = None
        self.latest_gga_time = 0.0
        self.latest_hdt = None
        self.latest_hdt_time = 0.0
        self.latest_ppp = None
        self.latest_ppp_time = 0.0
        self.ppp_seq = 0
        self.published_ppp_seq = -1

        try:
            self.serial_port = serial.Serial(self.dev_name, self.serial_baud, timeout=0.1)
        except serial.SerialException as e:
            self.get_logger().error(f"Serial open failed: {e}")
        self.reader_stop = threading.Event()
        self.reader_thread = threading.Thread(target=self.serial_reader_loop, daemon=True)
        self.reader_thread.start()

        self.get_logger().info("Start get_lonlat quat node")
        self.get_logger().info("-------------------------")
        
        # Timers
        self.timer = self.create_timer(1.0, self.timer_callback)

        self.first_heading = None
        
        self.gps_data_cache = None

        self.get_logger().info("Start get_lonlat_movingbase_quat_ttyUSB node")
        self.get_logger().info("-------------------------")

        # tkinter GUI setup
        self.root = tk.Tk()
        self.root.title("GPS Data Acquisition")
        self.start_button = tk.Button(self.root, text="Start GPS Acquisition", command=self.start_gps_acquisition, width=20, height = 5)
        self.start_button.pack()

        self.gps_acquisition_thread = None
        self.is_acquiring = False
        
        # init csv
        csv_dir = "/home/ubuntu/ros2_ws/gnss_log"
        run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_csv_dir = os.path.join(csv_dir, f"run_{run_time}")
        os.makedirs(self.run_csv_dir, exist_ok=True)
        self._open_new_csv()
    
    # service client
    def send_request(self):
        request = Avglatlon.Request()
        request.avg_lat = self.initial_coordinate[0]  # ← average lat
        request.avg_lon = self.initial_coordinate[1]  # ← average lon
        request.current_lat = self.current_coordinate[0]  # ← currennt lat
        request.current_lon = self.current_coordinate[1]  # ← currennt lon
        #request.theta = self.theta
        request.theta = self.tsukuba_theta # tsukuba start theta
        request.current_theta = self.theta # for tsukuba

        future = self.client.call_async(request)
        future.add_done_callback(self.response_callback)
        
    def response_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info('サービス送信成功')
            else:
                self.get_logger().warn('サービスは受け取られましたが、処理は失敗しました')
        except Exception as e:
            self.get_logger().error(f'サービス呼び出し失敗: {e}')
           
    # timer callback
    def timer_callback(self):
        if not self.initialized:
            # 初期化が完了していないので何もしない
            return    
        self.gps_data_cache = self.get_gps_quat(self.dev_name, self.country_id)

    # gps data collect
    def start_gps_acquisition(self):
        if not self.is_acquiring:
            self.is_acquiring = True
            self.gps_acquisition_thread = threading.Thread(target=self.acquire_gps_data)
            self.gps_acquisition_thread.start()

    def acquire_gps_data(self):
        lat_sum = 0.0
        lon_sum = 0.0
        count = 0
        last_used_time = 0.0

        start_time = time.time()
        while time.time() - start_time < 10:
            with self.cache_lock:
                line = self.latest_gga
                t    = self.latest_gga_time
            if line is not None and t != last_used_time and (time.time() - t) <= self.cache_max_age:
                last_used_time = t
                parsed = self.parse_gga(line)
                if parsed is not None and parsed[0] != 0:
                    _, lat, lon, _, _ = parsed
                    if lat != 0 and lon != 0:
                        lat_sum += lat
                        lon_sum += lon
                        count += 1
            time.sleep(0.1)

        if count > 0:
            self.initial_coordinate = self.start_GPS_coordinate
            self.current_coordinate = [lat_sum / count, lon_sum / count]
            self.initialized = True
            self.get_logger().info(f"Initial coordinate set to: {self.initial_coordinate}")
            self.get_logger().info(f"current coordinate set to: {self.current_coordinate} (n={count})")
            self.get_logger().info(f"Initial theta set to: {self.tsukuba_theta}")
            self.send_request()
        else:
            self.get_logger().error("GPS data not received. initialization failed")
        self.is_acquiring = False
        
    def serial_reader_loop(self):
        """シリアルを常時読み、種類ごとに最新1件だけ保持する"""
        hdt_headers = (b"$GNHDT", b"$GPHDT")
        if self.country_id == 0:
            gga_header = b"GNGGA"
        elif self.country_id == 1:
            gga_header = b"GPGGA"
        else:
            gga_header = None

        while not self.reader_stop.is_set():
            if self.serial_port is None or not self.serial_port.is_open:
                time.sleep(0.5)
                continue
            try:
                line = self.serial_port.readline()
            except serial.SerialException as e:
                self.get_logger().error(f"Serial read error: {e}")
                time.sleep(0.5)
                continue
            if not line:
                continue

            now = time.time()

            ppp = self.parse_pppnava(line)
            if ppp is not None:
                with self.cache_lock:
                    self.latest_ppp = ppp
                    self.latest_ppp_time = now
                    self.ppp_seq += 1
                continue

            if any(h in line for h in hdt_headers):
                with self.cache_lock:
                    self.latest_hdt = line
                    self.latest_hdt_time = now
                continue

            if gga_header is not None:
                idx = line.find(gga_header)
                if idx != -1:
                    with self.cache_lock:
                        self.latest_gga = line[max(idx - 1, 0):]
                        self.latest_gga_time = now

    def parse_gga(self, line):
        """GGA1行 → (Fixtype, lat, lon, alt, satcount)。パース失敗は None"""
        gps_data = line.split(b",")
        try:
            fixtype = int(gps_data[6])
            if fixtype == 0:
                return (0, 0, 0, 0, 0)
            satcount = int(gps_data[7])
            lat = float(gps_data[2]) / 100.0
            if gps_data[3] == b"S":
                lat *= -1
            lon = float(gps_data[4]) / 100.0
            if gps_data[5] == b"W":
                lon *= -1
            alt = float(gps_data[9])
            return (fixtype, lat, lon, alt, satcount)
        except (ValueError, IndexError):
            return None

    def parse_pppnava(self, line):
        """bytes 1行を受け取り、#PPPNAVA ならdictを返す。非該当は None。"""
        idx = line.find(b'#PPPNAVA')
        if idx == -1:
            return None
        parts = line[idx:].split(b';', 1)
        if len(parts) < 2:
            return None
        f = parts[1].split(b',')
        if len(f) < 15:
            return None
        try:
            pos_type = f[1].decode('ascii', errors='ignore')
            return {
                'pos_type':    pos_type,
                'lat_sd':      float(f[7]),
                'lon_sd':      float(f[8]),
                'alt_sd':      float(f[9]),
                'diff_age':    float(f[11]),
                'sol_age':     float(f[12]),
                'num_tracked': int(f[13]),
                'num_used':    int(f[14]),
                'valid':       (pos_type != 'NONE'),
            }
        except (ValueError, IndexError):
            return None

    def publish_ppp_status(self, ppp):

        msg = PppNav()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "gps"
        msg.pos_type    = ppp['pos_type']
        msg.diff_age    = ppp['diff_age']
        msg.lat_sd      = ppp['lat_sd']
        msg.lon_sd      = ppp['lon_sd']
        msg.alt_sd      = ppp['alt_sd']
        msg.sol_age     = ppp['sol_age']
        msg.num_tracked = min(ppp['num_tracked'], 255)
        msg.num_used    = min(ppp['num_used'], 255)
        msg.valid       = ppp['valid']
        self.ppp_pub.publish(msg)

    def get_gps_quat(self, dev_name, country_id):
        now = time.time()
        with self.cache_lock:
            line_latlon  = self.latest_gga
            gga_age      = now - self.latest_gga_time
            line_heading = self.latest_hdt
            hdt_age      = now - self.latest_hdt_time
            ppp          = self.latest_ppp
            ppp_seq      = self.ppp_seq

        # PPPNAVA は新着があったときだけ publish
        if ppp is not None and ppp_seq != self.published_ppp_seq:
            self.publish_ppp_status(ppp)
            self.published_ppp_seq = ppp_seq

        if line_latlon is None or gga_age > self.cache_max_age:
            self.get_logger().error("!--GGA stale or not received--!")
            return None

        heading = 0.0
        if line_heading is not None and hdt_age <= self.cache_max_age:
            hdt_fields = line_heading.split(b",")
            if len(hdt_fields) > 1 and hdt_fields[1] != b'':
                try:
                    heading = float(hdt_fields[1])
                except ValueError:
                    heading = 0.0
        else:
            line_heading = b""
            self.get_logger().warn("HDT stale or not received")

        parsed = self.parse_gga(line_latlon)
        if parsed is None:
            self.get_logger().error("!--GGA parse error--!")
            parsed = (0, 0, 0, 0, 0)
        Fixtype_data, latitude_data, longitude_data, altitude_data, satelitecount_data = parsed
        if Fixtype_data == 0:
            self.get_logger().error("!--not fix data--!")

        self.publish_raw_gps(line_latlon, line_heading)
        self.save_csv(line_latlon, line_heading)

        return (Fixtype_data, latitude_data, longitude_data,
                altitude_data, satelitecount_data, heading)
        
    def _open_new_csv(self):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.run_csv_dir, f"gps_raw_{now}.csv")
        self.csv_file = open(filename, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["timestamp", "raw_nmea"])
        self.current_file_time = now
    
    def save_csv(self, line_latlon,line_heading):
        now = datetime.now()
        now_str = now.strftime("%Y%m%d_%H%M%S")
        if line_latlon:
            if now_str != self.current_file_time:
                self.csv_file.close()
                self._open_new_csv()

            timestamp = now.isoformat()
            line_str_latlon = line_latlon.decode("ascii", errors="ignore").strip()
            line_str_heading = line_heading.decode("ascii", errors="ignore").strip()
            self.csv_writer.writerow([timestamp, line_str_latlon, line_str_heading])

    def publish_raw_gps(self, line_latlon, line_heading):
        self.raw_latlon_msg.data = line_latlon.decode("ascii", errors="ignore").strip()
        self.raw_heading_msg.data = line_heading.decode("ascii", errors="ignore").strip()
        self.raw_gps_msg.data = (
            f"HDT:{self.raw_heading_msg.data},"
            f"GGA:{self.raw_latlon_msg.data}"
        )
        self.raw_gps_pub.publish(self.raw_gps_msg)
        self.get_logger().info(f"Publish: {self.raw_gps_msg.data}")

    def publish_raw_latlon(self, line):
        if line:
            self.raw_latlon_msg.data = line.decode("ascii", errors="ignore").strip()
            self.raw_latlon_pub.publish(self.raw_latlon_msg)
    
    def publish_raw_heading(self, line):
        if line:
            self.raw_heading_msg.data = line.decode("ascii", errors="ignore").strip()
            self.raw_heading_pub.publish(self.raw_heading_msg)

def main(args=None):
    #rclpy.init(args=args)
    #gpslonlat = GPSData()
    #rclpy.spin(gpslonlat)
    #gpslonlat.root.mainloop()
    #gpslonlat.destroy_node()
    #rclpy.shutdown()
    rclpy.init(args=args)
    gpslonlat = GPSData()    
    ros_thread = threading.Thread(target=rclpy.spin, args=(gpslonlat,))
    ros_thread.start()
    gpslonlat.root.mainloop()  # tkinter GUI表示
    gpslonlat.reader_stop.set()                   
    gpslonlat.reader_thread.join(timeout=2.0)      
    if gpslonlat.serial_port is not None and gpslonlat.serial_port.is_open:
        gpslonlat.serial_port.close()
    gpslonlat.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
