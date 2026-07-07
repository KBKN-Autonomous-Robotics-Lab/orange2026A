#!/usr/bin/env python3
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

        self.declare_parameter('port', '/dev/sensors/gps')
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
        
        # service client
        self.client = self.create_client(Avglatlon, 'send_avg_gps')
        #while not self.client.wait_for_service(timeout_sec=1.0):
        #    self.get_logger().info("service not available...")

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

        start_time = time.time()
        while time.time() - start_time < 10:  # 10 seconds
            GPS_data = self.get_gps_quat(self.dev_name, self.country_id)
            if GPS_data and GPS_data[1] != 0 and GPS_data[2] != 0:
                lat_sum += GPS_data[1]
                lon_sum += GPS_data[2]
                #heading_sum += float(GPS_data[5])
                count += 1
            time.sleep(0.1)  # Slight delay to avoid overwhelming the GPS device

        if count > 0:
            #self.initial_coordinate = [lat_sum / count, lon_sum / count] # calculate average
            self.initial_coordinate = self.start_GPS_coordinate
            self.current_coordinate = [lat_sum / count, lon_sum / count] # for tsukuba
            #self.theta = (heading_sum / count) - 90
            self.initialized = True
            self.get_logger().info(f"Initial coordinate set to: {self.initial_coordinate}")
            self.get_logger().info(f"current coordinate set to: {self.current_coordinate}")
            self.get_logger().info(f"Initial theta set to: {self.tsukuba_theta}")
            self.send_request()
        self.is_acquiring = False

    def get_gps_quat(self, dev_name, country_id):
        # interface with sensor device(as a serial port)
        try:
            serial_port = serial.Serial(dev_name, self.serial_baud)
        except serial.SerialException as serialerror:
            self.get_logger().error(f"Serial error: {serialerror}")
            return None
        
        # country info 
        if country_id == 0:   # Japan
            initial_letters = b"GNGGA"
        elif country_id == 1: # USA
            initial_letters = b"GPGGA"
        else:                 # not certain
            initial_letters = None
        
        initial_letters_outdoor = b"$GNHDT"
        initial_letters_indoor = b"$GPHDT"

        while(1):
            line_heading = serial_port.readline()
            #self.get_logger().info(f"line: {line}")
            talker_ID_indoor = line_heading.find(initial_letters_indoor)
            talker_ID_outdoor = line_heading.find(initial_letters_outdoor)            
            if talker_ID_indoor != -1:
                #self.get_logger().info("GPHDT ok")
                #line = line[(talker_ID_indoor-1):]
                gps_data = line_heading.split(b",")
                #self.get_logger().info(f"gps_data: {gps_data}")
                heading = float(gps_data[1])
                if heading is None:
                    self.get_logger().error("not GPS heading data")
                    heading = 0
                break
            if talker_ID_outdoor != -1:
                #self.get_logger().info("GNHDT ok")
                #line = line[(talker_ID_outdoor-1):]
                gps_data = line_heading.split(b",")
                #self.get_logger().info(f"gps_data: {gps_data}")
                heading = float(gps_data[1])
                if heading is None:
                    self.get_logger().error("not GPS heading data")
                    heading = 0
                break

#    gps_data = ["$G?GGA", 
#                "UTC time", 
#                "Latitude (ddmm.mmmmm)", 
#                "latitude type (south/north)", 
#                "Longitude (ddmm.mmmmm)", 
#                "longitude type (east longitude/west longitude)", 
#                "Fixtype", 
#                "Number of satellites used for positioning", 
#                "HDOP", 
#                "Altitude", 
#                "M(meter)", 
#                "Elevation", 
#                "M(meter)", 
#                "", 
#                "checksum"]
    
        line_latlon = serial_port.readline()
        talker_ID = line_latlon.find(initial_letters)
        if talker_ID != -1:
            line_latlon = line_latlon[(talker_ID-1):]
            gps_data = line_latlon.split(b",")
            Fixtype_data = int(gps_data[6])
            if Fixtype_data != 0:
                satelitecount_data = int(gps_data[7])###
                if Fixtype_data != 0:
                    latitude_data = float(gps_data[2]) / 100.0  # ddmm.mmmmm to dd.ddddd
                    if gps_data[3] == b"S":#south
                        latitude_data *= -1
                    longitude_data = float(gps_data[4]) / 100.0  # ddmm.mmmmm to dd.ddddd
                    if gps_data[5] == b"W":#west
                        longitude_data *= -1
                    altitude_data = float(gps_data[9])
                else :
                    #not fix data
                    Fixtype_data = 0
                    latitude_data = 0
                    longitude_data = 0
                    altitude_data = 0
                    satelitecount_data = 0
                    self.get_logger().error("!--not fix data--!")
            else :
            #no GPS data
                Fixtype_data = 0
                latitude_data = 0
                longitude_data = 0
                altitude_data = 0
                satelitecount_data = 0
                self.get_logger().error("!--not GPS data--!")          
        
        serial_port.close()
        
        #self.publish_raw_latlon(line_latlon)
        #self.publish_raw_heading(line_heading)
        self.publish_raw_gps(line_latlon, line_heading)
        self.save_csv(line_latlon,line_heading)

        gnggadata = (Fixtype_data,latitude_data,longitude_data,altitude_data,satelitecount_data,heading)
        return gnggadata
        
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
        self.raw_gps_msg.data = f"HDT:{self.raw_heading_msg},GGA:{self.raw_latlon_msg}"
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
    gpslonlat.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
