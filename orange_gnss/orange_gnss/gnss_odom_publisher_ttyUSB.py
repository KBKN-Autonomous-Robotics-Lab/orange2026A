#!/usr/bin/env python3
import rclpy
import serial
import math
import tkinter as tk
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, NavSatStatus
from std_msgs.msg import Header, String
import threading
import time
from my_msgs.srv import Avglatlon

class GPSData(Node):
    def __init__(self):
        super().__init__('gps_data_acquisition')

        self.declare_parameter('Position_magnification', 1.675) # 1.675
        self.declare_parameter('heading', 0.0)
        self.declare_parameter('start_lat', 35.425952230280004) # tsukuba start point right 36.04974095972727, 140.04593633886364 , left 36.04976195993636, 140.04593755179093/nakaniwa 35.4257898377487,139.313807281254 /35.425952230280004, 139.31380123427
        self.declare_parameter('start_lon', 139.31380123427)

        self.Position_magnification = self.get_parameter('Position_magnification').get_parameter_value().double_value
        #self.theta = self.get_parameter('heading').get_parameter_value().double_value
        self.tsukuba_theta= self.get_parameter('heading').get_parameter_value().double_value # nakaniwa 180 tsukuba 93

        self.initial_coordinate = None
        self.start_lat = self.get_parameter('start_lat').get_parameter_value().double_value
        self.start_lon = self.get_parameter('start_lon').get_parameter_value().double_value
        self.start_GPS_coordinate = [self.start_lat, self.start_lon]
        self.fix_data = None
        self.count = 0
        
        # subscriber
        #self.latlon_sub = self.create_subscription(String, "/gps_raw_latlon", self.get_gps_latlon, 1)
        #self.heading_sub = self.create_subscription(String, "/gps_raw_heading", self.get_gps_heading, 1)
        self.gps_sub = self.create_subscription(String, "/gps_raw", self.get_gps_data, 1)
        
        # publisher
        self.odom_pub = self.create_publisher(Odometry, "/odom/UM982", 10)
        self.odom_msg = Odometry()
                
        #self.timer = self.create_timer(1.0, self.publish_GPS_lonlat_quat)
        self.initialized = False
        
        self.latlon_data = None
        self.heading = None

    def get_gps_data(self, msg):
        line = msg.data
        parts = line.split("GGA:")
        hdt_part = parts[0].replace("HDT:", "").strip()
        gga_part = parts[1].strip()

        # --- HDT --- 
        hdt_fields = hdt_part.split(",")
        heading = float(hdt_fields[1])  # 93.12°
        self.heading = heading

        # --- GGA ---
        gga_fields = gga_part.split(",")
        Fixtype_data = int(gga_fields[6])

        latitude_data = float(gga_fields[2]) / 100.0
        longitude_data = float(gga_fields[4]) / 100.0
        altitude_data = float(gga_fields[9])
        satelitecount_data = int(gga_fields[7])
        self.latlon_data = (Fixtype_data,latitude_data,longitude_data,altitude_data,satelitecount_data)
        
        self.publish_GPS_lonlat_quat()
    
    def quaternion_from_euler(self, roll, pitch, yaw):
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        q = [0] * 4
        q[0] = cy * cp * cr + sy * sp * sr
        q[1] = cy * cp * sr - sy * sp * cr
        q[2] = sy * cp * sr + cy * sp * cr
        q[3] = sy * cp * cr - cy * sp * sr
        return q
    
    
    def heading_to_quat(self ,real_heading):

        robotheading = real_heading - 90
        if robotheading >= 360:
            robotheading -= 360

        #self.get_logger().info(f"real_heading: {real_heading}")
        #self.get_logger().info(f"robotheading: {robotheading}")

        if self.count == 0:
            self.get_logger().info(f"!!!----------robotheading: {robotheading} deg----------!!!")
            #self.first_heading = robotheading
            #self.first_heading = self.tsukuba_theta
            self.first_heading = 0
            self.count = 1

        relative_heading = robotheading - self.first_heading
        if relative_heading < 0:
            relative_heading += 360

        if relative_heading > 180:
            relative_heading -= 360

        movingbaseyaw = relative_heading * (math.pi / 180)

        roll, pitch = 0.0, 0.0
        yaw = movingbaseyaw

        q = self.quaternion_from_euler(roll, pitch, yaw)
        # self.get_logger().info(f"Quaternion: {q}")
            
        return q
    
    def conversion(self, coordinate, origin, theta):
        ido = coordinate[0]
        keido = coordinate[1]
        ido0 = origin[0]
        keido0 = origin[1]

        # self.get_logger().info(f"theta: {theta}")

        a = 6378137
        f = 35/10439
        e1 = 734/8971
        e2 = 127/1547
        n = 35/20843
        a0 = 1
        a2 = 102/40495
        a4 = 1/378280
        a6 = 1/289634371
        a8 = 1/204422462123
        pi180 = 71/4068
        # %math.pi/180
        d_ido = ido - ido0
        d_keido = keido - keido0
        rd_ido = d_ido * pi180
        rd_keido = d_keido * pi180
        r_ido = ido * pi180
        r_keido = keido * pi180
        r_ido0 = ido0 * pi180
        W = math.sqrt(1-(e1**2)*(math.sin(r_ido)**2))
        N = a / W
        t = math.tan(r_ido)
        ai = e2*math.cos(r_ido)

       # %===Y===%
        S = a*(a0*r_ido - a2*math.sin(2*r_ido)+a4*math.sin(4*r_ido) -
               a6*math.sin(6*r_ido)+a8*math.sin(8*r_ido))/(1+n)
        S0 = a*(a0*r_ido0-a2*math.sin(2*r_ido0)+a4*math.sin(4*r_ido0) -
                a6*math.sin(6*r_ido0)+a8*math.sin(8*r_ido0))/(1+n)
        m0 = S/S0
        B = S-S0
        y1 = (rd_keido**2)*N*math.sin(r_ido)*math.cos(r_ido)/2
        y2 = (rd_keido**4)*N*math.sin(r_ido) * \
            (math.cos(r_ido)**3)*(5-(t**2)+9*(ai**2)+4*(ai**4))/24
        y3 = (rd_keido**6)*N*math.sin(r_ido)*(math.cos(r_ido)**5) * \
            (61-58*(t**2)+(t**4)+270*(ai**2)-330*(ai**2)*(t**2))/720
        gps_y = self.Position_magnification * m0 * (B + y1 + y2 + y3)

       # %===X===%
        x1 = rd_keido*N*math.cos(r_ido)
        x2 = (rd_keido**3)*N*(math.cos(r_ido)**3)*(1-(t**2)+(ai**2))/6
        x3 = (rd_keido**5)*N*(math.cos(r_ido)**5) * \
            (5-18*(t**2)+(t**4)+14*(ai**2)-58*(ai**2)*(t**2))/120
        gps_x = self.Position_magnification * m0 * (x1 + x2 + x3)

        # point = (gps_x, gps_y)Not match

        degree_to_radian = math.pi / 180
        r_theta = theta * degree_to_radian
        h_x = math.cos(r_theta) * gps_x - math.sin(r_theta) * gps_y
        h_y = math.sin(r_theta) * gps_x + math.cos(r_theta) * gps_y
        #point = (-h_y, h_x)
        point = (h_y, -h_x)

        return point
    
    def publish_GPS_lonlat_quat(self):        
        if self.heading is not None and self.latlon_data is not None:
            self.initialized = True
        
        if not self.initialized:
            return    
        
        GPS_data = self.latlon_data
        GPS_heading = self.heading
        #gnggadata = (Fixtype_data,latitude_data,longitude_data,altitude_data,satelitecount_data,heading)
        if GPS_data and GPS_data[1] != 0 and GPS_data[2] != 0:
            self.satelite = GPS_data[4]
            lonlat = [GPS_data[1], GPS_data[2]]
            
            # publish fix topic
            #self.lonlat_msg.header = Header()
            #self.lonlat_msg.frame_id = "gps"
            #self.lonlat_msg.header.stamp = self.get_clock().now().to_msg()
            
            #self.lonlat_msg.status.status = NavSatStatus.STATUS_FIX if lonlat[
            #    0] != 0 else NavSatStatus.STATUS_NO_FIX
            #self.lonlat_msg.latitude = GPS_data[1] # ido
            #self.lonlat_msg.longitude = GPS_data[2] # keido
            #self.lonlat_msg.altitude = GPS_data[3] # koudo
            
            #self.lonlat_pub.publish(self.lonlat_msg)
            # self.get_logger().info(f"Published GPS data: {lonlat}")           
            
            #if self.initial_coordinate is None:
            #    self.initial_coordinate = [GPS_data[1], GPS_data[2]]        
            GPSxy = self.conversion(lonlat, self.start_GPS_coordinate, 0)
            GPSquat = self.heading_to_quat(GPS_heading)       

            self.odom_msg.header.stamp = self.get_clock().now().to_msg()
            self.odom_msg.header.frame_id = "odom"
            self.odom_msg.child_frame_id = "base_footprint"
            self.odom_msg.pose.pose.position.x = GPSxy[0]
            self.odom_msg.pose.pose.position.y = GPSxy[1]

            self.odom_msg.pose.pose.orientation.x = GPSquat[1]
            self.odom_msg.pose.pose.orientation.y = GPSquat[2]
            self.odom_msg.pose.pose.orientation.z = -GPSquat[3]
            self.odom_msg.pose.pose.orientation.w = GPSquat[0]
            # Number of satellites
            self.odom_msg.pose.covariance[0] = float(self.satelite)
            self.odom_pub.publish(self.odom_msg)
        else:
            self.get_logger().error("!!!!-NOT RECIEVE-gps data error-!!!!")


def main(args=None):
    rclpy.init(args=args)
    gpslonlat = GPSData()
    rclpy.spin(gpslonlat)
    gpslonlat.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()