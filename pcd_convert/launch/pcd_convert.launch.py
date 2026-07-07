import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

# 標準出力と標準エラーを/dev/nullにリダイレクト
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

def generate_launch_description():
    return LaunchDescription([
        Node(package='pcd_convert',
            executable='pcd_rotation',
            name='pcd_rotation_node',
            output='screen',
            arguments=[]
        ),
        Node(package='pcd_convert',
            executable='pcd_rotation_lidar2',
            name='pcd_rotation_lidar2_node',
            output='screen',
            arguments=[]
        ),
        Node(package='pcd_convert',
            executable='pcd_merge',
            name='pointcloud_merger_node',
            output='screen',
            arguments=[]
        ),
        Node(package='pcd_convert',
            executable='pcd_height_segmentation',
            name='pcd_heigth_segmentation_node',
            output='screen',
            arguments=[]
        ),
    ])
    
