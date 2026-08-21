import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    rviz_config_dir = os.path.join(
        get_package_share_directory('try_navigation'),
        'config', 'config.rviz')
    assert os.path.exists(rviz_config_dir)
    #get livox data    
    livox_to_pointcloud2_launch_file = os.path.join(
        get_package_share_directory('livox_to_pointcloud2'),
        'launch',
        'livox_to_pointcloud2.launch.py'
    )
    
    # define config
    odom = LaunchConfiguration('odom')
    declare_odom_arg = DeclareLaunchArgument(
        'odom',
        default_value='/odom_ekf_match', # odom_ekf_match
        description='Odometry topic name'
    )
    
    waypoint_path = LaunchConfiguration('waypoint_path')
    declare_waypoint_arg = DeclareLaunchArgument(
        'waypoint_path',
        default_value='kbkn_maps/waypoints/tsukuba/2025/papa/tsukuba_waypoint.yaml',
        description='waypoint name'
    )
    #kbkn_maps/waypoints/tsukuba/2025/papa/tsukuba_waypoint.yaml
    #kbkn_maps/waypoints/hosei/2025/nakaniwa.yaml
    waypoint_start_index = LaunchConfiguration('waypoint_start_index')
    declare_waypoint_start_arg = DeclareLaunchArgument(
        'waypoint_start_index',
        default_value= '0',
        description='waypoint start index'
    )
    
    return LaunchDescription([
        declare_odom_arg, declare_waypoint_arg, declare_waypoint_start_arg,
        
        #rviz2
        Node(package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_dir],
            parameters=[{'use_sim_time': False}],
            output='screen'
        ),
        #get livox data
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(livox_to_pointcloud2_launch_file)
        ),
        
        #pcd rotation
        Node(package='pcd_convert',
            executable='pcd_rotation',
            name='pcd_rotation_node',
            output='screen',
            parameters=[{'use_sim_time': False}],
            arguments=[]
        ),
        Node(package='pcd_convert',
            executable='pcd_rotation_lidar2',
            name='pcd_rotation_lidar2_node',
            output='screen',
            parameters=[{'use_sim_time': False}],
            arguments=[]
        ),
        Node(package='pcd_convert',
            executable='pcd_merge',
            name='pointcloud_merger_node',
            output='screen',
            parameters=[{'use_sim_time': False}],
            arguments=[]
        ),
        
        #odom combination
        Node(package='orange_gnss',
            executable='odom_combination',
            name='odom_combination',
            output='screen',
            parameters=[{'use_sim_time': False}],
            arguments=[],
        ),
   
        #gps ekf edit
        Node(package='try_navigation',
            executable='ekf_myself_gps',
            name='sensor_fusion',
            output='screen',
            parameters=[{'use_sim_time': False}],
            arguments=[]
        ),
        
        #pcd segmentation
        Node(package='pcd_convert',
            executable='pcd_height_segmentation',
            name='pcd_heigth_segmentation_node',
            output='screen',
            parameters=[{'use_sim_time': False}],
            arguments=[]
        ),
        
        #pcd segmentation
        Node(package='pcd_convert',
            executable='pcd_reflect_segmentation',
            name='pcd_reflect_segmentation_node',
            output='screen',
            parameters=[{'use_sim_time': False}],
            arguments=[]
        ),
        
        #odom wheel
        #Node(package='try_navigation',
        #    executable='odom_wheel',
        #    name='odom_wheel_node',
        #    output='screen',
        #    arguments=[]
        #),
        
        #waypoint manager
        # waypoint gps command
        #Node(package='navigation_control',
        #    executable='gps_waypoint',
        #    name='gps_waypoint',
        #    output='screen',
        #    parameters=[{'odom': odom},
        #                {'waypoint_path': waypoint_path},
        #                {'waypoint_start_index': waypoint_start_index}],
        #    arguments=[],
        #),
        # $ ros2 run navigation_control gps_waypoint
        # file path /ros2_ws/src/Use_action/navigation_control/navigation_control/gps_waypoint
        
        #reflection intensity map
        #Node(package='try_navigation',
        #    executable='reflection_intensity_map',
        #    name='reflection_intensity_map_node',
        #    output='screen',
        #    arguments=[],
        #),
        ##path planning
        Node(package='try_navigation',
            executable='potential_astar',
            name='potential_astar_node',
            output='screen',
            parameters=[{'odom': odom}, {'use_sim_time': False}],
            arguments=[],
        ),
        #robot ctrl
        Node(package='try_navigation',
            executable='path_follower',
            name='path_follower_node',
            output='screen',
            parameters=[{'odom': odom}, {'use_sim_time': False}],
            arguments=[],
        ),
        
        #navigation start
        Node(package='navigation_control',
            executable='button',
            name='button',
            output='screen',
            parameters=[{'use_sim_time': False}],
            arguments=[],
        ),
        #takamori Autonav
        #Node(package='try_navigation',
        #    executable='reflection_to_pcd',
        #    name='reflection_to_pcd',
        #    output='screen',
        #    arguments=[],
        #),
        #takamori Selfdrive
        #Node(package='try_navigation',
        #    executable='self_drive_line',
        #    name='self_drive_line',
        #    output='screen',
        #    arguments=[],
        #),
    ])