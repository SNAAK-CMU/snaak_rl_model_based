import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("realsense2_camera"),
                    "launch",
                    "rs_launch.py",
                )
            ]
        ),
        launch_arguments={
            "camera_name": "camera",
            "device_type": "d405",
            "enable_color": "true",
            "enable_depth": "true",
            "depth_module.enable_auto_exposure": "false",
            "depth_module.exposure": "12000",
        }.items(),
    )
    
    dynamixel_node = Node(
        package='dynamixel_sdk_examples',
        executable='read_write_node',
        name='dynamixel_node'
    )
    
    manipulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('snaak_manipulation'),
                'launch',
                'snaak_manipulation_launch.py'
            )
        ])
    )
    weight_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('snaak_weight_read'),
                'launch',
                'snaak_weight_read_launch.py'
            )
        ])
    )

    return LaunchDescription([
        realsense_launch,
        dynamixel_node,
        manipulation_launch,
        weight_launch,
        Node(
                package="handeye_realsense",
                executable="eye2hand",
                name="handeye_publisher",
        ),
        Node(
            package='model_based_test',
            executable='test_model.py',
            name='model_based_test'
        ),
    ])