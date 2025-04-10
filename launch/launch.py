import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import GroupAction
from launch_ros.actions import Node

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('kftest'),
        'config', 'config.yaml'
    )

    load_nodes=GroupAction(
        actions=[
            Node(
                package='kftest',
                executable='kftest_node',
                output='screen',
                parameters=[config],
                emulate_tty=True
            ),
        ]
    )
    ld = LaunchDescription()
    ld.add_action(load_nodes)
    return ld