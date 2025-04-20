import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import GroupAction
from launch_ros.actions import Node

launch_name = "kftest"
executable_name = 'kftest_node'

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory(launch_name),
        'config', 'config.yaml'
    )

    load_nodes=GroupAction(
        actions=[
            Node(
                package=launch_name,
                executable=executable_name,
                output='screen',
                parameters=[config],
                emulate_tty=True
            ),
        ]
    )
    ld = LaunchDescription()
    ld.add_action(load_nodes)
    return ld