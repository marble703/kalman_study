from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 获取包的共享目录路径
    package_dir = get_package_share_directory('kftest')
    
    # 默认数据文件路径
    default_data_path = os.path.join(
        os.path.dirname(os.path.dirname(package_dir)), 
        'DataLoader/DataGenerator/output/trajectory.txt'
    )
    
    # 创建数据加载器节点
    data_loader_node = Node(
        package='kftest',
        executable='data_loader',
        name='data_loader_node',
        output='screen',
        arguments=[default_data_path]
    )
    
    return LaunchDescription([
        data_loader_node
    ])
