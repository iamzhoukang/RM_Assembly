from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    config_dir = Path(get_package_share_directory("enginer_net")) / "config"
    return LaunchDescription([
        DeclareLaunchArgument("show_window", default_value="true"),
        Node(
            package="enginer_net",
            executable="net_node",
            name="net_node",
            output="screen",
            parameters=[
                str(config_dir / "net.yaml"),
                str(config_dir / "pnp.yaml"),
                {"show_window": ParameterValue(LaunchConfiguration("show_window"), value_type=bool)},
            ],
        ),
    ])
