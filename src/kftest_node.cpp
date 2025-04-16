#include "KF/include/KF.hpp"
#include "EKF/include/EKF.hpp"

#include "std_msgs/msg/float32.hpp" // IWYU pragma: keep
#include <iostream>
#include <rclcpp/rclcpp.hpp>

int main() {
    rclcpp::init(0, nullptr);

    auto node = std::make_shared<rclcpp::Node>("kf_test_node");
    auto publisher = node->create_publisher<std_msgs::msg::Float32>("yaw_filter", 10);

    double dt = 0.03;
    double yaw = 0.0;

    Eigen::Matrix<double, 2, 2> f; // 状态转移矩阵
    f << 1, dt, 0, 1;
    Eigen::Matrix<double, 2, 2> h; // 观测矩阵
    h << 1, 0, 0, 1;
    Eigen::Matrix<double, 2, 1> b; // 控制输入矩阵
    b << 0, 1;
    Eigen::Matrix<double, 2, 2> q; // 过程噪声协方差矩阵
    q << 0.01, 0, 
         0, 0.01;
    Eigen::Matrix<double, 2, 2> r; // 观测噪声协方差矩阵
    r << 1, 0, 
         0, 1;
    Eigen::Matrix<double, 2, 1> initState; // 初始状态矩阵
    initState << 0, 0;
    KF kf(f, h, b, q, r, dt);
    kf.init(initState);

    auto subscriber = node->create_subscription<std_msgs::msg::Float32>(
        "shoot_info2",
        10,
        [&yaw, &kf, &publisher](const std_msgs::msg::Float32::SharedPtr msg) {
            yaw = msg->data;

            std::cout << "Received yaw: " << yaw << std::endl;
            Eigen::Matrix<double, 2, 1> measurement; // 观测值矩阵
            measurement << yaw, 0;
            kf.update(measurement);
            auto state = kf.getState();
            std::cout << "Filtered yaw: " << state(0, 0) << std::endl;
            std_msgs::msg::Float32 Filteredmsg;
            Filteredmsg.data = state(0, 0);
            publisher->publish(Filteredmsg);
        }
    );

    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}