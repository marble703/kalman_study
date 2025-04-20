#include "KF/include/KF.hpp"
#include "EKF/include/EKF.hpp"

#include "std_msgs/msg/float32.hpp" // IWYU pragma: keep
#include <iostream>
#include <memory>
#include <rclcpp/rclcpp.hpp>

enum FilterType {
    KALMANFLITER = 0,
    EXTENDKALMANFLITER,
    UNSCENTEDKALMANFLITER
};

int main() {
    rclcpp::init(0, nullptr);

    auto node = std::make_shared<rclcpp::Node>("kf_test_node");
    auto publisher = node->create_publisher<std_msgs::msg::Float32>("yaw_filter", 10);

    int fliter_type = node->declare_parameter("fliter_type", 0);

    double yaw = 0.0;

    std::shared_ptr<double> arg = std::make_shared<double>(0.003);

    // clang-format off
    auto bindMatrix = utils::BindableMatrixXd::create(
            2,
            2,
            false,
            1.0, [arg]() { return *arg; },
            0.0, 1.0
        );
    // clang-format on

    bindMatrix.setArg(arg);

    Eigen::Matrix<double, 1, 2> h; // 观测矩阵
    h << 1, 0;
    Eigen::Matrix<double, 2, 1> b; // 控制输入矩阵
    b << 0, 1;
    Eigen::Matrix<double, 2, 2> q; // 过程噪声协方差矩阵
    q << 0.01, 0, 0, 1;
    Eigen::Matrix<double, 1, 1> r; // 观测噪声协方差矩阵
    r << 100;
    Eigen::Matrix<double, 2, 1> initState; // 初始状态矩阵
    initState << 0, 0;
    KF kf(bindMatrix, h, b, q, r);

    if(fliter_type == KALMANFLITER) {
        KF kf(bindMatrix, h, b, q, r);
    } else if (fliter_type == EXTENDKALMANFLITER) {
        EKF kf(bindMatrix, h, b, q, r);
    } else if (fliter_type == UNSCENTEDKALMANFLITER) {
        // UKF kf(bindMatrix, h, b, q, r);
    }

    kf.init(initState);

    auto subscriber = node->create_subscription<std_msgs::msg::Float32>(
        "shoot_info2",
        10,
        [&yaw, &kf, &publisher](const std_msgs::msg::Float32::SharedPtr msg) {
            yaw = msg->data;
            std::cout << "Received yaw: " << yaw << std::endl;

            Eigen::Matrix<double, 1, 1> measurement; // 观测值矩阵
            measurement << yaw;
            kf.update(measurement);

            auto state = kf.getState();
            std::cout << "Filtered yaw: " << state(0, 0) << "\n" << std::endl;
            std::cout << "state: " << state << "\n" << std::endl;

            std_msgs::msg::Float32 Filteredmsg;
            Filteredmsg.data = state(0, 0);
            publisher->publish(Filteredmsg);
        }
    );

    rclcpp::WaitSet waitset;
    waitset.add_subscription(subscriber);

    // 启动定时预测线程
    std::thread waitSetThread([&waitset, &kf, &publisher]() {
        while (rclcpp::ok()) {
            auto wait_result = waitset.wait(std::chrono::milliseconds(9));
            if (wait_result.kind() == rclcpp::WaitResultKind::Timeout) {
                for (int i = 0; i < 3; ++i) {
                    kf.predict();
                    auto state = kf.getState();
                    std::cout << "Predicted state: " << state(0, 0) << "\n" << std::endl;
                    std::cout << "state: " << state << "\n" << std::endl;
                    std_msgs::msg::Float32 Filteredmsg;
                    Filteredmsg.data = state(0, 0);
                    publisher->publish(Filteredmsg);
                    std::this_thread::sleep_for(std::chrono::milliseconds(3));
                }
            }
        }
    });
    // 分离线程，让主线程运行 spin
    waitSetThread.detach();

    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}