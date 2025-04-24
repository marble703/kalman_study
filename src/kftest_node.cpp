#include "EKF/include/EKF.hpp"
#include "KF/include/KF.hpp"
#include "UKF/include/UKF.hpp"
#include <chrono>
#include <thread>

#include "std_msgs/msg/float32.hpp" // IWYU pragma: keep
#include <iostream>
#include <memory>
#include <rclcpp/rclcpp.hpp>

class FliterNode: public rclcpp::Node {
public:
    FliterNode(): Node("kftest_node") {
        this->declare_parameter<int>("fliter_type", 0);
        this->declare_parameter<bool>("debug", false);
    }
};
enum FilterType { KALMANFLITER = 0, EXTENDKALMANFLITER, UNSCENTEDKALMANFLITER };

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);

    auto node = std::make_shared<FliterNode>();
    auto publisher = node->create_publisher<std_msgs::msg::Float32>("yaw_filter", 10);
    auto diff_publisher = node->create_publisher<std_msgs::msg::Float32>("yaw_diff", 10);

    // 等待一小段时间让参数加载
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    int fliter_type = node->get_parameter("fliter_type").as_int();
    // bool debug = node->get_parameter("debug").as_bool();

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
    q << 0.1, 0, 0, 0.01;
    Eigen::Matrix<double, 1, 1> r; // 观测噪声协方差矩阵
    r << 100;
    Eigen::Matrix<double, 2, 1> initState; // 初始状态矩阵
    initState << 0, 0;

    std::shared_ptr<FilterBase> kf;

    if (fliter_type == KALMANFLITER) {
        std::cout << "Using Kalman Filter" << std::endl;
        kf = std::make_shared<KF>(bindMatrix, h, b, q, r);
    } else if (fliter_type == EXTENDKALMANFLITER) // 由于这里使用的线性模型，实际上退化为线性卡尔曼滤波器
    {
        std::cout << "Using Extended Kalman Filter" << std::endl;
        kf = std::make_shared<EKF>(bindMatrix, h, b, q, r);
    } else if (fliter_type == UNSCENTEDKALMANFLITER) {
        std::cout << "Using Unscented Kalman Filter" << std::endl;

        // 定义状态转移函数和观测函数
        auto f = [](const Eigen::MatrixXd& x) -> Eigen::MatrixXd {
            // 状态转移方程
            Eigen::MatrixXd result(2, 1);
            result(0, 0) = x(0, 0) + x(1, 0); // 位置 = 上次位置 + 速度
            result(1, 0) = x(1, 0);          // 速度不变
            return result;
        };

        // 观测函数
        auto h_ukf = [](const Eigen::MatrixXd& x) -> Eigen::MatrixXd {
            // 观测方程
            Eigen::MatrixXd result(1, 1);
            result(0, 0) = x(0, 0); // 观测位置
            return result;
        };

        kf = std::make_shared<UKF>(f, h_ukf, q, r, 2, 1, 0.003);
    }

    kf->init(initState);

    auto subscriber = node->create_subscription<std_msgs::msg::Float32>(
        "shoot_info2",
        10,
        [&yaw, &kf, &publisher, &diff_publisher](const std_msgs::msg::Float32::SharedPtr msg) {
            yaw = msg->data;
            std::cout << "Received yaw: " << yaw << std::endl;

            Eigen::Matrix<double, 1, 1> measurement; // 观测值矩阵
            measurement << yaw;
            kf->predict();
            kf->update(measurement);

            auto state = kf->getState();
            std::cout << "Filtered yaw: " << state(0, 0) << "\n" << std::endl;
            std::cout << "state: " << state << "\n" << std::endl;

            std_msgs::msg::Float32 Filteredmsg;
            Filteredmsg.data = state(0, 0);

            publisher->publish(Filteredmsg);

            std_msgs::msg::Float32 diffmsg;
            diffmsg.data = yaw - state(0, 0);

            diff_publisher->publish(diffmsg);
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
                    kf->predict();
                    auto state = kf->getState();
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