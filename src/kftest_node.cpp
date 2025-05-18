#include "Fliter.hpp"
#include "DataLoader.hpp"

#include <chrono>
#include <cstdint>
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
        this->declare_parameter<bool>("use_generated_data", false);
        this->declare_parameter<std::string>("data_path", "");
        
        // KF参数
        this->declare_parameter<int>("kf.state_dim", 2);
        this->declare_parameter<int>("kf.measure_dim", 1);
        this->declare_parameter<int>("kf.control_dim", 1);
        this->declare_parameter<std::vector<double>>("kf.H", {1.0, 0.0});
        this->declare_parameter<std::vector<double>>("kf.B", {0.0, 1.0});
        this->declare_parameter<std::vector<double>>("kf.Q", {0.1, 0.0, 0.0, 0.01});
        this->declare_parameter<std::vector<double>>("kf.R", {10.0});
        this->declare_parameter<std::vector<double>>("kf.init_state", {0.0, 0.0});
        
        // EKF参数
        this->declare_parameter<int>("ekf.state_dim", 3);
        this->declare_parameter<int>("ekf.measure_dim", 1);
        this->declare_parameter<int>("ekf.control_dim", 1);
        this->declare_parameter<std::vector<double>>("ekf.H", {1.0, 0.0, 0.0});
        this->declare_parameter<std::vector<double>>("ekf.B", {0.0, 0.0, 1.0});
        this->declare_parameter<std::vector<double>>("ekf.Q", {0.1, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.001});
        this->declare_parameter<std::vector<double>>("ekf.R", {10.0});
        this->declare_parameter<std::vector<double>>("ekf.init_state", {0.0, 0.0, 0.0});
        
        // UKF参数
        this->declare_parameter<int>("ukf.state_dim", 2);
        this->declare_parameter<int>("ukf.measure_dim", 1);
        this->declare_parameter<int>("ukf.control_dim", 1);
        this->declare_parameter<std::vector<double>>("ukf.H", {1.0, 0.0});
        this->declare_parameter<std::vector<double>>("ukf.B", {0.0, 1.0});
        this->declare_parameter<std::vector<double>>("ukf.Q", {0.001, 0.0, 0.0, 0.001});
        this->declare_parameter<std::vector<double>>("ukf.R", {1000.0});
        this->declare_parameter<std::vector<double>>("ukf.init_state", {0.0, 0.0});
    }
    
    // 辅助函数：从参数向量构建动态大小矩阵
    template<typename T>
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> getMatrixFromParams(const std::vector<T>& params, int rows, int cols) {
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> matrix(rows, cols);
        if (static_cast<int>(params.size()) == rows * cols) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    matrix(i, j) = params[i * cols + j];
                }
            }
        } else {
            RCLCPP_ERROR(this->get_logger(), "参数数量与矩阵维度不匹配: %ld != %d*%d", 
                        params.size(), rows, cols);
        }
        return matrix;
    }
    
    // 从配置中获取动态大小矩阵
    template<typename T>
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> getMatrixFromConfig(const std::string& param_name, int rows, int cols) {
        std::vector<T> param_values;
        this->get_parameter(param_name, param_values);
        return getMatrixFromParams<T>(param_values, rows, cols);
    }
    
    // 保留旧的固定大小矩阵方法（兼容现有代码）
    template<typename T, int Rows, int Cols>
    Eigen::Matrix<T, Rows, Cols> getMatrixFromParams(const std::vector<T>& params) {
        Eigen::Matrix<T, Rows, Cols> matrix;
        if (params.size() == Rows * Cols) {
            for (int i = 0; i < Rows; i++) {
                for (int j = 0; j < Cols; j++) {
                    matrix(i, j) = params[i * Cols + j];
                }
            }
        } else {
            RCLCPP_ERROR(this->get_logger(), "参数数量与矩阵维度不匹配: %ld != %d*%d", 
                        params.size(), Rows, Cols);
        }
        return matrix;
    }
    
    // 保留旧的固定大小矩阵方法（兼容现有代码）
    template<typename T, int Rows, int Cols>
    Eigen::Matrix<T, Rows, Cols> getMatrixFromConfig(const std::string& param_name) {
        std::vector<T> param_values;
        this->get_parameter(param_name, param_values);
        return getMatrixFromParams<T, Rows, Cols>(param_values);
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
    bool use_generated_data = node->get_parameter("use_generated_data").as_bool();
    std::string data_path = node->get_parameter("data_path").as_string();

    double yaw = 0.0;

    std::shared_ptr<double> dt = std::make_shared<double>(0.003);

    // clang-format off
    auto bindMatrix = utils::BindableMatrixXd::create(
            2,
            2,
            false,
            1.0, [dt]() { return *dt; },
            0.0, 1.0
        );
    // clang-format on

    bindMatrix.setArg(dt);

    std::shared_ptr<FilterBase> kf;

    // 具体滤波器的初始化
    if (fliter_type == KALMANFLITER) {
        std::cout << "Using Kalman Filter" << std::endl;

        // 从配置中读取矩阵维度
        int state_dim = node->get_parameter("kf.state_dim").as_int();
        int measure_dim = node->get_parameter("kf.measure_dim").as_int();
        int control_dim = node->get_parameter("kf.control_dim").as_int();

        RCLCPP_INFO(node->get_logger(), "KF: 配置维度为 state_dim=%d, measure_dim=%d, control_dim=%d",
                   state_dim, measure_dim, control_dim);
        
        // 从配置中读取矩阵（使用动态维度）
        auto h = node->getMatrixFromConfig<double>("kf.H", measure_dim, state_dim); // 观测矩阵
        auto b = node->getMatrixFromConfig<double>("kf.B", state_dim, control_dim); // 控制输入矩阵
        auto q = node->getMatrixFromConfig<double>("kf.Q", state_dim, state_dim); // 过程噪声协方差矩阵
        auto r = node->getMatrixFromConfig<double>("kf.R", measure_dim, measure_dim); // 观测噪声协方差矩阵
        auto initState = node->getMatrixFromConfig<double>("kf.init_state", state_dim, 1); // 初始状态矩阵

        // 打印读取到的矩阵参数
        RCLCPP_INFO(node->get_logger(), "KF参数已加载");
        std::cout << "H =\n" << h << std::endl;
        std::cout << "B =\n" << b << std::endl;
        std::cout << "Q =\n" << q << std::endl;
        std::cout << "R =\n" << r << std::endl;
        std::cout << "initState =\n" << initState << std::endl;

        // 创建动态状态转移矩阵
        // 根据状态维度动态创建状态转移矩阵
        utils::BindableMatrixXd dynamicBindMatrix(state_dim, state_dim);
        // 设置状态转移矩阵的值（根据实际情况调整）
        for (int i = 0; i < state_dim; i++) {
            for (int j = 0; j < state_dim; j++) {
                if (i == j) {
                    dynamicBindMatrix.getMatrix()(i, j) = 1.0;  // 对角线元素为1
                } else if (j == i + 1 && i < state_dim - 1) {
                    dynamicBindMatrix.getMatrix()(i, j) = *dt;  // 上对角线元素为dt
                } else {
                    dynamicBindMatrix.getMatrix()(i, j) = 0.0;  // 其他元素为0
                }
            }
        }
        dynamicBindMatrix.setArg(dt);

        kf = std::make_shared<KF>(dynamicBindMatrix, h, b, q, r);
        kf->init(initState);

    } else if (fliter_type == EXTENDKALMANFLITER) {
        std::cout << "Using Extended Kalman Filter" << std::endl;

        // 从配置中读取矩阵维度
        int state_dim = node->get_parameter("ekf.state_dim").as_int();
        int measure_dim = node->get_parameter("ekf.measure_dim").as_int();
        int control_dim = node->get_parameter("ekf.control_dim").as_int();

        RCLCPP_INFO(node->get_logger(), "EKF: 配置维度为 state_dim=%d, measure_dim=%d, control_dim=%d",
                   state_dim, measure_dim, control_dim);
        
        // 从配置中读取矩阵（使用动态维度）
        auto h = node->getMatrixFromConfig<double>("ekf.H", measure_dim, state_dim); // 观测矩阵
        auto b = node->getMatrixFromConfig<double>("ekf.B", state_dim, control_dim); // 控制输入矩阵
        auto q = node->getMatrixFromConfig<double>("ekf.Q", state_dim, state_dim); // 过程噪声协方差矩阵
        auto r = node->getMatrixFromConfig<double>("ekf.R", measure_dim, measure_dim); // 观测噪声协方差矩阵
        auto initState = node->getMatrixFromConfig<double>("ekf.init_state", state_dim, 1); // 初始状态矩阵

        // 打印读取到的矩阵参数
        RCLCPP_INFO(node->get_logger(), "EKF参数已加载");
        std::cout << "H =\n" << h << std::endl;
        std::cout << "B =\n" << b << std::endl;
        std::cout << "Q =\n" << q << std::endl;
        std::cout << "R =\n" << r << std::endl;
        std::cout << "initState =\n" << initState << std::endl;

        // 定义非线性状态转移函数
        auto stateFn = [state_dim](const Eigen::MatrixXd& x, std::shared_ptr<double> dt) -> Eigen::MatrixXd {
            Eigen::MatrixXd result(state_dim, 1);
            // 默认实现针对三维状态 [位置, 速度, 加速度]
            if (state_dim >= 3) {
                result(0, 0) = x(0, 0) + x(1, 0) * *dt + 0.5 * x(2, 0) * (*dt) * (*dt);
                result(1, 0) = x(1, 0) + x(2, 0) * *dt;
                result(2, 0) = x(2, 0);
                
                // 对于更高维度的状态，保持不变
                for (int i = 3; i < state_dim; i++) {
                    result(i, 0) = x(i, 0);
                }
            } else {
                // 对于低维状态，提供简单实现
                for (int i = 0; i < state_dim; i++) {
                    if (i == 0 && state_dim > 1) {
                        result(0, 0) = x(0, 0) + x(1, 0) * *dt; // 位置 = 上次位置 + 速度*dt
                    } else {
                        result(i, 0) = x(i, 0); // 其他状态保持不变
                    }
                }
            }
            return result;
        };

        // 定义非线性观测函数
        auto measureFn = [measure_dim, state_dim](const Eigen::MatrixXd& x) -> Eigen::MatrixXd {
            // 非线性观测方程，默认观测位置
            Eigen::MatrixXd result(measure_dim, 1);
            for (int i = 0; i < measure_dim; i++) {
                if (i < state_dim) {
                    result(i, 0) = x(i, 0); // 直接观测状态的对应分量
                } else {
                    result(i, 0) = 0.0; // 超出状态维度的观测设为0
                }
            }
            return result;
        };

        // 使用非线性EKF构造函数，传入动态维度
        kf = std::make_shared<EKF>(stateFn, measureFn, state_dim, measure_dim, b, q, r, dt);
        kf->init(initState);

    } else if (fliter_type == UNSCENTEDKALMANFLITER) {
        std::cout << "Using Unscented Kalman Filter" << std::endl;

        // 从配置中读取矩阵维度
        int state_dim = node->get_parameter("ukf.state_dim").as_int();
        int measure_dim = node->get_parameter("ukf.measure_dim").as_int();
        int control_dim = node->get_parameter("ukf.control_dim").as_int();

        RCLCPP_INFO(node->get_logger(), "UKF: 配置维度为 state_dim=%d, measure_dim=%d, control_dim=%d",
                   state_dim, measure_dim, control_dim);
        
        // 从配置中读取矩阵（使用动态维度）
        auto h = node->getMatrixFromConfig<double>("ukf.H", measure_dim, state_dim); // 观测矩阵
        auto b = node->getMatrixFromConfig<double>("ukf.B", state_dim, control_dim); // 控制输入矩阵
        auto q = node->getMatrixFromConfig<double>("ukf.Q", state_dim, state_dim); // 过程噪声协方差矩阵
        auto r = node->getMatrixFromConfig<double>("ukf.R", measure_dim, measure_dim); // 观测噪声协方差矩阵
        auto initState = node->getMatrixFromConfig<double>("ukf.init_state", state_dim, 1); // 初始状态矩阵

        // 打印读取到的矩阵参数
        RCLCPP_INFO(node->get_logger(), "UKF参数已加载");
        std::cout << "H =\n" << h << std::endl;
        std::cout << "B =\n" << b << std::endl;
        std::cout << "Q =\n" << q << std::endl;
        std::cout << "R =\n" << r << std::endl;
        std::cout << "initState =\n" << initState << std::endl;

        // 定义状态转移函数和观测函数
        auto f = [state_dim](const Eigen::MatrixXd& x) -> Eigen::MatrixXd {
            // 状态转移方程，动态适应状态维度
            Eigen::MatrixXd result(state_dim, 1);
            
            if (state_dim > 1) {
                // 简单运动学模型：位置 = 上次位置 + 速度
                result(0, 0) = x(0, 0) + x(1, 0);
                result(1, 0) = x(1, 0);  // 速度不变
                
                // 对于更高维度的状态，保持不变
                for (int i = 2; i < state_dim; i++) {
                    result(i, 0) = x(i, 0);
                }
            } else {
                // 只有一维状态时，保持不变
                result(0, 0) = x(0, 0);
            }
            
            return result;
        };

        // 观测函数
        auto h_ukf = [measure_dim, state_dim](const Eigen::MatrixXd& x) -> Eigen::MatrixXd {
            // 观测方程，默认观测位置
            Eigen::MatrixXd result(measure_dim, 1);
            for (int i = 0; i < measure_dim; i++) {
                if (i < state_dim) {
                    result(i, 0) = x(i, 0); // 直接观测状态的对应分量
                } else {
                    result(i, 0) = 0.0; // 超出状态维度的观测设为0
                }
            }
            return result;
        };

        // 使用UKF构造函数，传入动态维度
        kf = std::make_shared<UKF>(f, h_ukf, q, r, state_dim, measure_dim, 0.003);
        kf->init(initState);
    }

    // TODO: 等 UKF 修完了改成动态时间
    // 数据接收和处理
    auto subscriber = node->create_subscription<std_msgs::msg::Float32>(
        use_generated_data ? "shoot_info_data" : "shoot_info2",
        10,
        [&yaw, &kf, &publisher, &diff_publisher](const std_msgs::msg::Float32::SharedPtr msg) {
            yaw = msg->data;
            std::cout << "Received yaw: " << yaw << std::endl;

            // 创建动态大小的观测值矩阵
            Eigen::MatrixXd measurement(1, 1);
            measurement(0, 0) = yaw;
            
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

    // 若一段时间无数据，启动预测
    rclcpp::WaitSet waitset;
    waitset.add_subscription(subscriber);

    std::thread waitSetThread([&waitset, &kf, &publisher, &dt]() {
        while (rclcpp::ok()) {
            auto wait_result = waitset.wait(std::chrono::milliseconds(90));
            if (wait_result.kind() == rclcpp::WaitResultKind::Timeout) {
                *dt = 0.09;
                kf->predict();
                auto state = kf->getState();
                std::cout << "Predicted state: " << state(0, 0) << "\n" << std::endl;
                std::cout << "state: " << state << "\n" << std::endl;
                std_msgs::msg::Float32 Filteredmsg;
                Filteredmsg.data = state(0, 0);
                publisher->publish(Filteredmsg);
                *dt = 0.003;
                std::this_thread::sleep_for(std::chrono::milliseconds(3));
            }
        }
    });
    // 分离线程，让主线程运行 spin
    waitSetThread.detach();

    if(use_generated_data) {
        // 使用生成的数据进行测试
    }

    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}