#pragma once

#include <eigen3/Eigen/Dense>
#include <optional>

#include "Utils/BindMatrix.hpp"

class FilterBase {
public:
    virtual ~FilterBase() = default;

    // 传入初始状态,初始化滤波器
    virtual void init(const Eigen::MatrixXd& initState) = 0;

    // 使用观测值和时间间隔更新滤波器
    virtual void
    update(const Eigen::MatrixXd& measurement, double dt = 0.0, bool setDefault = false) = 0;

    // 使用控制量更新滤波器
    virtual void control(const Eigen::MatrixXd& controlInput, double dt = 0.0) = 0;

    // 使用传入的时间间隔预测
    virtual void predict(double dt = 0.0) = 0;

    // 重置状态到指定值
    virtual void resetState(const Eigen::MatrixXd& currentState) = 0;

    // 获取当前状态
    virtual Eigen::MatrixXd getState() const = 0;

protected:
    // 矩阵维度
    int ObservationSize_; // 观测维度
    int StateSize_;       // 状态维度
    int ControlSize_;     // 控制输入维度

    // 模型矩阵(固定)
    const utils::BindableMatrixXd f_; // 状态转移矩阵,大小为StateSize_ * StateSize_
    const Eigen::MatrixXd h_;         // 观测矩阵,大小为ObservationSize_ * StateSize_
    const Eigen::MatrixXd b_;         // 控制输入矩阵,大小为StateSize_ * ControlSize_

    // 初始量
    Eigen::Matrix<double, Eigen::Dynamic, 1> initState_; // 初始状态矩阵,大小为StateSize_ * 1
    Eigen::MatrixXd p0_; // 初始协方差矩阵,大小为StateSize_ * StateSize_
    Eigen::MatrixXd q0_; // 初始过程噪声协方差矩阵,大小为StateSize_ * StateSize_
    Eigen::MatrixXd r0_; // 初始观测噪声协方差矩阵,大小为ObservationSize_ * ObservationSize_

    // 噪声协方差矩阵
    Eigen::MatrixXd q_; // 过程噪声协方差矩阵,大小为StateSize_ * StateSize_
    Eigen::MatrixXd r_; // 观测噪声协方差矩阵,大小为ObservationSize_ * ObservationSize_
    Eigen::MatrixXd k_; // 卡尔曼增益矩阵,大小为StateSize_ * ObservationSize_

    // 当前量
    Eigen::MatrixXd current_state_; // 当前状态矩阵,大小为StateSize_*StateSize_
    Eigen::Matrix<double, Eigen::Dynamic, 1> controlInput_; // 控制输入矩阵,大小为ControlSize_ * 1
    Eigen::Matrix<double, Eigen::Dynamic, 1> measurement_; // 观测值矩阵,大小为ObservationSize_ * 1
    Eigen::MatrixXd KalmanGain_; // 卡尔曼增益矩阵,大小为StateSize_ * ObservationSize_
    Eigen::MatrixXd p_;          // 协方差矩阵,大小为StateSize_ * StateSize_

    // 预测量
    Eigen::MatrixXd predictedState_; // 预测状态矩阵

    double dt_; // 时间间隔
};