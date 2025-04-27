#pragma once

#include <eigen3/Eigen/Dense>

#include "Utils/BindMatrix.hpp"

class FilterBase {
public:
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
};