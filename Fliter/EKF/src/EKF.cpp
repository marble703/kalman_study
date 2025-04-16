#include "EKF/include/EKF.hpp"
#include <cassert>

EKF::EKF(
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> f,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> h,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> b,
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> q,
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> r,
    double dt
):
    f_(f),
    h_(h),
    b_(b),
    q_(q),
    r_(r),
    dt_(dt) {
    assert(f_.rows() == f_.cols() && "状态转移矩阵 f 必须是方阵");
    assert(q_.rows() == q_.cols() && "过程噪声协方差矩阵 q 必须是方阵");
    assert(r_.rows() == r_.cols() && "观测噪声协方差矩阵 r 必须是方阵");
    assert(f_.rows() == h_.cols() && "状态转移矩阵 f 的行数必须等于观测矩阵 h 的列数");
    assert(h_.rows() == r_.rows() && "观测矩阵 h 的行数必须等于观测噪声协方差矩阵 r 的行数");
    assert(f_.rows() == q_.rows() && "状态转移矩阵 f 的行数必须等于过程噪声协方差矩阵 q 的行数");
    assert(b_.rows() == f_.rows() && "控制输入矩阵 b 的行数必须等于状态转移矩阵 f 的行数");

    ObservationSize_ = h_.rows();
    StateSize_ = f_.rows();
    ControlSize_ = b_.rows();

    k_ = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(StateSize_, ObservationSize_);
    initState_ = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(StateSize_, 1);
    p0_ = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Identity(StateSize_, StateSize_);
    q0_ = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Identity(StateSize_, StateSize_);
    r0_ = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Identity(ObservationSize_, ObservationSize_);

    current_state_ = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(StateSize_, 1);
    controlInput_ = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(ControlSize_, 1);
    measurement_ = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(ObservationSize_, 1);
    KalmanGain_ = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(StateSize_, ObservationSize_);
    p_ = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Identity(StateSize_, StateSize_);
    predictedState_ = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(StateSize_, StateSize_);
}

void EKF::init(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& initState) {
    assert(initState.rows() == f_.rows() && initState.cols() == 1 && "初始状态矩阵维度不匹配");
    initState_ = initState;
    current_state_ = initState_;
}

void EKF::update(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& measurement) {
    assert(measurement.rows() == h_.cols() && "测量矩阵维度不匹配");
    assert(measurement.cols() == 1 && "测量矩阵必须是列向量");
    measurement_ = measurement;
    predict();
    updateKalmanGain();
}

void EKF::update(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& measurement, float dt) {
    assert(measurement.rows() == h_.cols() && "测量矩阵维度不匹配");
    assert(measurement.cols() == 1 && "测量矩阵必须是列向量");
    assert(dt > 0 && "时间间隔 dt 必须大于 0");
    measurement_ = measurement;
    dt_ = dt;
    predict();
    updateKalmanGain();
}

void EKF::predict() {
    assert(f_.cols() == current_state_.rows() && "f_ 的列数必须等于 current_state_ 的行数");
    current_state_ = f_ * current_state_;
    p_ = f_ * p_ * f_.transpose() + q_;
}

void EKF::predict(float dt) {
    dt_ = dt;
    predict();
}

void EKF::updateKalmanGain() {
    k_ = p_ * h_.transpose() * (h_ * p_ * h_.transpose() + r_).inverse();
    current_state_ = current_state_ + k_ * (measurement_ - h_ * current_state_);
    p_ = (Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Identity(f_.rows(), f_.cols()) - k_ * h_) * p_;
}

void EKF::control(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& controlInput) {
    controlInput_ = controlInput;
    current_state_ = f_ * current_state_ + b_ * controlInput_;
}

void EKF::resetState() {
    current_state_ = initState_;
}

void EKF::resetState(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& currentState) {
    current_state_ = currentState;
}

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> EKF::getState() const {
    return current_state_;
}