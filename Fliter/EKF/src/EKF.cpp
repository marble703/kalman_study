#include "EKF/include/EKF.hpp"
#include <cassert>

EKF::EKF(
    const Eigen::MatrixXd f,
    const Eigen::MatrixXd h,
    const Eigen::MatrixXd b,
    Eigen::MatrixXd q,
    Eigen::MatrixXd r,
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

    k_ = Eigen::MatrixXd::Zero(StateSize_, ObservationSize_);
    initState_ = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(StateSize_, 1);
    p0_ = Eigen::MatrixXd::Identity(StateSize_, StateSize_);
    q0_ = Eigen::MatrixXd::Identity(StateSize_, StateSize_);
    r0_ = Eigen::MatrixXd::Identity(ObservationSize_, ObservationSize_);

    current_state_ = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(StateSize_, 1);
    controlInput_ = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(ControlSize_, 1);
    measurement_ = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(ObservationSize_, 1);
    KalmanGain_ = Eigen::MatrixXd::Zero(StateSize_, ObservationSize_);
    p_ = Eigen::MatrixXd::Identity(StateSize_, StateSize_);
    predictedState_ = Eigen::MatrixXd::Zero(StateSize_, StateSize_);
}

void EKF::init(const Eigen::MatrixXd& initState) {
    assert(initState.rows() == f_.rows() && initState.cols() == 1 && "初始状态矩阵维度不匹配");
    initState_ = initState;
    current_state_ = initState_;
}

void EKF::update(const Eigen::MatrixXd& measurement) {
    assert(measurement.rows() == h_.cols() && "测量矩阵维度不匹配");
    assert(measurement.cols() == 1 && "测量矩阵必须是列向量");
    measurement_ = measurement;
    predict();
    updateKalmanGain();
}

void EKF::update(const Eigen::MatrixXd& measurement, float dt) {
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
    p_ = (Eigen::MatrixXd::Identity(f_.rows(), f_.cols()) - k_ * h_) * p_;
}

void EKF::control(const Eigen::MatrixXd& controlInput) {
    controlInput_ = controlInput;
    current_state_ = f_ * current_state_ + b_ * controlInput_;
}

void EKF::resetState() {
    current_state_ = initState_;
}

void EKF::resetState(const Eigen::MatrixXd& currentState) {
    current_state_ = currentState;
}

Eigen::MatrixXd EKF::getState() const {
    return current_state_;
}