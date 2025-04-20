#include "EKF/include/EKF.hpp"
#include <iostream>

EKF::EKF(
    const Eigen::MatrixXd f,
    const Eigen::MatrixXd h,
    const Eigen::MatrixXd b,
    const Eigen::MatrixXd q,
    const Eigen::MatrixXd r,
    double dt
):
    f_(f),
    h_(h),
    b_(b),
    q_(q),
    r_(r),
    dt_(dt) {
    initandCheck();
}

EKF::EKF(
    utils::BindableMatrixXd f,
    const Eigen::MatrixXd h,
    const Eigen::MatrixXd b,
    const Eigen::MatrixXd q,
    const Eigen::MatrixXd r
):
    f_(f),
    h_(h),
    b_(b),
    q_(q),
    r_(r),
    dt_(*f.getArg()) {
    initandCheck();
}

EKF::EKF(const Eigen::MatrixXd f, const Eigen::MatrixXd h, const Eigen::MatrixXd b, double dt):
    EKF(f,
       h,
       b,
       Eigen::MatrixXd::Identity(f.rows(), f.rows()),
       Eigen::MatrixXd::Identity(h.rows(), h.rows()),
       dt) {}

void EKF::init(const Eigen::MatrixXd& initState) {
    assert(initState.rows() == f_.rows() && initState.cols() == 1 && "初始状态矩阵维度不匹配");
    this->initState_ = initState;
    this->current_state_ = this->initState_;
}

void EKF::update(const Eigen::MatrixXd& measurement, double dt, bool setDefault) {
    assert(measurement.cols() == 1 && "测量矩阵必须是列向量");

    this->measurement_ = measurement;
    this->dt_ = dt == 0.0 ? dt_ : dt;

    if (setDefault) {
        this->current_state_ = f_ * this->current_state_;

        // 预测协方差矩阵
        p_ = f_ * p_ * f_.transpose() + q_;
        updateKalmanGain();
        return;
    }

    // EKF中，f和h矩阵可能需要在每次迭代中重新计算
    // 这里假设f_和h_已经是当前状态下的雅可比矩阵

    this->current_state_ = f_ * this->current_state_;

    // 预测协方差矩阵
    p_ = f_ * p_ * f_.transpose() + q_;
    updateKalmanGain();

    return;
}

void EKF::predict(double dt) {
    assert(f_.rows() == current_state_.rows() && "f_ 的列数必须等于 current_state_ 的行数");
    assert((dt > 0 || this->dt_ > 0) && "时间间隔 dt 必须大于 0");

    this->dt_ = dt == 0.0 ? dt_ : dt;

    // EKF中，f_应该是在当前状态下计算的雅可比矩阵
    this->current_state_ = f_ * this->current_state_;

    // 预测协方差矩阵
    p_ = f_ * p_ * f_.transpose() + q_;
}

void EKF::updateKalmanGain() {
    // 检查维度匹配
    assert(this->measurement_.rows() == h_.rows() && "测量矩阵维度不匹配");
    assert(h_.rows() > 0 && h_.cols() > 0 && "h矩阵维度不能为0");
    assert(p_.rows() == StateSize_ && p_.cols() == StateSize_ && "p矩阵维度不匹配");

    // 计算卡尔曼增益矩阵
    // 在EKF中，h_应该是在当前状态下计算的雅可比矩阵
    this->k_ = p_ * h_.transpose() * (h_ * p_ * h_.transpose() + r_).inverse();

    // 检查卡尔曼增益矩阵维度
    assert(k_.rows() == StateSize_ && k_.cols() == ObservationSize_ && "卡尔曼增益矩阵维度不匹配");

    // 更新状态
    // 在EKF中，观测残差计算可能需要使用非线性函数
    this->current_state_ =
        this->current_state_ + this->k_ * (this->measurement_ - h_ * this->current_state_);

    // 更新协方差矩阵
    p_ = (Eigen::MatrixXd::Identity(StateSize_, StateSize_) - this->k_ * h_) * p_;
}

void EKF::control(const Eigen::MatrixXd& controlInput, double dt) {
    assert(controlInput.rows() == b_.cols() && "控制输入矩阵维度不匹配");
    assert(controlInput.cols() == 1 && "控制输入矩阵必须是列向量");

    this->dt_ = dt == 0.0 ? dt_ : dt_;
    this->f_.update(dt_);

    this->controlInput_ = controlInput;
    // 在EKF中，状态更新可能需要使用非线性函数
    this->current_state_ = f_ * this->current_state_ + b_ * this->controlInput_;
}

void EKF::resetState(const Eigen::MatrixXd& currentState) {
    if (currentState.rows() != this->StateSize_ || currentState.cols() != 1) {
        throw std::invalid_argument("当前状态矩阵维度不匹配");
    }

    if (currentState.size() == 0) {
        this->current_state_ = Eigen::MatrixXd::Zero(this->StateSize_, 1);
    } else {
        this->current_state_ = currentState;
    }

    this->p_ = this->p0_;
    this->q_ = this->q0_;
}

Eigen::MatrixXd EKF::getState() const {
    return this->current_state_;
}

void EKF::initandCheck() {
    // assert(f_.rows() == f_.cols() && "状态转移矩阵 f 必须是方阵");
    assert(q_.rows() == q_.cols() && "过程噪声协方差矩阵 q 必须是方阵");
    assert(r_.rows() == r_.cols() && "观测噪声协方差矩阵 r 必须是方阵");

    assert(f_.rows() == h_.cols() && "状态转移矩阵 f 的行数必须等于观测矩阵 h 的列数");
    assert(h_.rows() == r_.rows() && "观测矩阵 h 的行数必须等于观测噪声协方差矩阵 r 的行数");
    assert(f_.rows() == q_.rows() && "状态转移矩阵 f 的行数必须等于过程噪声协方差矩阵 q 的行数");
    assert(b_.rows() == f_.rows() && "控制输入矩阵 b 的行数必须等于状态转移矩阵 f 的行数");
    assert(dt_ > 0 && "时间间隔 dt 必须大于 0");

    // 初始化矩阵维度
    this->ObservationSize_ = h_.rows(); // 观测维度
    this->StateSize_ = f_.rows();       // 状态维度
    this->ControlSize_ = b_.rows();     // 控制输入维度

    // 初始化其他矩阵大小
    this->k_ = Eigen::MatrixXd::Zero(this->StateSize_,
                                     this->ObservationSize_); // 卡尔曼增益矩阵
    this->initState_ = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(
        this->StateSize_,
        1
    ); // 初始状态矩阵
    this->p0_ = Eigen::MatrixXd::Identity(this->StateSize_,
                                          this->StateSize_); // 初始协方差矩阵
    this->q0_ = Eigen::MatrixXd::Identity(
        this->StateSize_,
        this->StateSize_
    ); // 初始过程噪声协方差矩阵
    this->r0_ = Eigen::MatrixXd::Identity(
        this->ObservationSize_,
        this->ObservationSize_
    ); // 初始观测噪声协方差矩阵

    this->current_state_ = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(
        this->StateSize_,
        1
    ); // 当前状态矩阵
    this->controlInput_ = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(
        this->ControlSize_,
        1
    ); // 控制输入矩阵
    this->measurement_ = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(
        this->ObservationSize_,
        1
    ); // 观测值矩阵
    this->KalmanGain_ = Eigen::MatrixXd::Zero(
        this->StateSize_,
        this->ObservationSize_
    ); // 卡尔曼增益矩阵
    this->p_ = Eigen::MatrixXd::Identity(this->StateSize_,
                                         this->StateSize_); // 协方差矩阵

    this->predictedState_ = Eigen::MatrixXd::Zero(
        this->StateSize_,
        this->StateSize_
    ); // 预测状态矩阵
}