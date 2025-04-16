#include "KF/include/KF.hpp"
#include <variant>

KF::KF(
    const Eigen::MatrixXd f,
    const Eigen::MatrixXd h,
    const Eigen::MatrixXd b,
    const Eigen::MatrixXd q,
    const Eigen::MatrixXd r,
    std::variant<double> dt
):
    f_(f),
    h_(h),
    b_(b),
    q_(q),
    r_(r),
    dt_(std::get<double>(dt)) {
    assert(f_.rows() == f_.cols() && "状态转移矩阵 f 必须是方阵");

    assert(q_.rows() == q_.cols() && "过程噪声协方差矩阵 q 必须是方阵");
    assert(r_.rows() == r_.cols() && "观测噪声协方差矩阵 r 必须是方阵");

    assert(f_.rows() == h_.cols() && "状态转移矩阵 f 的行数必须等于观测矩阵 h 的列数");
    assert(
        h_.rows() == r_.rows() && "观测矩阵 h 的行数必须等于观测噪声协方差矩阵 r 的行数"
    );
    assert(
        f_.rows() == q_.rows()
        && "状态转移矩阵 f 的行数必须等于过程噪声协方差矩阵 q 的行数"
    );
    assert(
        b_.rows() == f_.rows() && "控制输入矩阵 b 的行数必须等于状态转移矩阵 f 的行数"
    );
    assert(dt_ > 0 && "时间间隔 dt 必须大于 0");

    // 初始化矩阵维度
    this->ObservationSize_ = h_.rows(); // 观测维度
    this->StateSize_ = f_.rows();       // 状态维度
    this->ControlSize_ = b_.rows();     // 控制输入维度

    // 初始化其他矩阵大小
    this->k_ = Eigen::MatrixXd::Zero(
        this->StateSize_,
        this->ObservationSize_
    ); // 卡尔曼增益矩阵
    this->initState_ = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(
        this->StateSize_,
        1
    ); // 初始状态矩阵
    this->p0_ = Eigen::MatrixXd::Identity(
        this->StateSize_,
        this->StateSize_
    ); // 初始协方差矩阵
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
    this->p_ = Eigen::MatrixXd::Identity(
        this->StateSize_,
        this->StateSize_
    ); // 协方差矩阵

    this->predictedState_ = Eigen::MatrixXd::Zero(
        this->StateSize_,
        this->StateSize_
    ); // 预测状态矩阵
}

KF::KF(
    const Eigen::MatrixXd f,
    const Eigen::MatrixXd h,
    const Eigen::MatrixXd b,
    std::variant<double> dt
):
    KF(f,
       h,
       b,
       Eigen::MatrixXd::Identity(f.rows(), f.rows()),
       Eigen::MatrixXd::Identity(h.rows(), h.rows()),
       dt) {}

void KF::init(const Eigen::MatrixXd& initState) {
    assert(
        initState.rows() == f_.rows() && initState.cols() == 1 && "初始状态矩阵维度不匹配"
    );
    this->initState_ = initState;
    this->current_state_ = this->initState_;
}

void KF::update(const Eigen::MatrixXd& measurement, float dt, bool setDefault) {
    assert(measurement.rows() == h_.cols() && "测量矩阵维度不匹配");
    assert(measurement.cols() == 1 && "测量矩阵必须是列向量");
    assert(dt > 0 && "时间间隔 dt 必须大于 0");

    this->measurement_ = measurement;
    if(setDefault) {
        this->dt_ = dt;
        predict();
        updateKalmanGain();
        return;
    }

    predict(dt);
    updateKalmanGain();
    
    return;
}

void KF::update(const Eigen::MatrixXd& measurement) {
    assert(measurement.rows() == h_.cols() && "测量矩阵维度不匹配");
    assert(measurement.cols() == 1 && "测量矩阵必须是列向量");

    this->measurement_ = measurement;

    predict();
    updateKalmanGain();
}

void KF::predict() {
    assert(
        f_.cols() == current_state_.rows() && "f_ 的列数必须等于 current_state_ 的行数"
    );

    this->current_state_ = f_ * this->current_state_;

    // 预测协方差矩阵
    p_ = f_ * p_ * f_.transpose() + q_;
}

void KF::predict(float dt) {
    this->dt_ = dt;
    // 预测状态
    this->predict();
}

void KF::updateKalmanGain() {
    // 计算卡尔曼增益矩阵
    this->k_ = p_ * h_.transpose() * (h_ * p_ * h_.transpose() + r_).inverse();
    // 更新状态
    this->current_state_ = this->current_state_
        + this->k_ * (this->measurement_ - h_ * this->current_state_);
    // 更新协方差矩阵
    p_ = (Eigen::MatrixXd::Identity(f_.rows(), f_.cols()) - this->k_ * h_) * p_;
}

void KF::control(const Eigen::MatrixXd& controlInput) {
    this->controlInput_ = controlInput;
    this->current_state_ = f_ * this->current_state_ + b_ * this->controlInput_;
}

void KF::resetState() {
    this->current_state_ = this->initState_;
}
void KF::resetState(const Eigen::MatrixXd& currentState) {
    this->current_state_ = currentState;
}

Eigen::MatrixXd KF::getState() const {
    return this->current_state_;
}