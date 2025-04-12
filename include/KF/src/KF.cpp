#include "KF/include/KF.hpp"

KF::KF(
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>& f,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>& h,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>& b,
    const Eigen::Ref<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>& q,
    const Eigen::Ref<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>& r,
    double dt
):
    f_(f),
    h_(h),
    b_(b),
    q_(q),
    r_(r),
    dt_(dt) {
    assert(f_.rows() == f_.cols() && "状态转移矩阵 f 必须是方阵");
    assert(h_.rows() == h_.cols() && "观测矩阵 h 必须是方阵");
    assert(b_.rows() == b_.cols() && "控制输入矩阵 b 必须是方阵");
    assert(q_.rows() == q_.cols() && "过程噪声协方差矩阵 q 必须是方阵");
    assert(r_.rows() == r_.cols() && "观测噪声协方差矩阵 r 必须是方阵");
    assert(
        f_.rows() == h_.rows() && f_.rows() == b_.rows() && f_.rows() == q_.rows()
        && f_.rows() == r_.rows() && "矩阵维度不匹配"
    );
    assert(
        f_.cols() == h_.cols() && f_.cols() == b_.cols() && f_.cols() == q_.cols()
        && f_.cols() == r_.cols() && "矩阵维度不匹配"
    );

    // 初始化矩阵维度
    this->ObservationSize_ = h_.rows(); // 观测维度
    this->StateSize_ = f_.rows();       // 状态维度
    this->ControlSize_ = b_.rows();     // 控制输入维度

    // 初始化其他矩阵大小
    this->initState_ = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(
        this->StateSize_, 1
    ); // 初始状态矩阵
    this->p0_ = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Identity(
        this->StateSize_, this->StateSize_
    ); // 初始协方差矩阵
    this->q0_ = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Identity(
        this->StateSize_, this->StateSize_
    ); // 初始过程噪声协方差矩阵
    this->r0_ = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Identity(
        this->ObservationSize_, this->ObservationSize_
    ); // 初始观测噪声协方差矩阵

    this->current_state_ = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(
        this->StateSize_, 1
    ); // 当前状态矩阵
    this->controlInput_ = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(
        this->ControlSize_, 1
    ); // 控制输入矩阵
    this->measurement_ = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(
        this->ObservationSize_, 1
    ); // 观测值矩阵
    this->KalmanGain_ = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(
        this->StateSize_, this->ObservationSize_
    ); // 卡尔曼增益矩阵
    this->p_ = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Identity(
        this->StateSize_, this->StateSize_
    ); // 协方差矩阵

    this->predictedState_ = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(
        this->StateSize_, this->StateSize_
    ); // 预测状态矩阵

}

void KF::init(const Eigen::Ref<
              const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>& initState) {
    assert(
        initState.rows() == f_.rows() && initState.cols() == 1 && "初始状态矩阵维度不匹配"
    );
    this->initState_ = initState;
    this->current_state_ = this->initState_;
}

void KF::update(
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>&
        measurement
) {
    this->measurement_ = measurement;

    predict();
    updateKalmanGain();
}

void KF::update(
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>&
        measurement,
    float dt
) {
    this->measurement_ = measurement;
    this->dt_ = dt;

    predict(this->dt_);
    updateKalmanGain();
}

void KF::predict() {
    // 预测状态
    this->current_state_ =
        f_ * this->current_state_ * this->dt_ + b_ * this->controlInput_;
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
    p_ = (Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Identity(
              f_.rows(),
              f_.cols()
          )
          - this->k_ * h_)
        * p_;
}

void KF::control(
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>&
        controlInput
) {
    this->controlInput_ = controlInput;
    this->current_state_ = f_ * this->current_state_ + b_ * this->controlInput_;
}

void KF::resetState() {
    this->current_state_ = this->initState_;
}
void KF::resetState(
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>&
        currentState
) {
    this->current_state_ = currentState;
}

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> KF::getState() const {
    return this->current_state_;
}