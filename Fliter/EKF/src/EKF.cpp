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

// 非线性EKF构造函数实现
EKF::EKF(
    const StateTransitionFunction& stateTransitionFn,
    const MeasurementFunction& measurementFn,
    int stateSize,
    int observationSize,
    const Eigen::MatrixXd& b,
    const Eigen::MatrixXd& q,
    const Eigen::MatrixXd& r,
    double dt
):
    // 初始化为恒等矩阵，后续会被雅可比矩阵替换
    f_(Eigen::MatrixXd::Identity(stateSize, stateSize)),
    h_(Eigen::MatrixXd::Identity(observationSize, stateSize)),
    b_(b),
    q_(q),
    r_(r),
    dt_(dt),
    stateTransitionFn_(stateTransitionFn),
    measurementFn_(measurementFn),
    useNonlinearFunctions_(true) {
    // 初始化状态尺寸
    StateSize_ = stateSize;
    ObservationSize_ = observationSize;
    ControlSize_ = b.cols();

    // 初始化雅可比矩阵
    F_jacobian_ = Eigen::MatrixXd::Identity(StateSize_, StateSize_);
    H_jacobian_ = Eigen::MatrixXd::Identity(ObservationSize_, StateSize_);

    initandCheck();
}

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

    this->current_state_ = f_ * this->current_state_;

    // 预测协方差矩阵
    p_ = f_ * p_ * f_.transpose() + q_;
    updateKalmanGain();

    return;
}

void EKF::predict(double dt) {
    assert((dt > 0 || this->dt_ > 0) && "时间间隔 dt 必须大于 0");
    this->dt_ = dt == 0.0 ? dt_ : dt;

    if (useNonlinearFunctions_) {
        // 使用非线性方法计算预测状态
        this->current_state_ = predictNonlinearState(this->current_state_);

        // 计算状态转移函数的雅可比矩阵
        F_jacobian_ = computeStateJacobian(this->current_state_);

        // 更新协方差矩阵
        p_ = F_jacobian_ * p_ * F_jacobian_.transpose() + q_;
    } else {
        // 使用线性方法计算预测状态
        assert(f_.rows() == current_state_.rows() && "f_ 的列数必须等于 current_state_ 的行数");
        this->current_state_ = f_ * this->current_state_;

        // 预测协方差矩阵
        p_ = f_ * p_ * f_.transpose() + q_;
    }
}

void EKF::updateKalmanGain() {
    // 检查维度匹配
    assert(this->measurement_.rows() == ObservationSize_ && "测量矩阵维度不匹配");
    assert(p_.rows() == StateSize_ && p_.cols() == StateSize_ && "p矩阵维度不匹配");

    if (useNonlinearFunctions_) {
        // 计算观测函数的雅可比矩阵
        H_jacobian_ = computeMeasurementJacobian(this->current_state_);

        // 使用计算得到的雅可比矩阵计算卡尔曼增益
        this->k_ = p_ * H_jacobian_.transpose()
            * (H_jacobian_ * p_ * H_jacobian_.transpose() + r_).inverse();

        // 非线性观测预测
        Eigen::VectorXd predicted_measurement = predictNonlinearMeasurement(this->current_state_);

        // 计算观测残差
        Eigen::VectorXd innovation = this->measurement_ - predicted_measurement;

        // 更新状态
        this->current_state_ = this->current_state_ + this->k_ * innovation;

        // 更新协方差矩阵 - Joseph形式
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(StateSize_, StateSize_);
        p_ = (I - this->k_ * H_jacobian_) * p_ * (I - this->k_ * H_jacobian_).transpose()
            + this->k_ * r_ * this->k_.transpose();
    } else {
        // 线性实现
        // 计算卡尔曼增益矩阵
        this->k_ = p_ * h_.transpose() * (h_ * p_ * h_.transpose() + r_).inverse();

        // 检查卡尔曼增益矩阵维度
        assert(
            k_.rows() == StateSize_ && k_.cols() == ObservationSize_ && "卡尔曼增益矩阵维度不匹配"
        );

        // 更新状态
        Eigen::MatrixXd innovation = this->measurement_ - h_ * this->current_state_;
        this->current_state_ = this->current_state_ + this->k_ * innovation;

        // 更新协方差矩阵 - Joseph形式
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(StateSize_, StateSize_);
        p_ = (I - this->k_ * h_) * p_ * (I - this->k_ * h_).transpose()
            + this->k_ * r_ * this->k_.transpose();
    }

    // 保存当前卡尔曼增益矩阵用于外部访问
    this->KalmanGain_ = this->k_;
}

void EKF::control(const Eigen::MatrixXd& controlInput, double dt) {
    assert(controlInput.rows() == b_.cols() && "控制输入矩阵维度不匹配");
    assert(controlInput.cols() == 1 && "控制输入矩阵必须是列向量");

    this->dt_ = dt == 0.0 ? dt_ : dt_;
    this->f_.update(dt_);

    this->controlInput_ = controlInput;
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

Eigen::MatrixXd EKF::computeStateJacobian(const Eigen::VectorXd& state) {
    if (!useNonlinearFunctions_) {
        // 如果不使用非线性函数，直接返回线性状态转移矩阵
        return f_.getMatrix();
    }

    // 数值计算雅可比矩阵
    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(StateSize_, StateSize_);
    const double h = 1e-5; // 扰动量，用于数值计算导数

    // 对状态的每个维度进行扰动来计算偏导数
    for (int i = 0; i < StateSize_; ++i) {
        // 创建扰动状态
        Eigen::VectorXd state_plus = state;
        state_plus(i) += h;

        // 计算状态转移结果的差异
        Eigen::VectorXd f_state = stateTransitionFn_(state, dt_);
        Eigen::VectorXd f_state_plus = stateTransitionFn_(state_plus, dt_);

        // 计算雅可比矩阵的一列
        jacobian.col(i) = (f_state_plus - f_state) / h;
    }

    return jacobian;
}

Eigen::MatrixXd EKF::computeMeasurementJacobian(const Eigen::VectorXd& state) {
    if (!useNonlinearFunctions_) {
        return h_; // 如果不使用非线性函数，直接返回线性观测矩阵
    }

    // 数值计算雅可比矩阵
    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(ObservationSize_, StateSize_);
    const double h = 1e-5; // 扰动量，用于数值计算导数

    // 对状态的每个维度进行扰动来计算偏导数
    for (int i = 0; i < StateSize_; ++i) {
        // 创建扰动状态
        Eigen::VectorXd state_plus = state;
        state_plus(i) += h;

        // 计算观测结果的差异
        Eigen::VectorXd h_state = measurementFn_(state);
        Eigen::VectorXd h_state_plus = measurementFn_(state_plus);

        // 计算雅可比矩阵的一列
        jacobian.col(i) = (h_state_plus - h_state) / h;
    }

    return jacobian;
}

Eigen::VectorXd EKF::predictNonlinearState(const Eigen::VectorXd& state) {
    if (!useNonlinearFunctions_) {
        return f_ * state; // 如果不使用非线性函数，使用线性状态转移
    }

    return stateTransitionFn_(state, dt_);
}

Eigen::VectorXd EKF::predictNonlinearMeasurement(const Eigen::VectorXd& state) {
    if (!useNonlinearFunctions_) {
        return h_ * state; // 如果不使用非线性函数，使用线性观测
    }

    return measurementFn_(state);
}