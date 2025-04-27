#include "UKF/include/UKF.hpp"

UKF::UKF(
    std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> f,
    std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> h,
    const Eigen::MatrixXd& q,
    const Eigen::MatrixXd& r,
    int stateSize,
    int observationSize,
    double dt,
    double alpha,
    double beta,
    double kappa
):
    dt_(dt),
    alpha_(alpha),
    beta_(beta),
    kappa_(kappa),
    f_func_(f),
    h_func_(h) // Initialize dt_
{
    this->stateSize_ = stateSize;
    this->observationSize_ = observationSize;
    this->q_ = q;
    this->r_ = r;
    initandCheck();
}

UKF::UKF(const UKF& ukf): FilterBase(ukf) {
    alpha_ = ukf.alpha_;
    beta_ = ukf.beta_;
    kappa_ = ukf.kappa_;
    lambda_ = ukf.lambda_;
    sigmaPoints_ = ukf.sigmaPoints_;
    weights_m_ = ukf.weights_m_;
    weights_c_ = ukf.weights_c_;
    predictedSigmaPoints_ = ukf.predictedSigmaPoints_;
    predictedObservations_ = ukf.predictedObservations_;
    f_func_ = ukf.f_func_;
    h_func_ = ukf.h_func_;
    predictedMeasurementMean_ = ukf.predictedMeasurementMean_;
    Pyy_ = ukf.Pyy_;
    Pxy_ = ukf.Pxy_;

    this->stateSize_ = ukf.stateSize_;
    this->observationSize_ = ukf.observationSize_;
    this->ControlSize_ = ukf.ControlSize_;
    this->q_ = ukf.q_;
    this->r_ = ukf.r_;
    this->p_ = ukf.p_;
    this->current_state_ = ukf.current_state_;
    this->dt_ = ukf.dt_;
    this->initState_ = ukf.initState_;
    this->p0_ = ukf.p0_;
    this->q0_ = ukf.q0_;
    this->r0_ = ukf.r0_;
    this->k_ = ukf.k_;
    this->measurement_ = ukf.measurement_;
    this->controlInput_ = ukf.controlInput_;
    this->predictedState_ = ukf.predictedState_;
}

UKF& UKF::operator=(const UKF& ukf) {
    if (this == &ukf) {
        return *this;
    }

    // 拷贝 UKF 特定成员
    alpha_ = ukf.alpha_;
    beta_ = ukf.beta_;
    kappa_ = ukf.kappa_;
    lambda_ = ukf.lambda_;
    sigmaPoints_ = ukf.sigmaPoints_;
    weights_m_ = ukf.weights_m_;
    weights_c_ = ukf.weights_c_;
    predictedSigmaPoints_ = ukf.predictedSigmaPoints_;
    predictedObservations_ = ukf.predictedObservations_;
    f_func_ = ukf.f_func_;
    h_func_ = ukf.h_func_;
    predictedMeasurementMean_ = ukf.predictedMeasurementMean_;
    Pyy_ = ukf.Pyy_;
    Pxy_ = ukf.Pxy_;

    this->stateSize_ = ukf.stateSize_;
    this->observationSize_ = ukf.observationSize_;
    this->ControlSize_ = ukf.ControlSize_;
    this->q_ = ukf.q_;
    this->r_ = ukf.r_;
    this->p_ = ukf.p_;
    this->current_state_ = ukf.current_state_;
    this->dt_ = ukf.dt_;
    this->initState_ = ukf.initState_;
    this->p0_ = ukf.p0_;
    this->q0_ = ukf.q0_;
    this->r0_ = ukf.r0_;
    this->k_ = ukf.k_;
    this->measurement_ = ukf.measurement_;
    this->controlInput_ = ukf.controlInput_;
    this->predictedState_ = ukf.predictedState_;

    return *this;
}

void UKF::init(const Eigen::MatrixXd& initState) {
    assert(initState.rows() == stateSize_ && initState.cols() == 1 && "初始状态矩阵维度不匹配");
    this->initState_ = initState;
    this->current_state_ = this->initState_;
    this->p_ = this->p0_;
}

void UKF::predict(double dt) {
    assert((dt > 0 || this->dt_ > 0) && "时间间隔 dt 必须大于 0");

    double current_dt = (dt > 0.0) ? dt : this->dt_;

    if (p_.determinant() <= 0) {
        std::cerr << "Warning: 协方差矩阵 P 非正定，添加小的正定对角矩阵以修正。" << std::endl;
        p_ += Eigen::MatrixXd::Identity(stateSize_, stateSize_) * 1e-6; // 添加一个小的正定对角矩阵
    }

    this->dt_ = current_dt; // 更新 dt_

    // 1. 生成 Sigma 点
    generateSigmaPoints();

    // 2. 通过状态转移函数预测 Sigma 点
    predictSigmaPoints();

    // 3. 从预测的 Sigma 点预测均值和协方差
    predictMeanAndCovariance();

    // 存储预测状态
    this->predictedState_ = this->current_state_;
}

void UKF::update(const Eigen::MatrixXd& measurement, double dt, bool setDefault) {
    assert(
        measurement.rows() == observationSize_ && measurement.cols() == 1 && "观测矩阵维度不匹配"
    );
    this->measurement_ = measurement;

    if (dt > 0.0) {
        this->dt_ = dt;
        std::cerr
            << "Warning: UKF::update called with dt > 0. Updating internal dt, but assuming predict() was already called with appropriate timing."
            << std::endl;
    }

    // 确保协方差矩阵 P 是正定的
    // 如果不是，添加小的正定对角矩阵以修正

    // 4. 预测观测值
    predictObservation();

    // 5. 使用实际测量值更新状态
    updateState(measurement);

    // 6. 更新协方差矩阵
    // 更新步骤始终包含测量值。
    if (setDefault) {
        std::cerr << "Warning: setDefault=true has no standard effect in UKF::update." << std::endl;
        // 如果意图是丢弃测量值并恢复到预测状态：
        // this->current_state_ = this->predictedState_; // 恢复状态
        // this->p_ = predicted_covariance; // 恢复协方差（需要从 predictMeanAndCovariance 中存储预测的 p_）
    }
}

void UKF::control(const Eigen::MatrixXd& controlInput, double dt) {
    // 标准 UKF 不单独处理控制输入 'B*u'。
    // 控制输入应被合并到状态转移函数 f_func_ 中。
    // f_func_ 应定义为：x_k+1 = f(x_k, u_k, dt)
    std::cerr
        << "警告: 调用了 UKF::control，但标准 UKF 将控制输入合并到状态转移函数 'f' 中。请确保 'f_func_' 处理控制输入（如果需要）。"
        << std::endl;
    this->controlInput_ = controlInput; // 如果 f_func_ 需要控制输入，则存储控制输入

    // 如果提供了 dt，更新 dt
    if (dt > 0.0) {
        this->dt_ = dt;
    }

    // 注意：控制输入的实际应用发生在预测步骤中
    // 当调用 f_func_ 时，假设 f_func_ 使用了控制输入。
    // 在标准 UKF 模型中，这里不执行单独的状态更新。

    // 如果控制输入未被 f_func_ 处理，抛出错误可能更安全
    // throw std::logic_error("控制输入必须在 UKF 的状态转移函数 'f_func_' 中处理。");
}

void UKF::resetState(const Eigen::MatrixXd& currentState) {
    if (currentState.size() == 0) {
        // 重置为零状态和初始协方差
        this->current_state_ = Eigen::MatrixXd::Zero(this->stateSize_, 1);
        this->p_ = this->p0_;
    } else {
        if (currentState.rows() != this->stateSize_ || currentState.cols() != 1) {
            throw std::invalid_argument("提供的状态矩阵维度与 resetState 不匹配。");
        }
        this->current_state_ = currentState;
        // 可选地也重置协方差，这是常见做法
        this->p_ = this->p0_;
    }
    // 重置其他 UKF 内部状态（如 sigma 点）并非严格必要
    // 因为它们将在下一次预测步骤中重新计算。
}

Eigen::MatrixXd UKF::getState() const {
    return this->current_state_;
}

// --- 私有方法 ---

void UKF::generateSigmaPoints() {
    lambda_ = alpha_ * alpha_ * (static_cast<double>(stateSize_) + kappa_)
        - static_cast<double>(stateSize_);
    double gamma = std::sqrt(static_cast<double>(stateSize_) + lambda_); // 确保类型转换为 double

    sigmaPoints_.col(0) = current_state_; // 第一个 sigma 点是当前均值

    // 使用 LLT 计算协方差矩阵的 Cholesky 分解
    Eigen::MatrixXd P_sqrt;
    Eigen::LLT<Eigen::MatrixXd> lltOfP(p_);

    // 如果协方差矩阵 P 不是正定的
    if (lltOfP.info() == Eigen::NumericalIssue) {
        // 尝试添加小的正定对角矩阵以修正
        double epsilon = 1e-9;
        std::cerr << "Warning: 协方差矩阵 P 不是正定的。添加 epsilon * I." << std::endl;
        Eigen::MatrixXd perturbed_p =
            p_ + epsilon * Eigen::MatrixXd::Identity(stateSize_, stateSize_);
        lltOfP.compute(perturbed_p);
        if (lltOfP.info() == Eigen::NumericalIssue) {
            // 如果仍然失败，可能是因为矩阵的特征值为负
            // 方法 1: 使用伪逆（需要 SVD）
            p_ = perturbed_p.diagonal().asDiagonal();
            lltOfP.compute(p_);
            // 方法 2: 抛出异常
            // throw std::runtime_error(
            //     "Cholesky decomposition failed: Matrix is not positive definite even after adding epsilon."
            // );
        }
        P_sqrt = lltOfP.matrixL(); // 使用 L（下三角）
        p_ = perturbed_p;          // 如果 Cholesky 成功, 更新 p_ 为扰动版本
    } else {
        P_sqrt = lltOfP.matrixL(); // 使用 L（下三角）
    }

    for (int i = 0; i < stateSize_; ++i) {
        sigmaPoints_.col(i + 1) = current_state_ + gamma * P_sqrt.col(i);
        sigmaPoints_.col(i + 1 + stateSize_) = current_state_ - gamma * P_sqrt.col(i);
    }
}

void UKF::predictSigmaPoints() {
    assert(
        predictedSigmaPoints_.rows() == stateSize_
        && predictedSigmaPoints_.cols() == (2 * stateSize_ + 1)
        && "predictedSigmaPoints_ 维度不正确"
    );
    for (int i = 0; i < 2 * stateSize_ + 1; ++i) {
        // 对每个 sigma 点应用非线性状态转移函数
        // 如果 f_func_ 需要控制输入（需要修改 f_func_ 的签名或状态扩展），则传递控制输入
        predictedSigmaPoints_.col(i) = f_func_(sigmaPoints_.col(i));
    }
}

void UKF::predictMeanAndCovariance() {
    // 计算预测状态均值（预测 sigma 点的加权和）
    // 使用临时变量以避免过早修改 current_state_
    Eigen::MatrixXd predicted_mean = Eigen::MatrixXd::Zero(stateSize_, 1);
    for (int i = 0; i < 2 * stateSize_ + 1; ++i) {
        predicted_mean += weights_m_(i) * predictedSigmaPoints_.col(i);
    }

    // 计算预测状态协方差
    Eigen::MatrixXd predicted_cov = Eigen::MatrixXd::Zero(stateSize_, stateSize_);
    for (int i = 0; i < 2 * stateSize_ + 1; ++i) {
        Eigen::MatrixXd diff = predictedSigmaPoints_.col(i) - predicted_mean;
        // 如果需要，确保一致的归一化（例如角度归一化）
        // 示例：如果状态包含角度在索引 k：diff(k, 0) = normalize_angle(diff(k, 0));
        predicted_cov += weights_c_(i) * diff * diff.transpose();
    }
    predicted_cov += q_; // 添加过程噪声协方差

    // 更新滤波器的状态和协方差
    current_state_ = predicted_mean;
    p_ = predicted_cov;
}

void UKF::predictObservation() {
    // 通过非线性观测函数变换预测的 sigma 点
    assert(
        predictedObservations_.rows() == observationSize_
        && predictedObservations_.cols() == (2 * stateSize_ + 1)
        && "predictedObservations_ 维度不正确"
    );
    for (int i = 0; i < 2 * stateSize_ + 1; ++i) {
        predictedObservations_.col(i) = h_func_(predictedSigmaPoints_.col(i));
    }

    // 计算预测观测均值
    predictedMeasurementMean_.setZero(); // 在求和前重置
    for (int i = 0; i < 2 * stateSize_ + 1; ++i) {
        predictedMeasurementMean_ += weights_m_(i) * predictedObservations_.col(i);
    }

    // 计算预测观测协方差（Pyy）和交叉协方差（Pxy）
    Pyy_.setZero(); // 在求和前重置
    Pxy_.setZero(); // 在求和前重置
    for (int i = 0; i < 2 * stateSize_ + 1; ++i) {
        Eigen::MatrixXd diff_y = predictedObservations_.col(i) - predictedMeasurementMean_;
        Eigen::MatrixXd diff_x = predictedSigmaPoints_.col(i) - current_state_; // 使用预测状态均值
        // 如果必要，处理 diff_y 和 diff_x 的角度归一化或标准化
        // 示例：如果观测包含角度在索引 j：diff_y(j, 0) = normalize_angle(diff_y(j, 0));
        // 示例：如果状态包含角度在索引 k：diff_x(k, 0) = normalize_angle(diff_x(k, 0));

        Pyy_ += weights_c_(i) * diff_y * diff_y.transpose();
        Pxy_ += weights_c_(i) * diff_x * diff_y.transpose();
    }
    Pyy_ += r_; // 添加观测噪声协方差
}

void UKF::updateState(const Eigen::MatrixXd& measurement) {
    // 计算卡尔曼增益 K
    // K = Pxy * Pyy^-1
    // 如果 Pyy 可能条件较差，使用稳健的逆计算
    Eigen::MatrixXd Pyy_inv;
    // 在求逆前检查条件数或行列式
    if (Pyy_.determinant() == 0) { // 简单检查，最好使用条件数
        std::cerr << "警告: Pyy 是奇异的或接近奇异。使用伪逆或跳过更新。" << std::endl;
        // 选项 1：使用伪逆（需要 SVD）
        // Pyy_inv = Pyy_.completeOrthogonalDecomposition().pseudoInverse();
        // 选项 2：跳过更新（或使用非常小的增益）
        k_.setZero(); // 将增益设置为零
        return;       // 跳过状态和协方差更新
    } else {
        Pyy_inv = Pyy_.inverse();
    }

    k_ = Pxy_ * Pyy_inv; // 如果需要，将卡尔曼增益存储在基类成员中

    // 更新状态估计
    // x = x_predicted + K * (z - z_predicted)
    Eigen::MatrixXd innovation = measurement - predictedMeasurementMean_;
    // 如果必要，处理创新的角度归一化（例如，如果测量涉及角度）
    // 示例：如果测量包含角度在索引 j：innovation(j, 0) = normalize_angle(innovation(j, 0));
    current_state_ = current_state_ + k_ * innovation;

    // 更新状态协方差
    // P = P_predicted - K * Pyy * K^T
    p_ = p_ - k_ * Pyy_ * k_.transpose();

    // 确保 P 保持对称和正半定
    p_ = 0.5 * (p_ + p_.transpose()); // 强制对称
    // 可选：添加小的对角项或执行特征值截断以提高稳定性

    // 使用 Joseph 形式更新协方差（数值更稳定，但需要 H）：
    // P = (I - K * H) * P_predicted * (I - K * H)^T + K * R * K^T
    // 注意：这需要 UKF 的等效 'H'，这不是直接可用的。
    // 通常使用公式 P = P - K * Pyy * K^T。
}

void UKF::initandCheck() {
    // 参数检查
    assert(alpha_ > 0 && alpha_ <= 1 && "UKF 参数 alpha 必须在 (0, 1] 之间");

    // beta_ >= 0 是典型的。 beta_ = 2 对高斯分布是最优的。
    assert(beta_ >= 0 && "UKF 参数 beta 必须非负");

    // kappa_ + stateSize_ > 0 必须成立。 kappa_ = 0 或 kappa_ = 3 - stateSize_ 是常见选择。
    // 对于涉及 stateSize_ 和 kappa_ 的计算使用 static_cast<double>
    assert(
        static_cast<double>(stateSize_) + kappa_ > 1e-9 && "UKF 参数 kappa + 状态大小必须为正"
    ); // 对于浮点比较使用 epsilon

    // 维度检查
    assert(stateSize_ > 0 && "状态维度必须为正");
    assert(observationSize_ > 0 && "观测维度必须为正");
    assert(q_.rows() == stateSize_ && q_.cols() == stateSize_ && "过程噪声协方差 Q 维度不匹配");
    assert(
        r_.rows() == observationSize_ && r_.cols() == observationSize_
        && "观测噪声协方差 R 维度不匹配"
    );

    // 函数检查
    assert(f_func_ && "状态转移函数 f_func_ 未定义");
    assert(h_func_ && "观测函数 h_func_ 未定义");

    // 初始化 Sigma 点相关矩阵和向量
    lambda_ = alpha_ * alpha_ * (static_cast<double>(stateSize_) + kappa_)
        - static_cast<double>(stateSize_);
    int numSigmaPoints = 2 * stateSize_ + 1;

    sigmaPoints_.resize(stateSize_, numSigmaPoints);
    predictedSigmaPoints_.resize(stateSize_, numSigmaPoints);
    predictedObservations_.resize(observationSize_, numSigmaPoints);

    weights_m_.resize(numSigmaPoints);
    weights_c_.resize(numSigmaPoints);

    // 计算权重
    double w_m0 = lambda_ / (static_cast<double>(stateSize_) + lambda_);
    double w_c0 = w_m0 + (1.0 - alpha_ * alpha_ + beta_);
    double w_i = 0.5 / (static_cast<double>(stateSize_) + lambda_);

    weights_m_(0) = w_m0;
    weights_c_(0) = w_c0;
    for (int i = 1; i < numSigmaPoints; ++i) {
        weights_m_(i) = w_i;
        weights_c_(i) = w_i;
    }

    // 初始化成员
    this->initState_ = Eigen::MatrixXd::Zero(stateSize_, 1);
    this->p0_ = Eigen::MatrixXd::Identity(stateSize_, stateSize_); // 默认初始协方差
    this->q0_ = q_;                                                // 存储初始 Q
    this->r0_ = r_;                                                // 存储初始 R
    this->current_state_ = this->initState_;
    this->p_ = this->p0_;
    this->k_ = Eigen::MatrixXd::Zero(stateSize_, observationSize_); // K 在更新中计算
    this->measurement_ = Eigen::MatrixXd::Zero(observationSize_, 1);
    this->predictedState_ = Eigen::MatrixXd::Zero(stateSize_, 1); // 存储预测状态
    this->ControlSize_ = 0;                                       // 控制输入不设置
    this->controlInput_ = Eigen::MatrixXd::Zero(0, 1);            // 初始化控制输入

    // 初始化 UKF特定矩阵
    predictedMeasurementMean_.resize(observationSize_, 1);
    Pyy_.resize(observationSize_, observationSize_);
    Pxy_.resize(stateSize_, observationSize_);
}