#pragma once

#include <eigen3/Eigen/Dense>
#include <iostream>

#include "FilterBase.hpp"
#include "Utils/BindMatrix.hpp"

class UKF: public FilterBase {
public:
    /**
     * @brief 完整构造
     * 
     * @param f 状态转移函数 (非线性)
     * @param h 观测函数 (非线性)
     * @param q 过程噪声协方差矩阵
     * @param r 观测噪声协方差矩阵
     * @param alpha UKF 参数 alpha
     * @param beta UKF 参数 beta
     * @param kappa UKF 参数 kappa
     * @param dt 时间间隔 (固定)
     */
    UKF(std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> f,
        std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> h,
        const Eigen::MatrixXd& q,
        const Eigen::MatrixXd& r,
        int stateSize,
        int observationSize,
        double dt = 0.0,
        double alpha = 1e-3,
        double beta = 2.0,
        double kappa = 0.0);

    /**
     * @brief 拷贝构造函数
     */
    UKF(const UKF& ukf);

    /**
     * @brief 拷贝赋值函数
     */
    UKF& operator=(const UKF& ukf);

    UKF() = default;
    ~UKF() = default;

    void init(const Eigen::MatrixXd& initState) override;
    void
    update(const Eigen::MatrixXd& measurement, double dt = 0.0, bool setDefault = false) override;
    void control(const Eigen::MatrixXd& controlInput, double dt = 0.0)
        override; // UKF 通常不直接处理控制输入,可以留空或抛出异常
    void predict(double dt = 0.0) override;
    void resetState(const Eigen::MatrixXd& currentState = Eigen::MatrixXd()) override;
    Eigen::MatrixXd getState() const override;

private:
    void generateSigmaPoints();
    void predictSigmaPoints();
    void predictMeanAndCovariance();
    void predictObservation();
    void updateState(const Eigen::MatrixXd& measurement);
    void initandCheck();

    // 模型参数
    int stateSize_;
    int observationSize_;
    int ControlSize_;

    double dt_;                   // 时间间隔
    std::shared_ptr<double> arg_; // 绑定的参数

    // 模型矩阵
    Eigen::MatrixXd q_; // 过程噪声协方差矩阵
    Eigen::MatrixXd r_; // 观测噪声协方差矩阵
    Eigen::MatrixXd p_; // 协方差矩阵

    Eigen::MatrixXd p0_;             // 初始协方差矩阵
    Eigen::MatrixXd q0_;             // 初始过程噪声协方差矩阵
    Eigen::MatrixXd r0_;             // 初始观测噪声协方差矩阵
    Eigen::MatrixXd current_state_;  // 当前状态矩阵
    Eigen::MatrixXd initState_;      // 初始状态矩阵
    Eigen::MatrixXd measurement_;    // 观测矩阵
    Eigen::MatrixXd controlInput_;   // 控制输入矩阵
    Eigen::MatrixXd predictedState_; // 预测状态矩阵
    Eigen::MatrixXd k_;              // 卡尔曼增益矩阵

    // UKF 特定参数
    double alpha_;
    double beta_;
    double kappa_;
    double lambda_; // 派生参数 lambda = alpha_ * alpha_ * (StateSize_ + kappa_) - StateSize_

    // Sigma Points 相关
    Eigen::MatrixXd sigmaPoints_;           // Sigma 点矩阵 (2 * StateSize_ + 1 个点)
    Eigen::VectorXd weights_m_;             // 均值权重
    Eigen::VectorXd weights_c_;             // 协方差权重
    Eigen::MatrixXd predictedSigmaPoints_;  // 预测的 Sigma 点
    Eigen::MatrixXd predictedObservations_; // 预测的观测 Sigma 点

    // 非线性函数
    std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> f_func_; // 状态转移函数
    std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> h_func_; // 观测函数

    // 预测的观测均值和协方差
    Eigen::MatrixXd predictedMeasurementMean_;
    Eigen::MatrixXd Pyy_; // 预测观测协方差
    Eigen::MatrixXd Pxy_; // 状态与观测的互协方差
};