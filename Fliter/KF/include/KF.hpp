#pragma once

#include "FliterBase/FilterBase.hpp"
#include "Utils/BindMatrix.hpp"

class KF: public FilterBase {
public:
    /**
     * @brief 完整构造
     * 
     * @param f  // 状态转移矩阵(固定)
     * @param h  // 观测矩阵(固定)
     * @param b  // 控制输入矩阵(固定)
     * @param q  // 过程噪声协方差矩阵(固定)
     * @param r  // 观测噪声协方差矩阵(固定)
     * @param dt // 时间间隔(固定)
     */
    KF(const Eigen::MatrixXd f,
       const Eigen::MatrixXd h,
       const Eigen::MatrixXd b,
       const Eigen::MatrixXd q,
       const Eigen::MatrixXd r,
       double dt = 0.0);

    /**
     * @brief 动态 dt 构造
     * 
     * @param f  // 状态转移矩阵(固定)
     * @param h  // 观测矩阵(固定)
     * @param b  // 控制输入矩阵(固定)
     * @param q  // 过程噪声协方差矩阵(固定)
     * @param r  // 观测噪声协方差矩阵(固定)
     * @param dt // 时间间隔(可变)
     */
    KF(utils::BindableMatrixXd f,
       const Eigen::MatrixXd h,
       const Eigen::MatrixXd b,
       const Eigen::MatrixXd q,
       const Eigen::MatrixXd r);

    /**
    * @brief 仅模型构造,仅初始化模型
    * 
    * @param f // 状态转移矩阵(固定)
    * @param h // 观测矩阵(固定)
    * @param b // 控制输入矩阵(固定)
    */
    KF(const Eigen::MatrixXd f, const Eigen::MatrixXd h, const Eigen::MatrixXd b, double dt = 0.0);

    /**
     * @brief 拷贝构造函数,只拷贝模型
     */
    KF(const KF& kf);

    /**
     * @brief 拷贝赋值函数,复制所有成员变量
     * @param kf 需要复制的KF对象
     * @return 返回当前对象的引用
     */
    KF& operator=(const KF& kf);

    KF() = default;

    ~KF() = default;

    /**
     * @brief 传入初始状态,初始化卡尔曼滤波器
     * @param initState 初始状态
     */
    void init(const Eigen::MatrixXd& initState) override;

    /**
     * @brief 使用观测值和时间间隔更新卡尔曼滤波器,预测下一帧,并更新卡尔曼增益矩阵
     * @note 使用默认观测矩阵h
     * @param measurement 观测值
     * @param dt 时间间隔
     */
    void
    update(const Eigen::MatrixXd& measurement, double dt = 0.0, bool setDefault = false) override;

    /**
     * @brief 使用控制量更新卡尔曼滤波器
     * @note 使用默认控制输入矩阵b
     * @param controlInput 控制输入
     */
    void control(const Eigen::MatrixXd& controlInput, double dt = 0.0) override;

    /**
     * @brief 使用传入的时间间隔预测一帧
     * @param dt 时间间隔
     */
    void predict(double dt = 0.0) override;

    /**
     * @brief 重置状态
     * @param currentState 设置模型状态矩阵
     */
    void resetState(const Eigen::MatrixXd& currentState = Eigen::MatrixXd()) override;

    /**
     * @brief 获取当前状态
     * @return 当前状态矩阵
     */
    Eigen::MatrixXd getState() const override;

private:
    /**
     * @brief 使用观测值更新卡尔曼滤波器
     * @param measurement 观测值
     */
    void updateKalmanGain();

    void initandCheck();

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
    Eigen::MatrixXd current_state_; // 当前状态矩阵,大小为StateSize_*1
    Eigen::Matrix<double, Eigen::Dynamic, 1> controlInput_; // 控制输入矩阵,大小为ControlSize_ * 1
    Eigen::Matrix<double, Eigen::Dynamic, 1> measurement_; // 观测值矩阵,大小为ObservationSize_ * 1
    Eigen::MatrixXd KalmanGain_; // 卡尔曼增益矩阵,大小为StateSize_ * ObservationSize_
    Eigen::MatrixXd p_;          // 协方差矩阵,大小为StateSize_ * StateSize_

    // 预测量
    Eigen::MatrixXd predictedState_; // 预测状态矩阵

    double dt_; // 时间间隔
};