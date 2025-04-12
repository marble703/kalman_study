#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <eigen3/Eigen/src/Core/util/Constants.h>

class KF {
public:
    /**
     * @brief 完整模型构造,不初始化dt
     * 
     * @param f // 状态转移矩阵(固定)
     * @param h // 观测矩阵(固定)
     * @param b // 控制输入矩阵(固定)
     * @param q // 过程噪声协方差矩阵(可变)
     * @param r // 观测噪声协方差矩阵(可变)
     */
    KF(const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>& f,
       const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>& h,
       const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>& b,
       const Eigen::Ref<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>& q,
       const Eigen::Ref<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>& r);
    KF(const KF& kf);
    ~KF();

    /**
     * @brief 传入初始状态,初始化卡尔曼滤波器
     * @param initState 初始状态
     */
    void
    init(const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>&
             initState);

    /**
     * @brief 使用观察值和时间间隔更新卡尔曼滤波器,预测下一帧,并更新卡尔曼增益矩阵
     * @brief 使用默认观测矩阵h
     * @param measurement 观察值
     * @param dt 时间间隔
     */
    void update(
        const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>&
            measurement,
        float dt
    );

    void
    update(const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>&
               measurement);

    /**
     * @brief 使用观测值更新卡尔曼滤波器
     * @brief 使用默认观测矩阵h
     * @param measurement 观测值
     */
    void updateKalmanGain();

    /**
     * @brief 使用控制量更新卡尔曼滤波器
     * @brief 使用默认控制输入矩阵b
     * @param controlInput 控制输入
     */
    void
    control(const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>&
                controlInput);

    /**
     * @brief 预测一帧
     * 
     */
    void predict();
    void predict(float dt);

    void resetState();
    void resetState(
        const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>&
            measurement
    );

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> getState() const;

private:
    // 矩阵维度
    size_t ObservationSize_; // 观测维度
    size_t StateSize_;       // 状态维度
    size_t ControlSize_;     // 控制输入维度

    // 模型矩阵(固定)
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>&
        f_; // 状态转移矩阵,大小为StateSize_ * StateSize_
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>&
        h_; // 观测矩阵,大小为ObservationSize_ * StateSize_
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>&
        b_; // 控制输入矩阵,大小为StateSize_ * ControlSize_

    // 初始量
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
        initState_; // 初始状态矩阵,大小为StateSize_
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
        p0_; // 初始协方差矩阵,大小为StateSize_ * StateSize_
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
        q0_; // 初始过程噪声协方差矩阵,大小为StateSize_ * StateSize_
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
        r0_; // 初始观测噪声协方差矩阵,大小为ObservationSize_ * ObservationSize_

    // 噪声协方差矩阵
    const Eigen::Ref<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>&
        q_; // 过程噪声协方差矩阵,大小为StateSize_ * StateSize_
    const Eigen::Ref<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>&
        r_; // 观测噪声协方差矩阵,大小为ObservationSize_ * ObservationSize_
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
        k_; // 卡尔曼增益矩阵,大小为StateSize_ * ObservationSize_

    // 当前量
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
        current_state_; // 当前状态矩阵,大小为StateSize_*StateSize_
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
        controlInput_; // 控制输入矩阵,大小为ControlSize_
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
        measurement_; // 观测值矩阵,大小为ObservationSize_
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
        KalmanGain_; // 卡尔曼增益矩阵,大小为StateSize_ * ObservationSize_
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
        p_; // 协方差矩阵,大小为StateSize_ * StateSize_
    // 预测量
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> predictedState_; // 预测状态矩阵

    double dt_; // 时间间隔
};