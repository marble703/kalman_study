#include <eigen3/Eigen/Dense>

class FilterBase {
public:
    virtual ~FilterBase() = default;

    // 初始化滤波器
    virtual void
    init(const Eigen::VectorXd& initial_state, const Eigen::MatrixXd& initial_covariance) = 0;

    // 预测
    virtual void predict(const Eigen::MatrixXd& process_noise, double dt = 0.0) = 0;

    // 更新
    virtual void
    update(const Eigen::VectorXd& measurement, const Eigen::MatrixXd& measurement_noise) = 0;

    // 获取当前状态和协方差
    virtual Eigen::VectorXd getState() const = 0;
    virtual Eigen::MatrixXd getCovariance() const = 0;

protected:
    Eigen::VectorXd state_;      // 状态向量
    Eigen::MatrixXd covariance_; // 状态协方差矩阵
};