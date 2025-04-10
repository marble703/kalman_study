#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>

template<typename T>
class KF {
public:
    /**
     * @brief Construct a new KF object
     * 
     * @param f  // 状态转移矩阵
     * @param h  // 观测矩阵
     * @param b  // 控制输入矩阵
     * @param q  // 过程噪声协方差矩阵
     * @param r  // 观测噪声协方差矩阵
     * @param p // 估计误差协方差矩阵
     */
    KF(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> f,
       Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> h,
       Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> b,
       Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> q,
       Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> r,
       Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> p);
    KF(const KF& kf);
    ~KF();

    void init(Eigen::Vector<T, Eigen::Dynamic> initState // 初始状态
    );
    void predict();
    void update(float measurement, float dt);

    float getState() const;

private:
    void resetState();

    void setMeasurement(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> measurement);

    void setControlInput(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> controlInput);

    float getPredictedState() const;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> f_; // 状态转移矩阵
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> h_; // 观测矩阵
    
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> b_; // 控制输入矩阵
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> q_; // 过程噪声协方差矩阵
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> r_; // 观测噪声协方差矩阵
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> p_; // 估计误差协方差矩阵

    Eigen::Vector<T, Eigen::Dynamic> initState_; // 初始状态
    Eigen::Vector<T, Eigen::Dynamic> state_; // 当前状态
};