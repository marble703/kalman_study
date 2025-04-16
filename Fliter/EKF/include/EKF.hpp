#include <eigen3/Eigen/Dense>

class EKF {
public:
    EKF(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> f,
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> h,
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> b,
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> q,
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> r,
        double dt = 0.01);

    EKF(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> f,
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> h,
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> b);

    EKF(const EKF& ekf);

    EKF& operator=(const EKF& ekf);

    EKF() = default;

    ~EKF() = default;

    void init(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& initState);

    void update(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& measurement, float dt);

    void update(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& measurement);

    void updateKalmanGain();

    void control(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& controlInput);

    void predict(float dt);

    void predict();

    void resetState(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& currentState);

    void resetState();

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> getState() const;

private:
    size_t ObservationSize_;
    size_t StateSize_;
    size_t ControlSize_;

    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> f_;
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> h_;
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> b_;

    Eigen::Matrix<double, Eigen::Dynamic, 1> initState_;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> p0_;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> q0_;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> r0_;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> q_;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> r_;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> k_;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> current_state_;
    Eigen::Matrix<double, Eigen::Dynamic, 1> controlInput_;
    Eigen::Matrix<double, Eigen::Dynamic, 1> measurement_;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> KalmanGain_;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> p_;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> predictedState_;

    double dt_;
};