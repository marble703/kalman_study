#include <eigen3/Eigen/Dense>

class EKF {
public:
    EKF(const Eigen::MatrixXd f,
        const Eigen::MatrixXd h,
        const Eigen::MatrixXd b,
        Eigen::MatrixXd q,
        Eigen::MatrixXd r,
        double dt = 0.01);

    EKF(Eigen::MatrixXd f,
        Eigen::MatrixXd h,
        Eigen::MatrixXd b);

    EKF(const EKF& ekf);

    EKF& operator=(const EKF& ekf);

    EKF() = default;

    ~EKF() = default;

    void init(const Eigen::MatrixXd& initState);

    void update(const Eigen::MatrixXd& measurement, float dt);

    void update(const Eigen::MatrixXd& measurement);

    void updateKalmanGain();

    void control(const Eigen::MatrixXd& controlInput);

    void predict(float dt);

    void predict();

    void resetState(const Eigen::MatrixXd& currentState);

    void resetState();

    Eigen::MatrixXd getState() const;

private:
    size_t ObservationSize_;
    size_t StateSize_;
    size_t ControlSize_;

    const Eigen::MatrixXd f_;
    const Eigen::MatrixXd h_;
    const Eigen::MatrixXd b_;

    Eigen::Matrix<double, Eigen::Dynamic, 1> initState_;
    Eigen::MatrixXd p0_;
    Eigen::MatrixXd q0_;
    Eigen::MatrixXd r0_;

    Eigen::MatrixXd q_;
    Eigen::MatrixXd r_;
    Eigen::MatrixXd k_;

    Eigen::MatrixXd current_state_;
    Eigen::Matrix<double, Eigen::Dynamic, 1> controlInput_;
    Eigen::Matrix<double, Eigen::Dynamic, 1> measurement_;
    Eigen::MatrixXd KalmanGain_;
    Eigen::MatrixXd p_;

    Eigen::MatrixXd predictedState_;

    double dt_;
};