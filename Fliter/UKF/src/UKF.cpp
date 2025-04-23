#include "UKF/include/UKF.hpp"
#include <cassert> // for assert
#include <cmath>   // for sqrt
#include <iostream>
#include <stdexcept> // for exceptions

UKF::UKF(
    std::function<Eigen::MatrixXd(const utils::BindableMatrixXd&)> f,
    std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> h,
    const Eigen::MatrixXd& q,
    const Eigen::MatrixXd& r,
    int stateSize,
    int observationSize,
    double alpha,
    double beta,
    double kappa
):
    alpha_(alpha),
    beta_(beta),
    kappa_(kappa),
    f_func_(f),
    h_func_(h)
// Note: dt_ is initialized in FilterBase or needs explicit initialization if not inherited
{
    this->stateSize_ = stateSize;
    this->observationSize_ = observationSize;
    this->q_ = q;
    this->r_ = r;
    initandCheck();
}

UKF::UKF(const UKF& ukf): FilterBase(ukf) { // Call base class copy constructor
    // Copy UKF specific members
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

    // Base class members are handled by FilterBase(ukf) if implemented correctly
    // If FilterBase copy constructor is default or shallow, manual copy might be needed
    // e.g., this->q_ = ukf.q_; this->r_ = ukf.r_; etc.
    // Assuming FilterBase handles its members properly.
    // Explicitly copy base members if FilterBase doesn't have a proper copy constructor
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
    // Call base class assignment operator if available and appropriate
    // FilterBase::operator=(ukf); // Uncomment if FilterBase has operator=

    // Copy UKF specific members
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

    // Copy base class members manually if FilterBase assignment is default/shallow or not implemented
    this->stateSize_ = ukf.stateSize_;
    this->observationSize_ = ukf.observationSize_;
    this->ControlSize_ = ukf.ControlSize_; // Make sure ControlSize_ is copied
    this->q_ = ukf.q_;
    this->r_ = ukf.r_;
    this->p_ = ukf.p_;
    this->current_state_ = ukf.current_state_;
    this->dt_ = ukf.dt_;
    this->initState_ = ukf.initState_;
    this->p0_ = ukf.p0_;
    this->q0_ = ukf.q0_;
    this->r0_ = ukf.r0_;
    this->k_ = ukf.k_; // UKF's K is calculated differently, but copy for completeness
    this->measurement_ = ukf.measurement_;       // Copy measurement if needed
    this->controlInput_ = ukf.controlInput_;     // Copy control input if needed
    this->predictedState_ = ukf.predictedState_; // Copy predicted state if needed

    return *this;
}

void UKF::init(const Eigen::MatrixXd& initState) {
    assert(
        initState.rows() == stateSize_ && initState.cols() == 1
        && "Initial state matrix dimension mismatch"
    );
    this->initState_ = initState;
    this->current_state_ = this->initState_;
    this->p_ = this->p0_; // Use initial covariance
}

void UKF::predict(double dt) {
    assert((dt > 0 || this->dt_ > 0) && "Time interval dt must be positive");
    // Use provided dt if > 0, otherwise use the stored dt_
    double current_dt = (dt > 0.0) ? dt : this->dt_;
    if (current_dt <= 0.0) {
        throw std::invalid_argument("Time interval dt must be positive for prediction.");
    }
    this->dt_ = current_dt; // Update stored dt if a new one is provided

    // 1. Generate Sigma Points based on current state and covariance
    generateSigmaPoints();

    // 2. Predict Sigma Points through the state transition function
    predictSigmaPoints();

    // 3. Predict Mean and Covariance from predicted sigma points
    predictMeanAndCovariance();

    // Store the predicted state (mean) in the base class member if desired
    this->predictedState_ = this->current_state_;
}

void UKF::update(const Eigen::MatrixXd& measurement, double dt, bool setDefault) {
    assert(
        measurement.rows() == observationSize_ && measurement.cols() == 1
        && "Measurement matrix dimension mismatch"
    );
    this->measurement_ = measurement; // Store measurement in base class member

    // Note: UKF typically assumes predict() was called just before update().
    // If dt is provided here, it might imply a combined predict-update step,
    // or simply updating the internal dt for future predictions.
    // Standard UKF separates predict and update. If a predict is needed here,
    // it should ideally use the dt provided.
    if (dt > 0.0) {
        // Option 1: Re-run prediction if dt is different (might be computationally expensive)
        // predict(dt);
        // Option 2: Just update the internal dt_ for the next cycle
        this->dt_ = dt;
        std::cerr
            << "Warning: UKF::update called with dt > 0. Updating internal dt, but assuming predict() was already called with appropriate timing."
            << std::endl;
    }

    // Ensure prediction results (predictedSigmaPoints_, predicted state mean/covariance) are available
    // This relies on the user calling predict() appropriately before update().

    // 4. Predict Observation from predicted sigma points
    predictObservation();

    // 5. Update State using the actual measurement
    updateState(measurement);

    // setDefault doesn't have a direct equivalent in standard UKF update.
    // The update step always incorporates the measurement.
    if (setDefault) {
        std::cerr << "Warning: setDefault=true has no standard effect in UKF::update." << std::endl;
        // If the intention is to discard the measurement and revert to the predicted state:
        // this->current_state_ = this->predictedState_; // Revert state
        // this->p_ = predicted_covariance; // Revert covariance (need to store predicted p_ from predictMeanAndCovariance)
    }
}

void UKF::control(const Eigen::MatrixXd& controlInput, double dt) {
    // Standard UKF doesn't explicitly handle control input 'B*u' separately.
    // Control inputs should be incorporated into the state transition function f_func_.
    // f_func_ should be defined as: x_k+1 = f(x_k, u_k, dt)
    std::cerr
        << "Warning: UKF::control is called, but standard UKF incorporates control input within the state transition function 'f'. Ensure 'f_func_' handles control inputs if needed."
        << std::endl;
    this->controlInput_ = controlInput; // Store control input if needed by f_func_

    // If dt is provided, update the internal dt
    if (dt > 0.0) {
        this->dt_ = dt;
    }

    // Note: The actual application of control input happens during the predict step
    // when f_func_ is called, assuming f_func_ uses the control input.
    // No separate state update is performed here in the standard UKF model.

    // Throwing an error might be safer if control inputs aren't handled by f_func_
    // throw std::logic_error("Control input must be handled within the state transition function 'f_func_' for UKF.");
}

void UKF::resetState(const Eigen::MatrixXd& currentState) {
    if (currentState.size() == 0) {
        // Reset to zero state and initial covariance
        this->current_state_ = Eigen::MatrixXd::Zero(this->stateSize_, 1);
        this->p_ = this->p0_;
    } else {
        if (currentState.rows() != this->stateSize_ || currentState.cols() != 1) {
            throw std::invalid_argument("Provided state matrix dimension mismatch for resetState.");
        }
        this->current_state_ = currentState;
        // Optionally reset covariance as well, common practice
        this->p_ = this->p0_;
    }
    // Resetting other internal UKF states (like sigma points) isn't strictly necessary
    // as they will be recalculated in the next predict step.
}

Eigen::MatrixXd UKF::getState() const {
    return this->current_state_;
}

// --- Private Methods ---

void UKF::generateSigmaPoints() {
    lambda_ = alpha_ * alpha_ * (static_cast<double>(stateSize_) + kappa_)
        - static_cast<double>(stateSize_);
    double gamma = std::sqrt(static_cast<double>(stateSize_) + lambda_); // Ensure double cast

    sigmaPoints_.col(0) = current_state_; // First sigma point is the current mean

    // Calculate matrix square root of P using Cholesky decomposition
    // Ensure P is positive definite
    Eigen::MatrixXd P_sqrt;
    Eigen::LLT<Eigen::MatrixXd> lltOfP(p_); // p_ is the current state covariance

    if (lltOfP.info() == Eigen::NumericalIssue) {
        // Attempt to fix non-positive definiteness (e.g., add small identity matrix)
        double epsilon = 1e-9;
        std::cerr << "Warning: Covariance matrix P is not positive definite. Adding epsilon * I."
                  << std::endl;
        Eigen::MatrixXd perturbed_p =
            p_ + epsilon * Eigen::MatrixXd::Identity(stateSize_, stateSize_);
        lltOfP.compute(perturbed_p);
        if (lltOfP.info() == Eigen::NumericalIssue) {
            // If still not positive definite, try making it diagonal or throw error
            // Option 1: Make diagonal (loses correlation info)
            // p_ = perturbed_p.diagonal().asDiagonal();
            // lltOfP.compute(p_);
            // Option 2: Throw error
            throw std::runtime_error(
                "Cholesky decomposition failed: Matrix is not positive definite even after adding epsilon."
            );
        }
        P_sqrt = lltOfP.matrixL(); // Use L (lower triangular)
        p_ = perturbed_p;          // Update p_ to the perturbed version if Cholesky succeeded
    } else {
        P_sqrt = lltOfP.matrixL(); // Use L (lower triangular)
        // Alternatively, use matrixU() for upper triangular, adjust indexing if needed
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
        && "predictedSigmaPoints_ dimensions incorrect"
    );
    for (int i = 0; i < 2 * stateSize_ + 1; ++i) {
        // Apply the non-linear state transition function to each sigma point
        // Pass control input if f_func_ expects it (requires modification of f_func_ signature or state augmentation)
        predictedSigmaPoints_.col(i) = f_func_(sigmaPoints_.col(i));
    }
}

void UKF::predictMeanAndCovariance() {
    // Calculate predicted state mean (weighted sum of predicted sigma points)
    // Use a temporary variable to avoid modifying current_state_ prematurely
    Eigen::MatrixXd predicted_mean = Eigen::MatrixXd::Zero(stateSize_, 1);
    for (int i = 0; i < 2 * stateSize_ + 1; ++i) {
        predicted_mean += weights_m_(i) * predictedSigmaPoints_.col(i);
    }

    // Calculate predicted state covariance
    Eigen::MatrixXd predicted_cov = Eigen::MatrixXd::Zero(stateSize_, stateSize_);
    for (int i = 0; i < 2 * stateSize_ + 1; ++i) {
        Eigen::MatrixXd diff = predictedSigmaPoints_.col(i) - predicted_mean;
        // Ensure consistent normalization if needed (e.g., angle wrapping)
        // Example: if state includes angle at index k: diff(k, 0) = normalize_angle(diff(k, 0));
        predicted_cov += weights_c_(i) * diff * diff.transpose();
    }
    predicted_cov += q_; // Add process noise covariance

    // Update the filter's state and covariance
    current_state_ = predicted_mean;
    p_ = predicted_cov;
}

void UKF::predictObservation() {
    // Transform predicted sigma points through the non-linear observation function
    assert(
        predictedObservations_.rows() == observationSize_
        && predictedObservations_.cols() == (2 * stateSize_ + 1)
        && "predictedObservations_ dimensions incorrect"
    );
    for (int i = 0; i < 2 * stateSize_ + 1; ++i) {
        predictedObservations_.col(i) = h_func_(predictedSigmaPoints_.col(i));
    }

    // Calculate predicted observation mean
    predictedMeasurementMean_.setZero(); // Reset before summing
    for (int i = 0; i < 2 * stateSize_ + 1; ++i) {
        predictedMeasurementMean_ += weights_m_(i) * predictedObservations_.col(i);
    }

    // Calculate predicted observation covariance (Pyy) and cross-covariance (Pxy)
    Pyy_.setZero(); // Reset before summing
    Pxy_.setZero(); // Reset before summing
    for (int i = 0; i < 2 * stateSize_ + 1; ++i) {
        Eigen::MatrixXd diff_y = predictedObservations_.col(i) - predictedMeasurementMean_;
        Eigen::MatrixXd diff_x =
            predictedSigmaPoints_.col(i) - current_state_; // Use the predicted state mean
        // Handle angle wrapping or normalization for diff_y and diff_x if necessary
        // Example: if observation includes angle at index j: diff_y(j, 0) = normalize_angle(diff_y(j, 0));
        // Example: if state includes angle at index k: diff_x(k, 0) = normalize_angle(diff_x(k, 0));

        Pyy_ += weights_c_(i) * diff_y * diff_y.transpose();
        Pxy_ += weights_c_(i) * diff_x * diff_y.transpose();
    }
    Pyy_ += r_; // Add observation noise covariance
}

void UKF::updateState(const Eigen::MatrixXd& measurement) {
    // Calculate Kalman Gain K
    // K = Pxy * Pyy^-1
    // Use robust inverse calculation if Pyy might be ill-conditioned
    Eigen::MatrixXd Pyy_inv;
    // Check condition number or determinant before inverting
    if (Pyy_.determinant() == 0) { // Simple check, better use condition number
        std::cerr
            << "Warning: Pyy is singular or near-singular. Using pseudo-inverse or skipping update."
            << std::endl;
        // Option 1: Use pseudo-inverse (requires SVD)
        // Pyy_inv = Pyy_.completeOrthogonalDecomposition().pseudoInverse();
        // Option 2: Skip update (or use very small gain)
        k_.setZero(); // Set gain to zero
        return;       // Skip state and covariance update
    } else {
        Pyy_inv = Pyy_.inverse();
    }

    k_ = Pxy_ * Pyy_inv; // Store Kalman gain in base class member if needed

    // Update state estimate
    // x = x_predicted + K * (z - z_predicted)
    Eigen::MatrixXd innovation = measurement - predictedMeasurementMean_;
    // Handle angle wrapping for innovation if necessary (e.g., if measurement involves angles)
    // Example: if measurement includes angle at index j: innovation(j, 0) = normalize_angle(innovation(j, 0));
    current_state_ = current_state_ + k_ * innovation;

    // Update state covariance
    // P = P_predicted - K * Pyy * K^T
    p_ = p_ - k_ * Pyy_ * k_.transpose();

    // Ensure P remains symmetric and positive semi-definite
    p_ = 0.5 * (p_ + p_.transpose()); // Force symmetry
    // Optional: Add small diagonal term or perform eigenvalue clamping if needed for stability

    // Joseph form for covariance update (more numerically stable but requires H):
    // P = (I - K * H) * P_predicted * (I - K * H)^T + K * R * K^T
    // Note: This requires an equivalent 'H' for UKF, which isn't directly available.
    // The standard formula P = P - K * Pyy * K^T is commonly used.
}

void UKF::initandCheck() {
    // Parameter checks
    assert(alpha_ > 0 && alpha_ <= 1 && "UKF parameter alpha must be in (0, 1]");
    // beta_ >= 0 is typical. beta_ = 2 is optimal for Gaussian distributions.
    assert(beta_ >= 0 && "UKF parameter beta must be non-negative");
    // kappa_ + stateSize_ > 0 must hold. kappa_ = 0 or kappa_ = 3 - stateSize_ are common choices.
    // Use static_cast<double> for calculations involving stateSize_ and kappa_
    assert(
        static_cast<double>(stateSize_) + kappa_ > 1e-9
        && "UKF parameter kappa + StateSize must be positive"
    ); // Use epsilon for float comparison

    // Dimension checks
    assert(stateSize_ > 0 && "State dimension must be positive");
    assert(observationSize_ > 0 && "Observation dimension must be positive");
    assert(
        q_.rows() == stateSize_ && q_.cols() == stateSize_
        && "Process noise covariance Q dimension mismatch"
    );
    assert(
        r_.rows() == observationSize_ && r_.cols() == observationSize_
        && "Observation noise covariance R dimension mismatch"
    );

    // Function checks
    assert(f_func_ && "State transition function f_func_ is not defined");
    assert(h_func_ && "Observation function h_func_ is not defined");

    // Initialize Sigma Point related matrices and vectors
    lambda_ = alpha_ * alpha_ * (static_cast<double>(stateSize_) + kappa_)
        - static_cast<double>(stateSize_);
    int numSigmaPoints = 2 * stateSize_ + 1;

    sigmaPoints_.resize(stateSize_, numSigmaPoints);
    predictedSigmaPoints_.resize(stateSize_, numSigmaPoints);
    predictedObservations_.resize(observationSize_, numSigmaPoints); // Resize this here

    weights_m_.resize(numSigmaPoints);
    weights_c_.resize(numSigmaPoints);

    // Calculate weights
    double w_m0 = lambda_ / (static_cast<double>(stateSize_) + lambda_);
    double w_c0 = w_m0 + (1.0 - alpha_ * alpha_ + beta_);
    double w_i = 0.5 / (static_cast<double>(stateSize_) + lambda_);

    weights_m_(0) = w_m0;
    weights_c_(0) = w_c0;
    for (int i = 1; i < numSigmaPoints; ++i) {
        weights_m_(i) = w_i;
        weights_c_(i) = w_i;
    }

    // Initialize base class members (assuming they exist and are accessible)
    // These might be initialized in FilterBase constructor or need explicit initialization here
    this->initState_ = Eigen::MatrixXd::Zero(stateSize_, 1);
    this->p0_ = Eigen::MatrixXd::Identity(stateSize_, stateSize_); // Default initial covariance
    this->q0_ = q_;                                                // Store initial Q
    this->r0_ = r_;                                                // Store initial R
    this->current_state_ = this->initState_;
    this->p_ = this->p0_;
    this->k_ = Eigen::MatrixXd::Zero(stateSize_, observationSize_); // K is calculated during update
    this->measurement_ = Eigen::MatrixXd::Zero(observationSize_, 1);
    this->predictedState_ = Eigen::MatrixXd::Zero(stateSize_, 1); // Initialize predicted state
    this->ControlSize_ =
        0; // Standard UKF doesn't have explicit control size, set to 0 unless augmented
    this->controlInput_ = Eigen::MatrixXd::Zero(0, 1); // Initialize control input

    // Initialize UKF specific matrices
    predictedMeasurementMean_.resize(observationSize_, 1);
    Pyy_.resize(observationSize_, observationSize_);
    Pxy_.resize(stateSize_, observationSize_);
}