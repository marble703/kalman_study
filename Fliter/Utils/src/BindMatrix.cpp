#include "../BindMatrix.hpp"

namespace utils {

BindableMatrixXd::BindableMatrixXd() {
    matrix_.resize(0, 0);
}

BindableMatrixXd::BindableMatrixXd(int rows, int cols): matrix_(rows, cols) {
    matrix_.resize(rows, cols);
    matrix_.setIdentity();
}

BindableMatrixXd::BindableMatrixXd(const Eigen::MatrixXd& mat): matrix_(mat) {}

void BindableMatrixXd::bind(
    int row,
    int col,
    std::function<double(std::shared_ptr<double>)> func
) {
    assert(
        row >= 0 && row < matrix_.rows() && col >= 0 && col < matrix_.cols()
        && "非法矩阵索引"
    );
    assert(func && "绑定函数不能为空");
    assert(arg_ && "参数不能为空");
    bindings_.emplace_back(row, col, func);
}

void BindableMatrixXd::update() const {
    assert(arg_ && "参数不能为空");
    // 如果没有绑定函数，或者参数没有变化，则不更新
    if (bindings_.empty() || this->argValueCache_ == *arg_) {
        return;
    }
    for (const auto& binding: bindings_) {
        int i = std::get<0>(binding);
        int j = std::get<1>(binding);
        matrix_(i, j) = std::get<2>(binding)(arg_);
    }
    // 更新参数值缓存
    this->argValueCache_ = *arg_;
    return;
}

void BindableMatrixXd::update(double arg) const {
    // 如果没有绑定函数，或者参数没有变化，则不更新
    if (bindings_.empty() || this->argValueCache_ == arg) {
        return;
    }
    for (const auto& binding: bindings_) {
        int i = std::get<0>(binding);
        int j = std::get<1>(binding);
        matrix_(i, j) = std::get<2>(binding)(std::make_shared<double>(arg));
    }
    // 更新参数值缓存
    this->argValueCache_ = *arg_;
}

double BindableMatrixXd::operator()(int row, int col) const {
    assert(
        row >= 0 && row < matrix_.rows() && col >= 0 && col < matrix_.cols()
        && "非法矩阵索引"
    );

    return matrix_(row, col);
}

int BindableMatrixXd::rows() const {
    return matrix_.rows();
}

int BindableMatrixXd::cols() const {
    return matrix_.cols();
}

void BindableMatrixXd::setArg(std::shared_ptr<double> arg) {
    arg_ = arg;
}

std::shared_ptr<double> BindableMatrixXd::getArg() const {
    return arg_;
}

Eigen::MatrixXd& BindableMatrixXd::getMatrix() {
    if (this->autoUpdate_) {
        this->update();
    }
    return matrix_;
}

Eigen::MatrixXd BindableMatrixXd::operator+(const Eigen::MatrixXd& other) const {
    if (this->autoUpdate_) {
        this->update();
    }
    Eigen::MatrixXd result;
    result = this->matrix_ + other;
    return result;
}

Eigen::MatrixXd BindableMatrixXd::operator-(const Eigen::MatrixXd& other) const {
    if (this->autoUpdate_) {
        this->update();
    }
    Eigen::MatrixXd result;
    result = this->matrix_ - other;
    return result;
}

Eigen::MatrixXd BindableMatrixXd::operator*(const Eigen::MatrixXd& other) const {
    if (this->autoUpdate_) {
        this->update();
    }
    Eigen::MatrixXd result;
    result = this->matrix_ * other;
    return result;
}

Eigen::MatrixXd BindableMatrixXd::operator/(const Eigen::MatrixXd& other) const {
    if (this->autoUpdate_) {
        this->update();
    }
    Eigen::MatrixXd result;
    result = this->matrix_.cwiseQuotient(other);
    return result;
}

Eigen::MatrixXd BindableMatrixXd::operator-() const {
    if (this->autoUpdate_) {
        this->update();
    }
    Eigen::MatrixXd result;
    result = -this->matrix_;
    return result;
}

Eigen::MatrixXd BindableMatrixXd::transpose() const {
    if (this->autoUpdate_) {
        this->update();
    }
    Eigen::MatrixXd result;
    result = this->matrix_.transpose();
    return result;
}

Eigen::MatrixXd BindableMatrixXd::inverse() const {
    if (this->autoUpdate_) {
        this->update();
    }
    Eigen::MatrixXd result;
    result = this->matrix_.inverse();
    return result;
}

} // namespace utils