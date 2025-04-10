#include "KF/include/KF.hpp"

template<typename T>
KF<T>::KF(
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> f,
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> h,
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> b,
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> q,
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> r,
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> p
) : f_(f), h_(h), b_(b), q_(q), r_(r), p_(p) {
    assert(f_.rows() == f_.cols() && "状态转移矩阵 f 必须是方阵");
    assert(h_.rows() == h_.cols() && "观测矩阵 h 必须是方阵");
    assert(b_.rows() == b_.cols() && "控制输入矩阵 b 必须是方阵");
    assert(q_.rows() == q_.cols() && "过程噪声协方差矩阵 q 必须是方阵");
    assert(r_.rows() == r_.cols() && "观测噪声协方差矩阵 r 必须是方阵");
    assert(p_.rows() == p_.cols() && "初始估计协方差矩阵 p 必须是方阵");
    assert(f_.rows() == h_.rows() && f_.rows() == b_.rows() && f_.rows() == q_.rows() && f_.rows() == r_.rows() && f_.rows() == p_.rows() && "矩阵维度不匹配");
    assert(f_.cols() == h_.cols() && f_.cols() == b_.cols() && f_.cols() == q_.cols() && f_.cols() == r_.cols() && f_.cols() == p_.cols() && "矩阵维度不匹配");
}