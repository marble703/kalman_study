#include <eigen3/Eigen/Dense>
#include <memory>

/**
 * @brief 包装Eigen矩阵,实现使用lambda表达式绑定动态元素到矩阵
 * 
 */
class BindableMatrixXd {
public:
    /**
    * @brief 构造函数，仅初始化矩阵
    * 
    * @param rows 
    * @param cols 
    */
    BindableMatrixXd(int rows, int cols): matrix_(rows, cols) {
        matrix_.setZero(); // 初始化矩阵为零
    }

    BindableMatrixXd(const Eigen::MatrixXd& mat): matrix_(mat) {}
    /**
     * @brief 绑定 lambda 表达式到矩阵元素
     * 
     * @param row 矩阵行索引
     * @param col 矩阵列索引
     * @param func 绑定的 lambda 表达式
     */
    void bind(int row, int col, std::function<double(std::shared_ptr<double>)> func) {
        bindings_.emplace_back(row, col, func);
    }

    /**
     * @brief 更新所有绑定的矩阵元素
     * 
     */
    void update() {
        for (const auto& binding: bindings_) {
            int i = std::get<0>(binding);
            int j = std::get<1>(binding);
            matrix_(i, j) = std::get<2>(binding)(arg_);
        }
    }
    // 访问元素
    double operator()(int row, int col) const {
        assert(
            row >= 0 && row < matrix_.rows() && col >= 0 && col < matrix_.cols()
            && "非法矩阵索引"
        );

        return matrix_(row, col);
    }

    int rows() const {
        return matrix_.rows();
    }

    int cols() const {
        return matrix_.cols();
    }

    void setArg(std::shared_ptr<double> arg) {
        arg_ = arg;
    }
    std::shared_ptr<double> getArg() const {
        return arg_;
    }

private:
    std::shared_ptr<double> arg_; // 用于存储 lambda 表达式的参数

    Eigen::MatrixXd matrix_; // 基础矩阵
    std::vector<std::tuple<int, int, std::function<double(std::shared_ptr<double>)>>>
        bindings_; // 存储绑定关系
};