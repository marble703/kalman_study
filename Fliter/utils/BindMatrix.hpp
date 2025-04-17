#include <eigen3/Eigen/Dense>
#include <memory>

namespace utils {

// 编译期固定值的元素生成器
template<auto Value>
constexpr auto make_element() {
    return []() constexpr {
        return Value;
    };
}

// 运行时动态值的元素生成器（按值捕获）
template<typename T>
auto make_element(T&& value) {
    return [value = std::forward<T>(value)]() { return value; };
}

class BindableMatrixXd {
public:
    BindableMatrixXd();
    BindableMatrixXd(int rows, int cols);
    BindableMatrixXd(const Eigen::MatrixXd& mat);

    /**
     * @brief 构造函数，支持类似矩阵初始化的方式
     * 
     * @param rows 行数
     * @param cols 列数
     * @param initializer 初始化列表，包含固定值或lambda表达式
     */
    template <typename T>
    BindableMatrixXd(int rows, int cols, std::initializer_list<T> initializer);

    /**
     * @brief 绑定一个函数到矩阵的指定位置
     * 
     * @param row 行索引
     * @param col 列索引
     * @param func 绑定的函数
     */
    void bind(int row, int col, std::function<double(std::shared_ptr<double>)> func);

    /**
     * @brief 更新矩阵的值
     * 
     * @note 更新矩阵的值
     */
    void update();

    /**
     * @brief 获取矩阵的值
     * 
     * @param row 行索引
     * @param col 列索引
     * @return double 矩阵的值
     */
    double operator()(int row, int col) const;

    /**
     * @brief 获取矩阵的行数
     * 
     * @return int 矩阵的行数
     */
    int rows() const;

    /**
     * @brief 获取矩阵的列数
     * 
     * @return int 矩阵的列数
     */
    int cols() const;

    /**
     * @brief 设置参数
     * 
     * @param arg 参数
     */
    void setArg(std::shared_ptr<double> arg);

    /**
     * @brief 获取参数
     * 
     * @return std::shared_ptr<double> 参数
     */
    std::shared_ptr<double> getArg() const;

    /**
     * @brief 清除所有绑定
     * 
     * @note 该方法将清除所有已绑定的函数
     */
    void clearBindings();

    /**
     * @brief 获取矩阵的引用
     * 
     * @return Eigen::MatrixXd& 矩阵的引用
     */
    Eigen::MatrixXd& getMatrix() {
        return matrix_;
    }

private:
    std::shared_ptr<double> arg_; // 参数
    Eigen::MatrixXd matrix_; // 矩阵
    std::vector<std::tuple<int, int, std::function<double(std::shared_ptr<double>)>>>
        bindings_; // 绑定的函数
};

} // namespace utils