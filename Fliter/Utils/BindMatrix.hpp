#pragma once

#include <eigen3/Eigen/Dense>
#include <memory>

namespace utils {

class BindableMatrixXd {
public:
    BindableMatrixXd();
    BindableMatrixXd(int rows, int cols);
    BindableMatrixXd(const Eigen::MatrixXd& mat);

    /**
     * @brief 静态工厂方法，创建一个包含任意类型元素的 BindableMatrixXd
     * 
     * 用法示例：
     * auto matrix = utils::BindableMatrixXd::create(2, 2, 1.0, [arg](){ return *arg; }, 0.0, 1.0);
     * 
     * @tparam Args 参数包类型
     * @param rows 行数
     * @param cols 列数
     * @param args 任意类型的参数列表
     * @return BindableMatrixXd 
     */
    template<typename... Args>
    static BindableMatrixXd create(int rows, int cols, bool autoUpdate = false, Args&&... args) {
        BindableMatrixXd result;

        result.autoUpdate_ = autoUpdate;
        result.matrix_.resize(rows, cols);

        // 将参数包展开为元组
        std::tuple<Args...> argsTuple(std::forward<Args>(args)...);
        constexpr size_t numArgs = sizeof...(Args);

        if (numArgs != rows * cols) {
            throw std::invalid_argument("参数数量与矩阵尺寸不匹配");
        }

        // 处理每个参数
        processArgs(result, argsTuple, std::make_index_sequence<numArgs> {}, rows, cols);

        return result;
    }

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
     */
    void update() const;

    /**
     * @brief 更新矩阵的值
     * 
     * @param 传入值
     */
    void update(double arg) const;

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
     * @brief 获取矩阵的引用
     * 
     * @return Eigen::MatrixXd& 矩阵的引用
     */
    Eigen::MatrixXd& getMatrix();

    Eigen::MatrixXd operator+(const Eigen::MatrixXd& other) const;
    Eigen::MatrixXd operator-(const Eigen::MatrixXd& other) const;
    Eigen::MatrixXd operator*(const Eigen::MatrixXd& other) const;
    Eigen::MatrixXd operator/(const Eigen::MatrixXd& other) const;
    Eigen::MatrixXd operator-() const;
    Eigen::MatrixXd transpose() const;
    Eigen::MatrixXd inverse() const;

private:
    // 递归辅助函数，处理参数包中的每个元素
    template<typename Tuple, size_t... Is>
    static void processArgs(
        BindableMatrixXd& matrix,
        const Tuple& tuple,
        std::index_sequence<Is...>,
        int rows,
        int cols
    ) {
        // 计算行列索引
        auto processArg = [rows, cols](auto& matrix, size_t idx, const auto& value) {
            int i = idx / cols;
            int j = idx % cols;

            if constexpr (std::is_invocable_v<
                              std::decay_t<decltype(value)>,
                              std::shared_ptr<double>>) {
                // 如果是接受 std::shared_ptr<double> 参数的可调用对象
                auto func = [v = value](std::shared_ptr<double> arg) -> double { return v(arg); };
                matrix.bindings_.emplace_back(i, j, func);
                try {
                    // 尝试获取初始值，如果 arg_ 已设置则传入
                    matrix.matrix_(i, j) = matrix.arg_ ? value(matrix.arg_) : 0.0;
                } catch (...) {
                    matrix.matrix_(i, j) = 0.0;
                }
            } else if constexpr (std::is_invocable_v<std::decay_t<decltype(value)>>) {
                // 如果是不接受参数的可调用对象
                auto func = [v = value](std::shared_ptr<double>) -> double { return v(); };
                matrix.bindings_.emplace_back(i, j, func);
                try {
                    // 尝试获取初始值
                    matrix.matrix_(i, j) = value();
                } catch (...) {
                    matrix.matrix_(i, j) = 0.0;
                }
            } else {
                // 如果是常量值
                matrix.matrix_(i, j) = static_cast<double>(value);
            }
        };

        // 展开参数包并应用处理函数
        (processArg(matrix, Is, std::get<Is>(tuple)), ...);
    }

    std::shared_ptr<double> arg_;    // 参数
    mutable double argValueCache_;   // 参数值缓存
    mutable Eigen::MatrixXd matrix_; // 矩阵
    std::vector<std::tuple<int, int, std::function<double(std::shared_ptr<double>)>>>
        bindings_; // 绑定的函数
    bool autoUpdate_;
};

} // namespace utils