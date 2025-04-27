# 滤波器合集

`KF`, `EKF`, `UKF` 均继承自 `FilterBase` 滤波器基类
除了 `UKF` 未实现 `control` 方法(可调用，但是只更新参数，不实现控制)外, 其余方法均已实现

## 当前问题

`UKF` 还在开发中(25.04.24)，未实现动态时间更新，控制输入逻辑待优化

## 公共方法

### 构造

`EKF`, `UKF` 存在非线性函数而 `KF` 没有，目前没有公共的完整构造

来自 Eigen 矩阵的拷贝/移动构造在计划中

### 初始化 `init`

`void init(const Eigen::MatrixXd& initState)`

### 更新 `update`

`void update(const Eigen::MatrixXd& measurement, double dt = 0.0, bool setDefault = false)`

参数 `setDefault` 作用是设置传入的 `dt` 为默认

### 预测 `predict`

`void predict(double dt = 0.0)`

### 获取状态 `getState`

`Eigen::MatrixXd getState() const`

### 控制 `control`

`void control(const Eigen::MatrixXd& controlInput, double dt = 0.0)`

### 重置状态 `resetState`

`void resetState(const Eigen::MatrixXd& currentState)`