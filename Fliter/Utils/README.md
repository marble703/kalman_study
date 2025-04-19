# 工具空间

全部包含在命名空间 `utils` 下

## BindableMatrixXd 变量矩阵

实现了一个单变量混合矩阵，矩阵元素可以是 `double` 常量或是 `std::function<double(std::shared_ptr<double>)`
由于仅供 KF 使用，后面不会在这里拓展更多非必要功能

### 实现方法

通过绑定到一个 `double` 类型智能指针的一个可调用对象容器维护一个 `Eigen::MatrixXd`

### 构造

#### 建议使用 create

最简单的构造是使用 `create()`
直接使用传矩阵元素的方法混合传递

`template<typename... Args> static BindableMatrixXd create(int rows, int cols, bool autoUpdate = false, Args&&... args)`

示例：

```cpp
    std::shared_ptr<double> arg = std::make_shared<double>(0.0);

    auto bindMatrix = utils::BindableMatrixXd::create(
        2, 2,
        1.0, [arg]() { return *arg; },
        0.0, 1.0
    );
```

参数 `autoUpdate` 见[自动更新](./README.md#自动更新)

#### 其他

同时支持以下构造

`BindableMatrixXd()`
初始化大小为0,0的矩阵

`BindableMatrixXd(int rows, int cols)`
初始化指定大小矩阵

`BindableMatrixXd(const Eigen::MatrixXd& mat)`
传递矩阵

### 更新值

使用一个缓存变量，只有值更新时更新矩阵

#### 手动更新

`void update()`
使用绑定指针更新

`void update(double arg)`
使用传入值更新

#### 自动更新

在每次需要获取矩阵值时重新计算

### set

#### 设置绑定变量 setArg

`void setArg(std::shared_ptr<double> arg)`

#### 设置绑定关系 bind

`void bind(int row, int col, std::function<double(std::shared_ptr<double>)> func)`

### get

#### 获取参数

`std::shared_ptr<double> getArg() const`

#### 获取矩阵

`Eigen::MatrixXd& getMatrix();`
获取矩阵

`int rows() const`
`int cols() const`
获取行列

重载了(),可以直接通过行列索引获取矩阵值

### 矩阵运算

所有矩阵操作均不会影响类内值，而是返回一个新的 `Eigen::MatrixXd`