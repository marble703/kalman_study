#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace data {

struct Target {
    double t;                // 时间戳
    double x, y, z;          // 位置
    double vx, vy, vz;       // 速度
    double yaw, roll, pitch; // 姿态角
};

class DataLoader {
public:
    // 构造函数，接收数据文件路径
    explicit DataLoader(const std::string& filePath);

    // 析构函数
    ~DataLoader();

    // 从文件加载数据
    bool loadData();

    // 获取帧间时间间隔
    double getFrameInterval() const;

    // 获取主目标数据
    const std::vector<Target>& getMainTargetData() const;

    // 获取指定子目标数据
    const std::vector<Target>& getSubTargetData(int subTargetIndex) const;

    // 获取子目标数量
    int getSubTargetCount() const;

    // 获取总数据帧数
    size_t getFrameCount() const;

private:
    std::string filePath_;                  // 数据文件路径
    std::vector<std::vector<Target>> data_; // 所有目标的数据(主目标+子目标)
    int subTargetCount_ { 0 };              // 子目标数量
    double frameInterval_ { 0.005 };        // 帧间时间间隔，默认为0.005秒(200Hz)
};

} // namespace data
