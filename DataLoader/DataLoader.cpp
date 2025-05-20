#include "DataLoader.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

namespace data {

DataLoader::DataLoader(const std::string& filePath): filePath_(filePath) {
    // 加载数据
    if (!loadData()) {
        std::cerr << "Failed to load data from file: " << filePath_ << std::endl;
    }
}

DataLoader::~DataLoader() {}

bool DataLoader::loadData() {
    std::ifstream file(filePath_);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filePath_ << std::endl;
        return false;
    }

    std::string line;
    std::getline(file, line); // 读取表头

    // 解析表头，确定子目标数量
    std::stringstream headerStream(line);
    std::string header;
    std::vector<std::string> headers;

    while (std::getline(headerStream, header, ',')) {
        headers.push_back(header);
    }

    // 计算子目标数量 (每个子目标有6个字段)
    if (headers.size() <= 10) { // 主目标有10个字段
        std::cerr << "Invalid header format, no sub-targets found" << std::endl;
        return false;
    }

    subTargetCount_ = (headers.size() - 10) / 6; // 计算子目标数量
    std::cout << "Detected " << subTargetCount_ << " sub-targets from header" << std::endl;

    if (subTargetCount_ != 3 && subTargetCount_ != 4) {
        std::cerr << "Unexpected sub-target count: " << subTargetCount_ << " (expected 3 or 4)"
                  << std::endl;
    }

    // 初始化数据结构
    data_.resize(subTargetCount_ + 1); // 主目标 + 子目标

    // 读取每一行数据
    double prevTimestamp = -1.0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> values;

        while (std::getline(ss, value, ',')) {
            values.push_back(std::stod(value));
        }

        if (values.size() != headers.size()) {
            std::cerr << "Line has incorrect number of values: " << values.size()
                      << ", expected: " << headers.size() << std::endl;
            continue;
        }

        // 计算帧间时间间隔
        double currentTimestamp = values[0];
        if (prevTimestamp >= 0.0) {
            // 更新帧间间隔为当前观察到的平均值
            double interval = currentTimestamp - prevTimestamp;
            if (frameInterval_ <= 0.0) {
                frameInterval_ = interval;
            } else {
                // 简单平均计算
                frameInterval_ = (frameInterval_ + interval) / 2.0;
            }
        }
        prevTimestamp = currentTimestamp;

        // 提取主目标数据
        Target mainTarget;
        mainTarget.t = values[0];
        mainTarget.x = values[1];
        mainTarget.y = values[2];
        mainTarget.z = values[3];
        mainTarget.vx = values[4];
        mainTarget.vy = values[5];
        mainTarget.vz = values[6];
        mainTarget.yaw = values[7];
        mainTarget.roll = values[8];
        mainTarget.pitch = values[9];

        data_[0].push_back(mainTarget);

        // 提取子目标数据
        for (int i = 0; i < subTargetCount_; ++i) {
            Target subTarget;
            int baseIdx = 10 + i * 6; // 子目标数据起始位置

            subTarget.t = values[0]; // 使用相同的时间戳
            subTarget.x = values[baseIdx];
            subTarget.y = values[baseIdx + 1];
            subTarget.z = values[baseIdx + 2];
            subTarget.yaw = values[baseIdx + 3];
            subTarget.roll = values[baseIdx + 4];
            subTarget.pitch = values[baseIdx + 5];
            subTarget.vx = 0.0; // 速度信息不可用，设为0
            subTarget.vy = 0.0;
            subTarget.vz = 0.0;

            data_[i + 1].push_back(subTarget);
        }
    }

    file.close();

    if (data_[0].empty()) {
        std::cerr << "No data loaded" << std::endl;
        return false;
    }

    std::cout << "Loaded " << data_[0].size()
              << " data points with frame interval: " << frameInterval_ << " seconds" << std::endl;
    return true;
}

double DataLoader::getFrameInterval() const {
    return frameInterval_;
}

const std::vector<Target>& DataLoader::getMainTargetData() const {
    return data_[0];
}

const std::vector<Target>& DataLoader::getSubTargetData(int subTargetIndex) const {
    if (subTargetIndex < 0 || subTargetIndex >= subTargetCount_) {
        throw std::out_of_range("Sub-target index out of range");
    }
    return data_[subTargetIndex + 1];
}

int DataLoader::getSubTargetCount() const {
    return subTargetCount_;
}

size_t DataLoader::getFrameCount() const {
    return data_[0].size();
}

} // namespace data