#include "KF/include/KF.hpp"

#include <iostream>
#include <rclcpp/rclcpp.hpp>
// #include "msg/autoaim.hpp"     // IWYU pragma: keep

int main(){

    rclcpp::init(0, nullptr);
    

    auto node = std::make_shared<rclcpp::Node>("kf_test_node");

    rclcpp::spin(node);
    rclcpp::shutdown();
    
    return 0;
}