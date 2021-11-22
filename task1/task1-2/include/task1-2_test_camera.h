#pragma once
#include "math/vector.h"
#include <iostream>
class Camera
{
public:
    Camera()
    {
        //采用归一化坐标，不考虑图像尺寸
        c_[0] = c_[1] = 0.0;
    }
    math::Vec2d projection(math::Vec3d const &p3d);
    math::Vec3d pos_in_world();
    math::Vec3d dir_in_world();

public:
    // 焦距f
    double f_;

    // 径向畸变系数k1, k2
    double dist_[2];

    // 中心点坐标u0, v0
    double c_[2];

    // 旋转矩阵
    /*
     * [ R_[0], R_[1], R_[2] ]
     * [ R_[3], R_[4], R_[5] ]
     * [ R_[6], R_[7], R_[8] ]
     */
    double R_[9];
    
    // 平移向量
    double t_[3];
};