#include "task1-2_test_camera.h"
//相机投影过程
math::Vec2d Camera::projection(math::Vec3d const &p3d)
{
    //math::Vec2d p;
    /** TODO HERE
         *
         */
    //return p;

    //Reference
    // 世界坐标系到相机坐标系
    double xc = R_[0] * p3d[0] + R_[1] * p3d[1] + R_[2] * p3d[2] + t_[0];
    double yc = R_[3] * p3d[0] + R_[4] * p3d[1] + R_[5] * p3d[2] + t_[1];
    double zc = R_[6] * p3d[0] + R_[7] * p3d[1] + R_[8] * p3d[2] + t_[2];

    // 相机坐标系到像平面
    double x = xc / zc;
    double y = yc / zc;

    // 径向畸变过程
    double r2 = x * x + y * y; //r2为x，y到图像中心的距离
    double distort_ratio = 1 + dist_[0] * r2 + dist_[1] * r2 * r2;//x`=(1+k0*r^2+k1*r^4)x

    // 图像坐标系到屏幕坐标系
    math::Vec2d p;
    p[0] = f_ * distort_ratio * x + c_[0];
    p[1] = f_ * distort_ratio * y + c_[1];

    return p;
}

// 相机在世界坐标中的位置 -R^T*t
math::Vec3d Camera::pos_in_world()
{
    math::Vec3d pos;
    pos[0] = R_[0] * t_[0] + R_[3] * t_[1] + R_[6] * t_[2];
    pos[1] = R_[1] * t_[0] + R_[4] * t_[1] + R_[7] * t_[2];
    pos[2] = R_[2] * t_[0] + R_[5] * t_[1] + R_[8] * t_[2];
    return -pos;
}

// 相机在世界坐标中的方向
/*
R^T*[0 0 1]^T,也就是旋转矩阵的转置的第三列，也就是旋转矩阵的第三行
*/
math::Vec3d Camera::dir_in_world()
{
    math::Vec3d dir(R_[6], R_[7], R_[8]);
    return dir;
}