#ifndef SFM_BA_TYPES_HEADER
#define SFM_BA_TYPES_HEADER

#include <algorithm>

#include "sfm/defines.h"

SFM_NAMESPACE_BEGIN
SFM_BA_NAMESPACE_BEGIN

/** Camera representation for bundle adjustment. BA优化的相机类型*/
struct Camera
{
    Camera (void);

    double focal_length = 0.0;//焦距
    double distortion[2];//径向畸变系数
    double translation[3];//T
    double rotation[9];//R
    bool is_constant = false;
};

/** 3D point representation for bundle adjustment. BA优化的3D点坐标*/
struct Point3D
{
    double pos[3];//3D点坐标
    bool is_constant = false;
};

/** Observation of a 3D point for a camera. BA优化的3D-2D信息*/
struct Observation
{
    double pos[2];//2D点坐标
    int camera_id;//相机编号
    int point_id;//3D点编号
};

/* ------------------------ Implementation ------------------------ */

inline
Camera::Camera (void)
{
    std::fill(this->distortion, this->distortion + 2, 0.0);
    std::fill(this->translation, this->translation + 3, 0.0);
    std::fill(this->rotation, this->rotation + 9, 0.0);
}

SFM_BA_NAMESPACE_END
SFM_NAMESPACE_END

#endif /* SFM_BA_TYPES_HEADER */

