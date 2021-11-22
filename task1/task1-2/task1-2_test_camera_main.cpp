#include"task1-2_test_camera.h"
int main(int argc, char* argv[]){


    Camera cam;

    //焦距
    cam.f_ = 0.920227;

    // 径向畸变系数
    cam.dist_[0] = -0.106599; cam.dist_[1] = 0.104385;

    // 平移向量
    cam.t_[0] = 0.0814358; cam.t_[1] =  0.937498;   cam.t_[2] = -0.0887441;

    // 旋转矩阵
    cam.R_[0] = 0.999796 ; cam.R_[1] = -0.0127375;  cam.R_[2] =  0.0156807;
    cam.R_[3] = 0.0128557; cam.R_[4] =  0.999894 ;  cam.R_[5] = -0.0073718;
    cam.R_[6] = -0.0155846; cam.R_[7] = 0.00757181; cam.R_[8] = 0.999854;

    // 三维点坐标
    math::Vec3d p3d ={1.36939, -1.17123, 7.04869};

    /*计算相机的投影点*/
    math::Vec2d p2d = cam.projection(p3d);
    std::cout<<"projection coord:\n "<<p2d<<std::endl;
    std::cout<<"result should be:\n 0.208188 -0.035398\n\n";

    /*计算相机在世界坐标系中的位置*/
    math::Vec3d pos = cam.pos_in_world();
    std::cout<<"cam position in world is:\n "<< pos<<std::endl;
    std::cout<<"result should be: \n -0.0948544 -0.935689 0.0943652\n\n";

    /*计算相机在世界坐标系中的方向*/
    math::Vec3d dir = cam.dir_in_world();
    std::cout<<"cam direction in world is:\n "<<dir<<std::endl;
    std::cout<<"result should be: \n -0.0155846 0.00757181 0.999854\n";
}