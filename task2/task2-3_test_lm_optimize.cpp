//
// Created by caoqi on 2018/9/2.
/*
 * 实现Levenberg-Marquardt算法，该算法又称为 damped least squares 阻尼最小二乘，用来求解非线性最小二乘
 * 问题。LM找到的是局部最小值，该算法介于高斯牛顿法和梯度下降法之间，并通过控制信赖域尺寸的大小，在高斯牛顿法
 * 和梯度下降法之间进行调整。LM 算法比高斯牛顿法速度慢但是更为鲁棒，
 *
 * LM算法的原理是用模型函数f对待估向量p在邻域内做线性估计(泰勒展开），忽略掉二阶以上的导数项，从而转化为线性最小
 * 二乘问题。本质上就是用二次曲面对目标函数进行局部近似。LM算法属于一种信赖域法，即：从初始值开始，先假设一个可以
 * 信赖的最大的位移s, 然后以当前点为中心，以s为半径的区域内，通过寻找目标函数的一个近似二次函数的最优点来求得真正
 * 的位移。在得到位移之后，再计算目标函数值，如果其使得目标函数值得下降满足了一定条件，那么说明这个位移是可靠的
 * 则继续按照此规则迭代计算下去；如果其不能使目标函数的下降满足一定的条件，则应该减少信赖域的范围，重新求解。
 *
 * LM算法的一般流程是：
 *       1） 初始化
 *       2） 计算雅阁比矩阵J，构造正规方程(JTJ + lambdaI) = JTf
 *       3） 求解正规方程（共轭梯度或者预定共轭梯度法）
 *       4） 判断若求解成功
 *               增加信赖域(1/lambda)，使得求解算法接近于高斯牛顿法,加快收敛速度
 *               判断终止条件
 *           判断若求解失败
 *               减少信赖域(1/lambda), 使得求解算法解决域梯度下降法
 *       5)  重复1), 2), 3)，4)
 *
 * （注意，信赖域的大小为正规方程中lambda的倒数)
 */
//
#include <cassert>
#include <fstream>
#include <sstream>
#include <sfm/camera_pose.h>
#include <iomanip>
#include "sfm/ba_conjugate_gradient.h"
#include "sfm/bundle_adjustment.h"
#include "sfm/ba_sparse_matrix.h"
#include "sfm/ba_dense_vector.h"
#include "sfm/ba_linear_solver.h"
#include "sfm/ba_sparse_matrix.h"
#include "sfm/ba_dense_vector.h"
#include "sfm/ba_cholesky.h"

typedef sfm::ba::SparseMatrix<double> SparseMatrixType; //稀疏矩阵
typedef sfm::ba::DenseVector<double> DenseVectorType;   //稠密矩阵

// global variables
std::vector<sfm::ba::Camera> cameras; //存储所有的相机的信息
std::vector<sfm::ba::Point3D> points; //存储所有三维点的坐标
std::vector<sfm::ba::Observation> observations;

#define TRUST_REGION_RADIUS_INIT (1000)
#define TRUST_REGION_RADIUS_DECREMENT (1.0 / 10.0)
#define TRUST_REGION_RADIUS_GAIN (10.0)

// lm 算法最多迭代次数
const int lm_max_iterations = 100;
// mean square error
double initial_mse = 0.0;
double final_mse = 0.0;
int num_lm_iterations = 0;
int num_lm_successful_iterations = 0;
int num_lm_unsuccessful_iterations = 0;

// lm 算法终止条件
double lm_mse_threshold = 1e-16;
double lm_delta_threshold = 1e-8;

// 信赖域大小
double trust_region_radius = 1000;
int cg_max_iterations = 1000;
//相机参数个数
int camera_block_dim = 9;

const int num_cam_params = 9;

/**
 * /decription 加载相关数据，包括相机的初始内外参数，三维点，观察点
 * @param file_name
 * @param cams
 * @param pts3D
 * @param observations
 */
void load_data(const std::string &file_name, std::vector<sfm::ba::Camera> &cams, std::vector<sfm::ba::Point3D> &pts3D, std::vector<sfm::ba::Observation> &observations)
{

    /* 加载数据 */
    std::ifstream in(file_name);
    assert(in.is_open());
    std::string line, word;

    // 加载相机参数
    {
        int n_cams = 0;
        getline(in, line);
        std::stringstream stream(line);
        stream >> word >> n_cams; // word=n_cam；n_cams=2
        cams.resize(n_cams);
        for (int i = 0; i < cams.size(); i++)
        {
            getline(in, line);
            std::stringstream stream(line);
            stream >> cams[i].focal_length;                           //相机1：focal_length[1]=0.972222  相机2:focal_length[2]=0.972222
            stream >> cams[i].distortion[0] >> cams[i].distortion[1]; // k0=0,k1=0
            for (int j = 0; j < 3; j++)
                stream >> cams[i].translation[j]; // t1=t2=t3=0 相机2的T：[0.0796625 0.99498 0.0605768]
            for (int j = 0; j < 9; j++)
                stream >> cams[i].rotation[j]; //旋转矩阵R=
            //[1 0 0] [0.999827 -0.0119578 0.0142419]
            //[0 1 0] [0.0122145 0.999762 -0.0180719]
            //[0 0 1] [-0.0140224 0.0182427 0.999735]
        }
    }

    // 加载三维点
    {
        int n_points = 0;
        getline(in, line);
        std::stringstream stream(line);
        stream >> word >> n_points; // word=n_points;n_points=272
        pts3D.resize(n_points);
        for (int i = 0; i < n_points; i++)
        {
            getline(in, line);
            std::stringstream stream(line);
            stream >> pts3D[i].pos[0] >> pts3D[i].pos[1] >> pts3D[i].pos[2]; //给3D点的X，Y，Z赋值
        }
    }

    //加载观察点
    {
        int n_observations = 0;
        getline(in, line);
        std::stringstream stream(line);
        stream >> word >> n_observations; // word=n_observations;n_observations=544
        observations.resize(n_observations);
        for (int i = 0; i < observations.size(); i++)
        {
            getline(in, line);
            std::stringstream stream(line);
            stream >> observations[i].camera_id //二维点在所在相机的序号
                >> observations[i].point_id     //点的序号
                >> observations[i].pos[0]       //二维点的x
                >> observations[i].pos[1];      //二维点的y
        }
    }
}

/*
 * Computes for a given matrix A the square matrix A^T * A for the
 * case that block columns of A only need to be multiplied with itself.
 * Becase the resulting matrix is symmetric（对称）, only about half the work
 * needs to be performed.
 */
/*计算A^T * A*/
void matrix_block_column_multiply(sfm::ba::SparseMatrix<double> const &A,
                                  std::size_t block_size, sfm::ba::SparseMatrix<double> *B)
{
    sfm::ba::SparseMatrix<double>::Triplets triplets;                      //存放三元组的容器
    triplets.reserve(A.num_cols() * block_size);                           // A是一个1088*18的稀疏矩阵，triplets是一个1088*9的容器                          //为容器triplets申请A.行*block_size
    for (std::size_t block = 0; block < A.num_cols(); block += block_size) // block_size就是每一个块儿的长度
    {
        std::vector<sfm::ba::DenseVector<double>> columns(block_size); //定义一个储存DenseVector的容器
        for (std::size_t col = 0; col < block_size; ++col)
            A.column_nonzeros(block + col, &columns[col]);
        for (std::size_t col = 0; col < block_size; ++col)
        {
            double dot = columns[col].dot(columns[col]);
            triplets.emplace_back(block + col, block + col, dot);
            for (std::size_t row = col + 1; row < block_size; ++row)
            {
                dot = columns[col].dot(columns[row]);
                triplets.emplace_back(block + row, block + col, dot);
                triplets.emplace_back(block + col, block + row, dot);
            }
        }
    }
    B->allocate(A.num_cols(), A.num_cols());
    B->set_from_triplets(triplets);
}

/*
 * Inverts a matrix with 3x3 bocks on its diagonal. All other entries
 * must be zero. Reading blocks is thus very efficient.
 */
//求解3*3矩阵的逆
void invert_block_matrix_3x3_inplace(sfm::ba::SparseMatrix<double> *A)
{
    if (A->num_rows() != A->num_cols())
        throw std::invalid_argument("Block matrix must be square");
    if (A->num_non_zero() != A->num_rows() * 3)
        throw std::invalid_argument("Invalid number of non-zeros");

    for (double *iter = A->begin(); iter != A->end();)
    {
        double *iter_backup = iter;
        math::Matrix<double, 3, 3> rot;
        for (int i = 0; i < 9; ++i)
            rot[i] = *(iter++);

        double det = math::matrix_determinant(rot);
        if (MATH_DOUBLE_EQ(det, 0.0))
            continue;

        rot = math::matrix_inverse(rot, det);
        iter = iter_backup;
        for (int i = 0; i < 9; ++i)
            *(iter++) = rot[i];
    }
}

/*
 * Inverts a symmetric, positive definite matrix with NxN bocks on its
 * diagonal using Cholesky decomposition. All other entries must be zero.
 */
//用Cholesky分解求一个N*N的对称正定矩阵，A=L*L^T,L为下三角矩阵，且所有元素都大于0
void invert_block_matrix_NxN_inplace(sfm::ba::SparseMatrix<double> *A, int blocksize)
{
    if (A->num_rows() != A->num_cols())
        throw std::invalid_argument("Block matrix must be square");
    if (A->num_non_zero() != A->num_rows() * blocksize)
        throw std::invalid_argument("Invalid number of non-zeros");

    int const bs2 = blocksize * blocksize;
    std::vector<double> matrix_block(bs2);
    for (double *iter = A->begin(); iter != A->end();)
    {
        double *iter_backup = iter;
        for (int i = 0; i < bs2; ++i)
            matrix_block[i] = *(iter++);

        sfm::ba::cholesky_invert_inplace(matrix_block.data(), blocksize);

        iter = iter_backup;
        for (int i = 0; i < bs2; ++i)
            if (std::isfinite(matrix_block[i]))
                *(iter++) = matrix_block[i];
            else
                *(iter++) = 0.0;
    }
}

/**
 * /descrition 将角轴法转化成旋转矩阵
 * @param r 角轴向量
 * @param m 旋转矩阵
 */
void rodrigues_to_matrix(double const *r, double *m)
{
    /* Obtain angle from vector length. */
    double a = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]); //根号下：x^2+y^2+z^2,也就是角轴向量的角度斯塔
    /* Precompute sine and cosine terms. */
    double ct = (a == 0.0) ? 0.5f : (1.0f - std::cos(a)) / (2.0 * a);
    double st = (a == 0.0) ? 1.0 : std::sin(a) / a;
    /* R = I + st * K + ct * K^2 (with cross product matrix K). */
    m[0] = 1.0 - (r[1] * r[1] + r[2] * r[2]) * ct;
    m[1] = r[0] * r[1] * ct - r[2] * st;
    m[2] = r[2] * r[0] * ct + r[1] * st;
    m[3] = r[0] * r[1] * ct + r[2] * st;
    m[4] = 1.0f - (r[2] * r[2] + r[0] * r[0]) * ct;
    m[5] = r[1] * r[2] * ct - r[0] * st;
    m[6] = r[2] * r[0] * ct - r[1] * st;
    m[7] = r[1] * r[2] * ct + r[0] * st;
    m[8] = 1.0 - (r[0] * r[0] + r[1] * r[1]) * ct;
}

/**
 * \description 根据求解得到的增量，对相机参数进行更新
 * @param cam
 * @param update
 * @param out
 */
void update_camera(sfm::ba::Camera const &cam,
                   double const *update, sfm::ba::Camera *out)
{
    out->focal_length = cam.focal_length + update[0];
    out->distortion[0] = cam.distortion[0] + update[1];
    out->distortion[1] = cam.distortion[1] + update[2];

    out->translation[0] = cam.translation[0] + update[3];
    out->translation[1] = cam.translation[1] + update[4];
    out->translation[2] = cam.translation[2] + update[5];

    double rot_orig[9];
    std::copy(cam.rotation, cam.rotation + 9, rot_orig); //将旋转矩阵复制到rot_orig
    double rot_update[9];
    rodrigues_to_matrix(update + 6, rot_update);                         //将角轴转化为旋转矩阵
    math::matrix_multiply(rot_update, 3, 3, rot_orig, 3, out->rotation); //将delta_R*R
}

/**
 * \description 根据求解的增量，对三维点坐标进行更新
 * @param pt
 * @param update
 * @param out
 */
void update_point(sfm::ba::Point3D const &pt,
                  double const *update, sfm::ba::Point3D *out)
{
    out->pos[0] = pt.pos[0] + update[0];
    out->pos[1] = pt.pos[1] + update[1];
    out->pos[2] = pt.pos[2] + update[2];
}

/**
 * /descripition 根据求得的delta_x, 更新相机参数和三维点
 * @param delta_x
 * @param cameras
 * @param points
 */
void update_parameters(DenseVectorType const &delta_x, std::vector<sfm::ba::Camera> *cameras, std::vector<sfm::ba::Point3D> *points)
{
    /* Update cameras. */
    std::size_t total_camera_params = 0;
    for (std::size_t i = 0; i < cameras->size(); ++i)
    {
        update_camera(cameras->at(i),
                      delta_x.data() + num_cam_params * i,
                      &cameras->at(i));
        total_camera_params = cameras->size() * num_cam_params;
    }

    /* Update points. */
    for (std::size_t i = 0; i < points->size(); ++i)
    {
        update_point(points->at(i),
                     delta_x.data() + total_camera_params + i * 3,
                     &points->at(i));
    }
}

/**
 * \description 对像素进行径向畸变
 * @param x
 * @param y
 * @param dist
 */
void radial_distort(double *x, double *y, double const *dist)
{
    double const radius2 = *x * *x + *y * *y;                            // r^2=x^2+y^2
    double const factor = 1.0 + radius2 * (dist[0] + dist[1] * radius2); // d(k0,k1,k2)=1+(k0+k1r^2)r^2
    *x *= factor;                                                        // d(k0,k1,k2)*x
    *y *= factor;
}

/**
 * \description 计算重投影误差
 * @param vector_f
 * @param delta_x
 * @param cameras
 * @param points
 * @param observations
 */
void compute_reprojection_errors(DenseVectorType *vector_f, DenseVectorType const *delta_x, std::vector<sfm::ba::Camera> *cameras, std::vector<sfm::ba::Point3D> *points, std::vector<sfm::ba::Observation> *observations)
{
    if (vector_f->size() != observations->size() * 2) //若稠密矩阵的尺寸！=观测的两倍
        vector_f->resize(observations->size() * 2);

#pragma omp parallel for
    for (std::size_t i = 0; i < observations->size(); ++i) // observations->size()=544
    {
        sfm::ba::Observation const &obs = observations->at(i); // observations储存Observation的容器
        // Observation则储存像素坐标系坐标及相机标号与对应3D点的标号
        sfm::ba::Point3D const &p3d = points->at(obs.point_id);  // 3D点坐标
        sfm::ba::Camera const &cam = cameras->at(obs.camera_id); //相机模型

        double const *flen = &cam.focal_length; // 相机焦距
        double const *dist = cam.distortion;    // 径向畸变系数
        double const *rot = cam.rotation;       // 相机旋转矩阵
        double const *trans = cam.translation;  // 相机平移向量
        double const *point = p3d.pos;          // 三维点坐标

        sfm::ba::Point3D new_point;
        sfm::ba::Camera new_camera;

        // 如果delta_x 不为空，则先利用delta_x对相机和结构进行更新，然后再计算重投影误差
        if (delta_x != nullptr)
        {
            std::size_t cam_id = obs.camera_id * num_cam_params; // size_t为无符号长整形
            std::size_t pt_id = obs.point_id * 3;

            update_camera(cam, delta_x->data() + cam_id, &new_camera);
            flen = &new_camera.focal_length;
            dist = new_camera.distortion;
            rot = new_camera.rotation;
            trans = new_camera.translation;
            pt_id += cameras->size() * num_cam_params;

            update_point(p3d, delta_x->data() + pt_id, &new_point);
            point = new_point.pos;
        }

        /* Project point onto image plane. */ //将3D点投影到像素平面
        double rp[] = {0.0, 0.0, 0.0};
        for (int d = 0; d < 3; ++d) //将世界坐标系坐标转换到相机坐标系
        {
            rp[0] += rot[0 + d] * point[d]; // Xc=rp[0]=R1*X+R2*Y+R3*Z
            rp[1] += rot[3 + d] * point[d]; // Yc=rp[1]=R4*X+R5*Y+R6*Z
            rp[2] += rot[6 + d] * point[d]; // Zc=rp[2]=R7*X+R8*Y+R9*Z
        }
        rp[2] = (rp[2] + trans[2]);         //->Zc
        rp[0] = (rp[0] + trans[0]) / rp[2]; //->Xc/Zc;此时rp[0]表示U
        rp[1] = (rp[1] + trans[1]) / rp[2]; //->Yc/Zc；此时的rp[1]表示V

        /* Distort reprojections. */
        radial_distort(rp + 0, rp + 1, dist); //对图像进行畸变矫正，矫正之后的u，v保存在原变量中

        /* Compute reprojection error. */
        vector_f->at(i * 2 + 0) = rp[0] * (*flen) - obs.pos[0];
        vector_f->at(i * 2 + 1) = rp[1] * (*flen) - obs.pos[1]; //计算投影误差：就是拿3D重新投影到像素坐标，然后用投影
        //的坐标减去对应的像素坐标，计算投影误差
    }
}

/**
 * \description 计算均方误差
 * @param vector_f
 * @return
 */
double compute_mse(DenseVectorType const &vector_f)
{
    double mse = 0.0;
    for (std::size_t i = 0; i < vector_f.size(); ++i)
        mse += vector_f[i] * vector_f[i];                  //投影误差u^2+v^2
    return mse / static_cast<double>(vector_f.size() / 2); // static_cast<double>将数据类型转换为double
}

/**
 * /description 计算观察点坐标(x,y)，相遇对相机参数和三维点坐标的雅阁比矩阵
 * @param cam
 * @param point
 * @param cam_x_ptr
 * @param cam_y_ptr
 * @param point_x_ptr
 * @param point_y_ptr
 */
void my_jacobian(sfm::ba::Camera const &cam,
                 sfm::ba::Point3D const &point,
                 double *cam_x_ptr, double *cam_y_ptr,
                 double *point_x_ptr, double *point_y_ptr) // cam_x_ptr,cam_y_ptr储存相机偏导，point_x_ptr，point_y_ptr储存3D点偏导
{
    const double f = cam.focal_length;   //相机焦距
    const double *R = cam.rotation;      //相机的旋转矩阵
    const double *t = cam.translation;   //相机的平移矩阵
    const double *X = point.pos;         //三维点的xyz
    const double k0 = cam.distortion[0]; //畸变参数k0
    const double k1 = cam.distortion[1]; //畸变参数k1

    //世界坐标系转相机坐标系
    const double xc = R[0] * X[0] + R[1] * X[1] + R[2] * X[2] + t[0];
    const double yc = R[3] * X[0] + R[4] * X[1] + R[5] * X[2] + t[1];
    const double zc = R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + t[2];

    //相机坐标系转归一化像平面
    const double x = xc / zc;
    const double y = yc / zc;

    //畸变矫正
    const double r2 = x * x + y * y;
    const double distort = 1.0 + (k0 + k1 * r2) * r2; // d(k0,k1,r^2)

    const double u = f * distort * x;
    const double v = f * distort * y;

    /*关于焦距的偏导数*/
    cam_x_ptr[0] = distort * x; // P_u/P_f
    cam_y_ptr[0] = distort * y; // P_v/P_f

    /*计算关于径向畸变函数k0, k1的偏导数*/
    // 计算中间变量
    const double u_deriv_distort = f * x;    // P_u/P_d=fx
    const double v_deriv_distort = f * y;    // P_v/P_d=fy
    const double distort_deriv_k0 = r2;      // P_d/P_k0=r^2
    const double distort_deriv_k1 = r2 * r2; // P_d/P_k1=r^4

    cam_x_ptr[1] = u_deriv_distort * distort_deriv_k0; // fxr^2
    cam_x_ptr[2] = u_deriv_distort * distort_deriv_k1; // fxr^4

    cam_y_ptr[1] = v_deriv_distort * distort_deriv_k0; // fyr^2
    cam_y_ptr[2] = v_deriv_distort * distort_deriv_k1; // fyr^4

    // 计算中间变量 (x,y)关于(xc, yc, zc)的偏导数
    const double x_deriv_xc = 1 / zc;  // P_x/P_xc=1/zc
    const double x_deriv_yc = 0;       // P_x/P_yc=0
    const double x_deriv_zc = -x / zc; // P_x/P_zc=-x/zc
    const double y_deriv_xc = 0;       // P_y/xc=0
    const double y_deriv_yc = 1 / zc;  // P_y/yc=1/zc
    const double y_deriv_zc = -y / zc; // P_y/zc=-y/zc

    // 计算u, v关于x, y的偏导数
    const double u_deriv_x = f * distort; // P_u/P_x=fd
    const double v_deriv_y = f * distort; // P_v/P_y=fd

    // 计算中间变量distort关于r2的偏导数
    const double distort_deriv_r2 = k0 + 2 * k1 * r2; // P_d/P_r^2=k0+2k1r^2

    // 计算中间变量r2关于xc, yc, zc的偏导数
    const double r2_deriv_xc = 2 * x / zc;   // P_r^2/P_xc=2x/zc
    const double r2_deriv_yc = 2 * y / zc;   // P_r^2/P_yc=2y/zc
    const double r2_deriv_zc = -2 * r2 / zc; // P_r^2/zc=-2r^2/zc

    // 计算中间变量distort关于xc, yc, zc的偏导数
    const double distort_deriv_xc = distort_deriv_r2 * r2_deriv_xc; // P_d/P_xc
    const double distort_deriv_yc = distort_deriv_r2 * r2_deriv_yc; // P_d/P_yc
    const double distort_deriv_zc = distort_deriv_r2 * r2_deriv_zc; // P_d/P_zc

    // 计算(u,v)关于xc, yc, zc的偏导数
    const double u_deriv_xc = u_deriv_distort * distort_deriv_xc + u_deriv_x * x_deriv_xc; // P_u/P_xc
    const double u_deriv_yc = u_deriv_distort * distort_deriv_yc + u_deriv_x * x_deriv_yc; // P_u/P_yc
    const double u_deriv_zc = u_deriv_distort * distort_deriv_zc + u_deriv_x * x_deriv_zc; // P_u/P_zc

    const double v_deriv_xc = v_deriv_distort * distort_deriv_xc + v_deriv_y * y_deriv_xc; // P_v/P_xc
    const double v_deriv_yc = v_deriv_distort * distort_deriv_yc + v_deriv_y * y_deriv_yc; // P_v/P_yc
    const double v_deriv_zc = v_deriv_distort * distort_deriv_zc + v_deriv_y * y_deriv_zc; // P_v/P_zc

    /* 计算关于平移向量的t0, t1, t2的偏导数*/
    const double xc_deriv_t0 = 1; // P_xc/P_t0
    const double yc_deriv_t1 = 1; // P_yc/P_t1
    const double zc_deriv_t2 = 1; // P_zc/P_t2

    cam_x_ptr[3] = u_deriv_xc * xc_deriv_t0; // P_u/P_t0
    cam_x_ptr[4] = u_deriv_yc * yc_deriv_t1; // P_u/P_t1
    cam_x_ptr[5] = u_deriv_zc * zc_deriv_t2; // P_u/P_t2

    cam_y_ptr[3] = v_deriv_xc * xc_deriv_t0; // P_v/P_t0
    cam_y_ptr[4] = v_deriv_yc * yc_deriv_t1; // P_v/P_t1
    cam_y_ptr[5] = v_deriv_zc * zc_deriv_t2; // P_v/P_t2

    /* 计算关于旋转矩阵(表示为角轴向量w0, w1, w2)的偏导数 */
    const double rx = R[0] * X[0] + R[1] * X[1] + R[2] * X[2]; // r1*X
    const double ry = R[3] * X[0] + R[4] * X[1] + R[5] * X[2]; // r2*X
    const double rz = R[6] * X[0] + R[7] * X[1] + R[8] * X[2]; // r3*X
    const double xc_deriv_w0 = 0;                              // P_xc/P_w0
    const double xc_deriv_w1 = rz;                             // P_xc/P_w1
    const double xc_deriv_w2 = -ry;                            // P_xc/P_w2
    const double yc_deriv_w0 = -rz;                            // P_yc/P_w0
    const double yc_deriv_w1 = 0;                              // P_yc/P_w1
    const double yc_deriv_w2 = rx;                             // P_yc/P_w2
    const double zc_deriv_w0 = ry;                             // P_zc/P_w0
    const double zc_deriv_w1 = -rx;                            // P_zc/P_w1
    const double zc_deriv_w2 = 0;                              // P_zc/P_w2

    cam_x_ptr[6] = u_deriv_yc * yc_deriv_w0 + u_deriv_zc * zc_deriv_w0; // P_u/P_w0
    cam_x_ptr[7] = u_deriv_xc * xc_deriv_w1 + u_deriv_zc * zc_deriv_w1; // P_u/P_w1
    cam_x_ptr[8] = u_deriv_xc * xc_deriv_w2 + u_deriv_yc * yc_deriv_w2; // P_u/P_w2

    cam_y_ptr[6] = v_deriv_yc * yc_deriv_w0 + v_deriv_zc * zc_deriv_w0; // P_v/P_w0
    cam_y_ptr[7] = v_deriv_xc * xc_deriv_w1 + v_deriv_zc * zc_deriv_w1; // P_v/P_w1
    cam_y_ptr[8] = v_deriv_xc * xc_deriv_w2 + v_deriv_yc * yc_deriv_w2; // P_v/P_w2

    /* 计算关于三维点坐标X,Y,X的偏导数*/
    const double xc_deriv_X = R[0]; // P_xc/P_X
    const double xc_deriv_Y = R[1]; // P_xc/P_Y
    const double xc_deriv_Z = R[2]; // P_xc/P_Z
    const double yc_deriv_X = R[3]; // P_yc/P_X
    const double yc_deriv_Y = R[4]; // P_yc/P_Y
    const double yc_deriv_Z = R[5]; // P_yc/P_Z
    const double zc_deriv_X = R[6]; // P_zc/P_X
    const double zc_deriv_Y = R[7]; // P_zc/P_Y
    const double zc_deriv_Z = R[8]; // P_zc/P_Z

    point_x_ptr[0] = u_deriv_xc * xc_deriv_X + u_deriv_yc * yc_deriv_X + u_deriv_zc * zc_deriv_X; // P_u/P_X
    point_x_ptr[1] = u_deriv_xc * xc_deriv_Y + u_deriv_yc * yc_deriv_Y + u_deriv_zc * zc_deriv_Y; // P_u/P_Y
    point_x_ptr[2] = u_deriv_xc * xc_deriv_Z + u_deriv_yc * yc_deriv_Z + u_deriv_zc * zc_deriv_Z; // P_u/P_Z

    point_y_ptr[0] = v_deriv_xc * xc_deriv_X + v_deriv_yc * yc_deriv_X + v_deriv_zc * zc_deriv_X; // P_v/P_X
    point_y_ptr[1] = v_deriv_xc * xc_deriv_Y + v_deriv_yc * yc_deriv_Y + v_deriv_zc * zc_deriv_Y; // P_v/P_Y
    point_y_ptr[2] = v_deriv_xc * xc_deriv_Z + v_deriv_yc * yc_deriv_Z + v_deriv_zc * zc_deriv_Z; // P_v/P_Z
}

/**
 * \description 构造雅阁比矩阵，采用稀疏矩阵形式，
 *       关于相机参数的雅阁比矩阵大小为：(2*observations.size()) x (num_cameras*9)
 *       关于三维点坐标的雅阁比矩阵大小为：(2*observation.size()) x (num_points*3)
 * @param jac_cam-- 观察点相对于相机参数的雅阁比矩阵
 * @param jac_points--观察点相对于三维点的雅阁比矩阵
 */
void analytic_jacobian(SparseMatrixType *jac_cam, SparseMatrixType *jac_points)
{
    assert(jac_cam); //判断矩阵是否为空，为空则终止程序
    assert(jac_points);
    // 相机和三维点jacobian矩阵的行数都是n_observations*2
    // 相机jacobian矩阵jac_cam的列数是n_cameras* n_cam_params
    // 三维点jacobian矩阵jac_points的列数是n_points*3
    std::size_t const camera_cols = cameras.size() * num_cam_params; //相机数量2*相机参数个数9
                                                                     //(相机内参：f，k0，k1；外参矩阵有6个自由度)
    std::size_t const point_cols = points.size() * 3;                //三维点雅克比矩阵的列数
    std::size_t const jacobi_rows = observations.size() * 2;         //相机和三维点雅克比矩阵的行数，所以这里并没有写列数

    // 定义稀疏矩阵的基本元素
    SparseMatrixType::Triplets cam_triplets, point_triplets;        //存放相机和三维点三元组的容器(row,col,value)
    cam_triplets.reserve(observations.size() * 2 * num_cam_params); //观测点的数量*2*相机的参数，也就是雅克比矩阵的行数*相机参数
    point_triplets.reserve(observations.size() * 2 * 3);            //观测点的数量*2*3，每个3D点提供雅克比矩阵3*2，则n个3D点则是n*3*2

    double cam_x_ptr[9], cam_y_ptr[9], point_x_ptr[3], point_y_ptr[3];
    // 对于每一个观察到的二维点
    for (std::size_t i = 0; i < observations.size(); ++i) // 544
    {

        // 获取二维点，obs.point_id 三维点的索引，obs.camera_id 相机的索引
        sfm::ba::Observation const &obs = observations[i];
        // 三维点坐标
        sfm::ba::Point3D const &p3d = points[obs.point_id];
        // 相机参数
        sfm::ba::Camera const &cam = cameras[obs.camera_id];

        /*对一个三维点和相机求解偏导数*/
        my_jacobian(cam, p3d,
                    cam_x_ptr, cam_y_ptr, point_x_ptr, point_y_ptr); // cam_x_ptr存储u的偏导，cam_y_ptr存储v相关的偏导

        /*观察点对应雅各比矩阵的行，第i个观察点在雅各比矩阵的位置是2*i, 2*i+1*/
        std::size_t row_x = i * 2 + 0;
        std::size_t row_y = i * 2 + 1;

        /*jac_cam中相机对应的列数为camera_id* n_cam_params*/
        std::size_t cam_col = obs.camera_id * num_cam_params; //相当于将两个相机的偏导合并到稀疏矩阵的相同列

        /*jac_points中三维点对应的列数为point_id* 3*/
        std::size_t point_col = obs.point_id * 3;

        for (int j = 0; j < num_cam_params; ++j) //存储相机的雅克比矩阵的稀疏矩阵
        {
            cam_triplets.push_back(SparseMatrixType::Triplet(row_x, cam_col + j, cam_x_ptr[j]));
            cam_triplets.push_back(SparseMatrixType::Triplet(row_y, cam_col + j, cam_y_ptr[j]));
        }

        for (int j = 0; j < 3; ++j) //存储三维点的雅克比矩阵的稀疏矩阵
        {
            point_triplets.push_back(SparseMatrixType::Triplet(row_x, point_col + j, point_x_ptr[j]));
            point_triplets.push_back(SparseMatrixType::Triplet(row_y, point_col + j, point_y_ptr[j]));
        }
    }

    if (jac_cam != nullptr)
    {
        jac_cam->allocate(jacobi_rows, camera_cols);
        jac_cam->set_from_triplets(cam_triplets);
    }

    if (jac_points != nullptr)
    {
        jac_points->allocate(jacobi_rows, point_cols);
        jac_points->set_from_triplets(point_triplets);
    }
}

sfm::ba::LinearSolver::Status my_solve_schur(
    SparseMatrixType const &jac_cams,
    SparseMatrixType const &jac_points,
    DenseVectorType const &values,
    DenseVectorType *delta_x) //这个函数的主要目的就是找到最佳的下降方向delta_x
{
    /*
     *   雅阁比矩阵：
     *           J = [Jc Jp]
     *   Jc是与相机相关的模块，Jp是与三维点相关的模块。
     *   正规方程
     *          (J^TJ + lambda*I)delta_x = J^T(x - F)
     *   进一步写为
     *   [ Jcc+ lambda*Icc   Jcx            ][delta_c]= [v]
     *   [ Jxc               Jxx+lambda*Ixx ][delta_x]  [w]
     *
     *   B = Jcc, E = Jcx, C = Jxx
     *  其中 Jcc = Jc^T* Jc, Jcx = Jc^T*Jx, Jxc = Jx^TJc, Jxx = Jx^T*Jx
     *      v = Jc^T(F-x), w = Jx^T(F-x), deta_x = [delta_c; delta_x]
     */

    // 误差向量
    DenseVectorType const &F = values; // F就是误差值
    // 关于相机的雅阁比矩阵
    SparseMatrixType const &Jc = jac_cams;
    // 关于三维点的雅阁比矩阵
    SparseMatrixType const &Jp = jac_points;
    SparseMatrixType JcT = Jc.transpose();
    SparseMatrixType JpT = Jp.transpose();

    // 构造正规方程
    SparseMatrixType B, C;
    // B = Jc^T* Jc
    matrix_block_column_multiply(Jc, camera_block_dim, &B);
    // C = Jx^T*Jx
    matrix_block_column_multiply(Jp, 3, &C); //稀疏矩阵自己乘自己
    // E = Jc^T*Jx
    SparseMatrixType E = JcT.multiply(Jp); //两个稀疏矩阵相乘

    /* Assemble two values vectors. */
    DenseVectorType v = JcT.multiply(F); // bc
    DenseVectorType w = JpT.multiply(F); // bx
    v.negate_self();
    w.negate_self();

    /* 以矩阵B和C的对角元素重新构建对角阵*/
    //    SparseMatrixType B_diag = B.diagonal_matrix();
    //    SparseMatrixType C_diag = C.diagonal_matrix();

    /* 添加信赖域 */
    C.mult_diagonal(1.0 + 1.0 / trust_region_radius); // C = Jx^T*Jx
    B.mult_diagonal(1.0 + 1.0 / trust_region_radius); // B = Jc^T* Jc

    /* 求解C矩阵的逆C = inv(Jx^T+Jx + lambda*Ixx)*/
    invert_block_matrix_3x3_inplace(&C); // Jxx^(-1)

    /* 计算S矩阵的Schur补用于高斯消元. */
    SparseMatrixType ET = E.transpose(); // E = Jc^T*Jx

    // S = (Jcc+lambda*Icc) - Jc^T*Jx*inv(Jxx+ lambda*Ixx)*Jx^T*Jc
    SparseMatrixType S = B.subtract(E.multiply(C).multiply(ET)); // B = Jcc+lambda*Icc
    // rhs = v -  Jc^T*Jx*inv(Jxx+ lambda*Ixx)*w
    DenseVectorType rhs = v.subtract(E.multiply(C.multiply(w))); // v=bc,w=bx

    /* Compute pre-conditioner for linear system. */
    // SparseMatrixType precond = S.diagonal_matrix();
    // precond.cwise_invert();
    SparseMatrixType precond = B;                                // Jcc+lambda*Icc
    invert_block_matrix_NxN_inplace(&precond, camera_block_dim); // camera_block_dim=9

    /* 用共轭梯度法求解相机参数. */
    DenseVectorType delta_y(Jc.num_cols());
    typedef sfm::ba::ConjugateGradient<double> CGSolver;
    CGSolver::Options cg_opts;
    cg_opts.max_iterations = cg_max_iterations; //最大迭代次数1000
    cg_opts.tolerance = 1e-20;
    CGSolver solver(cg_opts);
    CGSolver::Status cg_status;
    cg_status = solver.solve(S, rhs, &delta_y, &precond);

    sfm::ba::LinearSolver::Status status;
    status.num_cg_iterations = cg_status.num_iterations;
    switch (cg_status.info)
    {
    case CGSolver::CG_CONVERGENCE:
        status.success = true;
        break;
    case CGSolver::CG_MAX_ITERATIONS:
        status.success = true;
        break;
    case CGSolver::CG_INVALID_INPUT:
        std::cout << "BA: CG failed (invalid input)" << std::endl;
        status.success = false;
        return status;
    default:
        break;
    }

    /* 将相机参数带入到第二个方程中，求解三维点的参数. */
    /*E= inv(Jp^T Jp) (JpT.multiply(F)-Jc^T * Jp * delta_y)*/
    DenseVectorType delta_z =
        C.multiply(w.subtract(ET.multiply(delta_y)));
    std::size_t const jac_cam_cols = Jc.num_cols();
    std::size_t const jac_point_cols = Jp.num_cols();
    std::size_t const jac_cols = jac_cam_cols + jac_point_cols;
    //进行赋值
    if (delta_x->size() != jac_cols)
        delta_x->resize(jac_cols, 0.0);
    for (std::size_t i = 0; i < jac_cam_cols; ++i) //相当于delta_x记录了f,k0,k1,w1,w2,w3,w4,w5,w6,x,y,z的变化量
        delta_x->at(i) = delta_y[i];
    for (std::size_t i = 0; i < jac_point_cols; ++i)
        delta_x->at(jac_cam_cols + i) = delta_z[i];

    return status;
}
/**
 * /description  LM 算法流程
 * @param cameras
 * @param points
 * @param observations
 */
void lm_optimization(std::vector<sfm::ba::Camera> *cameras, std::vector<sfm::ba::Point3D> *points, std::vector<sfm::ba::Observation> *observations)
{

    /*1.0 初始化*/
    // 计算重投影误差向量
    DenseVectorType F, F_new;
    // compute_reprojection_errors函数也对点和相机参数进行了更新，但是只是实时更新了该函数里面的参数，
    //整个cam和point的参数并没有更新，故在后面又进行了一次更新
    compute_reprojection_errors(&F, nullptr, cameras, points, observations); // todo F 是误差向量，F储存误差
    // 计算初始的均方误差
    double current_mse = compute_mse(F); // F是存储所有的误差的向量
    initial_mse = current_mse;
    final_mse = current_mse;

    // 设置共轭梯度法的相关参数
    trust_region_radius = TRUST_REGION_RADIUS_INIT; // 1000

    /* Levenberg-Marquard 算法. */
    for (int lm_iter = 0;; ++lm_iter)
    {

        // 判断终止条件，均方误差小于一定阈值
        if (current_mse < lm_mse_threshold) // 1e-16
        {
            std::cout << "BA: Satisfied MSE threshold." << std::endl;
            break;
        }

        // 1.0 计算雅阁比矩阵
        SparseMatrixType Jc, Jp;
        analytic_jacobian(&Jc, &Jp);

        // 2.0 预置共轭梯梯度法对正规方程进行求解*/
        DenseVectorType delta_x;
        sfm::ba::LinearSolver::Status cg_status = my_solve_schur(Jc, Jp, F, &delta_x);

        // 3.0 根据计算得到的偏移量，重新计算冲投影误差和均方误差，用于判断终止条件和更新条件.
        double new_mse, delta_mse, delta_mse_ratio = 1.0;

        // 正规方程求解成功的情况下
        if (cg_status.success)
        {
            /*重新计算相机和三维点，计算重投影误差，注意原始的相机参数没有被更新*/
            compute_reprojection_errors(&F_new, &delta_x, cameras, points, observations);
            /* 计算新的残差值 */
            new_mse = compute_mse(F_new);
            /* 均方误差的绝对变化值和相对变化率*/
            delta_mse = current_mse - new_mse; //用于判断正则方程是否成功计算
            delta_mse_ratio = 1.0 - new_mse / current_mse;
        }
        // 正规方程求解失败的情况下
        else
        {
            new_mse = current_mse;
            delta_mse = 0.0;
        }

        // new_mse < current_mse表示残差值减少
        bool successful_iteration = delta_mse > 0.0;

        /*
         * 如果正规方程求解成功，则更新相机参数和三维点坐标，并且增大信赖域的尺寸，使得求解方式
         * 趋近于高斯牛顿法
         */
        if (successful_iteration) //正规化方程求解成功
        {
            std::cout << "BA: #" << std::setw(2) << std::left << lm_iter
                      << " success" << std::right
                      << ", MSE " << std::setw(11) << current_mse
                      << " -> " << std::setw(11) << new_mse
                      << ", CG " << std::setw(3) << cg_status.num_cg_iterations
                      << ", TRR " << trust_region_radius
                      << ", MSE Ratio: " << delta_mse_ratio
                      << std::endl;

            num_lm_iterations += 1;
            num_lm_successful_iterations += 1;

            /* 对相机参数和三点坐标进行更新 */
            update_parameters(delta_x, cameras, points);

            std::swap(F, F_new);
            current_mse = new_mse;

            if (delta_mse_ratio < lm_delta_threshold)
            {
                std::cout << "BA: Satisfied delta mse ratio threshold of "
                          << lm_delta_threshold << std::endl;
                break;
            }

            // 增大信赖域大小
            trust_region_radius *= TRUST_REGION_RADIUS_GAIN;
        }
        else
        {
            std::cout << "BA: #" << std::setw(2) << std::left << lm_iter // std::setw(2)为下一个输出值指定宽度为2
                      << " failure" << std::right                        // std::left使后续输出左对齐
                      << ", MSE " << std::setw(11) << current_mse
                      << ",    " << std::setw(11) << " "
                      << " CG " << std::setw(3) << cg_status.num_cg_iterations
                      << ", TRR " << trust_region_radius
                      << std::endl;

            num_lm_iterations += 1;
            num_lm_unsuccessful_iterations += 1;
            // 求解失败的减小信赖域尺寸
            trust_region_radius *= TRUST_REGION_RADIUS_DECREMENT; // 1/10
        }

        /* 判断是否超过最大的迭代次数. */
        if (lm_iter + 1 >= lm_max_iterations)
        {
            std::cout << "BA: Reached maximum LM iterations of "
                      << lm_max_iterations << std::endl;
            break;
        }
    }

    final_mse = current_mse;
}

int main(int argc, char *argv[])
{

    /* 加载数据 */
    load_data("/home/lei/git_for_homework/task2/test_ba.txt", cameras, points, observations);

    lm_optimization(&cameras, &points, &observations);

    // ba优化
    //    sfm::ba::BundleAdjustment::Options ba_opts;
    //    ba_opts.verbose_output = true;
    //    ba_opts.lm_mse_threshold = 1e-16;
    //    ba_opts.lm_delta_threshold = 1e-8;
    //    sfm::ba::BundleAdjustment ba(ba_opts);
    //    ba.set_cameras(&cameras);
    //    ba.set_points(&points);
    //    ba.set_observations(&observations);
    //    ba.optimize();
    //    ba.print_status();

    // 将优化后的结果重新赋值
    std::vector<sfm::CameraPose> new_cam_poses(2);
    std::vector<math::Vec2f> radial_distortion(2);
    std::vector<math::Vec3f> new_pts_3d(points.size());
    for (int i = 0; i < cameras.size(); i++)
    {
        std::copy(cameras[i].translation, cameras[i].translation + 3, new_cam_poses[i].t.begin());
        std::copy(cameras[i].rotation, cameras[i].rotation + 9, new_cam_poses[i].R.begin());
        radial_distortion[i] = math::Vec2f(cameras[i].distortion[0], cameras[i].distortion[1]);
        new_cam_poses[i].set_k_matrix(cameras[i].focal_length, 0.0, 0.0);
    }
    for (int i = 0; i < new_pts_3d.size(); i++)
    {
        std::copy(points[i].pos, points[i].pos + 3, new_pts_3d[i].begin());
    }

    // 输出优化信息
    std::cout << "Params after BA: " << std::endl;
    std::cout << "  f: " << new_cam_poses[0].get_focal_length() << std::endl;
    std::cout << "  distortion: " << radial_distortion[0][0] << ", " << radial_distortion[0][1] << std::endl;
    std::cout << "  R: " << new_cam_poses[0].R << std::endl;
    std::cout << "  t: " << new_cam_poses[0].t << std::endl;

    // 输出优化信息
    std::cout << "Params after BA: " << std::endl;
    std::cout << "  f: " << new_cam_poses[1].get_focal_length() << std::endl;
    std::cout << "  distortion: " << radial_distortion[1][0] << ", " << radial_distortion[1][1] << std::endl;
    std::cout << "  R: " << new_cam_poses[1].R << std::endl;
    std::cout << "  t: " << new_cam_poses[1].t << std::endl;

    std::cout << "points 3d: " << std::endl;
    for (int i = 0; i < points.size(); i++)
    {
        std::cout << points[i].pos[0] << ", " << points[i].pos[1] << ", " << points[i].pos[2] << std::endl;
    }

    //    Params after BA:
    //    f: 0.919446
    //    distortion: -0.108421, 0.103782
    //    R: 0.999999 -0.00068734 -0.00135363
    //    0.000675175 0.999952 -0.0104268
    //    0.0013597 0.0104261 0.999952
    //    t: 0.00276221 0.0588868 -0.128463

    //    Params after BA:
    //    f: 0.920023
    //    distortion: -0.106701, 0.104344
    //    R: 0.999796 -0.0127484 0.0156791
    //    0.0128673 0.999897 -0.00735337
    //              -0.0155827 0.00755345 0.999857
    //    t: 0.0814124 0.93742 -0.0895658

    //    points 3d:
    //    1.36957, -1.17132, 7.04854
    //    0.0225931, 0.978747, 7.48085

    return 0;
}