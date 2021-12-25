/*
 * Copyright (C) 2015, Simon Fuhrmann, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <thread>
#include <iostream>
#include <iomanip>

#include "math/matrix_tools.h"
#include "util/timer.h"
#include "sfm/ba_sparse_matrix.h"
#include "sfm/ba_dense_vector.h"
#include "sfm/bundle_adjustment.h"

#define LOG_E this->log.error()
#define LOG_W this->log.warning()
#define LOG_I this->log.info()
#define LOG_V this->log.verbose()
#define LOG_D this->log.debug()

#define TRUST_REGION_RADIUS_INIT (1000)
#define TRUST_REGION_RADIUS_DECREMENT (1.0 / 2.0)

SFM_NAMESPACE_BEGIN
SFM_BA_NAMESPACE_BEGIN

BundleAdjustment::Status
BundleAdjustment::optimize(void)
{
    util::WallTimer timer; //计时器
    this->sanity_checks(); //参数检查
    this->status = Status();
    this->lm_optimize(); // lm优化
    this->status.runtime_ms = timer.get_elapsed();
    return this->status;
}

void BundleAdjustment::sanity_checks(void)
{
    /* Check for null arguments. 检查空参数*/
    if (this->cameras == nullptr) //检查相机参数是否为空
        throw std::invalid_argument("No cameras given");
    if (this->points == nullptr) //检查track中的3D点信息是否为空
        throw std::invalid_argument("No tracks given");
    if (this->observations == nullptr) //检查2D点信息是否为空
        throw std::invalid_argument("No observations given");

    /* Check for valid focal lengths. 检查相机焦距*/
    for (std::size_t i = 0; i < this->cameras->size(); ++i) //遍历所有相机
        if (this->cameras->at(i).focal_length <= 0.0)       //如果焦距小于0
            throw std::invalid_argument("Camera with invalid focal length");

    /* Check for valid IDs in the observations. 检查2D点的ID*/
    for (std::size_t i = 0; i < this->observations->size(); ++i) //遍历所有2D点
    {
        Observation const &obs = this->observations->at(i);
        if (obs.camera_id < 0 || obs.camera_id >= static_cast<int>(this->cameras->size())) //如果相机编号小于0或者大于所有相机的数量
            throw std::invalid_argument("Observation with invalid camera ID");
        if (obs.point_id < 0 || obs.point_id >= static_cast<int>(this->points->size())) //如果2D点编号小于0或者大于所有2D点的数量
            throw std::invalid_argument("Observation with invalid track ID");
    }
}

void BundleAdjustment::lm_optimize(void)
{
    /* Setup linear solver. 设置线性求解器*/
    LinearSolver::Options pcg_opts;                          //设置参数
    pcg_opts = this->opts.linear_opts;                       //共轭梯度法求解delta_x,delta_y的参数设置
    pcg_opts.trust_region_radius = TRUST_REGION_RADIUS_INIT; //置信半径，1000

    /* Compute reprojection error for the first time. */
    DenseVectorType F, F_new;                  //定义两个稠密矩阵
    this->compute_reprojection_errors(&F);     // 计算重投影误差并储存在F中  todo F 是误差向量
    double current_mse = this->compute_mse(F); // 计算残差值
    this->status.initial_mse = current_mse;
    this->status.final_mse = current_mse;

    /* Levenberg-Marquard main loop. Levenberg-Marquard主循环*/
    for (int lm_iter = 0;; ++lm_iter)
    {
        if (lm_iter + 1 > this->opts.lm_min_iterations && (current_mse < this->opts.lm_mse_threshold)) //循环终止条件：达到循环次数|残差满足条件
        {
            LOG_V << "BA: Satisfied MSE threshold." << std::endl;
            break;
        }

        /* Compute Jacobian. */ //计算雅克比矩阵
        SparseMatrixType Jc, Jp;
        switch (this->opts.bundle_mode)
        {
        /*同时优化相机和三维点*/
        case BA_CAMERAS_AND_POINTS:
            this->analytic_jacobian(&Jc, &Jp);
            break;
        /*固定三维点，只优化相机参数*/
        case BA_CAMERAS:
            this->analytic_jacobian(&Jc, nullptr);
            break;
        /*固定相机优化三维点的坐标*/
        case BA_POINTS:
            this->analytic_jacobian(nullptr, &Jp);
            break;
        default:
            throw std::runtime_error("Invalid bundle mode");
        }

        /* Perform linear step. */
        // 预置共轭梯梯度法进行求解*/
        DenseVectorType delta_x;
        LinearSolver pcg(pcg_opts);
        LinearSolver::Status cg_status = pcg.solve(Jc, Jp, F, &delta_x);

        /* Update reprojection errors and MSE after linear step. */
        double new_mse, delta_mse, delta_mse_ratio = 1.0;
        if (cg_status.success)
        {
            /*重新计算相机和三维点，计算重投影误差，注意原始的相机参数没有被更新*/
            this->compute_reprojection_errors(&F_new, &delta_x);
            /* 计算新的残差值 */
            new_mse = this->compute_mse(F_new);
            /* 残差值的变化*/
            delta_mse = current_mse - new_mse;
            delta_mse_ratio = 1.0 - new_mse / current_mse;
            this->status.num_cg_iterations += cg_status.num_cg_iterations;
        }
        else
        {
            new_mse = current_mse;
            delta_mse = 0.0;
        }

        // new_mse < current_mse表示残差值减少
        bool successful_iteration = delta_mse > 0.0;

        /*
         * Apply delta to parameters after successful step.
         * Adjust the trust region to increase/decrease regulariztion.
         */
        if (successful_iteration)
        {
            LOG_V << "BA: #" << std::setw(2) << std::left << lm_iter
                  << " success" << std::right
                  << ", MSE " << std::setw(11) << current_mse
                  << " -> " << std::setw(11) << new_mse
                  << ", CG " << std::setw(3) << cg_status.num_cg_iterations
                  << ", TRR " << pcg_opts.trust_region_radius
                  << ", MSE Ratio: " << delta_mse_ratio
                  << std::endl;

            this->status.num_lm_iterations += 1;
            this->status.num_lm_successful_iterations += 1;

            /* 对相机参数进行更新 */
            this->update_parameters(delta_x);

            std::swap(F, F_new);
            current_mse = new_mse;

            // todo trust region 是用来做什么的？
            /* Compute trust region update. FIXME delta_norm or mse? */
            double const gain_ratio = delta_mse * (F.size() / 2) / cg_status.predicted_error_decrease;
            double const trust_region_update = 1.0 / std::max(1.0 / 3.0,
                                                              (1.0 - MATH_POW3(2.0 * gain_ratio - 1.0)));
            pcg_opts.trust_region_radius *= trust_region_update;
        }
        else
        {
            LOG_V << "BA: #" << std::setw(2) << std::left << lm_iter
                  << " failure" << std::right
                  << ", MSE " << std::setw(11) << current_mse
                  << ",    " << std::setw(11) << " "
                  << " CG " << std::setw(3) << cg_status.num_cg_iterations
                  << ", TRR " << pcg_opts.trust_region_radius
                  << std::endl;

            this->status.num_lm_iterations += 1;
            this->status.num_lm_unsuccessful_iterations += 1;
            pcg_opts.trust_region_radius *= TRUST_REGION_RADIUS_DECREMENT;
        }

        /* Check termination due to LM iterations. */
        if (lm_iter + 1 < this->opts.lm_min_iterations)
            continue;
        if (lm_iter + 1 >= this->opts.lm_max_iterations)
        {
            LOG_V << "BA: Reached maximum LM iterations of "
                  << this->opts.lm_max_iterations << std::endl;
            break;
        }

        /* Check threshold on the norm of delta_x. */
        if (successful_iteration)
        {
            if (delta_mse_ratio < this->opts.lm_delta_threshold)
            {
                LOG_V << "BA: Satisfied delta mse ratio threshold of "
                      << this->opts.lm_delta_threshold << std::endl;
                break;
            }
        }
    }

    this->status.final_mse = current_mse;
}

void BundleAdjustment::compute_reprojection_errors(DenseVectorType *vector_f,
                                                   DenseVectorType const *delta_x) //计算重投影误差
{
    if (vector_f->size() != this->observations->size() * 2) //若稠密矩阵的尺寸！=观测的两倍
        vector_f->resize(this->observations->size() * 2);   // observations存储2D-3D，以及相机标号信息的列表

#pragma omp parallel for
    for (std::size_t i = 0; i < this->observations->size(); ++i) //遍历所有观测点
    {
        Observation const &obs = this->observations->at(i);   //获取其中一观测点
        Point3D const &p3d = this->points->at(obs.point_id);  //获取3D点
        Camera const &cam = this->cameras->at(obs.camera_id); //获取相机

        double const *flen = &cam.focal_length; // 相机焦距
        double const *dist = cam.distortion;    // 径向畸变系数
        double const *rot = cam.rotation;       // 相机旋转矩阵
        double const *trans = cam.translation;  // 相机平移向量
        double const *point = p3d.pos;          // 三维点坐标

        Point3D new_point; //定义一个BA优化的3D点坐标
        Camera new_camera; //定义一个BA优化的相机类型

        // 如果delta_x 不为空，则先利用delta_x对相机和结构进行更新，然后再计算重投影误差
        if (delta_x != nullptr)
        {
            std::size_t cam_id = obs.camera_id * this->num_cam_params; //相机编号*相机变量的个数
            std::size_t pt_id = obs.point_id * 3;                      // 3D点编号*3

            if (this->opts.bundle_mode & BA_CAMERAS) //如果只是优化相机，则只更新相机的参数
            {
                this->update_camera(cam, delta_x->data() + cam_id, &new_camera); //根据优化的不同情况，更新相机参数，其中包括：焦距，径向畸变系数，R，T共9个参数
                flen = &new_camera.focal_length;                                 //获得更新后的相机焦距
                dist = new_camera.distortion;                                    //获得更新之后的径向畸变系数
                rot = new_camera.rotation;                                       //获得更新之后的旋转矩阵R
                trans = new_camera.translation;                                  //获得更新之后的平移向量
                pt_id += this->cameras->size() * this->num_cam_params;           //为了便于下一步获得点的增量
            }

            if (this->opts.bundle_mode & BA_POINTS) //如果只优化3D点，则只更新3D点的参数
            {
                this->update_point(p3d, delta_x->data() + pt_id, &new_point);
                point = new_point.pos; //获得点的最新坐标
            }
        }

        /* Project point onto image plane. 将3D点投影到图像平面*/
        double rp[] = {0.0, 0.0, 0.0};
        for (int d = 0; d < 3; ++d) // R*P
        {
            rp[0] += rot[0 + d] * point[d];
            rp[1] += rot[3 + d] * point[d];
            rp[2] += rot[6 + d] * point[d];
        }
        rp[2] = (rp[2] + trans[2]);         // R*P+T
        rp[0] = (rp[0] + trans[0]) / rp[2]; //转换到归一化像平面
        rp[1] = (rp[1] + trans[1]) / rp[2];

        /* Distort reprojections. */
        this->radial_distort(rp + 0, rp + 1, dist); //去径向畸变

        /* Compute reprojection error. 计算重投影误差*/
        vector_f->at(i * 2 + 0) = rp[0] * (*flen) - obs.pos[0];
        vector_f->at(i * 2 + 1) = rp[1] * (*flen) - obs.pos[1];
    }
}

double
BundleAdjustment::compute_mse(DenseVectorType const &vector_f) //计算残差值
{
    double mse = 0.0;
    for (std::size_t i = 0; i < vector_f.size(); ++i)
        mse += vector_f[i] * vector_f[i];
    return mse / static_cast<double>(vector_f.size() / 2);
}

void BundleAdjustment::radial_distort(double *x, double *y, double const *dist) //去径向畸变
{
    double const radius2 = *x * *x + *y * *y; // r^2=x^2+y^2
    double const factor = 1.0 + radius2 * (dist[0] + dist[1] * radius2);
    *x *= factor; // u=x(1+k1*r^2+k2*r^4)
    *y *= factor;
}

void BundleAdjustment::rodrigues_to_matrix(double const *r, double *m) //将角轴法转化成旋转矩阵
{
    /* Obtain angle from vector length. 从向量长度获取角度*/
    double a = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]); // w0^2+w1^2+w2^2，角轴的三个分量的平方和
    /* Precompute sine and cosine terms. 预先计算正弦和余弦项*/
    double ct = (a == 0.0) ? 0.5f : (1.0f - std::cos(a)) / (a * a); //(1-cos||w||)/2||w||
    double st = (a == 0.0) ? 1.0 : std::sin(a) / a;                 // sin||w||/||w||
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

void BundleAdjustment::analytic_jacobian(SparseMatrixType *jac_cam,
                                         SparseMatrixType *jac_points) //计算雅克比矩阵
{
    // 相机和三维点jacobian矩阵的行数都是n_observations*2，只要有一个观测关系就有两列雅克比矩阵
    // 相机jacobian矩阵jac_cam的列数是n_cameras* n_cam_params-相机个数*相机参数
    // 三维点jacobian矩阵jac_points的列数是n_points*3
    std::size_t const camera_cols = this->cameras->size() * this->num_cam_params; //相机数量*相机参数
    std::size_t const point_cols = this->points->size() * 3;                      // 3D点数量*3
    std::size_t const jacobi_rows = this->observations->size() * 2;               //观测关系*2

    // 定义稀疏矩阵的基本元素
    SparseMatrixType::Triplets cam_triplets, point_triplets; //存放相机和三维点三元组的容器(row,col,value)
    // 对相机进行优化
    if (jac_cam != nullptr)
        cam_triplets.reserve(this->observations->size() * 2 * this->num_cam_params); //重新定义存放相机三元组容器的大小
    // 对三维点进行优化
    if (jac_points != nullptr)
        point_triplets.reserve(this->observations->size() * 3 * 2); //重新定义存放3D点三元组容器的大小

        /*jac_cam的尺寸大小是 n_observations  x 2*n_cam_params--每一个观察点对应一个相机
         *jac_points的尺寸是 n_observations   x  2*3 --每一个观察点对应一个三维点
         */

#pragma omp parallel
    {
        double cam_x_ptr[9], cam_y_ptr[9], point_x_ptr[3], point_y_ptr[3];
#pragma omp for

        // 对于每一个观察到的二维点
        for (std::size_t i = 0; i < this->observations->size(); ++i)
        {

            // 获取二维点，obs.point_id 三维点的索引，obs.camera_id 相机的索引
            Observation const &obs = this->observations->at(i);
            // 三维点坐标
            Point3D const &p3d = this->points->at(obs.point_id);
            // 相机参数
            Camera const &cam = this->cameras->at(obs.camera_id);

            /*对一个三维点和相机求解偏导数*/
            this->analytic_jacobian_entries(cam, p3d,
                                            cam_x_ptr, cam_y_ptr, point_x_ptr, point_y_ptr);

            /*如果三维点是固定的，即只优化相机参数，则将三维点的偏导数设置为0*/
            if (p3d.is_constant)
            {
                std::fill(point_x_ptr, point_x_ptr + 3, 0.0);
                std::fill(point_y_ptr, point_y_ptr + 3, 0.0);
            }

            /*观察点对应雅各比矩阵的行，第i个观察点在雅各比矩阵的位置是2*i, 2*i+1*/
            std::size_t row_x = i * 2, row_y = row_x + 1;

            /*jac_cam中相机对应的列数为camera_id* n_cam_params*/
            std::size_t cam_col = obs.camera_id * this->num_cam_params;

            /*jac_points中三维点对应的列数为point_id* 3*/
            std::size_t point_col = obs.point_id * 3;

            for (int j = 0; jac_cam != nullptr && j < this->num_cam_params; ++j)
            {
                cam_triplets.push_back(SparseMatrixType::Triplet(row_x, cam_col + j, cam_x_ptr[j]));
                cam_triplets.push_back(SparseMatrixType::Triplet(row_y, cam_col + j, cam_y_ptr[j]));
            }

            for (int j = 0; jac_points != nullptr && j < 3; ++j)
            {
                point_triplets.push_back(SparseMatrixType::Triplet(row_x, point_col + j, point_x_ptr[j]));
                point_triplets.push_back(SparseMatrixType::Triplet(row_y, point_col + j, point_y_ptr[j]));
            }
        }

#pragma omp sections
        {
#pragma omp section
            if (jac_cam != nullptr)
            {
                jac_cam->allocate(jacobi_rows, camera_cols);
                jac_cam->set_from_triplets(cam_triplets);
            }

#pragma omp section
            if (jac_points != nullptr)
            {
                jac_points->allocate(jacobi_rows, point_cols);
                jac_points->set_from_triplets(point_triplets);
            }
        }
    }
}

void BundleAdjustment::analytic_jacobian_entries(
    Camera const &cam,
    Point3D const &point,
    double *cam_x_ptr, double *cam_y_ptr,
    double *point_x_ptr, double *point_y_ptr) //求偏导
{
    /*
     * This function computes the Jacobian entries for the given camera and
     * 3D point pair that leads to one observation
     * 这个函数计算给定摄像机和3D点对的雅可比矩阵条目，从而得到一个观测结果.
     *
     * The camera block 'cam_x_ptr' and 'cam_y_ptr' is:
     * - ID 0: Derivative of focal length f
     * - ID 0：焦距的偏导数
     * - ID 1-2: Derivative of distortion parameters k0, k1
     * - ID 1-2：k0,k1的偏导数
     * - ID 3-5: Derivative of translation t0, t1, t2
     * - ID 3-5：平移向量t0、t1、t2的偏倒数
     * - ID 6-8: Derivative of rotation r0, r1, r2
     * - ID 6-8：角轴r0，r1，r2的偏导数
     *
     * The 3D point block 'point_x_ptr' and 'point_y_ptr' is:
     * - ID 0-2: Derivative in x, y, and z direction.
     * - ID 0-2：3D点x,y,z的偏导数
     *
     * The function that leads to the observation is given as follows:
     *
     *   Px = f * D(ix,iy) * ix  (image observation x coordinate)
     *   Py = f * D(ix,iy) * iy  (image observation y coordinate)
     *
     * with the following definitions:
     *
     *   x = R0 * X + t0  (homogeneous projection)
     *   y = R1 * X + t1  (homogeneous projection)
     *   z = R2 * X + t2  (homogeneous projection)
     *   ix = x / z  (central projection)
     *   iy = y / z  (central projection)
     *   D(ix, iy) = 1 + k0 (ix^2 + iy^2) + k1 (ix^2 + iy^2)^2  (distortion)
     *
     * The derivatives for intrinsics (f, k0, k1) are easy to compute exactly.
     * The derivatives for extrinsics (r, t) and point coordinates Xx, Xy, Xz,
     * are a bit of a pain to compute.
     */

    /* Aliases. */
    double const *r = cam.rotation;    //获取旋转矩阵
    double const *t = cam.translation; //获取平移向量
    double const *k = cam.distortion;  //获取畸变
    double const *p3d = point.pos;     //获取3D点位姿

    /* Temporary values. 临时值，将3D点投影到像素平面的过程*/
    double const rx = r[0] * p3d[0] + r[1] * p3d[1] + r[2] * p3d[2];
    double const ry = r[3] * p3d[0] + r[4] * p3d[1] + r[5] * p3d[2];
    double const rz = r[6] * p3d[0] + r[7] * p3d[1] + r[8] * p3d[2];
    double const px = rx + t[0];//相机坐标系xc
    double const py = ry + t[1];//相机坐标系yc
    double const pz = rz + t[2];//相机坐标系zc
    double const ix = px / pz;//归一化坐标x
    double const iy = py / pz;//归一化坐标y
    double const fz = cam.focal_length / pz;//为啥焦距也除pz？？？
    double const radius2 = ix * ix + iy * iy;
    double const rd_factor = 1.0 + (k[0] + k[1] * radius2) * radius2;

    /* Compute exact camera and point entries if intrinsics are fixed 如果内参矩阵是固定的，计算精确的相机和点入口*/
    if (this->opts.fixed_intrinsics)
    {
        cam_x_ptr[0] = fz * rd_factor;//P_u/P_f=(f/zc)*d
        cam_x_ptr[1] = 0.0;//P_u/P_k0=0
        cam_x_ptr[2] = -fz * rd_factor * ix;//P_u/P_k1=-(f/zc)*d*x
        cam_x_ptr[3] = -fz * rd_factor * ry * ix;//P_u/P_t0=-(f/zc)*d*
        cam_x_ptr[4] = fz * rd_factor * (rz + rx * ix);
        cam_x_ptr[5] = -fz * rd_factor * ry;

        cam_y_ptr[0] = 0.0;
        cam_y_ptr[1] = fz * rd_factor;
        cam_y_ptr[2] = -fz * rd_factor * iy;
        cam_y_ptr[3] = -fz * rd_factor * (rz + ry * iy);
        cam_y_ptr[4] = fz * rd_factor * rx * iy;
        cam_y_ptr[5] = fz * rd_factor * rx;

        /*
         * Compute point derivatives in x, y, and z.
         */
        point_x_ptr[0] = fz * rd_factor * (r[0] - r[6] * ix);
        point_x_ptr[1] = fz * rd_factor * (r[1] - r[7] * ix);
        point_x_ptr[2] = fz * rd_factor * (r[2] - r[8] * ix);

        point_y_ptr[0] = fz * rd_factor * (r[3] - r[6] * iy);
        point_y_ptr[1] = fz * rd_factor * (r[4] - r[7] * iy);
        point_y_ptr[2] = fz * rd_factor * (r[5] - r[8] * iy);
        return;
    }

    /* The intrinsics are easy to compute exactly. */
    cam_x_ptr[0] = ix * rd_factor;
    cam_x_ptr[1] = cam.focal_length * ix * radius2;
    cam_x_ptr[2] = cam.focal_length * ix * radius2 * radius2;

    cam_y_ptr[0] = iy * rd_factor;
    cam_y_ptr[1] = cam.focal_length * iy * radius2;
    cam_y_ptr[2] = cam.focal_length * iy * radius2 * radius2;

#define JACOBIAN_APPROX_CONST_RD 0
#define JACOBIAN_APPROX_PBA 0
#if JACOBIAN_APPROX_CONST_RD
    /*
     * Compute approximations of the Jacobian entries for the extrinsics
     * by assuming the distortion coefficent D(ix, iy) is constant.
     */
    cam_x_ptr[3] = fz * rd_factor;
    cam_x_ptr[4] = 0.0;
    cam_x_ptr[5] = -fz * rd_factor * ix;
    cam_x_ptr[6] = -fz * rd_factor * ry * ix;
    cam_x_ptr[7] = fz * rd_factor * (rz + rx * ix);
    cam_x_ptr[8] = -fz * rd_factor * ry;

    cam_y_ptr[3] = 0.0;
    cam_y_ptr[4] = fz * rd_factor;
    cam_y_ptr[5] = -fz * rd_factor * iy;
    cam_y_ptr[6] = -fz * rd_factor * (rz + ry * iy);
    cam_y_ptr[7] = fz * rd_factor * rx * iy;
    cam_y_ptr[8] = fz * rd_factor * rx;

    /*
     * Compute point derivatives in x, y, and z.
     */
    point_x_ptr[0] = fz * rd_factor * (r[0] - r[6] * ix);
    point_x_ptr[1] = fz * rd_factor * (r[1] - r[7] * ix);
    point_x_ptr[2] = fz * rd_factor * (r[2] - r[8] * ix);

    point_y_ptr[0] = fz * rd_factor * (r[3] - r[6] * iy);
    point_y_ptr[1] = fz * rd_factor * (r[4] - r[7] * iy);
    point_y_ptr[2] = fz * rd_factor * (r[5] - r[8] * iy);
#elif JACOBIAN_APPROX_PBA
    /* Computation of Jacobian approximation with one distortion argument. */

    double rd_derivative_x;
    double rd_derivative_y;

    rd_derivative_x = 2.0 * ix * ix * (k[0] + 2.0 * k[1] * radius2);
    rd_derivative_y = 2.0 * iy * iy * (k[0] + 2.0 * k[1] * radius2);
    rd_derivative_x += rd_factor;
    rd_derivative_y += rd_factor;

    cam_x_ptr[3] = fz * rd_derivative_x;
    cam_x_ptr[4] = 0.0;
    cam_x_ptr[5] = -fz * rd_derivative_x * ix;
    cam_x_ptr[6] = -fz * rd_derivative_x * ry * ix;
    cam_x_ptr[7] = fz * rd_derivative_x * (rz + rx * ix);
    cam_x_ptr[8] = -fz * rd_derivative_x * ry;

    cam_y_ptr[3] = 0.0;
    cam_y_ptr[4] = fz * rd_derivative_y;
    cam_y_ptr[5] = -fz * rd_derivative_y * iy;
    cam_y_ptr[6] = -fz * rd_derivative_y * (rz + ry * iy);
    cam_y_ptr[7] = fz * rd_derivative_y * rx * iy;
    cam_y_ptr[8] = fz * rd_derivative_y * rx;

    /*
     * Compute point derivatives in x, y, and z.
     */
    point_x_ptr[0] = fz * rd_derivative_x * (r[0] - r[6] * ix);
    point_x_ptr[1] = fz * rd_derivative_x * (r[1] - r[7] * ix);
    point_x_ptr[2] = fz * rd_derivative_x * (r[2] - r[8] * ix);

    point_y_ptr[0] = fz * rd_derivative_y * (r[3] - r[6] * iy);
    point_y_ptr[1] = fz * rd_derivative_y * (r[4] - r[7] * iy);
    point_y_ptr[2] = fz * rd_derivative_y * (r[5] - r[8] * iy);

#else
    /* Computation of the full Jacobian. 全雅可比矩阵的计算*/

    /*
     * To keep everything comprehensible the chain rule
     * is applied excessively
     * 为了使一切都易于理解，使用链式法则*
     */
    double const f = cam.focal_length;

    // rd--ratial distortion  rad--radius2
    double const rd_deriv_rad = k[0] + 2.0 * k[1] * radius2;//k0+2k1*r^2

    double const rad_deriv_px = 2.0 * ix / pz;//2*x/zc
    double const rad_deriv_py = 2.0 * iy / pz;//2*y/zc
    /*
     * rad_deriv_pz =
     */
    double const rad_deriv_pz = -2.0 * radius2 / pz;//-2*r^2/zc

    double const rd_deriv_px = rd_deriv_rad * rad_deriv_px; //
    double const rd_deriv_py = rd_deriv_rad * rad_deriv_py; //
    double const rd_deriv_pz = rd_deriv_rad * rad_deriv_pz; //

    double const ix_deriv_px = 1 / pz;   //
    double const ix_deriv_pz = -ix / pz; //

    double const iy_deriv_py = 1 / pz;   //
    double const iy_deriv_pz = -iy / pz; //

    double const ix_deriv_r0 = -ix * ry / pz;
    double const ix_deriv_r1 = (rz + rx * ix) / pz;
    double const ix_deriv_r2 = -ry / pz;

    double const iy_deriv_r0 = -(rz + ry * iy) / pz;
    double const iy_deriv_r1 = rx * iy / pz;
    double const iy_deriv_r2 = rx / pz;

    double const rad_deriv_r0 = 2.0 * ix * ix_deriv_r0 + 2.0 * iy * iy_deriv_r0;
    double const rad_deriv_r1 = 2.0 * ix * ix_deriv_r1 + 2.0 * iy * iy_deriv_r1;
    double const rad_deriv_r2 = 2.0 * ix * ix_deriv_r2 + 2.0 * iy * iy_deriv_r2;

    double const rd_deriv_r0 = rd_deriv_rad * rad_deriv_r0;
    double const rd_deriv_r1 = rd_deriv_rad * rad_deriv_r1;
    double const rd_deriv_r2 = rd_deriv_rad * rad_deriv_r2;

    double const ix_deriv_X0 = (r[0] - r[6] * ix) / pz;
    double const ix_deriv_X1 = (r[1] - r[7] * ix) / pz;
    double const ix_deriv_X2 = (r[2] - r[8] * ix) / pz;

    double const iy_deriv_X0 = (r[3] - r[6] * iy) / pz;
    double const iy_deriv_X1 = (r[4] - r[7] * iy) / pz;
    double const iy_deriv_X2 = (r[5] - r[8] * iy) / pz;

    double const rad_deriv_X0 = 2.0 * ix * ix_deriv_X0 + 2.0 * iy * iy_deriv_X0;
    double const rad_deriv_X1 = 2.0 * ix * ix_deriv_X1 + 2.0 * iy * iy_deriv_X1;
    double const rad_deriv_X2 = 2.0 * ix * ix_deriv_X2 + 2.0 * iy * iy_deriv_X2;

    double const rd_deriv_X0 = rd_deriv_rad * rad_deriv_X0;
    double const rd_deriv_X1 = rd_deriv_rad * rad_deriv_X1;
    double const rd_deriv_X2 = rd_deriv_rad * rad_deriv_X2;

    /*
     * Compute translation derivatives
     * NOTE: px_deriv_t0 = 1
     */
    cam_x_ptr[3] = f * (rd_deriv_px * ix + rd_factor * ix_deriv_px);
    cam_x_ptr[4] = f * (rd_deriv_py * ix); // + rd_factor * ix_deriv_py = 0
    cam_x_ptr[5] = f * (rd_deriv_pz * ix + rd_factor * ix_deriv_pz);

    cam_y_ptr[3] = f * (rd_deriv_px * iy); // + rd_factor * iy_deriv_px = 0
    cam_y_ptr[4] = f * (rd_deriv_py * iy + rd_factor * iy_deriv_py);
    cam_y_ptr[5] = f * (rd_deriv_pz * iy + rd_factor * iy_deriv_pz);

    /*
     * Compute rotation derivatives
     */
    cam_x_ptr[6] = f * (rd_deriv_r0 * ix + rd_factor * ix_deriv_r0);
    cam_x_ptr[7] = f * (rd_deriv_r1 * ix + rd_factor * ix_deriv_r1);
    cam_x_ptr[8] = f * (rd_deriv_r2 * ix + rd_factor * ix_deriv_r2);

    cam_y_ptr[6] = f * (rd_deriv_r0 * iy + rd_factor * iy_deriv_r0);
    cam_y_ptr[7] = f * (rd_deriv_r1 * iy + rd_factor * iy_deriv_r1);
    cam_y_ptr[8] = f * (rd_deriv_r2 * iy + rd_factor * iy_deriv_r2);

    /*
     * Compute point derivatives in x, y, and z.
     */
    point_x_ptr[0] = f * (rd_deriv_X0 * ix + rd_factor * ix_deriv_X0);
    point_x_ptr[1] = f * (rd_deriv_X1 * ix + rd_factor * ix_deriv_X1);
    point_x_ptr[2] = f * (rd_deriv_X2 * ix + rd_factor * ix_deriv_X2);

    point_y_ptr[0] = f * (rd_deriv_X0 * iy + rd_factor * iy_deriv_X0);
    point_y_ptr[1] = f * (rd_deriv_X1 * iy + rd_factor * iy_deriv_X1);
    point_y_ptr[2] = f * (rd_deriv_X2 * iy + rd_factor * iy_deriv_X2);

#endif
}

void BundleAdjustment::update_parameters(DenseVectorType const &delta_x)
{
    /* Update cameras. */
    std::size_t total_camera_params = 0;
    if (this->opts.bundle_mode & BA_CAMERAS)
    {
        for (std::size_t i = 0; i < this->cameras->size(); ++i)
            this->update_camera(this->cameras->at(i),
                                delta_x.data() + this->num_cam_params * i,
                                &this->cameras->at(i));
        total_camera_params = this->cameras->size() * this->num_cam_params;
    }

    /* Update points. */
    if (this->opts.bundle_mode & BA_POINTS)
    {
        for (std::size_t i = 0; i < this->points->size(); ++i)
            this->update_point(this->points->at(i),
                               delta_x.data() + total_camera_params + i * 3,
                               &this->points->at(i));
    }
}

void BundleAdjustment::update_camera(Camera const &cam,
                                     double const *update, Camera *out) //更新相机参数
{
    if (opts.fixed_intrinsics) //如果固定内参数矩阵K，只优化R，T
    {
        out->focal_length = cam.focal_length;   //输入焦距等于输出焦距
        out->distortion[0] = cam.distortion[0]; //输入径向畸变等于输出径向畸变
        out->distortion[1] = cam.distortion[1];
    }
    else //若内参矩阵不固定，则同时优化相机的九个参数
    {
        out->focal_length = cam.focal_length + update[0];   //更新相机的焦距
        out->distortion[0] = cam.distortion[0] + update[1]; //更新相机的径向畸变
        out->distortion[1] = cam.distortion[1] + update[2];
    }

    int const offset = this->opts.fixed_intrinsics ? 0 : 3;        //内参矩阵固定=0，否则=3
    out->translation[0] = cam.translation[0] + update[0 + offset]; //更新T
    out->translation[1] = cam.translation[1] + update[1 + offset];
    out->translation[2] = cam.translation[2] + update[2 + offset];

    double rot_orig[9]; //定义数组储存旋转矩阵R
    std::copy(cam.rotation, cam.rotation + 9, rot_orig);
    double rot_update[9];                                                //定义数组储存旋转矩阵的变化量
    this->rodrigues_to_matrix(update + 3 + offset, rot_update);          //将角轴法转化成旋转矩阵;update + 3 + offset：指向delta_R矩阵的第一个元素
    math::matrix_multiply(rot_update, 3, 3, rot_orig, 3, out->rotation); //将旋转矩阵的增量加上
}

void BundleAdjustment::update_point(Point3D const &pt,
                                    double const *update, Point3D *out) //更新点的坐标
{
    out->pos[0] = pt.pos[0] + update[0];
    out->pos[1] = pt.pos[1] + update[1];
    out->pos[2] = pt.pos[2] + update[2];
}

void BundleAdjustment::print_status(bool detailed) const
{
    if (!detailed)
    {
        std::cout << "BA: MSE " << this->status.initial_mse
                  << " -> " << this->status.final_mse << ", "
                  << this->status.num_lm_iterations << " LM iters, "
                  << this->status.num_cg_iterations << " CG iters, "
                  << this->status.runtime_ms << "ms."
                  << std::endl;
        return;
    }

    std::cout << "Bundle Adjustment Status:" << std::endl;
    std::cout << "  Initial MSE: "
              << this->status.initial_mse << std::endl;
    std::cout << "  Final MSE: "
              << this->status.final_mse << std::endl;
    std::cout << "  LM iterations: "
              << this->status.num_lm_iterations << " ("
              << this->status.num_lm_successful_iterations << " successful, "
              << this->status.num_lm_unsuccessful_iterations << " unsuccessful)"
              << std::endl;
    std::cout << "  CG iterations: "
              << this->status.num_cg_iterations << std::endl;
}

SFM_BA_NAMESPACE_END
SFM_NAMESPACE_END
