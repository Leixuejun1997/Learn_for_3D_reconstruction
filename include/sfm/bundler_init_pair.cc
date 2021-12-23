/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <iostream>
#include <vector>
#include <sstream>
#include <random>

#include "sfm/ransac_homography.h"
#include "sfm/triangulate.h"
#include "sfm/bundler_init_pair.h"

SFM_NAMESPACE_BEGIN
SFM_BUNDLER_NAMESPACE_BEGIN

void InitialPair::compute_pair(Result *result) // Result：视角1、2的ID，视角1、2的相机姿态
{
    if (this->viewports == nullptr || this->tracks == nullptr)
        throw std::invalid_argument("Null viewports or tracks"); //如果视角和track都为空，抛出异常

    std::cout << "Searching for initial pair..." << std::endl;
    result->view_1_id = -1;
    result->view_2_id = -1;

    // 根据匹配的点的个数，找到候选的匹配对
    /* Convert tracks to pairwise information. */
    std::vector<CandidatePair> candidates;      //储存所有候选图像对的信息；CandidatePair：视角1、2的ID，视角1-2的特征匹配点的信息
    this->compute_candidate_pairs(&candidates); //通过tracks反过来计算匹配对信息，因为track是经过过滤的
    /* Sort the candidate pairs by number of matches. 根据匹配点的多少来排序这些候选图片对*/
    std::sort(candidates.rbegin(), candidates.rend()); //从小到大排序

    /*
     * Search for a good initial pair and return the first pair that
     * satisfies all thresholds (min matches, max homography inliers,
     * min triangulation angle). If no pair satisfies all thresholds, the
     * pair with the best score is returned.
     * 搜索一个最好的初始配对，并返回满足所有阈值（最小匹配、最大单应性内点数、最小三角化的角度）的第一个配对
     * 如果没有图像对满足所有阈值，则返回得分最高的图像对
     */
    bool found_pair = false;
    std::size_t found_pair_id = std::numeric_limits<std::size_t>::max(); //定义ID为最大数
    std::vector<float> pair_scores(candidates.size(), 0.0f);             //定义一个float的容器，大小为candidates的大小，并赋值为0，用于储存每个匹配对的分数
#pragma omp parallel for schedule(dynamic)                               //动态多线程
    for (std::size_t i = 0; i < candidates.size(); ++i)
    {
        if (found_pair)
            continue;

        //标准1： 匹配点的个数大50对
        /* Reject pairs with 8 or fewer matches. */
        CandidatePair const &candidate = candidates[i];                         //获得其中的第i个候选匹配对：视角1-2的ID，视角1-2的匹配对信息
        std::size_t num_matches = candidate.matches.size();                     //获得匹配对的数量
        if (num_matches < static_cast<std::size_t>(this->opts.min_num_matches)) //如果不满足最小匹配点数量：8，换下一对
        {
            this->debug_output(candidate);
            continue;
        }

        //标准2： 单应矩阵矩阵的内点比例数过高
        /* Reject pairs with too high percentage of homograhy inliers. */
        std::size_t num_inliers = this->compute_homography_inliers(candidate); //计算单应性矩阵的内点数
        float percentage = static_cast<float>(num_inliers) / num_matches;
        if (percentage > this->opts.max_homography_inliers) //如果单应性矩阵的内点数过高，就换下一对
        {
            this->debug_output(candidate, num_inliers);
            continue;
        }

        // 标准3：相机基线要足够长(用三角量测的夹角衡量）
        /* Compute initial pair pose. 计算两个匹配对的位姿*/
        CameraPose pose1, pose2;
        bool const found_pose = this->compute_pose(candidate, &pose1, &pose2); //计算基础矩阵，本质矩阵，
        //分解本质矩阵获得R，T，pose1-2存储的是两个视角的内参矩阵
        if (!found_pose)
        {
            this->debug_output(candidate, num_inliers);
            continue;
        }
        /* Rejects pairs with bad triangulation angle. */
        double const angle = this->angle_for_pose(candidate, pose1, pose2);//计算两个视角的角度
        pair_scores[i] = this->score_for_pair(candidate, num_inliers, angle);//计算每对视角的得分
        this->debug_output(candidate, num_inliers, angle);
        if (angle < this->opts.min_triangulation_angle)//如果三角化的角度小于阈值，则换下一对
            continue;

        // 标准4： 成功的三角量测的个数>50%
        /* Run triangulation to ensure correct pair */
        Triangulate::Options triangulate_opts;//都使用默认参数
        Triangulate triangulator(triangulate_opts);//赋值
        std::vector<CameraPose const *> poses;
        poses.push_back(&pose1);
        poses.push_back(&pose2);
        std::size_t successful_triangulations = 0;
        std::vector<math::Vec2f> positions(2);
        Triangulate::Statistics stats;
        for (std::size_t j = 0; j < candidate.matches.size(); ++j)
        {
            positions[0] = math::Vec2f(candidate.matches[j].p1);
            positions[1] = math::Vec2f(candidate.matches[j].p2);
            math::Vec3d pos3d;
            if (triangulator.triangulate(poses, positions, &pos3d, &stats))//进行三角化
                successful_triangulations += 1;
        }
        if (successful_triangulations * 2 < candidate.matches.size())//如果成功三角化的数量不满足要求，换下一对
            continue;

#pragma omp critical
        if (i < found_pair_id)
        {
            result->view_1_id = candidate.view_1_id;
            result->view_2_id = candidate.view_2_id;
            result->view_1_pose = pose1;
            result->view_2_pose = pose2;
            found_pair_id = i;
            found_pair = true;
        }
    }

    /* Return if a pair satisfying all thresholds has been found. 如果有一对图像对满足所有条件则返回*/
    if (found_pair)
        return;

    /* Return pair with best score (larger than 0.0). 如果没有找到满足所有条件的图像对，则返回分数最大的图像对（大于0.0）*/
    std::cout << "Searching for pair with best score..." << std::endl;
    float best_score = 0.0f;//最大的分数
    std::size_t best_pair_id = 0;//最好的图像对ID
    for (std::size_t i = 0; i < pair_scores.size(); ++i)//找最大分数
    {
        if (pair_scores[i] <= best_score)
            continue;

        best_score = pair_scores[i];
        best_pair_id = i;
    }

    /* Recompute pose for pair with best score. 为有最大分数的图像对重新计算pos*/
    if (best_score > 0.0f)
    {
        result->view_1_id = candidates[best_pair_id].view_1_id;
        result->view_2_id = candidates[best_pair_id].view_2_id;
        this->compute_pose(candidates[best_pair_id],
                           &result->view_1_pose, &result->view_2_pose);
    }
}

void InitialPair::compute_pair(int view_1_id, int view_2_id, Result *result)
{
    if (view_1_id > view_2_id)
        std::swap(view_1_id, view_2_id);

    /* Convert tracks to pairwise information. */
    std::vector<CandidatePair> candidates;
    this->compute_candidate_pairs(&candidates);

    /* Find candidate pair. */
    CandidatePair *candidate = nullptr;
    for (std::size_t i = 0; candidate == nullptr && i < candidates.size(); ++i)
    {
        if (view_1_id == candidates[i].view_1_id && view_2_id == candidates[i].view_2_id)
            candidate = &candidates[i];
    }
    if (candidate == nullptr)
        throw std::runtime_error("No matches for initial pair");

    /* Compute initial pair pose. */
    result->view_1_id = view_1_id;
    result->view_2_id = view_2_id;
    bool const found_pose = this->compute_pose(*candidate,
                                               &result->view_1_pose, &result->view_2_pose);
    if (!found_pose)
        throw std::runtime_error("Cannot compute pose for initial pair");
}

void InitialPair::compute_candidate_pairs(CandidatePairs *candidates) //通过tracks反过来计算匹配对信息
{
    /*
     * Convert the tracks to pairwise information. This is similar to using
     * the pairwise matching result directly, however, the tracks have been
     * further filtered during track generation.
     * 将tracks转换为匹配对的信息。这类似于直接使用两两匹配结果，然而，在tracks生成过程中，轨迹被进一步过滤过
     */
    int const num_viewports = static_cast<int>(this->viewports->size()); //获得视角的数量
    std::vector<int> candidate_lookup(MATH_POW2(num_viewports), -1);     //定义一个视角数量平方大小的vector，并初始化为-1
    candidates->reserve(1000);
    for (std::size_t i = 0; i < this->tracks->size(); ++i)
    {
        Track const &track = this->tracks->at(i);               //获得其中一条track
        for (std::size_t j = 1; j < track.features.size(); ++j) //也就是该track中由多少个特征点组成
            for (std::size_t k = 0; k < j; ++k)
            {

                int v1id = track.features[j].view_id; // 每取一个j的视角，用j序号之前的视角与之一一组合
                int v2id = track.features[k].view_id; //
                int f1id = track.features[j].feature_id;
                int f2id = track.features[k].feature_id;
                if (v1id > v2id)
                {
                    std::swap(v1id, v2id);
                    std::swap(f1id, f2id);
                }

                /* Lookup pair. */
                int const lookup_id = v1id * num_viewports + v2id;
                int pair_id = candidate_lookup[lookup_id];
                if (pair_id == -1)
                {
                    pair_id = static_cast<int>(candidates->size());
                    candidate_lookup[lookup_id] = pair_id;
                    candidates->push_back(CandidatePair());
                    candidates->back().view_1_id = v1id;
                    candidates->back().view_2_id = v2id;
                }

                /* Fill candidate with additional info. */
                Viewport const &view1 = this->viewports->at(v1id); // this存储了所有的视角的信息
                Viewport const &view2 = this->viewports->at(v2id);
                math::Vec2f const v1pos = view1.features.positions[f1id]; //获得视角的pos
                math::Vec2f const v2pos = view2.features.positions[f2id];
                Correspondence2D2D match; //储存匹配对的
                std::copy(v1pos.begin(), v1pos.end(), match.p1);
                std::copy(v2pos.begin(), v2pos.end(), match.p2);
                candidates->at(pair_id).matches.push_back(match);
            }
    }
}

std::size_t
InitialPair::compute_homography_inliers(CandidatePair const &candidate)
{
    /* Execute homography RANSAC. */
    RansacHomography::Result ransac_result;
    RansacHomography homography_ransac(this->opts.homography_opts);
    homography_ransac.estimate(candidate.matches, &ransac_result);
    return ransac_result.inliers.size();
}

bool InitialPair::compute_pose(CandidatePair const &candidate,
                               CameraPose *pose1, CameraPose *pose2)
{
    /* Compute fundamental matrix from pair correspondences. 从匹配对中计算基础矩阵*/
    FundamentalMatrix fundamental;
    {
        Correspondences2D2D matches = candidate.matches;
        if (matches.size() > 1000ul)
        {
            std::mt19937 g;
            std::shuffle(matches.begin(), matches.end(), g);
            matches.resize(1000ul);
        }
        fundamental_least_squares(matches, &fundamental);//最小二乘法计算基础矩阵
        enforce_fundamental_constraints(&fundamental);
    }

    /* Populate K-matrices.填充内参矩阵K */
    Viewport const &view_1 = this->viewports->at(candidate.view_1_id);//获得匹配点对应的视角
    Viewport const &view_2 = this->viewports->at(candidate.view_2_id);
    pose1->set_k_matrix(view_1.focal_length, 0.0, 0.0);
    pose1->init_canonical_form();
    pose2->set_k_matrix(view_2.focal_length, 0.0, 0.0);

    /* Compute essential matrix from fundamental matrix (HZ (9.12)). 从基础矩阵中计算本质矩阵*/
    EssentialMatrix E = pose2->K.transposed() * fundamental * pose1->K;

    /* Compute pose from essential. */
    std::vector<CameraPose> poses;
    pose_from_essential(E, &poses);//分解E矩阵，获得R，T

    /* Find the correct pose using point test (HZ Fig. 9.12). 使用第四个点验证正确的R，T*/
    bool found_pose = false;
    for (std::size_t i = 0; i < poses.size(); ++i)
    {
        poses[i].K = pose2->K;
        if (is_consistent_pose(candidate.matches[0], *pose1, poses[i]))//将一个点转换到相机坐标系下，Z值大于0，则分解的R，T满足条件
        {
            *pose2 = poses[i];
            found_pose = true;
            break;
        }
    }
    return found_pose;
}

double
InitialPair::angle_for_pose(CandidatePair const &candidate,
                            CameraPose const &pose1, CameraPose const &pose2)
{
    /* Compute transformation from image coordinates to viewing direction. */
    math::Matrix3d T1 = pose1.R.transposed() * math::matrix_inverse(pose1.K);//T1=R1^T*K1^-1
    math::Matrix3d T2 = pose2.R.transposed() * math::matrix_inverse(pose2.K);//T2=R2^T*K2^-1

    /* Compute triangulation angle for each correspondence. 计算每对对应点的角度*/
    std::vector<double> cos_angles;
    cos_angles.reserve(candidate.matches.size());
    for (std::size_t i = 0; i < candidate.matches.size(); ++i)
    {
        Correspondence2D2D const &match = candidate.matches[i];
        math::Vec3d p1(match.p1[0], match.p1[1], 1.0);
        p1 = T1.mult(p1).normalized();//获得3D点与2D点的单位方向向量
        math::Vec3d p2(match.p2[0], match.p2[1], 1.0);
        p2 = T2.mult(p2).normalized();//获得3D点与2D点的单位方向向量
        cos_angles.push_back(p1.dot(p2));//两个方向向量点乘cos0=n1*n2
    }

    /* Return 50% median. */
    std::size_t median_index = cos_angles.size() / 2;
    std::nth_element(cos_angles.begin(),
                     cos_angles.begin() + median_index, cos_angles.end());//获得中位数
    double const cos_angle = math::clamp(cos_angles[median_index], -1.0, 1.0);//取（-1，1）之间最大值
    return std::acos(cos_angle);//arcos()
}

float InitialPair::score_for_pair(CandidatePair const &candidate,
                                  std::size_t num_inliers, double angle)
{
    float const matches = static_cast<float>(candidate.matches.size());//匹配对的数量
    float const inliers = static_cast<float>(num_inliers) / matches;//内点/总匹配对数
    float const angle_d = MATH_RAD2DEG(angle);//将弧度转换为角度

    /* Score for matches (min: 20, good: 200). 匹配点的分数，最少20，最多200对*/
    float f1 = 2.0 / (1.0 + std::exp((20.0 - matches) * 6.0 / 200.0)) - 1.0;
    /* Score for angle (min 1 degree, good 8 degree). 角度分数，最小1度，最大8度*/
    float f2 = 2.0 / (1.0 + std::exp((1.0 - angle_d) * 6.0 / 8.0)) - 1.0;
    /* Score for H-Inliers (max 70%, good 40%). 单应性矩阵的内点比例，最大70%，最小40%*/
    float f3 = 2.0 / (1.0 + std::exp((inliers - 0.7) * 6.0 / 0.4)) - 1.0;

    f1 = math::clamp(f1, 0.0f, 1.0f);
    f2 = math::clamp(f2, 0.0f, 1.0f);
    f3 = math::clamp(f3, 0.0f, 1.0f);
    return f1 * f2 * f3;
}

void InitialPair::debug_output(CandidatePair const &candidate,
                               std::size_t num_inliers, double angle)
{
    if (!this->opts.verbose_output)
        return;

    std::stringstream message;
    std::size_t num_matches = candidate.matches.size();
    message << "  Pair " << std::setw(3) << candidate.view_1_id
            << "," << std::setw(3) << candidate.view_2_id
            << ": " << std::setw(4) << num_matches << " matches";

    if (num_inliers > 0)
    {
        float percentage = static_cast<float>(num_inliers) / num_matches;
        message << ", " << std::setw(4) << num_inliers
                << " H-inliers (" << (int)(100.0f * percentage) << "%)";
    }

    if (angle > 0.0)
    {
        message << ", " << std::setw(5)
                << util::string::get_fixed(MATH_RAD2DEG(angle), 2)
                << " pair angle";
    }

#pragma omp critical
    std::cout << message.str() << std::endl;
}

SFM_BUNDLER_NAMESPACE_END
SFM_NAMESPACE_END
