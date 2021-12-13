//
// Created by caoqi on 2018/8/28.
//
#include "defines.h"
#include "functions.h"
#include "sfm/bundler_common.h"
#include "sfm/bundler_features.h"
#include "sfm/bundler_matching.h"
#include "sfm/bundler_intrinsics.h"
#include "sfm/bundler_init_pair.h"
#include "sfm/bundler_tracks.h"
#include "sfm/bundler_incremental.h"
#include "core/scene.h"
#include "util/timer.h"

#include <util/file_system.h>
#include <core/bundle_io.h>
#include <core/camera.h>
#include <fstream>
#include <iostream>
#include <cassert>
/**
 *\description 创建一个场景
 * @param image_folder_path
 * @param scene_path
 * @return
 */
core::Scene::Ptr
make_scene(const std::string &image_folder_path, const std::string &scene_path) //输入图片路径，场景路径
{

    util::WallTimer timer; //计时器

    /*** 创建文件夹 ***/
    const std::string views_path = util::fs::join_path(scene_path, "views/"); //输出文件夹
    util::fs::mkdir(scene_path.c_str());
    util::fs::mkdir(views_path.c_str());

    /***扫描文件夹，获取所有的图像文件路径***/
    util::fs::Directory dir;
    try
    {
        dir.scan(image_folder_path); //扫描输入路径的文件夹文件
    }
    catch (std::exception &e)
    {
        std::cerr << "Error scanning input dir: " << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::cout << "Found " << dir.size() << " directory entries." << std::endl;

    core::Scene::Ptr scene = core::Scene::create("");

    /**** 开始加载图像 ****/
    std::sort(dir.begin(), dir.end()); //从小到大排序
    int num_imported = 0;
    for (std::size_t i = 0; i < dir.size(); i++)
    {
        // 是一个文件夹
        if (dir[i].is_dir)
        {
            std::cout << "Skipping directory " << dir[i].name << std::endl;
            continue;
        }

        std::string fname = dir[i].name;                 //获得每一个文件的名字
        std::string afname = dir[i].get_absolute_name(); //获得文件的绝对名字

        // 从可交换信息文件中读取图像焦距
        std::string exif;
        core::ImageBase::Ptr image = load_any_image(afname, &exif); //获得图片的信息（图像长宽、焦距）
        if (image == nullptr)
        {
            continue;
        }

        core::View::Ptr view = core::View::create();
        view->set_id(num_imported);                   //设置ID编号
        view->set_name(remove_file_extension(fname)); //将去除扩展名的文件名赋予给view

        // 限制图像尺寸
        int orig_width = image->width();                               //图像宽度
        image = limit_image_size(image, MAX_PIXELS);                   //设置图像的大小
        if (orig_width == image->width() && has_jpeg_extension(fname)) //若文件宽度符合且有扩展名
            view->set_image_ref(afname, "original");                   //把图片的名字改为original
        else
            view->set_image(image, "original"); //重命名且将图片额尺寸重新定义

        add_exif_to_view(view, exif); //文件夹的exif.bolb文件

        scene->get_views().push_back(view);

        /***保存视角信息到本地****/
        std::string mve_fname = make_image_name(num_imported);
        std::cout << "Importing image: " << fname
                  << ", writing MVE view: " << mve_fname << "..." << std::endl;
        view->save_view_as(util::fs::join_path(views_path, mve_fname));

        num_imported += 1;
    }

    std::cout << "Imported " << num_imported << " input images, "
              << "took " << timer.get_elapsed() << " ms." << std::endl;

    return scene;
}

/**
 *
 * @param scene
 * @param viewports
 * @param pairwise_matching
 */
void features_and_matching(core::Scene::Ptr scene,
                           sfm::bundler::ViewportList *viewports,
                           sfm::bundler::PairwiseMatching *pairwise_matching) // ViewportList存放所有视角的信息
//包括：焦距、径向畸变参数、相机姿态、特征点信息、以及特征点所在的tracks
//PairwiseMatching：记录所有匹配的图像对的特征点对应关系
{

    /****************************************** 计算特征点 ***********************************/
    //定义计算特征的一些参数
    sfm::bundler::Features::Options feature_opts;
    feature_opts.image_embedding = "original";//要计算特征的图像名
    feature_opts.max_image_size = MAX_PIXELS;//最大图像大小（以像素为单位）6000000
    feature_opts.feature_options.feature_types = sfm::FeatureSet::FEATURE_SIFT;//定义描述子的类型，可以在这里修改你使用的描述子

    std::cout << "Computing image features..." << std::endl;
    {
        util::WallTimer timer;//计时器
        sfm::bundler::Features bundler_features(feature_opts);//把参数赋值给bundler_features
        bundler_features.compute(scene, viewports);//计算特征点，储存在viewport中

        std::cout << "Computing features took " << timer.get_elapsed()
                  << " ms." << std::endl;
        std::cout << "Feature detection took " + util::string::get(timer.get_elapsed()) + "ms." << std::endl;
    }

    /* Exhaustive matching between all pairs of views. */
    sfm::bundler::Matching::Options matching_opts;//定义特征匹配的一些参数
    // matching_opts.ransac_opts.max_iterations = 1000;
    // matching_opts.ransac_opts.threshold = 0.0015;
    matching_opts.ransac_opts.verbose_output = false;//不在控制台产生状态消息
    matching_opts.use_lowres_matching = false;//不使用低分辨率图像剔除误匹配点
    matching_opts.match_num_previous_frames = false;
    matching_opts.matcher_type = sfm::bundler::Matching::MATCHER_EXHAUSTIVE;//匹配器类型

    std::cout << "Performing feature matching..." << std::endl;
    {
        util::WallTimer timer;//计时器
        sfm::bundler::Matching bundler_matching(matching_opts);//赋值特征匹配参数
        bundler_matching.init(viewports);
        bundler_matching.compute(pairwise_matching);//进行特征匹配
        std::cout << "Matching took " << timer.get_elapsed()
                  << " ms." << std::endl;
        std::cout << "Feature matching took " + util::string::get(timer.get_elapsed()) + "ms." << std::endl;
    }

    if (pairwise_matching->empty())
    {
        std::cerr << "Error: No matching image pairs. Exiting." << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[])
{

    if (argc < 3)
    {
        std::cout << "Usage: [input]image_dir [output]scene_dir" << std::endl;
        return -1;
    }

    core::Scene::Ptr scene = make_scene(argv[1], argv[2]); //输入源文件夹路径与输出文件夹
    std::cout << "Scene has " << scene->get_views().size() << " views. " << std::endl;

    /*进行特征匹配*/
    sfm::bundler::ViewportList viewports;//存储图像的信息
    sfm::bundler::PairwiseMatching pairwise_matching;//存储图像对信息
    features_and_matching(scene, &viewports, &pairwise_matching);//进行特征匹配

    /* Drop descriptors and embeddings to save memory. */
    scene->cache_cleanup();//清空内存
    for (std::size_t i = 0; i < viewports.size(); ++i)
        viewports[i].features.clear_descriptors();

    /* Check if there are some matching images. */
    if (pairwise_matching.empty())//看看是否有匹配的图像
    {
        std::cerr << "No matching image pairs. Exiting." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // 计算相机内参数，从Exif中读取
    {
        sfm::bundler::Intrinsics::Options intrinsics_opts;
        std::cout << "Initializing camera intrinsics..." << std::endl;
        sfm::bundler::Intrinsics intrinsics(intrinsics_opts);
        intrinsics.compute(scene, &viewports);
    }

    /****** 开始增量的BA*****/
    //图像连接图构建
    util::WallTimer timer; //时间计时器
    /* Compute connected feature components, i.e. feature tracks. */
    sfm::bundler::TrackList tracks;
    {
        sfm::bundler::Tracks::Options tracks_options;
        tracks_options.verbose_output = true;//不在控制台显示状态信息

        sfm::bundler::Tracks bundler_tracks(tracks_options);
        std::cout << "Computing feature tracks..." << std::endl;
        bundler_tracks.compute(pairwise_matching, &viewports, &tracks);//将所有的特征匹配转换为tracks
        std::cout << "Created a total of " << tracks.size()
                  << " tracks." << std::endl;
    }

    /* Remove color data and pairwise matching to save memory. */
    for (std::size_t i = 0; i < viewports.size(); ++i)
        viewports[i].features.colors.clear();
    pairwise_matching.clear();

    // 计算初始的匹配对
    sfm::bundler::InitialPair::Result init_pair_result;
    sfm::bundler::InitialPair::Options init_pair_opts;
    // init_pair_opts.homography_opts.max_iterations = 1000;
    // init_pair_opts.homography_opts.threshold = 0.005f;
    init_pair_opts.homography_opts.verbose_output = false;
    init_pair_opts.max_homography_inliers = 0.8f;
    init_pair_opts.verbose_output = true;

    // 开始计算初始的匹配对
    sfm::bundler::InitialPair init_pair(init_pair_opts);
    init_pair.initialize(viewports, tracks);
    init_pair.compute_pair(&init_pair_result);
    if (init_pair_result.view_1_id < 0 || init_pair_result.view_2_id < 0 || init_pair_result.view_1_id >= static_cast<int>(viewports.size()) || init_pair_result.view_2_id >= static_cast<int>(viewports.size()))
    {

        std::cerr << "Error finding initial pair, exiting!" << std::endl;
        std::cerr << "Try manually specifying an initial pair." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::cout << "Using views " << init_pair_result.view_1_id
              << " and " << init_pair_result.view_2_id
              << " as initial pair." << std::endl;

    /* Incrementally compute full bundle. */
    sfm::bundler::Incremental::Options incremental_opts;
    incremental_opts.pose_p3p_opts.max_iterations = 1000;
    incremental_opts.pose_p3p_opts.threshold = 0.005f;
    incremental_opts.pose_p3p_opts.verbose_output = false;
    incremental_opts.track_error_threshold_factor = TRACK_ERROR_THRES_FACTOR;
    incremental_opts.new_track_error_threshold = NEW_TRACK_ERROR_THRES;
    incremental_opts.min_triangulation_angle = MATH_DEG2RAD(1.0);
    incremental_opts.ba_fixed_intrinsics = false;
    // incremental_opts.ba_shared_intrinsics = conf.shared_intrinsics;
    incremental_opts.verbose_output = true;
    incremental_opts.verbose_ba = true;

    /* Initialize viewports with initial pair. */
    viewports[init_pair_result.view_1_id].pose = init_pair_result.view_1_pose;
    viewports[init_pair_result.view_2_id].pose = init_pair_result.view_2_pose;

    /* Initialize the incremental bundler and reconstruct first tracks. */
    sfm::bundler::Incremental incremental(incremental_opts);
    incremental.initialize(&viewports, &tracks);

    // 对当前两个视角进行track重建，并且如果track存在外点，则将每个track的外点剥离成新的track
    incremental.triangulate_new_tracks(2);

    // 根据重投影误差进行筛选
    incremental.invalidate_large_error_tracks();

    /* Run bundle adjustment. */
    std::cout << "Running full bundle adjustment..." << std::endl;
    incremental.bundle_adjustment_full();

    /* Reconstruct remaining views. */
    int num_cameras_reconstructed = 2;
    int full_ba_num_skipped = 0;
    while (true)
    {
        /* Find suitable next views for reconstruction. */
        std::vector<int> next_views;
        incremental.find_next_views(&next_views);

        /* Reconstruct the next view. */
        int next_view_id = -1;
        for (std::size_t i = 0; i < next_views.size(); ++i)
        {
            std::cout << std::endl;
            std::cout << "Adding next view ID " << next_views[i]
                      << " (" << (num_cameras_reconstructed + 1) << " of "
                      << viewports.size() << ")..." << std::endl;
            if (incremental.reconstruct_next_view(next_views[i]))
            {
                next_view_id = next_views[i];
                break;
            }
        }

        if (next_view_id < 0)
        {
            if (full_ba_num_skipped == 0)
            {
                std::cout << "No valid next view." << std::endl;
                std::cout << "SfM reconstruction finished." << std::endl;
                break;
            }
            else
            {
                incremental.triangulate_new_tracks(MIN_VIEWS_PER_TRACK);
                std::cout << "Running full bundle adjustment..." << std::endl;
                incremental.invalidate_large_error_tracks();
                incremental.bundle_adjustment_full();
                full_ba_num_skipped = 0;
                continue;
            }
        }

        /* Run single-camera bundle adjustment. */
        std::cout << "Running single camera bundle adjustment..." << std::endl;
        incremental.bundle_adjustment_single_cam(next_view_id);
        num_cameras_reconstructed += 1;

        /* Run full bundle adjustment only after a couple of views. */
        int const full_ba_skip_views = std::min(100, num_cameras_reconstructed / 10);
        if (full_ba_num_skipped < full_ba_skip_views)
        {
            std::cout << "Skipping full bundle adjustment (skipping "
                      << full_ba_skip_views << " views)." << std::endl;
            full_ba_num_skipped += 1;
        }
        else
        {
            incremental.triangulate_new_tracks(MIN_VIEWS_PER_TRACK);
            std::cout << "Running full bundle adjustment..." << std::endl;

            /*去除错误的track: 首先统计所有tracks的平均重投影误差找到阈值，根据统计的阈值筛选掉误差较大的tracks*/
            incremental.invalidate_large_error_tracks();

            /*全局的BA*/
            incremental.bundle_adjustment_full();
            full_ba_num_skipped = 0;
        }
    }

    sfm::bundler::TrackList valid_tracks;
    for (int i = 0; i < tracks.size(); i++)
    {
        if (tracks[i].is_valid())
        {
            valid_tracks.push_back(tracks[i]);
        }
    }

    std::cout << "SfM reconstruction took " << timer.get_elapsed()
              << " ms." << std::endl;
    std::cout << "SfM reconstruction took " + util::string::get(timer.get_elapsed()) + "ms." << std::endl;

    /***** 保存输出结果***/
    std::ofstream out_file("./points.ply");
    assert(out_file.is_open());
    out_file << "ply" << std::endl;
    out_file << "format ascii 1.0" << std::endl;
    out_file << "element vertex " << valid_tracks.size() << std::endl;
    out_file << "property float x" << std::endl;
    out_file << "property float y" << std::endl;
    out_file << "property float z" << std::endl;
    out_file << "property uchar red" << std::endl;
    out_file << "property uchar green" << std::endl;
    out_file << "property uchar blue" << std::endl;
    out_file << "end_header" << std::endl;

    for (int i = 0; i < valid_tracks.size(); i++)
    {
        out_file << valid_tracks[i].pos[0] << " " << valid_tracks[i].pos[1] << " " << valid_tracks[i].pos[2] << " "
                 << (int)valid_tracks[i].color[0] << " " << (int)valid_tracks[i].color[1] << " " << (int)valid_tracks[i].color[2] << std::endl;
    }
    out_file.close();

    /* Normalize scene if requested. */
    //    if (conf.normalize_scene)
    //    {
    //        std::cout << "Normalizing scene..." << std::endl;
    //        incremental.normalize_scene();
    //    }
    /* Save bundle file to scene. */
    std::cout << "Creating bundle data structure..." << std::endl;
    core::Bundle::Ptr bundle = incremental.create_bundle();
    core::save_mve_bundle(bundle, std::string(argv[2]) + "/synth_0.out");

    /* Apply bundle cameras to views. */
    core::Bundle::Cameras const &bundle_cams = bundle->get_cameras();
    core::Scene::ViewList const &views = scene->get_views();
    if (bundle_cams.size() != views.size())
    {
        std::cerr << "Error: Invalid number of cameras!" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    /*利用估计的相机内参数进行去劲向畸变操作*/
#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t i = 0; i < bundle_cams.size(); ++i)
    {
        core::View::Ptr view = views[i];
        core::CameraInfo const &cam = bundle_cams[i];
        if (view == nullptr)
            continue;
        if (view->get_camera().flen == 0.0f && cam.flen == 0.0f)
            continue;

        view->set_camera(cam);

        /* Undistort image. */
        if (!undistorted_name.empty())
        {
            core::ByteImage::Ptr original = view->get_byte_image(original_name);
            if (original == nullptr)
                continue;
            core::ByteImage::Ptr undist = core::image::image_undistort_k2k4<uint8_t>(original, cam.flen, cam.dist[0], cam.dist[1]);
            view->set_image(undist, undistorted_name);
        }

#pragma omp critical
        std::cout << "Saving view " << view->get_directory() << std::endl;
        view->save_view();
        view->cache_cleanup();
    }

    // log_message(conf, "SfM reconstruction done.\n");

    return 0;
}
