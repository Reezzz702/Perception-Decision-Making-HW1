import numpy as np
from PIL import Image
import numpy as np
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import os
import csv
import open3d as o3d
import copy
from tqdm import tqdm


SIZE = 512
fov = 90 / 180 * np.pi
focal_length = SIZE / (2 * np.tan(fov / 2.))
depth_scale = 1000
voxel_size = 0.00005

def transform_depth(image):
    img = np.asarray(image, dtype = np.float32)
    depth_img = (img / 255 * 10)
    depth_img = o3d.geometry.Image(depth_img)
    return depth_img

def depth_to_point_cloud(rgb, depth):
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, convert_rgb_to_intensity=False)
    
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(SIZE, SIZE, focal_length, focal_length, SIZE/2, SIZE/2))

    return pcd
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # o3d.visualization.draw_geometries([pcd])

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def preprocess_point_cloud(pcd, voxel_size):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, global_transformation):
    distance_threshold = voxel_size * 0.4
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, global_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result


data_size = len(os.listdir('data/task2/depth'))
pcd_list = []
transformation_seq = []
whole_scene_pcd = o3d.geometry.PointCloud() 

for i in tqdm(range(170)):
# for i in range(20):
    target_rgb = o3d.io.read_image(f"data/task2/rgb/rgb_{i}.png")
    target_depth = transform_depth(o3d.io.read_image(f"data/task2/depth/depth_{i}.png"))
        
    source_rgb = o3d.io.read_image(f"data/task2/rgb/rgb_{i + 1}.png")
    source_depth = transform_depth(o3d.io.read_image(f"data/task2/depth/depth_{i + 1}.png"))


    target_pcd = depth_to_point_cloud(target_rgb, target_depth)
    source_pcd = depth_to_point_cloud(source_rgb, source_depth)

    target_pcd_down, target_pcd_fpfh = preprocess_point_cloud(target_pcd, voxel_size)
    source_pcd_down, source_pcd_fpfh = preprocess_point_cloud(source_pcd, voxel_size)


    # visualization downsample point cloud
    # target_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # o3d.visualization.draw_geometries([target_pcd.Image])

    result_ransac = execute_global_registration(source_pcd_down, target_pcd_down,
                                            source_pcd_fpfh, target_pcd_fpfh,
                                            voxel_size)
    result_icp = refine_registration(source_pcd_down, target_pcd_down, source_pcd_fpfh, target_pcd_fpfh,
                                 voxel_size, result_ransac.transformation)
    # print(result_icp)
    if i == 0:
        target_pcd_down.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        points = np.asarray(target_pcd_down.points)
        target_pcd_down = target_pcd_down.select_by_index(np.where(points[:,1] < 0.0005)[0])
        pcd_list.append(target_pcd_down)
        whole_scene_pcd += target_pcd_down

    transformation_seq.insert(0, result_icp.transformation)
    for transformation in transformation_seq:
        source_pcd_down.transform(transformation)

    source_pcd_down.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    points = np.asarray(source_pcd_down.points)
    source_pcd_down = source_pcd_down.select_by_index(np.where(points[:,1] < 0.0005)[0])
    whole_scene_pcd += source_pcd_down
    pcd_list.append(source_pcd_down)   

print(transformation_seq[0])
# for pcd in pcd_list:
#     pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
# o3d.visualization.draw_geometries(pcd_list)
o3d.visualization.draw_geometries([whole_scene_pcd])