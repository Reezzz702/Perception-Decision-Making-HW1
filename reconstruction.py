import numpy as np
import numpy as np
import os
import csv
import open3d as o3d
import copy
from tqdm import tqdm
from argparse import ArgumentParser
from sklearn.neighbors import NearestNeighbors


SIZE = 512
fov = 90 / 180 * np.pi
f = SIZE / (2 * np.tan(fov / 2.0))
depth_scale = 1000
K = np.array([[f, 0, SIZE / 2], [0, f, SIZE / 2], [0, 0, 1]])
K_inverse = np.linalg.inv(K)


def transform_depth(image):
    img = np.asarray(image, dtype=np.float32)
    depth_img = img / 255 * 10
    depth_img = o3d.geometry.Image(depth_img)
    return depth_img


def depth_to_point_cloud(rgb, depth):
    colors = np.zeros((512*512, 3))
    points = np.zeros((512*512, 3))
    u = np.array([range(512)]*512).reshape(512,512) - 256
    v = np.array([[i]*512 for i in range(512)]).reshape(512,512) - 256
    z = np.asarray(depth)
    colors = (np.asarray(rgb)/255).reshape(512*512, 3)
    points[:, 0] = (u * z / f).reshape(512*512)
    points[:, 1] = (v * z / f).reshape(512*512)
    points[:, 2] = z.reshape(512*512)
    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.select_by_index(np.where(points[:, 2] != 0)[0])
    pcd.transform(np.array(([[1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, -1, 0],
                            [0, 0, 0, 1]])))
    #o3d.visualization.draw_geometries([pcd])
    return pcd


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, pcd_fpfh


def execute_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, voxel_size
):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result


def refine_registration(
    source, target, source_fpfh, target_fpfh, voxel_size, global_transformation
):
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_threshold,
        global_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    return result


def best_fit_transform(A, B):
    """
    Calculates the least-squares best-fit transform between corresponding 3D points A->B
    Input:
      A: Nx3 numpy array of corresponding 3D points
      B: Nx3 numpy array of corresponding 3D points
    Returns:
      T: 4x4 homogeneous transformation matrix
      R: 3x3 rotation matrix
      t: 3x1 column vector
    """

    assert len(A) == len(B)

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    W = np.dot(BB.T, AA)
    U, s, VT = np.linalg.svd(W)
    R = np.dot(U, VT)

    # special reflection case
    if np.linalg.det(R) < 0:
        VT[2, :] *= -1
        R = np.dot(U, VT)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t

    return T, R, t


def nearest_neighbor(src, dst):
    """
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nx3 array of points
        dst: Nx3 array of points
    Output:
        distances: Euclidean distances (errors) of the nearest neighbor
        indecies: dst indecies of the nearest neighbor
    """
    # print(src.shape[0])
    indecies = np.zeros(src.shape[0], dtype=np.int)
    distances = np.zeros(src.shape[0])
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(src)
    distances, indices = nbrs.kneighbors(dst)
    # for i, s in enumerate(src):
    #     min_dist = np.inf
    #     for j, d in enumerate(dst):
    #         dist = np.linalg.norm(s - d)
    #         if dist < min_dist:
    #             min_dist = dist
    #             indecies[i] = j
    #             distances[i] = dist
    return distances, indecies


def ICP(source, target, init_pose = None, max_iterations=500, voxel_size=1):
    """
    The Iterative Closest Point method
    Input:
        A: Nx3 numpy array of source 3D points
        B: Nx3 numpy array of destination 3D point
        init_pose: 4x4 homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation
    """
    tolerance = voxel_size * 0.4
    # make points homogeneous, copy them so as to maintain the originals
    src = np.ones((4, source.shape[0]))
    dst = np.ones((4, target.shape[0]))
    src[0:3, :] = np.copy(source.T)
    dst[0:3, :] = np.copy(target.T)

    # apply the initial pose estimation
    src = np.dot(init_pose, src)

    for i in range(max_iterations):
        # find the nearest neighbours between the current source and destination points
        distances, indices = nearest_neighbor(src[0:3, :].T, dst[0:3, :].T)

        # compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform(src[0:3, :].T, dst[0:3, indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        # print(mean_error)
        if abs(mean_error) < tolerance:
            print("early stop")
            break

    # calculcate final transformation
    T, _, _ = best_fit_transform(source, src[0:3, :].T)

    return T


def main(args):
    # parameters
    voxel_size = args.voxel_size
    data_size = len(os.listdir(f"data/task2/floor{args.floor}/depth"))

    GT_pose = []
    GT_link = []
    with open(f"data/task2/floor{args.floor}/GT/GT.csv", "r") as f:
        csvreader = csv.reader(f)
        next(csvreader)
        first_row = next(csvreader)
        first_coor = None
        count = 1
        for i in range(3):
            first_row[i] = float(first_row[i])
            first_coor = np.asarray(first_row[:3], dtype=np.float32)
            coor = first_coor - first_coor
        GT_pose.append(coor)

        for row in csvreader:
            for i in range(3):
                row[i] = float(row[i])
                coor = np.asarray(row[:3], dtype=np.float32)
                coor -= first_coor
            GT_pose.append(coor)
            GT_link.append([count - 1, count])
            count += 1
        f.close()

    # create GT lineset
    colors = [[0, 0, 0] for i in range(len(GT_link))]
    GT_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(GT_pose),
        lines=o3d.utility.Vector2iVector(GT_link),
    )
    GT_line_set.colors = o3d.utility.Vector3dVector(colors)

    transformation_seq = []
    whole_scene_pcd = o3d.geometry.PointCloud()
    prev_pcd_down = None
    prev_pcd_fpfh = None
    for i in tqdm(range(data_size - 1)):
        if i == 0:
            target_rgb = o3d.io.read_image(f"data/task2/floor{args.floor}/rgb/rgb_{i}.png")
            target_depth = transform_depth(
                o3d.io.read_image(f"data/task2/floor{args.floor}/depth/depth_{i}.png")
            )
            
            source_rgb = o3d.io.read_image(f"data/task2/floor{args.floor}/rgb/rgb_{i + 1}.png")
            source_depth = transform_depth(
                o3d.io.read_image(f"data/task2/floor{args.floor}/depth/depth_{i + 1}.png")
            )

            target_pcd = depth_to_point_cloud(target_rgb, target_depth)
            target_points = np.asarray(target_pcd.points)
            target_pcd = target_pcd.select_by_index(np.where(target_points[:, 1] < 0.5)[0])

            source_pcd = depth_to_point_cloud(source_rgb, source_depth)
            source_points = np.asarray(source_pcd.points)
            source_pcd = source_pcd.select_by_index(np.where(source_points[:, 1] < 0.5)[0])

            if args.icp == "own":
                # filter far points
                target_pcd_down, target_pcd_fpfh = preprocess_point_cloud(
                    target_pcd, voxel_size
                )
                source_pcd_down, source_pcd_fpfh = preprocess_point_cloud(
                    source_pcd, voxel_size
                )

                target_points = np.asarray(target_pcd_down.points)
                target_pcd_down = target_pcd_down.select_by_index(np.where(target_points[:,2] > -2.5)[0])
                source_points = np.asarray(source_pcd_down.points)
                source_pcd_down = source_pcd_down.select_by_index(np.where(source_points[:,2] > -2.5)[0])

            else:
                target_pcd_down, target_pcd_fpfh = preprocess_point_cloud(
                    target_pcd, voxel_size
                )
                source_pcd_down, source_pcd_fpfh = preprocess_point_cloud(
                    source_pcd, voxel_size
                )

        else:
            source_rgb = o3d.io.read_image(f"data/task2/floor{args.floor}/rgb/rgb_{i + 1}.png")
            source_depth = transform_depth(
                o3d.io.read_image(f"data/task2/floor{args.floor}/depth/depth_{i + 1}.png")
            )
            source_pcd = depth_to_point_cloud(source_rgb, source_depth)
            source_points = np.asarray(source_pcd.points)
            source_pcd = source_pcd.select_by_index(np.where(source_points[:, 1] < 0.5)[0])

            if args.icp == "own":
                source_pcd_down, source_pcd_fpfh = preprocess_point_cloud(
                    source_pcd, voxel_size
                )
                source_points = np.asarray(source_pcd_down.points)
                source_pcd_down = source_pcd_down.select_by_index(np.where(source_points[:,2] > -2.5)[0])
            else:
                source_pcd_down, source_pcd_fpfh = preprocess_point_cloud(
                    source_pcd, voxel_size
                )

            target_pcd_down = copy.deepcopy(prev_pcd_down)
            target_pcd_fpfh = copy.deepcopy(prev_pcd_fpfh)

        prev_pcd_down = copy.deepcopy(source_pcd_down)
        prev_pcd_fpfh = copy.deepcopy(source_pcd_fpfh)
        # o3d.visualization.draw_geometries([source_pcd_down])

        result_ransac = execute_global_registration(
            source_pcd_down,
            target_pcd_down,
            source_pcd_fpfh,
            target_pcd_fpfh,
            voxel_size,
        )
        # print(result_ransac.transformation)
        # get local icp transformation
        if args.icp == "open3d":
            result_icp = refine_registration(
                source_pcd_down,
                target_pcd_down,
                source_pcd_fpfh,
                target_pcd_fpfh,
                voxel_size,
                result_ransac.transformation,
            )
            result_transformation = result_icp.transformation
            # source_pcd.transform(result_transformation)
        elif args.icp == "own":
            # do gobal registration first
            # source_pcd_down.transform(result_ransac.transformation)
            # source_pcd.transform(result_ransac.transformation)
            result_transformation = ICP(
                np.asarray(source_pcd_down.points), np.asarray(target_pcd_down.points), init_pose = result_ransac.transformation, voxel_size = voxel_size
            )
            # source_pcd.transform(local_transformation)
            # result_transformation = local_transformation @ result_ransac.transformation

        else:
            print("unknown icp strategy")
            return

        if i == 0:
            whole_scene_pcd += target_pcd
            cur_transformation = result_transformation
        else:
            cur_transformation = transformation_seq[-1] @ result_transformation

        source_pcd.transform(cur_transformation)
        transformation_seq.append(cur_transformation)
        whole_scene_pcd += source_pcd
        
    # record reconstructed poses
    reconstruct_pose = [[0, 0, 0]]
    reconstruct_link = []
    for i, t in enumerate(transformation_seq):
        translation = [t[0, -1], t[1, -1], t[2, -1]]
        reconstruct_pose.append(translation)
        reconstruct_link.append([i, i + 1])
    reconstruct_link.pop(-1)

    
    # create reconstructed lineset
    colors = [[1, 0, 0] for i in range(len(reconstruct_link))]
    reconstruct_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(reconstruct_pose),
        lines=o3d.utility.Vector2iVector(reconstruct_link),
    )
    reconstruct_line_set.colors = o3d.utility.Vector3dVector(colors)
    
    # show result
    o3d.visualization.draw_geometries(
        [whole_scene_pcd, reconstruct_line_set, GT_line_set]
    )

    # calculate mean L2 norm
    GT = np.asarray(GT_pose)
    reconstruct = np.asarray(reconstruct_pose)
    print(np.mean(np.linalg.norm(GT - reconstruct, axis = 1)))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-i", dest="icp", help="select ipc strategy: [open3d, own]", type=str
    )
    parser.add_argument("-v", dest="voxel_size", help="voxel size", type=float)
    parser.add_argument("-f", dest="floor", help="select floor: [0, 1]", type=int)
    main(parser.parse_args())
