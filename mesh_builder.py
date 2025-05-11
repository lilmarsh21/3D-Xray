import open3d as o3d
import numpy as np

def merge_and_save_point_clouds(depth_maps, output_path):
    all_points = []

    for depth in depth_maps:
        h, w = depth.shape
        fx = fy = 500
        cx, cy = w / 2, h / 2

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        z = depth.astype(np.float32)
        x = (x - cx) * z / fx
        y = (y - cy) * z / fy

        points = np.stack((x, y, z), axis=2).reshape(-1, 3)
        all_points.append(points)

    all_points = np.concatenate(all_points, axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd = pcd.voxel_down_sample(voxel_size=0.01)
    o3d.io.write_point_cloud(output_path, pcd)