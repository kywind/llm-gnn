import open3d as o3d

pcd_o3d = o3d.io.read_point_cloud(
    f"/home/zhangkaifeng/projects/llm-gnn/src/vis/multiview-0/global_pcd.pcd")
o3d.visualization.draw_geometries([pcd_o3d])

for i in range(8):
    pcd_o3d = o3d.io.read_point_cloud(
        f"/home/zhangkaifeng/projects/llm-gnn/src/vis/multiview-0/obj_{i}.pcd")
    o3d.visualization.draw_geometries([pcd_o3d])
