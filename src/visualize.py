import open3d as o3d
import numpy as np


def visualize_o3d(
    geometry_list,
    title="O3D",
    view_point=None,
    point_size=5,
    pcd_color=[0, 0, 0],
    mesh_color=[0.5, 0.5, 0.5],
    show_normal=False,
    show_frame=True,
    path="",
):
    vis = o3d.visualization.Visualizer()
    vis.create_window(title)
    types = []

    for geometry in geometry_list:
        type = geometry.get_geometry_type()
        # Point Cloud
        # if type == o3d.geometry.Geometry.Type.PointCloud:
        #     geometry.paint_uniform_color(pcd_color)
        # Triangle Mesh
        if type == o3d.geometry.Geometry.Type.TriangleMesh:
            geometry.paint_uniform_color(mesh_color)
        types.append(type)

        vis.add_geometry(geometry)
        vis.update_geometry(geometry)

    vis.get_render_option().background_color = np.array([1.0, 1.0, 1.0])
    if show_frame:
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
        vis.add_geometry(mesh)
        vis.update_geometry(mesh)

    if o3d.geometry.Geometry.Type.PointCloud in types:
        vis.get_render_option().point_size = point_size
        vis.get_render_option().point_show_normal = show_normal
    if o3d.geometry.Geometry.Type.TriangleMesh in types:
        vis.get_render_option().mesh_show_back_face = True
        vis.get_render_option().mesh_show_wireframe = True

    vis.poll_events()
    vis.update_renderer()

    if view_point is None:
        vis.get_view_control().set_front(np.array([0.305, -0.463, 0.832]))
        vis.get_view_control().set_lookat(np.array([0.4, -0.1, 0.0]))
        vis.get_view_control().set_up(np.array([-0.560, 0.620, 0.550]))
        vis.get_view_control().set_zoom(0.2)
    else:
        vis.get_view_control().set_front(view_point["front"])
        vis.get_view_control().set_lookat(view_point["lookat"])
        vis.get_view_control().set_up(view_point["up"])
        vis.get_view_control().set_zoom(view_point["zoom"])

    # cd = os.path.dirname(os.path.realpath(sys.argv[0]))
    # path = os.path.join(cd, '..', 'figures', 'images', f'{title}_{datetime.now().strftime("%b-%d-%H:%M:%S")}.png')

    if len(path) > 0:
        vis.capture_screen_image(path, True)
        vis.destroy_window()
    else:
        vis.run()


if __name__ == "__main__":
    pcd_o3d = o3d.io.read_point_cloud(
        f"/home/zhangkaifeng/projects/llm-gnn/src/vis/multiview-0-debug/global_pcd.pcd")
    o3d.visualization.draw_geometries([pcd_o3d])

    for i in range(8):
        pcd_o3d = o3d.io.read_point_cloud(
            f"/home/zhangkaifeng/projects/llm-gnn/src/vis/multiview-0-debug/obj_{i}.pcd")
        o3d.visualization.draw_geometries([pcd_o3d])
