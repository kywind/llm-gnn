
import os
import numpy as np
import cv2
import torch
from PIL import Image
import open3d as o3d
import pymeshfix
import trimesh

from data.utils import load_yaml, set_seed, fps, fps_rad, recenter, \
        opengl2cam, depth2fgpcd, pcd2pix, find_relations_neighbor, label_colormap

from visualize import visualize_o3d

class MultiviewParticleDataset:

    def __init__(self, args, depths=None, masks=None, rgbs=None, cams=None, 
            text_labels_list=None, material_dict=None, vis_dir=None, visualize=False, verbose=False):
        self.args = args
        self.depths = depths  # list of PIL depth images
        self.masks = masks  # list of masks
        self.rgbs = rgbs  # list of PIL rgb images
        self.cam_params, self.cam_extrinsics = cams  # (4,) * n_cameras, (4, 4) * n_cameras
        self.text_labels = text_labels_list  # list of text labels
        self.material_dict = material_dict  # dict of {obj_name: material_name}

        # self.global_scale = 24
        # self.depth_thres = 0.599 / 0.8
        self.particle_num = 50
        self.adj_thresh = 0.1
        self.vis_dir = vis_dir
        self.visualize = visualize
        self.verbose = verbose

        self.n_cameras = len(self.depths)

        # change coordinates to world frame
        pcds = self.depth_to_pcd()

        # remove outliers based on table plane (RANSAC)
        pcds = self.remove_outliers(pcds)

        # find corresponding objects in different views
        global_pcds, global_labels, global_ids = self.pcd_grouping(pcds)

        # merge objects from different views
        objs = self.merge_views(global_pcds)

        # merge invisible points to reduce ovelapping effects
        # objs = self.remove_invisible_points(objs)

        save_pcd = True
        if save_pcd:
            self.save_view_pcd(pcds)
            self.save_global_pcd(objs, global_labels)

        self.objs = objs
        self.labels = global_labels

        # mesh reconstruction
        meshes = self.mesh_reconstruction()

        # sampling particles
        particle_pcds = self.sample_particles_from_mesh(meshes)

    def depth_to_pcd(self):
        pcd_list_all = []
        for camera_index in range(self.n_cameras):
            depth = np.array(self.depths[camera_index])
            depth = depth * 0.001
            rgb = np.array(self.rgbs[camera_index]) / 255.0
            masks = np.array(self.masks[camera_index])
            cam_param = self.cam_params[camera_index]
            cam_extrinsic = self.cam_extrinsics[camera_index]
            pcd_list, pcd_rgb_list, pcd_normal_list = self.parse_pcd(depth, masks, rgb, cam_param, cam_extrinsic)
            pcds = []
            for obj_index in range(len(pcd_list)):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pcd_list[obj_index])
                pcd.colors = o3d.utility.Vector3dVector(pcd_rgb_list[obj_index])
                # pcd.normals = o3d.utility.Vector3dVector(pcd_normal_list[obj_index])
                pcd = self.estimate_normal(pcd, cam_extrinsic)
                pcds.append(pcd)
            pcd_list_all.append(pcds)
        return pcd_list_all

    def estimate_normal(self, pcd, cam_extrinsic):
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        # make normals consistent
        pcd.orient_normals_consistent_tangent_plane(100)

        # normal in camera frame should point to the camera
        # normals = normals * np.where(normals_cam[:, 2:] > 0, -1, 1)
        normals = np.array(pcd.normals)
        normals_cam = cam_extrinsic @ np.hstack((normals, np.ones((normals.shape[0], 1)))).T
        normals_cam = normals_cam[:3, :].T
        if np.where(normals_cam[:, 2:] > 0, 1, 0).sum() > 0.5 * normals_cam.shape[0]:
            normals = normals * -1
        pcd.normals = o3d.utility.Vector3dVector(normals)
        return pcd
    
    def remove_outliers(self, pcd_list_all):
        total_pcd = o3d.geometry.PointCloud()
        for i in range(self.n_cameras):
            pcd_list = pcd_list_all[i]
            for j in range(len(pcd_list)):
                pcd = pcd_list[j]
                if pcd is None:
                    continue
                total_pcd += pcd
        
        total_pcd_np = np.array(total_pcd.points)

        # RANSAC to get the plane
        n_ransac_sample = 8
        n_ransac_iter = 100
        dist_thres = 0.3
        for i in range(n_ransac_iter):
            indices = np.random.choice(total_pcd_np.shape[0], n_ransac_sample, replace=False)
            pcd_ransac = total_pcd_np[indices]
            norm, intercept = self.compute_plane(pcd_ransac)
            dist = np.abs(np.matmul(total_pcd_np, norm) - intercept)  # distance to the plane
            inliers = total_pcd_np[dist < dist_thres]
            if inliers.shape[0] / total_pcd_np.shape[0] > 0.98:
                break
        else:
            raise AssertionError("RANSAC failed")
        
        # remove outliers based on the plane and the distance threshold
        for i in range(self.n_cameras):
            for j in range(len(pcd_list_all[i])):
                pcd = pcd_list_all[i][j]
                if pcd is None:
                    continue
                dist = np.abs(np.matmul(np.array(pcd.points), norm) - intercept)
                inliers_idx = np.where(dist < dist_thres)[0]
                pcd = pcd.select_by_index(inliers_idx)
                pcd_list_all[i][j] = pcd

        return pcd_list_all

    def merge_views(self, global_pcd_list):  
        ## using open3d remove statistical outliers
        objs = []
        for i in range(len(global_pcd_list)):
            obj = o3d.geometry.PointCloud()
            for j in range(len(global_pcd_list[i])):
                obj_pcd = global_pcd_list[i][j]
                if obj_pcd is None: continue

                # local refinement based on icp
                # if len(obj.points) > 0: # first view as the reference
                #     obj_pcd = self.local_refinement(obj_pcd, obj)

                obj += obj_pcd

            outliers = None
            new_outlier = None
            # remove until there's no new outlier
            rm_iter = 0
            # if True:
            while new_outlier is None or len(new_outlier.points) > 0:
                _, inlier_idx = obj.remove_statistical_outlier(
                    nb_neighbors=100, std_ratio=1.5+0.5*rm_iter
                )
                new_obj = obj.select_by_index(inlier_idx)
                new_outlier = obj.select_by_index(inlier_idx, invert=True)
                if outliers is None:
                    outliers = new_outlier
                else:
                    outliers += new_outlier
                obj = new_obj
                rm_iter += 1
                if self.verbose:
                    print("object {}, iter {}, removed {} outliers".format(i, rm_iter, len(new_outlier.points)))

            objs.append(obj)
        
            # if self.visualize:
            #     outliers.paint_uniform_color([0.0, 0.8, 0.0])
            #     visualize_o3d([obj, outliers], title="obj_{} and outliers".format(i))

        return objs  # no need to separate views

    def compute_plane(self, pcd):
        # pcd: [n, 3]
        # apply PCA to get the normal of the plane and the distance to the origin
        n = pcd.shape[0]
        pcd_mean = np.mean(pcd, axis=0)
        pcd_centered = pcd - pcd_mean
        cov = np.matmul(pcd_centered.T, pcd_centered) / n
        eig_val, eig_vec = np.linalg.eig(cov)
        eig_vec = eig_vec[:, np.argmin(eig_val)]
        eig_vec = eig_vec / np.linalg.norm(eig_vec)

        # calculate the distance to the origin
        dist = np.matmul(pcd_mean, eig_vec)
        return eig_vec, dist

    def pcd_grouping(self, pcd_list_all):
        global_pcd_list = []  # num_objects * num_cameras * (num_points, 3)
        global_id_list = []  # num_objects * num_cameras * [i, j]
        global_label_list = []  # num_objects

        for i in range(self.n_cameras):
            picked_k = []  # store the index of already merged objects in this view
            for j in range(len(pcd_list_all[i])):
                pcd = pcd_list_all[i][j]

                # check whether pcd belongs to an object
                dist_list = []  # store distance between pcd and each existing object
                for k in range(len(global_pcd_list)): 
                    if k in picked_k:  # already merged
                        dist_list.append(100000)
                        continue
                    view_dist_list = []  # store distance between pcd and each view of the object
                    for l in range(len(global_pcd_list[k])):
                        obj_pcd = global_pcd_list[k][l]  # the view's point cloud
                        if obj_pcd is None: # if no point cloud in this view
                            continue
                        # calculate distance between pcd and obj_pcd
                        dist = pcd.compute_point_cloud_distance(obj_pcd)
                        view_dist_list.append(dist)
                    assert len(view_dist_list) > 0

                    view_dist_list = np.array(view_dist_list)
                    dist_list.append(np.mean(view_dist_list))  # mean distance over views

                dist_list = np.array(dist_list)

                # merge objects by distance threshold
                if len(dist_list) == 0 or np.min(dist_list) > 100:
                    global_pcd_list.append([])  # new object
                    global_pcd_list[-1].extend([None] * i)  # fill to same length
                    global_pcd_list[-1].extend([pcd])

                    global_id_list.append([])  # new object
                    global_id_list[-1].extend([None] * i)  # fill to same length
                    global_id_list[-1].extend([(i, j)])
                    
                    # store the id of the new object
                    global_label_list.append(self.text_labels[i][j])
                    picked_k.append(len(global_pcd_list) - 1)

                else:
                    min_dist_index = np.argmin(dist_list)  # closest object's index

                    # check if id match
                    text_label = self.text_labels[i][j]
                    min_dist_text_label = global_label_list[min_dist_index]

                    if text_label == min_dist_text_label:
                        global_pcd_list[min_dist_index].append(pcd)
                        picked_k.append(min_dist_index)
                    else:
                        global_pcd_list.append([])  # new object
                        global_pcd_list[-1].extend([None] * i)  # fill to same length
                        global_pcd_list[-1].extend([pcd])

                        global_id_list.append([])  # new object
                        global_id_list[-1].extend([None] * i)  # fill to same length
                        global_id_list[-1].extend([(i, j)])

                        # store the id of the new object
                        global_label_list.append(self.text_labels[i][j])
                        picked_k.append(len(global_pcd_list) - 1)

            # fill to same length
            for m in range(len(global_pcd_list)):
                global_pcd_list[m].extend([None] * (i + 1 - len(global_pcd_list[m])))
                global_id_list[m].extend([None] * (i + 1 - len(global_id_list[m])))

        return global_pcd_list, global_label_list, global_id_list

    def save_view_pcd(self, pcd_list_all):
        for camera_index in range(self.n_cameras):
            for pcd_index in range(len(pcd_list_all[camera_index])):
                pcd = pcd_list_all[camera_index][pcd_index]
                o3d.io.write_point_cloud(
                    os.path.join(self.vis_dir, "pcd_{}_{}.pcd".format(camera_index, pcd_index)), pcd)
                # if self.visualize:
                #     visualize_o3d([pcd], title="pcd_{}_{}".format(camera_index, pcd_index))

    def save_global_pcd(self, objs, global_label_list):
        # concat each pcd that blongs to the same object
        n_obj = len(objs)
        for i in range(n_obj):
            obj = objs[i]
            o3d.io.write_point_cloud(
                os.path.join(self.vis_dir, "obj_{}.pcd".format(i)), obj)
            if self.visualize:
                visualize_o3d([obj], title="obj_{} ({})".format(i, global_label_list[i]))

            # save object text label
            with open(os.path.join(self.vis_dir, "obj_label_{}.txt".format(i)), "w") as f:
                f.write(global_label_list[i])

        # save global pcd
        global_pcd_o3d = o3d.geometry.PointCloud()
        for i in range(n_obj):
            obj = objs[i]
            if obj is None: continue
            global_pcd_o3d += obj
        o3d.io.write_point_cloud(
            os.path.join(self.vis_dir, "global_pcd.pcd"), global_pcd_o3d)
        if self.visualize:
            visualize_o3d([global_pcd_o3d], title="global_pcd_o3d")
        print("saved {} objects".format(n_obj))

    def parse_pcd(self, depth, masks, rgb, cam_param, cam_extrinsic):
        pcd_list = []
        pcd_rgb_list = []
        pcd_normal_list = []
        for i in range(masks.shape[0]):
            mask = masks[i]
            mask = np.logical_and(mask, depth > 0)

            # to camera frame
            fgpcd = np.zeros((mask.sum(), 3))
            fx, fy, cx, cy = cam_param
            pos_x, pos_y = np.meshgrid(np.arange(mask.shape[1]), np.arange(mask.shape[0]))  # w, h
            pos_x = pos_x[mask]
            pos_y = pos_y[mask]
            fgpcd[:, 0] = (pos_x - cx) * depth[mask] / fx
            fgpcd[:, 1] = (pos_y - cy) * depth[mask] / fy
            fgpcd[:, 2] = depth[mask]

            # add rgb color
            fgpcd_color = np.zeros((mask.sum(), 3))
            fgpcd_color[:, 0] = rgb[:, :, 0][mask]
            fgpcd_color[:, 1] = rgb[:, :, 1][mask]
            fgpcd_color[:, 2] = rgb[:, :, 2][mask]

            # estimate normal from depth
            # fgpcd_normal = np.zeros((mask.sum(), 3))
            # depth *= 1000.0  # magnify depth
            # fgpcd_normal[:, 0] = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)[mask] * fx
            # fgpcd_normal[:, 1] = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)[mask] * fy
            # fgpcd_normal[:, 2] = -depth[mask]
            # fgpcd_normal = fgpcd_normal / np.linalg.norm(fgpcd_normal, axis=1, keepdims=True)
            # depth /= 1000.0  # recover depth

            # to world frame
            fgpcd = np.hstack((fgpcd, np.ones((fgpcd.shape[0], 1))))
            fgpcd = np.matmul(fgpcd, np.linalg.inv(cam_extrinsic).T)[:, :3]

            pcd_list.append(fgpcd)
            pcd_rgb_list.append(fgpcd_color)
            # pcd_normal_list.append(fgpcd_normal)
        return pcd_list, pcd_rgb_list, pcd_normal_list

    def remove_invisible_points(self, objs):
        for i in range(len(objs)):
            visible = np.arange(len(objs[i].points))
            for j in range(self.n_cameras):
                cam_extrinsic = self.cam_extrinsics[j]
                cam_location = np.matmul(cam_extrinsic, np.array([0, 0, 0, 1]))[:3]
                radius = np.linalg.norm(cam_location) * 100
                _, visible_indices = objs[i].hidden_point_removal(cam_location, radius)
                visible = np.intersect1d(visible, visible_indices)
            objs[i] = objs[i].select_by_index(visible)
        return objs

    def local_refinement(self, source, target, distance_threshold=0.001):
        # local refinement
        result = o3d.pipelines.registration.registration_icp(
            source, target, 
            max_correspondence_distance=distance_threshold, 
            init=np.eye(4)
        )
        transformation = result.transformation
        print(transformation)
        source.transform(transformation)
        return source

    def mesh_reconstruction(self):
        # convert pcd to mesh with alpha shape
        alpha = 0.15
        all_meshes = []
        for i in range(len(self.objs)):
            obj = self.objs[i]
            if obj is None: continue
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(obj, alpha)
            # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(obj,
            #         o3d.utility.DoubleVector([0.04, 0.08, 0.16]))
            # if we have good normals, we can use poisson reconstruction
            # mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(obj, depth=4)

            fix_mesh = False
            if fix_mesh:
                mf = pymeshfix.MeshFix(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
                mf.repair()
                mesh.vertices = o3d.utility.Vector3dVector(mf.v)
                mesh.triangles = o3d.utility.Vector3iVector(mf.f)
                # mesh = mesh.remove_degenerate_triangles()
                # mesh = mesh.remove_duplicated_triangles()
                # mesh = mesh.remove_duplicated_vertices()
                # mesh = mesh.remove_non_manifold_edges()
                # mesh = mesh.remove_unreferenced_vertices()
                # mesh = mesh.filter_smooth_simple(number_of_iterations=10)
                vertices = np.asarray(mesh.vertices)
                faces = np.asarray(mesh.triangles)
                vertices, faces = trimesh.remesh.subdivide_to_size(vertices, faces, 
                        max_edge=0.01, max_iter=10, return_index=False)
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(vertices)
                mesh.triangles = o3d.utility.Vector3iVector(faces)
                mesh = mesh.filter_smooth_simple(number_of_iterations=2)

            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()
            o3d.io.write_triangle_mesh(
                os.path.join(self.vis_dir, "obj_{}.ply".format(i)), mesh)
            all_meshes.append(mesh)
            if self.visualize:
                visualize_o3d([mesh], title="mesh_{}".format(i))
        if self.visualize:
            visualize_o3d(all_meshes, title="all_meshes")
        return all_meshes

    def sample_particles_from_mesh(self, meshes):
        # sample particles from mesh with poisson disk sampling
        particle_pcds = []
        color_map = label_colormap() / 255.0
        for i in range(len(meshes)):
            pcd = meshes[i].sample_points_poisson_disk(self.particle_num)
            particle_pcds.append(pcd)
            pcd.paint_uniform_color(color_map[i])
            if self.visualize:
                visualize_o3d([pcd], title="sampled pcd_{}".format(i))
        if self.visualize:
            visualize_o3d(particle_pcds, title="sampled pcds")
        return particle_pcds

    ### legacy functions ###

    def update(self, state):
        self.history_states.append(self.state.copy())
        self.state = state

    def get_grouping(self):
        n_instance = np.unique(self.pcd_label).shape[0]
        n_p = self.particle_num * n_instance
        p_instance = torch.ones((1, n_p, n_instance), dtype=torch.float32) # the group each particle belongs to
        p_rigid = torch.zeros((1, n_instance), dtype=torch.float32) # the rigidness of each group

        # for i in range(n_instance):
        #     # import ipdb; ipdb.set_trace()
        #     p_instance[0, :, i] = torch.tensor((self.pcd_label[:, 0] == i).astype(np.float32))

        return n_p, n_instance, p_instance, p_rigid

    def subsample(self, pcd, particle_num=None, particle_r=None, particle_den=None):
        if particle_num is not None:
            sampled_pts, particle_r = fps(pcd, particle_num)
            particle_den = 1 / particle_r ** 2
        elif particle_r is not None:
            sampled_pts = fps_rad(pcd, particle_r) # [particle_num, 3]
            particle_den = 1 / particle_r ** 2
        elif particle_den is not None:
            particle_r = 1 / np.sqrt(particle_den)
            sampled_pts = fps_rad(pcd, particle_r) # [particle_num, 3]
        else:
            raise AssertionError("No subsampling method specified")
        particle_num = sampled_pts.shape[0]
        sampled_pts = recenter(pcd, sampled_pts, r = min(0.02, 0.5 * particle_r)) # [particle_num, 3]
        return sampled_pts, particle_num, particle_r, particle_den

    def generate_relation(self):
        args = self.args
        B = 1
        N = self.particle_num * self.n_instance
        rels = []

        s_cur = torch.tensor(self.state).unsqueeze(0)
        s_delta = torch.tensor(self.action).unsqueeze(0)

        # s_receiv, s_sender: B x particle_num x particle_num x 3
        s_receiv = (s_cur + s_delta)[:, :, None, :].repeat(1, 1, N, 1)
        s_sender = (s_cur + s_delta)[:, None, :, :].repeat(1, N, 1, 1)

        # dis: B x particle_num x particle_num
        # adj_matrix: B x particle_num x particle_num
        threshold = self.adj_thresh * self.adj_thresh
        dis = torch.sum((s_sender - s_receiv)**2, -1)
        max_rel = min(10, N)
        topk_res = torch.topk(dis, k=max_rel, dim=2, largest=False)
        topk_idx = topk_res.indices
        topk_bin_mat = torch.zeros_like(dis, dtype=torch.float32)
        topk_bin_mat.scatter_(2, topk_idx, 1)
        adj_matrix = ((dis - threshold) < 0).float()
        adj_matrix = adj_matrix * topk_bin_mat

        n_rels = adj_matrix.sum(dim=(1,2))
        n_rel = n_rels.max().long().item()
        rels_idx = []
        rels_idx = [torch.arange(n_rels[i]) for i in range(B)]
        rels_idx = torch.hstack(rels_idx).to(dtype=torch.long)
        rels = adj_matrix.nonzero()
        Rr = torch.zeros((B, n_rel, N), dtype=s_cur.dtype)
        Rs = torch.zeros((B, n_rel, N), dtype=s_cur.dtype)
        Rr[rels[:, 0], rels_idx, rels[:, 1]] = 1  # batch_idx, rel_idx, receiver_particle_idx
        Rs[rels[:, 0], rels_idx, rels[:, 2]] = 1  # batch_idx, rel_idx, sender_particle_idx

        self.Rr = Rr.squeeze(0).numpy() # n_rel, n_particle
        self.Rs = Rs.squeeze(0).numpy() # n_rel, n_particle
        return self.Rr, self.Rs

    def parse_action(self, action):
        # action: [start_x, start_y, end_x, end_y]
        s = action[:2]
        e = action[2:]
        h = 0.0
        pusher_w = 0.8 / 24.0
        s_3d = np.array([s[0], h, -s[1]])
        e_3d = np.array([e[0], h, -e[1]])
        s_3d_cam = opengl2cam(s_3d[None, :], self.cam_extrinsics, self.global_scale)[0]
        e_3d_cam = opengl2cam(e_3d[None, :], self.cam_extrinsics, self.global_scale)[0]
        push_dir_cam = e_3d_cam - s_3d_cam
        push_l = np.linalg.norm(push_dir_cam)
        push_dir_cam = push_dir_cam / push_l
        assert abs(push_dir_cam[2]) < 1e-6

        push_dir_ortho_cam = np.array([-push_dir_cam[1], push_dir_cam[0], 0.0])
        pos_diff_cam = self.state - s_3d_cam[None, :] # [particle_num, 3]
        pos_diff_ortho_proj_cam = (pos_diff_cam * np.tile(push_dir_ortho_cam[None, :], (self.particle_num * self.n_instance, 1))).sum(axis=1) # [particle_num,]
        pos_diff_proj_cam = (pos_diff_cam * np.tile(push_dir_cam[None, :], (self.particle_num * self.n_instance, 1))).sum(axis=1) # [particle_num,]
        pos_diff_l_mask = ((pos_diff_proj_cam < push_l) & (pos_diff_proj_cam > 0.0)).astype(np.float32) # hard mask
        pos_diff_w_mask = np.maximum(np.maximum(-pusher_w - pos_diff_ortho_proj_cam, 0.), # soft mask
                                    np.maximum(pos_diff_ortho_proj_cam - pusher_w, 0.))
        pos_diff_w_mask = np.exp(-pos_diff_w_mask / 0.01) # [particle_num,]
        pos_diff_to_end_cam = (e_3d_cam[None, :] - self.state) # [particle_num, 3]
        pos_diff_to_end_cam = (pos_diff_to_end_cam * np.tile(push_dir_cam[None, :], (self.particle_num * self.n_instance, 1))).sum(axis=1) # [particle_num,]
        states_delta = pos_diff_to_end_cam[:, None] * push_dir_cam[None, :] * pos_diff_l_mask[:, None] * pos_diff_w_mask[:, None]

        self.action = states_delta
        return states_delta
