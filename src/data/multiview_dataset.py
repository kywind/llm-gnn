
import os
import numpy as np
import cv2
import torch
from PIL import Image
import open3d as o3d
import pymeshfix
import trimesh

from data.utils import load_yaml, set_seed, fps, fps_rad, recenter, \
        opengl2cam, cam2opengl, depth2fgpcd, pcd2pix, find_relations_neighbor, label_colormap

from visualize import visualize_o3d

class MultiviewParticleDataset:

    def __init__(self, args, depths=None, masks=None, rgbs=None, cams=None, text_labels_list=None, 
            material_dict=None, vis_dir=None, visualize=False, verbose=False, save=False):
        self.args = args
        self.depths = depths  # list of PIL depth images
        self.masks = masks  # list of masks
        self.rgbs = rgbs  # list of PIL rgb images
        self.cam_params, self.cam_extrinsics = cams  # (4,) * n_cameras, (4, 4) * n_cameras
        self.text_labels = text_labels_list  # list of text labels
        self.material_dict = material_dict  # dict of {obj_name: material_name}

        # self.global_scale = 24
        # self.depth_thres = 0.599 / 0.8
        # self.particle_num = 50
        self.adj_thresh = 0.02
        self.num_max_rel = 7  # maximum number of relations per particle
        self.vis_dir = vis_dir
        self.visualize = visualize
        self.verbose = verbose
        self.save = save

        self.n_cameras = len(self.depths)

        # change coordinates to world frame
        pcds = self.depth_to_pcd()

        # remove outliers based on table plane (RANSAC)
        pcds = self.remove_outliers(pcds)

        # points_list = []
        # for i in range(self.n_cameras):
        #     for j in range(len(pcds[i])):
        #         pcd = pcds[i][j]
        #         if pcd is None: continue
        #         points_list.append(np.array(pcd.points))
        # points = np.vstack(points_list)
        # print("points min, max: {}, {}".format(points.min(0), points.max(0)))

        # find corresponding objects in different views
        global_pcds, global_labels, global_ids = self.pcd_grouping(pcds)

        # merge objects from different views
        objs = self.merge_views(global_pcds)

        # merge invisible points to reduce ovelapping effects
        # objs = self.remove_invisible_points(objs)

        if save:
            self.save_view_pcd(pcds)
            self.save_global_pcd(objs, global_labels)

        self.objs = objs
        self.labels = global_labels

        # mesh reconstruction
        meshes = self.mesh_reconstruction()

        # sampling particles
        particle_pcds = self.sample_particles_from_mesh(meshes)


        # generate attributes
        # n_types = len(set(self.labels)) + 1  # including robot end effector
        self.particle_num = sum([np.array(pcd.points).shape[0] for pcd in particle_pcds])
        attrs = np.zeros((self.particle_num, args.attr_dim), dtype=np.float32) # [particle_num, attr_dim]
        # import ipdb; ipdb.set_trace()

        # generate relations
        Rr, Rs, rel_attrs = self.generate_relation(particle_pcds)
        if verbose: print("Rr shape: {}, Rs shape: {}".format(Rr.shape, Rs.shape))
        # import ipdb; ipdb.set_trace()
        
        # generate state
        state = np.vstack([np.array(pcd.points) for pcd in particle_pcds])
        if verbose: print("state shape: {}".format(state.shape))

        self.state = state
        self.attrs = attrs
        self.Rr = Rr
        self.Rs = Rs
        self.rel_attrs = rel_attrs


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
            if self.save:
                o3d.io.write_point_cloud(
                    os.path.join(self.vis_dir, "obj_{}.pcd".format(i)), obj)
            # if self.visualize:
            #     visualize_o3d([obj], title="obj_{} ({})".format(i, global_label_list[i]))

            # save object text label
            if self.save:
                with open(os.path.join(self.vis_dir, "obj_label_{}.txt".format(i)), "w") as f:
                    f.write(global_label_list[i])

        # save global pcd
        global_pcd_o3d = o3d.geometry.PointCloud()
        for i in range(n_obj):
            obj = objs[i]
            if obj is None: continue
            global_pcd_o3d += obj
        if self.save:
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

            # to world frame (opengl)
            # import ipdb; ipdb.set_trace()
            # fgpcd = cam2opengl(fgpcd, cam_extrinsic)

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
            if self.save:
                o3d.io.write_triangle_mesh(
                    os.path.join(self.vis_dir, "obj_{}.ply".format(i)), mesh)
            all_meshes.append(mesh)
            # if self.visualize:
            #     visualize_o3d([mesh], title="mesh_{}".format(i))
        if self.visualize:
            visualize_o3d(all_meshes, title="all_meshes")
        return all_meshes

    def sample_particles_from_mesh(self, meshes):
        # sample particles from mesh with poisson disk sampling
        particle_pcds = []
        color_map = label_colormap() / 255.0
        particle_r = self.adj_thresh
        for i in range(len(meshes)):
            surface_area = meshes[i].get_surface_area()
            particle_num = int(surface_area * 0.5 / (particle_r ** 2))
            pcd = meshes[i].sample_points_poisson_disk(particle_num)  # adaptive sampling
            particle_pcds.append(pcd)
            pcd.paint_uniform_color(color_map[i * 2])
            # if self.visualize:
            #     visualize_o3d([pcd], title="sampled pcd_{}".format(i))
        if self.visualize:
            visualize_o3d(particle_pcds, title="sampled pcds")
        return particle_pcds

    ### legacy functions ###

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

    def generate_relation(self, particle_pcds):
        Rs_list = []
        Rr_list = []
        n_points_list = []
        n_rels_list = []

        for pcd in particle_pcds:
            s_cur = torch.tensor(np.asarray(pcd.points))
            s_delta = torch.zeros_like(s_cur)
            n_points = s_cur.shape[0]

            # s_receiv, s_sender: particle_num x particle_num x 3
            s_receiv = (s_cur + s_delta)[:, None, :].repeat(1, n_points, 1)
            s_sender = (s_cur + s_delta)[None, :, :].repeat(n_points, 1, 1)

            # dis: particle_num x particle_num
            # adj_matrix: particle_num x particle_num
            threshold = (2.5 * self.adj_thresh) ** 2
            dis = torch.sum((s_sender - s_receiv) ** 2, -1)
            max_rel = min(self.num_max_rel, n_points)
            topk_res = torch.topk(dis, k=max_rel, dim=1, largest=False)
            topk_idx = topk_res.indices
            topk_bin_mat = torch.zeros_like(dis, dtype=torch.float32)  # particle_num x particle_num
            topk_bin_mat.scatter_(1, topk_idx, 1)
            adj_matrix = ((dis - threshold) < 0).float()  # particle_num x particle_num
            adj_matrix = adj_matrix * topk_bin_mat

            # remove self relations
            anti_self = torch.ones_like(adj_matrix) - torch.eye(n_points)
            adj_matrix = adj_matrix * anti_self

            # import ipdb; ipdb.set_trace()
            n_rel = adj_matrix.sum().long().item()
            rels_idx = torch.arange(n_rel).to(dtype=torch.long)
            rels = adj_matrix.nonzero()
            Rr = torch.zeros((n_rel, n_points), dtype=s_cur.dtype)
            Rs = torch.zeros((n_rel, n_points), dtype=s_cur.dtype)
            # import ipdb; ipdb.set_trace()
            Rr[rels_idx, rels[:, 0]] = 1  # el_idx, receiver_particle_idx
            Rs[rels_idx, rels[:, 1]] = 1  # el_idx, sender_particle_idx

            Rr = Rr.numpy() # n_rel, n_particle
            Rs = Rs.numpy() # n_rel, n_particle

            Rs_list.append(Rs)
            Rr_list.append(Rr)
            n_points_list.append(n_points)
            n_rels_list.append(n_rel)
        
        n_points_cum = np.cumsum(np.array([0] + n_points_list))
        n_rels_cum = np.cumsum(np.array([0] + n_rels_list))
        
        # print(n_points_cum)
        # print(n_rels_cum)
        # import ipdb; ipdb.set_trace()
        Rs_all = np.zeros((n_rels_cum[-1], n_points_cum[-1]), dtype=np.float32)
        Rr_all = np.zeros((n_rels_cum[-1], n_points_cum[-1]), dtype=np.float32)

        for i in range(len(Rs_list)):
            Rs_all[n_rels_cum[i]:n_rels_cum[i+1], n_points_cum[i]:n_points_cum[i+1]] = Rs_list[i]
            Rr_all[n_rels_cum[i]:n_rels_cum[i+1], n_points_cum[i]:n_points_cum[i+1]] = Rr_list[i]
        
        Rs_list = []
        Rr_list = []
        n_rels_list = []

        # generate cross-object relations (Rr abd Rs will have shape n_rel x n_particle_all)
        # import ipdb; ipdb.set_trace()
        for i in range(len(particle_pcds)):  # receiver
            pcd = particle_pcds[i]
            s_cur = torch.tensor(np.asarray(pcd.points))  # n_points x 3
            s_delta = torch.zeros_like(s_cur)
            n_points = s_cur.shape[0]
            for j in range(i+1, len(particle_pcds)):  # sender
                pcd_dist = np.asarray(pcd.compute_point_cloud_distance(particle_pcds[j]))
                if pcd_dist.min() > self.adj_thresh:
                    # Rs = np.zeros((0, n_points), dtype=s_cur.dtype)
                    # Rr = np.zeros((0, n_points), dtype=s_cur.dtype)
                    continue
                pcd2 = particle_pcds[j]
                s_cur2 = torch.tensor(pcd2.points)  # n_points x 3
                s_delta2 = torch.zeros_like(s_cur2)
                n_points2 = s_cur2.shape[0]
                r = (s_cur + s_delta)[:, None, :].repeat(1, n_points2, 1)  # n_points x n_points2 x 3
                s = (s_cur2 + s_delta2)[None, :, :].repeat(n_points, 1, 1)  # n_points x n_points2 x 3
                dis = torch.sum((r - s)**2, -1)  # n_points x n_points2

                threshold = self.adj_thresh * self.adj_thresh * 4
                max_rel = min(self.num_max_rel, min(n_points, n_points2))  # 5
                topk_res = torch.topk(dis, k=max_rel, dim=1, largest=False)  # n_points x max_rel
                topk_idx = topk_res.indices  # n_points x max_rel
                topk_bin_mat = torch.zeros_like(dis, dtype=torch.float32)  # n_points x n_points2
                topk_bin_mat.scatter_(1, topk_idx, 1)  # n_points x n_points2
                adj_matrix = ((dis - threshold) < 0).float()  # n_points x n_points2
                adj_matrix = adj_matrix * topk_bin_mat  # n_points x n_points2

                n_rel = adj_matrix.sum().long().item()
                rels_idx = torch.arange(n_rel).to(dtype=torch.long)
                rels = adj_matrix.nonzero()  # n_rel x 3, [receiver_idx, sender_idx]
                Rr = torch.zeros((n_rel, n_points_cum[-1]), dtype=s_cur.dtype)
                Rs = torch.zeros((n_rel, n_points_cum[-1]), dtype=s_cur.dtype)
                Rr[rels_idx, rels[:, 0] + n_points_cum[i]] = 1  # rel_idx, receiver_particle_idx
                Rs[rels_idx, rels[:, 1] + n_points_cum[j]] = 1  # rel_idx, sender_particle_idx

                Rr = Rr.numpy() # n_rel, n_particle
                Rs = Rs.numpy() # n_rel, n_particle

                Rs_list.append(Rs)
                Rr_list.append(Rr)
                n_rels_list.append(n_rel)

        n_rels_cum = np.cumsum(np.array([0] + n_rels_list))
        Rs_all_inter = np.zeros((n_rels_cum[-1], n_points_cum[-1]), dtype=np.float32)
        Rr_all_inter = np.zeros((n_rels_cum[-1], n_points_cum[-1]), dtype=np.float32)

        for i in range(len(Rs_list)):
            Rs_all_inter[n_rels_cum[i]:n_rels_cum[i+1], :] = Rs_list[i]
            Rr_all_inter[n_rels_cum[i]:n_rels_cum[i+1], :] = Rr_list[i]

        Rs = np.concatenate([Rs_all, Rs_all_inter], axis=0)
        Rr = np.concatenate([Rr_all, Rr_all_inter], axis=0)

        rel_attrs = np.zeros((Rs.shape[0], 1), dtype=np.float32)
        rel_attrs[:Rs_all.shape[0], 0] = 0.0  # intra-object relations
        rel_attrs[Rs_all.shape[0]:, 0] = 1.0  # inter-object relations

        # import ipdb; ipdb.set_trace()
        return Rr, Rs, rel_attrs

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
