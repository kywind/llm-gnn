import os
import torch
import numpy as np
import glob
from PIL import Image
import cv2
import pickle as pkl
import open3d as o3d

from config import gen_args
from data.utils import label_colormap
from gnn.model_wrapper import gen_model


def construct_edges_from_states(states, adj_thresh, exclude_last_N):  # helper function for construct_graph
    # :param states: (B, N, state_dim) torch tensor
    # :param adj_thresh: float
    # :param exclude_last_N: int exclude connection between last N particles (end effectors)
    # :return:
    # - Rr: (B, n_rel, N) torch tensor
    # - Rs: (B, n_rel, N) torch tensor
    B, N, state_dim = states.shape
    s_receiv = states[:, :, None, :].repeat(1, 1, N, 1)
    s_sender = states[:, None, :, :].repeat(1, N, 1, 1)

    # dis: B x particle_num x particle_num
    # adj_matrix: B x particle_num x particle_num
    threshold = adj_thresh * adj_thresh
    dis = torch.sum((s_sender - s_receiv)**2, -1)
    dis[:, -exclude_last_N:, -exclude_last_N:] = 1e10
    adj_matrix = ((dis - threshold) < 0).float()
    
    # add topk constraints
    topk = 5
    topk_idx = torch.topk(dis, k=topk, dim=-1, largest=False)[1]
    topk_matrix = torch.zeros_like(adj_matrix)
    topk_matrix.scatter_(-1, topk_idx, 1)
    adj_matrix = adj_matrix * topk_matrix
    
    n_rels = adj_matrix.sum(dim=(1,2))
    n_rel = n_rels.max().long().item()
    rels_idx = []
    rels_idx = [torch.arange(n_rels[i]) for i in range(B)]
    rels_idx = torch.hstack(rels_idx).to(device=states.device, dtype=torch.long)
    rels = adj_matrix.nonzero()
    Rr = torch.zeros((B, n_rel, N), device=states.device, dtype=states.dtype)
    Rs = torch.zeros((B, n_rel, N), device=states.device, dtype=states.dtype)
    Rr[rels[:, 0], rels_idx, rels[:, 1]] = 1
    Rs[rels[:, 0], rels_idx, rels[:, 2]] = 1
    return Rr, Rs


def postprocess_graph(args, data_dir, orig_data_dir=None):
    graph_dir = os.path.join(data_dir, f"pred_graphs_{orig_data_dir.split('/')[-1]}")
    save_dir = os.path.join(data_dir, f"vis_{orig_data_dir.split('/')[-1]}")
    os.makedirs(save_dir, exist_ok=True)

    graphs = sorted(list(glob.glob(os.path.join(graph_dir, '*.pkl'))))

    num_graphs = len(graphs)

    img_vis = True
    if img_vis:
        vis_dir = os.path.join("vis", f"vis_pred_{orig_data_dir.split('/')[-1]}")
        os.makedirs(vis_dir, exist_ok=True)
        t_list = os.path.join(orig_data_dir, "push_t_list.txt")
        save_interval = 1
        t_list = np.loadtxt(t_list)
        t_list = t_list[::save_interval]
        graph_frame_dict = {int(i): [int(t[0]), int(t[1])] for i, t in enumerate(t_list)}

    camera_indices = [0, 1, 2, 3]

    pts_prev_proj_pred = [None] * 4
    pts_prev_proj_orig = [None] * 4
    pts_prev_proj_gt = [None] * 4

    for graph_id in range(num_graphs):
        # get keypoints 
        graph_path = graphs[graph_id]
        graph = pkl.load(open(graph_path, 'rb')) # list of (rand_ptcl_num, 3)
        # import ipdb; ipdb.set_trace()
        # graph.keys: ['orig_state_p', 'orig_state_s', 'pred_state_p', 'gt_state_p', 'action', 'p_rigid', 'p_instance', 'Rr', 'Rs']

        orig_state_p = graph['orig_state_p']  # (N, 3)
        orig_state_s = graph['orig_state_s']  # (M, 3)
        pred_state_p = graph['pred_state_p']  # (N, 3)
        gt_state_p = graph['gt_state_p']  # (N, 3)
        action = graph['action']  # (M, 3)
        p_rigid = graph['p_rigid']
        p_instance = graph['p_instance']
        Rr = graph['Rr']
        Rs = graph['Rs']
        attrs = graph['attrs']

        orig_state = np.concatenate([orig_state_p, orig_state_s], axis=0)  # (N+M, 3)

        # filter invalid particles
        valid_idx = attrs.sum(1).nonzero()[0]
        valid_idx_p = attrs[:, :-1].sum(1).nonzero()[0]
        valid_idx_s = attrs[:, -1].nonzero()[0]
        orig_state_p_valid = orig_state_p[valid_idx_p]
        pred_state_p_valid = pred_state_p[valid_idx_p]
        gt_state_p_valid = gt_state_p[valid_idx_p]
        orig_state_valid = orig_state[valid_idx]
        orig_state_s_valid = orig_state[valid_idx_s]
        Rr = Rr[:, valid_idx]
        Rs = Rs[:, valid_idx]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(orig_state_valid)
        colors = np.zeros_like(orig_state_valid)
        colors[:len(valid_idx_p)] = [0, 1, 0]
        colors[len(valid_idx_p):] = [1, 0, 0]
        pcd.colors = o3d.utility.Vector3dVector(colors)

        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(pred_state_p_valid)
        pcd_pred.paint_uniform_color([0, 0, 1])

        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(gt_state_p_valid)
        pcd_gt.paint_uniform_color([1, 1, 0])

        lst = o3d.geometry.LineSet()
        lst.points = o3d.utility.Vector3dVector(orig_state_valid)
        lines = []
        line_colors = []
        for i in range(Rs.shape[0]):
            if Rs[i].sum() == 0: continue
            assert Rs[i].sum() == 1
            assert Rr[i].sum() == 1
            idx1 = Rs[i].argmax()
            idx2 = Rr[i].argmax()
            if idx1 >= len(valid_idx_p) or idx2 >= len(valid_idx_p): 
                assert not (idx1 >= len(valid_idx_p) and idx2 >= len(valid_idx_p))
                line_colors.append([1, 0, 0])
            else:
                line_colors.append([0, 1, 0])
            lines.append([idx1, idx2])
        lines = np.array(lines)
        lst.lines = o3d.utility.Vector2iVector(lines)
        lst.colors = o3d.utility.Vector3dVector(line_colors)

        # o3d.visualization.draw_geometries([pcd, pcd_pred, pcd_gt, lst])
        # o3d.visualization.draw_geometries([pcd, lst])

        # draw on images
        if img_vis:
            frame_id_start, frame_id_end = graph_frame_dict[graph_id]
            for cam_id in camera_indices:
                img_dir = os.path.join(orig_data_dir, f"camera_{cam_id}", "color", f"color_{frame_id_start}.png")
                intr = np.load(os.path.join(orig_data_dir, f"camera_{cam_id}", "camera_params.npy"))
                extr = np.load(os.path.join(orig_data_dir, f"camera_{cam_id}", "camera_extrinsics.npy"))
                fx, fy, cx, cy = intr
                img = cv2.imread(img_dir)

                vis_orig = False
                vis_pred = True
                vis_gt = False
                if vis_orig:
                    pts = orig_state_valid
                    pts_cam = np.matmul(extr, np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1).T).T
                    pts_cam = pts_cam[:, :3] / pts_cam[:, 3:]

                    pts_proj = np.zeros((pts_cam.shape[0], 2))
                    pts_proj[:, 0] = fx * pts_cam[:, 0] / pts_cam[:, 2] + cx
                    pts_proj[:, 1] = fy * pts_cam[:, 1] / pts_cam[:, 2] + cy
                    pts_proj = pts_proj.astype(np.int32)

                    for j in range(orig_state_p_valid.shape[0]):
                        cv2.circle(img, (pts_proj[j, 0], pts_proj[j, 1]), 3, (0, 0, 255), -1)
                    
                    for j in range(orig_state_p_valid.shape[0], orig_state_valid.shape[0]):
                        cv2.circle(img, (pts_proj[j, 0], pts_proj[j, 1]), 3, (0, 255, 255), -1)

                if vis_pred:
                    pts = pred_state_p_valid
                    pts_cam = np.matmul(extr, np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1).T).T
                    pts_cam = pts_cam[:, :3] / pts_cam[:, 3:]

                    pts_proj = np.zeros((pts_cam.shape[0], 2))
                    pts_proj[:, 0] = fx * pts_cam[:, 0] / pts_cam[:, 2] + cx
                    pts_proj[:, 1] = fy * pts_cam[:, 1] / pts_cam[:, 2] + cy
                    pts_proj = pts_proj.astype(np.int32)

                    for j in range(pts_proj.shape[0]):
                        cv2.circle(img, (pts_proj[j, 0], pts_proj[j, 1]), 3, (255, 0, 0), -1)
                    
                    if pts_prev_proj_pred[cam_id] is not None:
                        for j in range(pts_prev_proj_pred[cam_id].shape[0]):
                            cv2.circle(img, (pts_prev_proj_pred[cam_id][j, 0], pts_prev_proj_pred[cam_id][j, 1]), 3, (0, 255, 255), -1)
                    
                    pts_prev_proj_pred[cam_id] = pts_proj
                
                if vis_gt:
                    pts = gt_state_p_valid
                    pts_cam = np.matmul(extr, np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1).T).T
                    pts_cam = pts_cam[:, :3] / pts_cam[:, 3:]

                    pts_proj = np.zeros((pts_cam.shape[0], 2))
                    pts_proj[:, 0] = fx * pts_cam[:, 0] / pts_cam[:, 2] + cx
                    pts_proj[:, 1] = fy * pts_cam[:, 1] / pts_cam[:, 2] + cy
                    pts_proj = pts_proj.astype(np.int32)

                    for j in range(pts_proj.shape[0]):
                        cv2.circle(img, (pts_proj[j, 0], pts_proj[j, 1]), 3, (0, 255, 0), -1)

                cv2.imwrite(os.path.join(vis_dir, f"{graph_id}_{cam_id}.png"), img)



if __name__ == "__main__":
    args = gen_args()
    _ = gen_model(args, material_dict=None, debug=True)

    # shoe-pushing, sparse
    # data_dir = "../log/shoe_debug_6/"
    # orig_data_dir = "../data/2023-08-23-12-08-12-201998"  # not in training set
    # orig_data_dir = "../data/2023-09-04-18-42-27-707743"  # not in training set
    # orig_data_dir = "../data/2023-08-23-12-23-07-775716"  # in training set

    # shoe-pushing, dense
    # data_dir = "../log/shoe_debug_7/"
    # orig_data_dir = "../data/2023-08-23-12-08-12-201998"  # not in training set
    # orig_data_dir = "../data/2023-09-04-18-42-27-707743"  # not in training set
    # orig_data_dir = "../data/2023-08-23-12-23-07-775716"  # in training set

    # cloth-pushing
    # data_dir = "../log/shirt_debug_3"
    # orig_data_dir = "../data/shirt"

    # shoe pushing, new random dataset
    data_dir = "../log/rigid_debug_1"
    orig_data_dir = "../data/2023-08-23-12-08-12-201998"
    # orig_data_dir = "../data/2023-09-04-18-42-27-707743"

    postprocess_graph(args, data_dir, orig_data_dir)
