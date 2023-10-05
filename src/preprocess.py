import os
import torch
import numpy as np
import glob
from PIL import Image
import cv2
import pickle as pkl

from config import gen_args
from data.utils import label_colormap

# img_dir = 'vis/apples_640.png'
# depth_dir = 'vis/apples_640_depth.npy'

# os.system(f"python MiDaS/run.py --input {img_dir} --output {depth_dir} --model MiDaS/weights/dpt_beit_large_512.pt")


def load_fingers(args):
    # filter raw data
    # generate save_dir/valid_frames.txt
    data_dir = "../data/2023-09-04-18-42-27-707743/"
    # save_dir = "../data/graph-2023-08-23-12-08-12-201998/"
    # dataset_name = "d3fields"

    # os.makedirs(save_dir, exist_ok=True)
    # frame_save_dir = os.path.join(save_dir, "valid_frames.txt")

    len_img_paths = len(list(glob.glob(os.path.join(data_dir, "camera_1/color/color_*.png"))))
    img_paths = [os.path.join(data_dir, "camera_1/color/color_{}.png".format(i)) for i in range(len_img_paths)]

    intr = np.load(os.path.join(data_dir, "camera_1/camera_params.npy"))
    extr = np.load(os.path.join(data_dir, "camera_1/camera_extrinsics.npy"))
    print(intr.shape, extr.shape)

    base_pose_in_tag = np.load(os.path.join(data_dir, "base_pose_in_tag.npy"))

    finger_points_in_finger_frame = np.array([[0.01,  -0.025, 0.014], # bottom right
                                              [-0.01, -0.025, 0.014], # bottom left
                                              [-0.01, -0.025, 0.051], # top left
                                              [0.01,  -0.025, 0.051]]) # top right

    # for i, img_path in enumerate(img_paths):
    i = 0
    img_path = img_paths[i]
    print(img_path)
    if True:
        left_finger_in_base_frame = np.loadtxt(os.path.join(data_dir, 'pose', 'left_finger', f'{i}.txt'))
        right_finger_in_base_frame = np.loadtxt(os.path.join(data_dir, 'pose', 'right_finger', f'{i}.txt'))

        left_finger_in_tag_frame = base_pose_in_tag @ left_finger_in_base_frame
        right_finger_in_tag_frame = base_pose_in_tag @ right_finger_in_base_frame

        N = finger_points_in_finger_frame.shape[0]
        finger_points_in_finger_frame_homo = np.concatenate([finger_points_in_finger_frame, np.ones((N, 1))], axis=1) # (N, 4)
        
        left_finger_points = left_finger_in_tag_frame @ finger_points_in_finger_frame_homo.T # (4, N)
        right_finger_points = right_finger_in_tag_frame @ finger_points_in_finger_frame_homo.T # (4, N)

        left_finger_points = left_finger_points.T # (N, 4)
        right_finger_points = right_finger_points.T # (N, 4)

        # img = Image.open(img_path).convert('RGB')
        # depth = Image.open(img_path.replace("color", "depth"))
        img = cv2.imread(img_path)

        # transform finger points to camera coordinate
        left_finger_points = left_finger_points @ extr.T  # (N, 4)
        right_finger_points = right_finger_points @ extr.T  # (N, 4)

        # get other quantities: grasper_dist, grasper_pose (used in grasp planner)
        grasper_dist = np.linalg.norm(left_finger_points - right_finger_points, axis=1).mean()
        robotiq_pose_in_base_frame = np.loadtxt(os.path.join(data_dir, 'pose', 'robotiq_base', f'{i}.txt'))
        robotiq_pose_in_tag_frame = base_pose_in_tag @ robotiq_pose_in_base_frame

        # project finger points
        fx, fy, cx, cy = intr
        left_finger_projs = np.zeros((N, 2))
        left_finger_projs[:, 0] = left_finger_points[:, 0] * fx / left_finger_points[:, 2] + cx
        left_finger_projs[:, 1] = left_finger_points[:, 1] * fy / left_finger_points[:, 2] + cy
        
        for j in range(N):
            cv2.circle(img, (int(left_finger_projs[j, 0]), int(left_finger_projs[j, 1])), 3, (0, 0, 255), -1)
        
        right_finger_projs = np.zeros((N, 2))
        right_finger_projs[:, 0] = right_finger_points[:, 0] * fx / right_finger_points[:, 2] + cx
        right_finger_projs[:, 1] = right_finger_points[:, 1] * fy / right_finger_points[:, 2] + cy

        # mean_x = (left_finger_projs[:, 0].mean() + right_finger_projs[:, 0].mean()) / 2
        # mean_y = (left_finger_projs[:, 1].mean() + right_finger_projs[:, 1].mean()) / 2
        # mean_depth = (left_finger_points[:, 2].mean() + right_finger_points[:, 2].mean()) / 2
        # print(mean_x, mean_y, mean_depth)

        for j in range(N):
            cv2.circle(img, (int(right_finger_projs[j, 0]), int(right_finger_projs[j, 1])), 3, (255, 0, 0), -1)
        
        cv2.imwrite('test.jpg', img)
        # import ipdb; ipdb.set_trace()
    
    def get_grasper_dist(self, sel_time):
        left_finger_points, right_finger_points = self.get_finger_points(sel_time)
        return 

    def get_grasper_pose(self, sel_time):
        robotiq_pose_in_base_frame = np.loadtxt(os.path.join(self.root_dir, 'pose', 'robotiq_base', f'{sel_time}.txt'))
        robotiq_pose_in_tag_frame = self.base_in_tag_frame @ robotiq_pose_in_base_frame
        return robotiq_pose_in_tag_frame


def load_keypoints(args):
    data_dir = "../data/2023-09-04-18-42-27-707743/"
    # save_dir = "../data/graph-2023-08-23-12-08-12-201998/"
    # dataset_name = "d3fields"
    # os.makedirs(save_dir, exist_ok=True)
    # frame_save_dir = os.path.join(save_dir, "valid_frames.txt")

    
    pkl_paths = sorted(list(glob.glob(os.path.join(data_dir, 'obj_kypts', '*.pkl'))))

    frame_list = [324, 966]
    for i in frame_list:
        pkl_path = pkl_paths[i]
        print(pkl_path)
        # print(img_path)
        for cam in range(4):
            intr = np.load(os.path.join(data_dir, f"camera_{cam}/camera_params.npy"))
            extr = np.load(os.path.join(data_dir, f"camera_{cam}/camera_extrinsics.npy"))
            img_path = pkl_path.replace('obj_kypts', f'camera_{cam}/color').replace(f'{i:06d}.pkl', f'color_{i}.png')
            obj_kp = pkl.load(open(pkl_path, 'rb')) # list of (rand_ptcl_num, 3)

            top_k = 20
            obj_kp = [kp[:top_k] for kp in obj_kp]
            instance_num = len(obj_kp)
            ptcl_num = obj_kp[0].shape[0]
            obj_kp = np.concatenate(obj_kp, axis=0) # (N = instance_num * rand_ptcl_num, 3)

            img = cv2.imread(img_path)

            # transform keypoints
            obj_kp = np.concatenate([obj_kp, np.ones((obj_kp.shape[0], 1))], axis=1) # (N, 4)
            obj_kp = obj_kp @ extr.T  # (N, 4)

            # project keypoints
            fx, fy, cx, cy = intr
            obj_kp_proj = np.zeros((instance_num * ptcl_num, 2))
            obj_kp_proj[:, 0] = obj_kp[:, 0] * fx / obj_kp[:, 2] + cx
            obj_kp_proj[:, 1] = obj_kp[:, 1] * fy / obj_kp[:, 2] + cy

            colormap = label_colormap()
            
            for j in range(obj_kp_proj.shape[0]):
                cv2.circle(img, (int(obj_kp_proj[j, 0]), int(obj_kp_proj[j, 1])), 3, 
                    (int(colormap[j, 2]), int(colormap[j, 1]), int(colormap[j, 0])), -1)
            
            cv2.imwrite(f'test_kp_{i}_{cam}.jpg', img)


def preprocess_raw_finger_keypoints(args):  # select frames where fingers are visible (deprecated)
    data_dir = "../data/2023-09-04-18-42-27-707743/"

    len_img_paths = len(list(glob.glob(os.path.join(data_dir, "camera_0/color/color_*.png"))))
    img_paths = [os.path.join(data_dir, "camera_0/color/color_{}.png".format(i)) for i in range(len_img_paths)]

    intr_list = []
    extr_list = []
    for cam_index in range(4):
        intr = np.load(os.path.join(data_dir, f"camera_{cam_index}/camera_params.npy"))
        extr = np.load(os.path.join(data_dir, f"camera_{cam_index}/camera_extrinsics.npy"))
        intr_list.append(intr)
        extr_list.append(extr)
    intr_all = np.stack(intr_list, axis=0)  # (4, 4)
    extr_all = np.stack(extr_list, axis=0)  # (4, 4, 4)
    # print(intr.shape, extr.shape)

    base_pose_in_tag = np.load(os.path.join(data_dir, "base_pose_in_tag.npy"))

    finger_points_in_finger_frame = np.array([[0.01,  -0.025, 0.014], # bottom right
                                              [-0.01, -0.025, 0.014], # bottom left
                                              [-0.01, -0.025, 0.051], # top left
                                              [0.01,  -0.025, 0.051]]) # top right

    # for i, img_path in enumerate(img_paths):
    # i = 2120
    img_path = img_paths[0]
    img = cv2.imread(img_path)
    h, w = img.shape[0], img.shape[1]
    # print(img_path)
    valid_list = []
    for i in range(len(img_paths)):
        left_finger_in_base_frame = np.loadtxt(os.path.join(data_dir, 'pose', 'left_finger', f'{i}.txt'))
        right_finger_in_base_frame = np.loadtxt(os.path.join(data_dir, 'pose', 'right_finger', f'{i}.txt'))

        left_finger_in_tag_frame = base_pose_in_tag @ left_finger_in_base_frame
        right_finger_in_tag_frame = base_pose_in_tag @ right_finger_in_base_frame

        N = finger_points_in_finger_frame.shape[0]
        finger_points_in_finger_frame_homo = np.concatenate([finger_points_in_finger_frame, np.ones((N, 1))], axis=1) # (N, 4)
        
        left_finger_points = left_finger_in_tag_frame @ finger_points_in_finger_frame_homo.T # (4, N)
        right_finger_points = right_finger_in_tag_frame @ finger_points_in_finger_frame_homo.T # (4, N)

        left_finger_points = left_finger_points.T # (N, 4)
        right_finger_points = right_finger_points.T # (N, 4)

        # img = Image.open(img_path).convert('RGB')
        # depth = Image.open(img_path.replace("color", "depth"))
        # img = cv2.imread(img_path)

        cnt = 0
        for j in range(4):
            extr = extr_all[j]
            intr = intr_all[j]
            # transform finger points to camera coordinate
            left_finger_points_cam = left_finger_points @ extr.T  # (N, 4)
            right_finger_points_cam = right_finger_points @ extr.T  # (N, 4)

            # project finger points
            fx, fy, cx, cy = intr
            left_finger_projs = np.zeros((N, 2))
            left_finger_projs[:, 0] = left_finger_points_cam[:, 0] * fx / left_finger_points_cam[:, 2] + cx
            left_finger_projs[:, 1] = left_finger_points_cam[:, 1] * fy / left_finger_points_cam[:, 2] + cy
            
            right_finger_projs = np.zeros((N, 2))
            right_finger_projs[:, 0] = right_finger_points_cam[:, 0] * fx / right_finger_points_cam[:, 2] + cx
            right_finger_projs[:, 1] = right_finger_points_cam[:, 1] * fy / right_finger_points_cam[:, 2] + cy

            mean_x = (left_finger_projs[:, 0].mean() + right_finger_projs[:, 0].mean()) / 2
            mean_y = (left_finger_projs[:, 1].mean() + right_finger_projs[:, 1].mean()) / 2
            mean_depth = (left_finger_points_cam[:, 2].mean() + right_finger_points_cam[:, 2].mean()) / 2

            # import ipdb; ipdb.set_trace()
            
            if mean_x >= 0 and mean_x <= w and mean_y >= 0 and mean_y <= h and mean_depth > 0:
                # print(i, j, mean_x, mean_y, mean_depth)
                cnt += 1

            if cnt >= 3:  # finger is visible in at least 2 cameras
                valid_list.append(i)  # valid frame that contains pushes
                break
        
    valid_list = np.array(valid_list)
    np.savetxt(os.path.join(data_dir, 'valid_frames.txt'), valid_list, fmt='%d')

    print(len(valid_list), len(img_paths))


def push_to_eef_pts(pushes):  # helper function for extract_pushes
    # :param pushes: (6) first 3 are starting point, and last 3 are ending point
    # :return: eef_pts (2, N, 3) eef_pts[0] are starting eef points, eef_pts[1] are ending eef points
    pushes_start = pushes[:3]
    pushes_end = pushes[3:]
    
    # Avoid NaN backward
    epsilon = 1e-7
    denominator = pushes_end[0] - pushes_start[0]
    numerator = pushes_end[1] - pushes_start[1]
    nudge = (denominator == 0) * epsilon
    denominator = denominator + nudge
    pushes_theta =  np.arctan2(numerator, denominator)
    
    pts_in_origin = np.array([[ 0.02355468, -0.0106243 ,  0.00315776],
                              [ 0.02356897,  0.0093757 ,  0.00315507],
                              [ 0.02356909,  0.00938067,  0.04015507],
                              [ 0.0235548 , -0.01061932,  0.04015776],
                              [-0.02359648,  0.00940939,  0.00315522],
                              [-0.02361077, -0.01059061,  0.00315791],
                              [-0.02361065, -0.01058563,  0.04015791],
                              [-0.02359636,  0.00941436,  0.04015522]]) # (N, 3)
    N = pts_in_origin.shape[0]
    pts_in_origin_homo = np.concatenate([pts_in_origin, np.ones((N, 1))], axis=1) # (N, 4)
    pts_in_origin_homo = pts_in_origin_homo.T # (4, N)

    def eef_pose_to_trans_mat(eef_pos, eef_rot):
        # :param eef_pos: (3)
        # :param eef_rot: (1)
        trans_mat = np.zeros((4, 4))
        trans_mat[0, 0] = np.sin(eef_rot)
        trans_mat[0, 1] = np.cos(eef_rot)
        trans_mat[1, 0] = -np.cos(eef_rot)
        trans_mat[1, 1] = np.sin(eef_rot)
        trans_mat[2, 2] = 1.0
        trans_mat[3, 3] = 1.0
        trans_mat[:3, 3] = eef_pos
        return trans_mat

    trans_mat_s = eef_pose_to_trans_mat(pushes_start, pushes_theta) # (4, 4)
    trans_mat_e = eef_pose_to_trans_mat(pushes_end, pushes_theta) # (4, 4)
    
    eef_pts = np.zeros((2, pts_in_origin.shape[0], 3)) # (2, N, 3)
    
    eef_pts[0, :, :] = (trans_mat_s @ pts_in_origin_homo).T[:, :3]
    eef_pts[1, :, :] = (trans_mat_e @ pts_in_origin_homo).T[:, :3]
    assert np.isnan(eef_pts).sum() == 0
    return eef_pts


def extract_pushes(args):  # save obj and eef keypoints
    data_dir = "../data/2023-09-04-18-42-27-707743/"
    
    # create output dir
    eef_kypts_dir = os.path.join(data_dir, 'eef_kypts') # list of (2, eef_ptcl_num, 3) for each push, indexed by push_num (dense)
    os.makedirs(eef_kypts_dir, exist_ok=True)
    obj_kypts_dir = os.path.join(data_dir, 'obj_kypts') # list of (ptcl_num, 3) for each push, indexed by push_num (dense)
    os.makedirs(obj_kypts_dir, exist_ok=True)
    dense_eef_kypts_dir = os.path.join(data_dir, 'dense_eef_kypts')
    os.makedirs(dense_eef_kypts_dir, exist_ok=True)
    dense_obj_kypts_dir = os.path.join(data_dir, 'dense_obj_kypts')
    os.makedirs(dense_obj_kypts_dir, exist_ok=True)
    
    # input dir
    database_obj_kypts_dir = os.path.join(data_dir, 'obj_kypts_orig')

    # extract push info
    push_num = len(os.listdir(os.path.join(data_dir, 'push')))
    push_pts = [np.load(os.path.join(data_dir, 'push', f'{i}', 'push_pts.npy')) for i in range(push_num)]
    push_time = [np.load(os.path.join(data_dir, 'push', f'{i}', 'push_time.npy')) for i in range(push_num)]
    push_pts = np.stack(push_pts, axis=0) # (push_num, 2, 3)
    push_time = np.stack(push_time, axis=0) # (push_num, 2)

    total_time = len(glob.glob(os.path.join(data_dir, 'camera_0/color/color_*.png')))
    image_times = list(range(total_time))
    timestamps = np.loadtxt(os.path.join(data_dir, 'timestamp.txt')) 
    record_idx = 0

    # extract finger info
    base_pose_in_tag = np.load(os.path.join(data_dir, "base_pose_in_tag.npy"))
    finger_points_in_finger_frame = np.array([[0.01,  -0.025, 0.014], # bottom right
                                              [-0.01, -0.025, 0.014], # bottom left
                                              [-0.01, -0.025, 0.051], # top left
                                              [0.01,  -0.025, 0.051]]) # top right

    last_mode = 'idle'
    curr_mode = 'idle'
    last_push_pt = np.zeros(3)
    curr_push_pt = np.zeros(3)
    push_idx = 0
    for t in image_times:
        is_start = False
        is_end = False
        # detect whether in push
        for p_idx, pt in enumerate(push_time):
            if timestamps[t] >= pt[0] and timestamps[t] < pt[1]:
                curr_mode = 'push'
                assert push_idx == p_idx
                # for sparse
                if timestamps[t - 1] < pt[0]:
                    is_start = True
                elif timestamps[t + 1] >= pt[1]:
                    is_end = True
                break
        else:
            curr_mode = 'idle'
        
        if curr_mode == 'push' and last_mode == 'idle':
            assert is_start
            # initialize obj keypoints for each push
            if record_idx == 0:
                os.system(f'cp {os.path.join(database_obj_kypts_dir, f"{t:06}.pkl")} {os.path.join(dense_obj_kypts_dir, f"{record_idx:06}.pkl")}')
            # initialize eef keypoints for each push
            last_push_pt = push_pts[push_idx, 0]
        
        # extract finger points
        left_finger_in_base_frame = np.loadtxt(os.path.join(data_dir, 'pose', 'left_finger', f'{t}.txt'))
        right_finger_in_base_frame = np.loadtxt(os.path.join(data_dir, 'pose', 'right_finger', f'{t}.txt'))

        left_finger_in_tag_frame = base_pose_in_tag @ left_finger_in_base_frame
        right_finger_in_tag_frame = base_pose_in_tag @ right_finger_in_base_frame

        N = finger_points_in_finger_frame.shape[0]
        finger_points_in_finger_frame_homo = np.concatenate([finger_points_in_finger_frame, np.ones((N, 1))], axis=1) # (N, 4)
        
        left_finger_points = left_finger_in_tag_frame @ finger_points_in_finger_frame_homo.T # (4, N)
        right_finger_points = right_finger_in_tag_frame @ finger_points_in_finger_frame_homo.T # (4, N)

        left_finger_points = left_finger_points.T # (N, 4)
        right_finger_points = right_finger_points.T # (N, 4)
        
        curr_push_pt = np.concatenate([left_finger_points[:, :3], right_finger_points[:, :3]], axis=0).mean(axis=0) # [3]  # ocurrent push location
        
        ## dense
        # save points in push
        # TODO: 0.02 is a hyperparameter worth tuning
        if (np.linalg.norm(curr_push_pt - last_push_pt) >= 0.02 and curr_mode == 'push') or (curr_mode == 'idle' and last_mode == 'push'):
            # store obj kypts at current state to next record_idx
            os.system(f'cp {os.path.join(database_obj_kypts_dir, f"{t:06}.pkl")} {os.path.join(dense_obj_kypts_dir, f"{(record_idx + 1):06}.pkl")}')
            # store the push between last state and current state to current record_idx
            curr_push = np.concatenate([last_push_pt, curr_push_pt]) # [6]  # start and end for one timestep

            eef_pts = push_to_eef_pts(curr_push) # (2, eef_ptcl_num, 3)
            np.save(os.path.join(dense_eef_kypts_dir, f'{record_idx:06}.npy'), eef_pts)
            record_idx += 1
            last_push_pt = curr_push_pt.copy()
        
        ## sparse
        if (is_start and push_idx == 0):  # only satisfy once
            os.system(f'cp {os.path.join(database_obj_kypts_dir, f"{t:06}.pkl")} {os.path.join(obj_kypts_dir, f"{push_idx:06}.pkl")}')
        elif is_end:  # once per push
            os.system(f'cp {os.path.join(database_obj_kypts_dir, f"{t:06}.pkl")} {os.path.join(obj_kypts_dir, f"{(push_idx+1):06}.pkl")}')
        if is_end:  # once per push
            # push_pts[push_idx] should be storing the same points as curr_push_pt?
            eef_pts = push_to_eef_pts(push_pts[push_idx].reshape(-1)) # (2, eef_ptcl_num, 3)
            np.save(os.path.join(eef_kypts_dir, f'{push_idx:06}.npy'), eef_pts)

        # update push index
        if curr_mode == 'idle' and last_mode == 'push':
            push_idx += 1
        last_mode = curr_mode


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


def preprocess_graph(args):  # save states, relations and attributes; use results of extract_pushes
    data_dir = "../data/2023-09-04-18-42-27-707743/"

    save_dir = os.path.join(data_dir, "graph")
    os.makedirs(save_dir, exist_ok=True)

    obj_kp_dir = os.path.join(data_dir, "dense_obj_kypts")
    eef_kp_dir = os.path.join(data_dir, "dense_eef_kypts")

    obj_kp_paths = sorted(list(glob.glob(os.path.join(obj_kp_dir, '*.pkl'))))

    num_frames = len(obj_kp_paths) - 1  # last frame do not have corresponding eef keypoints

    # all in world space, don't need to load camera params
    for i in range(num_frames):
        # get keypoints 
        obj_kp_path = obj_kp_paths[i]
        obj_kp = pkl.load(open(obj_kp_path, 'rb')) # list of (rand_ptcl_num, 3)
        eef_kp_path = os.path.join(eef_kp_dir, f'{i:06}.npy')
        eef_kp = np.load(eef_kp_path).astype(np.float32)   # (2, 8, 3)
        eef_kp_num = eef_kp.shape[1]

        top_k = 20
        obj_kp = [kp[:top_k] for kp in obj_kp]  # identical subsampling for all instances
        instance_num = len(obj_kp)
        ptcl_per_instance = obj_kp[0].shape[0]
        obj_kp = np.concatenate(obj_kp, axis=0) # (N = instance_num * rand_ptcl_num, 3)
        obj_kp_num = obj_kp.shape[0]

        # construct attributes
        p_rigid = np.ones(instance_num + 1, dtype=np.float32)  # TODO extend to nonrigid
        p_instance = np.zeros((obj_kp_num + eef_kp_num, instance_num + 1), dtype=np.float32)
        for j in range(instance_num):
            p_instance[j * ptcl_per_instance : (j + 1) * ptcl_per_instance, j] = 1
        p_instance[-eef_kp_num:, -1] = 1
        physics_param = np.zeros(obj_kp_num + eef_kp_num, dtype=np.float32)  # 1-dim

        attr_dim = 3
        attrs = np.zeros((obj_kp_num + eef_kp_num, attr_dim))
        for j in range(instance_num + 1):  # instances and end-effector
            assert instance_num + 1 <= attr_dim  # TODO make attr_dim instance_num independent
            one_hot = np.zeros(attr_dim)
            if j == instance_num:  # end-effector
                one_hot[-1] = 1.
                attrs[obj_kp_num:] = one_hot
            else:
                one_hot[j] = 1.
                attrs[j * ptcl_per_instance: (j + 1) * ptcl_per_instance] = one_hot

        # construct relations (density as hyperparameter)
        adj_thresh = 0.1
        states = np.concatenate([obj_kp, eef_kp[0]], axis=0)  # (N, 3)  # the current eef_kp
        Rr, Rs = construct_edges_from_states(torch.tensor(states).unsqueeze(0), adj_thresh, exclude_last_N=eef_kp_num)
        Rr, Rs = Rr.squeeze(0).numpy(), Rs.squeeze(0).numpy()

        # action encoded as state_delta (only stored in eef keypoints)
        states_delta = np.zeros((obj_kp_num + eef_kp_num, states.shape[-1]))
        states_delta[-eef_kp_num:] = eef_kp[1] - eef_kp[0]

        # save graph
        graph = {
            "attrs": attrs, 
            "states": states, 
            "states_delta": states_delta, 
            "Rr": Rr, 
            "Rs": Rs,
            "p_rigid": p_rigid,
            "p_instance": p_instance,
            "physics_param": physics_param,
            "adj_thresh": np.array([adj_thresh])
        }
        save_path = os.path.join(save_dir, f'{i:06}.pkl')
        pkl.dump(graph, open(save_path, 'wb'))


def preprocess_graph_old(args):  # save states, relations and attributes; use valid_list, deprecated
    data_dir = "../data/2023-09-04-18-42-27-707743/"

    save_dir = os.path.join(data_dir, "graph")
    os.makedirs(save_dir, exist_ok=True)

    len_img_paths = len(list(glob.glob(os.path.join(data_dir, "camera_0/color/color_*.png"))))
    img_paths = [os.path.join(data_dir, "camera_0/color/color_{}.png".format(i)) for i in range(len_img_paths)]

    valid_list = np.loadtxt(os.path.join(data_dir, 'valid_frames.txt'), dtype=np.int32)
    print(len(valid_list), len(img_paths))

    pkl_paths = sorted(list(glob.glob(os.path.join(data_dir, 'obj_kypts', '*.pkl'))))

    base_pose_in_tag = np.load(os.path.join(data_dir, "base_pose_in_tag.npy"))

    finger_points_in_finger_frame = np.array([[0.01,  -0.025, 0.014], # bottom right
                                              [-0.01, -0.025, 0.014], # bottom left
                                              [-0.01, -0.025, 0.051], # top left
                                              [0.01,  -0.025, 0.051]]) # top right

    # all in world space, don't need to load camera params
    for i in valid_list:
        # get keypoints 
        pkl_path = pkl_paths[i]
        obj_kp = pkl.load(open(pkl_path, 'rb')) # list of (rand_ptcl_num, 3)

        top_k = 20
        obj_kp = [kp[:top_k] for kp in obj_kp]  # identical subsampling for all instances
        instance_num = len(obj_kp)
        ptcl_per_instance = obj_kp[0].shape[0]
        obj_kp = np.concatenate(obj_kp, axis=0) # (N = instance_num * rand_ptcl_num, 3)
        obj_kp_num = obj_kp.shape[0]

        # get finger points
        left_finger_in_base_frame = np.loadtxt(os.path.join(data_dir, 'pose', 'left_finger', f'{i}.txt'))
        right_finger_in_base_frame = np.loadtxt(os.path.join(data_dir, 'pose', 'right_finger', f'{i}.txt'))

        left_finger_in_tag_frame = base_pose_in_tag @ left_finger_in_base_frame
        right_finger_in_tag_frame = base_pose_in_tag @ right_finger_in_base_frame

        nf = finger_points_in_finger_frame.shape[0]
        finger_points_in_finger_frame_homo = np.concatenate([finger_points_in_finger_frame, np.ones((nf, 1))], axis=1) # (N, 4)
        
        left_finger_points = left_finger_in_tag_frame @ finger_points_in_finger_frame_homo.T # (4, N)
        right_finger_points = right_finger_in_tag_frame @ finger_points_in_finger_frame_homo.T # (4, N)

        left_finger_points = left_finger_points.T # (N, 4)
        right_finger_points = right_finger_points.T # (N, 4)

        left_finger_points = left_finger_points[:, :3]
        right_finger_points = right_finger_points[:, :3]

        eef_kp = np.concatenate([left_finger_points, right_finger_points], axis=0)
        eef_kp_num = eef_kp.shape[0]

        # construct attributes
        p_rigid = np.ones(instance_num + 1, dtype=np.float32)  # TODO extend to nonrigid
        p_instance = np.zeros((ptcl_per_instance * instance_num + eef_kp_num, instance_num + 1), dtype=np.float32)
        for i in range(instance_num):
            p_instance[i * ptcl_per_instance : (i + 1) * ptcl_per_instance, i] = 1
        p_instance[-eef_kp_num:, -1] = 1
        physics_param = np.zeros(ptcl_per_instance * instance_num + eef_kp_num, dtype=np.float32)  # 1-dim

        attr_dim = 3
        attrs = np.zeros(obj_kp_num + eef_kp_num, attr_dim)
        for i in range(instance_num + 1):  # instances and end-effector
            assert instance_num + 1 <= attr_dim  # TODO make attr_dim instance_num independent
            one_hot = np.zeros(attr_dim)
            if i == instance_num:  # end-effector
                one_hot[-1] = 1.
                attrs[:, obj_kp_num:] = one_hot
            else:
                one_hot[i] = 1.
                attrs[:, i * ptcl_per_instance: (i + 1) * ptcl_per_instance] = one_hot

        # construct relations

        # save graph


if __name__ == "__main__":
    args = gen_args()
    # load_fingers(args)
    # load_keypoints(args)
    # preprocess_raw_finger_keypoints(args)
    # extract_pushes(args)
    preprocess_graph(args)
