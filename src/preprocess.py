import os
import torch
import numpy as np
import glob
from PIL import Image
import cv2
import pickle as pkl
import copy
from dgl.geometry import farthest_point_sampler

from config import gen_args
from data.utils import label_colormap
from gnn.model_wrapper import gen_model


def load_fingers(args, data_dir, idx=None):
    # filter raw data
    # generate save_dir/valid_frames.txt
    # data_dir = "../data/2023-09-04-18-42-27-707743/"
    # save_dir = "../data/graph-2023-08-23-12-08-12-201998/"
    # dataset_name = "d3fields"

    # os.makedirs(save_dir, exist_ok=True)
    # frame_save_dir = os.path.join(save_dir, "valid_frames.txt")

    len_img_paths = len(list(glob.glob(os.path.join(data_dir, "camera_0/color/color_*.png"))))
    img_paths = [os.path.join(data_dir, "camera_0/color/color_{}.png".format(i)) for i in range(len_img_paths)]

    intr = np.load(os.path.join(data_dir, "camera_0/camera_params.npy"))
    extr = np.load(os.path.join(data_dir, "camera_0/camera_extrinsics.npy"))
    print(intr.shape, extr.shape)

    base_pose_in_tag = np.load(os.path.join(data_dir, "base_pose_in_tag.npy"))

    finger_points_in_finger_frame = np.array([[0.01,  -0.025, 0.014], # bottom right
                                              [-0.01, -0.025, 0.014], # bottom left
                                              [-0.01, -0.025, 0.051], # top left
                                              [0.01,  -0.025, 0.051]]) # top right

    push_num = len(os.listdir(os.path.join(data_dir, 'push')))
    push_pts = [np.load(os.path.join(data_dir, 'push', f'{i}', 'push_pts.npy')) for i in range(push_num)]
    push_time = [np.load(os.path.join(data_dir, 'push', f'{i}', 'push_time.npy')) for i in range(push_num)]
    push_pts = np.stack(push_pts, axis=0) # (push_num, 2, 3)
    push_time = np.stack(push_time, axis=0) # (push_num, 2)

    t_list = np.loadtxt(os.path.join(data_dir, 'push_t_list.txt'), dtype=np.int32)

    # for i, img_path in enumerate(img_paths):
    # i = 920
    if idx is None: i = 2
    else: i = idx
    t = t_list[i][0]
    print("i:", i, "t:", t)
    img_path = img_paths[t]
    print(img_path)
    if True:
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

        # get other quantities: grasper_dist, grasper_pose (used in grasp planner)
        grasper_dist = np.linalg.norm(left_finger_points - right_finger_points, axis=1).mean()
        robotiq_pose_in_base_frame = np.loadtxt(os.path.join(data_dir, 'pose', 'robotiq_base', f'{t}.txt'))
        robotiq_pose_in_tag_frame = base_pose_in_tag @ robotiq_pose_in_base_frame

        # TODO check the potential calibration difference between finger positions and push info
        mean_push_pt = np.concatenate([left_finger_points[:, :3], right_finger_points[:, :3]], axis=0).mean(axis=0) # [3]  # ocurrent push location
        push_pt = push_pts[i, 0]
        print(mean_push_pt)
        print(push_pt)
        left_finger_points[:, :3] += (push_pt - mean_push_pt)
        right_finger_points[:, :3] += (push_pt - mean_push_pt)

        # print(left_finger_points)
        # print(right_finger_points)
        # left_finger_points[:, 2] += (-0.05 - left_finger_points[:, 2].mean())
        # right_finger_points[:, 2] += (-0.05 - right_finger_points[:, 2].mean())


        # img = Image.open(img_path).convert('RGB')
        # depth = Image.open(img_path.replace("color", "depth"))
        img = cv2.imread(img_path)

        # transform finger points to camera coordinate
        left_finger_points = left_finger_points @ extr.T  # (N, 4)
        right_finger_points = right_finger_points @ extr.T  # (N, 4)

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


def load_keypoints(args, data_dir):
    # data_dir = "../data/2023-09-04-18-42-27-707743/"
    # save_dir = "../data/graph-2023-08-23-12-08-12-201998/"
    # dataset_name = "d3fields"
    # os.makedirs(save_dir, exist_ok=True)
    # frame_save_dir = os.path.join(save_dir, "valid_frames.txt")

    
    pkl_paths = sorted(list(glob.glob(os.path.join(data_dir, 'obj_kypts_orig', '*.pkl'))))

    frame_list = [324, 966]
    for i in frame_list:
        pkl_path = pkl_paths[i]
        print(pkl_path)
        # print(img_path)
        for cam in range(4):
            intr = np.load(os.path.join(data_dir, f"camera_{cam}/camera_params.npy"))
            extr = np.load(os.path.join(data_dir, f"camera_{cam}/camera_extrinsics.npy"))
            img_path = pkl_path.replace('obj_kypts_orig', f'camera_{cam}/color').replace(f'{i:06d}.pkl', f'color_{i}.png')
            obj_kp = pkl.load(open(pkl_path, 'rb')) # list of (rand_ptcl_num, 3)

            top_k = 20
            # top_k_samples = np.random.choice(obj_kp[0].shape[0], top_k, replace=False)
            # obj_kp = [kp[top_k_samples] for kp in obj_kp]  # identical subsampling for all instances
            for j in range(len(obj_kp)):
                # farthest point sampling
                particle_tensor = torch.from_numpy(obj_kp[j]).float()[None, ...]
                fps_idx_tensor = farthest_point_sampler(particle_tensor, top_k, start_idx=np.random.randint(0, obj_kp[j].shape[0]))[0]
                fps_idx = fps_idx_tensor.numpy().astype(np.int32)
                obj_kp[j] = obj_kp[j][fps_idx]
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


def load_pushes(args, data_dir, idx=None):
    # data_dir = "../data/2023-09-04-18-42-27-707743/"
    push_num = len(os.listdir(os.path.join(data_dir, 'push')))
    push_pts = [np.load(os.path.join(data_dir, 'push', f'{i}', 'push_pts.npy')) for i in range(push_num)]
    push_time = [np.load(os.path.join(data_dir, 'push', f'{i}', 'push_time.npy')) for i in range(push_num)]
    push_pts = np.stack(push_pts, axis=0) # (push_num, 2, 3)
    push_time = np.stack(push_time, axis=0) # (push_num, 2)

    base_pose_in_tag = np.load(os.path.join(data_dir, "base_pose_in_tag.npy"))

    finger_points_in_finger_frame = np.array([[0.01,  -0.025, 0.014], # bottom right
                                              [-0.01, -0.025, 0.014], # bottom left
                                              [-0.01, -0.025, 0.051], # top left
                                              [0.01,  -0.025, 0.051]]) # top right

    eef_paths = sorted(list(glob.glob(os.path.join(data_dir, 'eef_kypts', '*.npy'))))
    num_frames = len(eef_paths)

    t_list = np.loadtxt(os.path.join(data_dir, 'push_t_list.txt'), dtype=np.int32)

    # for i in range(num_frames):
    if idx is None: i = 2
    else: i = idx
    if True:
        push_pt = push_pts[i, 0]
        t = t_list[i][0]
        print("i:", i, "t:", t)
        
        # eef_path = eef_paths[i]
        # eef_kp = np.load(eef_path) # (N, 3)

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
        
        mean_push_pt = np.concatenate([left_finger_points[:, :3], right_finger_points[:, :3]], axis=0).mean(axis=0) # [3]  # ocurrent push location

        # consistency check
        if np.max(mean_push_pt[:2] - push_pt[:2]) > 2e-3:
            print(i, t, mean_push_pt, push_pt)
        
        # TODO check the potential calibration difference between finger positions and push info
        print(mean_push_pt)
        print(push_pt)
        mean_push_pt[:3] += (push_pt - mean_push_pt)

        # transform mean push to finger poses again
        next_push_pt = push_pts[i, 1]
        mean_push_pt_double = np.concatenate([mean_push_pt, next_push_pt], axis=0)
        eef_pts = push_to_eef_pts(mean_push_pt_double)[0]

        img_path = os.path.join(data_dir, f"camera_0/color/color_{t}.png")
        intr = np.load(os.path.join(data_dir, "camera_0/camera_params.npy"))
        extr = np.load(os.path.join(data_dir, "camera_0/camera_extrinsics.npy"))

        img = cv2.imread(img_path)

        # transform finger points to camera coordinate
        eef_pts_homo = np.concatenate([eef_pts, np.ones((eef_pts.shape[0], 1))], axis=1)
        eef_pts_cam = eef_pts_homo @ extr.T  # (N, 4)
        eef_pts_cam = eef_pts_cam[:, :3]

        # project finger points
        fx, fy, cx, cy = intr
        eef_pts_projs = np.zeros((eef_pts_cam.shape[0], 2))
        eef_pts_projs[:, 0] = eef_pts_cam[:, 0] * fx / eef_pts_cam[:, 2] + cx
        eef_pts_projs[:, 1] = eef_pts_cam[:, 1] * fy / eef_pts_cam[:, 2] + cy
        
        for j in range(eef_pts.shape[0]):
            cv2.circle(img, (int(eef_pts_projs[j, 0]), int(eef_pts_projs[j, 1])), 3, (0, 0, 255), -1)
        
        cv2.imwrite('test_push.jpg', img)

        # mean_x = (left_finger_projs[:, 0].mean() + right_finger_projs[:, 0].mean()) / 2
        # mean_y = (left_finger_projs[:, 1].mean() + right_finger_projs[:, 1].mean()) / 2
        # mean_depth = (left_finger_points[:, 2].mean() + right_finger_points[:, 2].mean()) / 2
        # print(mean_x, mean_y, mean_depth)


def preprocess_raw_finger_keypoints(args, data_dir):  # select frames where fingers are visible (deprecated)
    # data_dir = "../data/2023-09-04-18-42-27-707743/"

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


def extract_pushes(args, data_dir):  # save obj and eef keypoints
    # data_dir = "../data/2023-09-04-18-42-27-707743/"
    
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
    t_list_dense = []
    t_list = []
    offset = None
    for t in image_times:
        is_start = False
        is_end = False
        # detect whether in push
        for p_idx, pt in enumerate(push_time):
            if timestamps[t] >= pt[0] and timestamps[t] < pt[1]:
                curr_mode = 'push'
                try:
                    assert push_idx == p_idx
                except:
                    print(f"no frames recorded in directory {data_dir}, push {push_idx - 1}")
                # for sparse
                if timestamps[t - 1] < pt[0]:
                    is_start = True
                    t_last = t
                elif timestamps[t + 1] >= pt[1]:
                    is_end = True
                    t_curr = t
                    t_list.append([t_last, t_curr])
                    t_last, t_curr = None, None
                break
        else:
            curr_mode = 'idle'
        
        if curr_mode == 'push' and last_mode == 'idle':
            assert is_start
            # initialize obj keypoints for each push
            if record_idx == 0:
                os.system(f'cp {os.path.join(database_obj_kypts_dir, f"{t:06}.pkl")} {os.path.join(dense_obj_kypts_dir, f"{record_idx:06}.pkl")}')
                # print('loading obj keypoints for push 0')
            # initialize eef keypoints for each push
            last_push_pt = push_pts[push_idx, 0]
            offset = None
            t_last_dense = t
        
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
        
        # calculate calibration offset between finger points and pushes
        if curr_mode == 'push' and last_mode == 'idle':
            assert np.linalg.norm(curr_push_pt - last_push_pt) < 0.02
            offset = last_push_pt - curr_push_pt
        
        # fix current push with the offset
        if (curr_mode == 'push') or (curr_mode == 'idle' and last_mode == 'push'):
            assert offset is not None
            curr_push_pt += offset

        ## dense
        # save points in push
        # TODO: 0.02 is a hyperparameter worth tuning
        if (np.linalg.norm(curr_push_pt - last_push_pt) >= 0.02 and curr_mode == 'push') or (curr_mode == 'idle' and last_mode == 'push'):
            # store obj kypts at current state to next record_idx
            os.system(f'cp {os.path.join(database_obj_kypts_dir, f"{t:06}.pkl")} {os.path.join(dense_obj_kypts_dir, f"{(record_idx + 1):06}.pkl")}')
            # print('loading obj keypoints for push', record_idx + 1)
            # store the push between last state and current state to current record_idx
            curr_push = np.concatenate([last_push_pt, curr_push_pt]) # [6]  # start and end for one timestep
            t_curr_dense = t
            t_list_dense.append([t_last_dense, t_curr_dense])

            eef_pts = push_to_eef_pts(curr_push) # (2, eef_ptcl_num, 3)
            np.save(os.path.join(dense_eef_kypts_dir, f'{record_idx:06}.npy'), eef_pts)
            record_idx += 1
            last_push_pt = curr_push_pt.copy()
            t_last_dense = t_curr_dense
        
        ## sparse (does not use curr_push)
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

    np.savetxt(os.path.join(data_dir, 'push_t_list.txt'), np.array(t_list).astype(np.int32), fmt='%i')
    np.savetxt(os.path.join(data_dir, 'push_t_list_dense.txt'), np.array(t_list_dense).astype(np.int32), fmt='%i')


def extract_pushes_2(args, data_dir):  # save obj and eef keypoints, obj keypoints containing both start and end points
    # pred_gap = 1  # TODO make this code compatible with pred_gap > 1
    verbose = False
    # create output dir
    eef_kypts_dir = os.path.join(data_dir, 'eef_kypts_2') # list of (2, eef_ptcl_num, 3) for each push, indexed by push_num (dense)
    os.makedirs(eef_kypts_dir, exist_ok=True)
    obj_kypts_dir = os.path.join(data_dir, 'obj_kypts_2') # list of (ptcl_num, 3) for each push, indexed by push_num (dense)
    os.makedirs(obj_kypts_dir, exist_ok=True)
    dense_eef_kypts_dir = os.path.join(data_dir, 'dense_eef_kypts_2')
    os.makedirs(dense_eef_kypts_dir, exist_ok=True)
    dense_obj_kypts_dir = os.path.join(data_dir, 'dense_obj_kypts_2')
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
    start_push_pt = np.zeros(3)
    last_push_pt = np.zeros(3)
    curr_push_pt = np.zeros(3)
    start_obj_pt = None
    last_obj_pt = None
    curr_obj_pt = None
    push_idx = 0
    t_list_dense = []
    t_list = []
    offset = None
    for t in image_times:
        is_start = False
        is_end = False
        # detect whether in push
        for p_idx, pt in enumerate(push_time):
            if timestamps[t] >= pt[0] and timestamps[t] < pt[1]:
                curr_mode = 'push'
                try:
                    assert push_idx == p_idx
                except:
                    print(f"no frames recorded in directory {data_dir}, push {push_idx - 1}")
                # for sparse
                if timestamps[t - 1] < pt[0]:
                    is_start = True
                    t_last = t
                elif timestamps[t + 1] >= pt[1]:
                    is_end = True
                    t_curr = t
                    t_list.append([t_last, t_curr])
                    t_last, t_curr = None, None
                break
        else:
            curr_mode = 'idle'
        
        if curr_mode == 'push' and last_mode == 'idle':
            assert is_start
            # initialize obj keypoints for each push
            if verbose: print(f'starting push at time {t}')
            # NOTE new obj_kypts loading method
            with open(os.path.join(os.path.join(database_obj_kypts_dir, f"{t:06}.pkl")), 'rb') as f:
                start_obj_pt = pkl.load(f)
                last_obj_pt = start_obj_pt  # if record_idx is not zero, last_obj_pt should be roughly the same as start_obj_pt
                if verbose: print(f'starting obj pt set at time {t}')
            # initialize eef keypoints for each push
            last_push_pt = push_pts[push_idx, 0]
            start_push_pt = push_pts[push_idx, 0]
            if verbose: print(f'starting eef pt set at time {t}')
            offset = None
            t_last_dense = t
        
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
        
        # calculate calibration offset between finger points and pushes
        if curr_mode == 'push' and last_mode == 'idle':
            assert np.linalg.norm(curr_push_pt - last_push_pt) < 0.02
            offset = last_push_pt - curr_push_pt
        
        # fix current push with the offset
        if (curr_mode == 'push') or (curr_mode == 'idle' and last_mode == 'push'):
            assert offset is not None
            curr_push_pt += offset

        ## dense
        # save points in push
        # TODO: 0.02 is a hyperparameter worth tuning
        if (np.linalg.norm(curr_push_pt - last_push_pt) >= 0.02 and curr_mode == 'push') or (curr_mode == 'idle' and last_mode == 'push'):
            # store obj kypts at current state to next record_idx
            # os.system(f'cp {os.path.join(database_obj_kypts_dir, f"{t:06}.pkl")} {os.path.join(dense_obj_kypts_dir, f"{(record_idx + 1):06}.pkl")}')
            
            # NOTE new obj_kypts saving method
            with open(os.path.join(os.path.join(database_obj_kypts_dir, f"{t:06}.pkl")), 'rb') as f:
                curr_obj_pt = pkl.load(f)
            obj_pts = [last_obj_pt, curr_obj_pt]
            with open(os.path.join(dense_obj_kypts_dir, f"{record_idx:06}.pkl"), 'wb') as f:
                pkl.dump(obj_pts, f)
            
            # print('loading obj keypoints for push', record_idx + 1)
            # store the push between last state and current state to current record_idx
            curr_push = np.concatenate([last_push_pt, curr_push_pt]) # [6]  # start and end for one timestep
            t_curr_dense = t
            t_list_dense.append([t_last_dense, t_curr_dense])

            eef_pts = push_to_eef_pts(curr_push) # (2, eef_ptcl_num, 3)
            np.save(os.path.join(dense_eef_kypts_dir, f'{record_idx:06}.npy'), eef_pts)
            record_idx += 1
            last_push_pt = curr_push_pt.copy()
            last_obj_pt = copy.deepcopy(curr_obj_pt)

            t_last_dense = t_curr_dense
        
        ## sparse (does not use curr_push)
        # if (is_start and push_idx == 0):  # only satisfy once
        #     os.system(f'cp {os.path.join(database_obj_kypts_dir, f"{t:06}.pkl")} {os.path.join(obj_kypts_dir, f"{push_idx:06}.pkl")}')
        # elif is_end:  # once per push
        #     os.system(f'cp {os.path.join(database_obj_kypts_dir, f"{t:06}.pkl")} {os.path.join(obj_kypts_dir, f"{(push_idx+1):06}.pkl")}')
        
        # NOTE new obj_kypts saving method
        if is_end:  # once per push
            with open(os.path.join(os.path.join(database_obj_kypts_dir, f"{t:06}.pkl")), 'rb') as f:
                curr_obj_pt = pkl.load(f)
            obj_pts = [start_obj_pt, curr_obj_pt]
            if verbose: print(f'ending obj pt set at time {t}')
            with open(os.path.join(obj_kypts_dir, f"{push_idx:06}.pkl"), 'wb') as f:
                pkl.dump(obj_pts, f)
            if verbose: print(f'saving obj pt pairs to push idx {push_idx}')

        if is_end:  # once per push
            # push_pts[push_idx] should be storing the same points as curr_push_pt?
            assert np.linalg.norm(start_push_pt - push_pts[push_idx, 0]) < 0.01
            assert np.linalg.norm(curr_push_pt - push_pts[push_idx, 1]) < 0.01
            if verbose: print(f'ending eef pt set at time {t}')
            eef_pts = push_to_eef_pts(push_pts[push_idx].reshape(-1)) # (2, eef_ptcl_num, 3)
            np.save(os.path.join(eef_kypts_dir, f'{push_idx:06}.npy'), eef_pts)
            if verbose: print(f'saving eef pt pairs to push idx {push_idx}')

        # update push index
        if curr_mode == 'idle' and last_mode == 'push':
            push_idx += 1
        last_mode = curr_mode

    np.savetxt(os.path.join(data_dir, 'push_t_list_2.txt'), np.array(t_list).astype(np.int32), fmt='%i')
    np.savetxt(os.path.join(data_dir, 'push_t_list_2_dense.txt'), np.array(t_list_dense).astype(np.int32), fmt='%i')


def kypts_verification(args, data_dir):  # ensure dense/sparse obj/eef kypts derived by extract_pushes_2 are continuous in time
    eef_subdirs = ['eef_kypts_2', 'dense_eef_kypts_2']
    obj_subdirs = ['obj_kypts_2', 'dense_obj_kypts_2']
    t_list_subdirs = ['push_t_list_2', 'push_t_list_2_dense']
    for i in range(2):
        eef_subdir = eef_subdirs[i]
        obj_subdir = obj_subdirs[i]
        eef_kypts_dir = os.path.join(data_dir, eef_subdir)
        obj_kypts_dir = os.path.join(data_dir, obj_subdir)
        eef_kypts_paths = sorted(list(glob.glob(os.path.join(eef_kypts_dir, '*.npy'))))
        obj_kypts_paths = sorted(list(glob.glob(os.path.join(obj_kypts_dir, '*.pkl'))))
        t_list = np.loadtxt(os.path.join(data_dir, t_list_subdirs[i] + '.txt'))
        assert len(eef_kypts_paths) == len(obj_kypts_paths)
        for j in range(len(eef_kypts_paths) - 1):
            eef_kypts_path = eef_kypts_paths[j]
            obj_kypts_path = obj_kypts_paths[j]
            t = t_list[j]
            eef_kypts = np.load(eef_kypts_path)
            with open(obj_kypts_path, 'rb') as f:
                obj_kypts = pkl.load(f)
            
            eef_kypts_next_path = eef_kypts_paths[j + 1]
            obj_kypts_next_path = obj_kypts_paths[j + 1]
            t_next = t_list[j + 1]
            eef_kypts_next = np.load(eef_kypts_next_path)
            with open(obj_kypts_next_path, 'rb') as f:
                obj_kypts_next = pkl.load(f)
            
            if data_dir == "../data/2023-08-30-03-27-55-702301":
                if i == 0 and j == 9 or i == 1 and j == 44: continue  # error in data collection, frame 2721 ~ 2784
            
            if t[1] == t_next[0]:  # pushes should be continuous
                try: assert np.linalg.norm(eef_kypts[1, :, :].mean(0) - eef_kypts_next[0, :, :].mean(0)) < 0.01
                except: import ipdb; ipdb.set_trace()
            
            # obj kypts should be continuous at any time
            obj_len = len(obj_kypts[0])  # number of instances
            for k in range(obj_len):
                try: 
                    assert np.linalg.norm(obj_kypts[1][k].mean(0) - obj_kypts_next[0][k].mean(0)) < 0.04
                    assert np.linalg.norm(obj_kypts[1][k].max(0) - obj_kypts_next[0][k].max(0)) < 0.04
                    assert np.linalg.norm(obj_kypts[1][k].min(0) - obj_kypts_next[0][k].min(0)) < 0.04
                except: import ipdb; ipdb.set_trace()
    print("kypts verification passed")


def construct_edges_from_states(states, adj_thresh, mask, eef_mask, no_self_edge=False):  # helper function for construct_graph
    # :param states: (B, N, state_dim) torch tensor
    # :param adj_thresh: float
    # :param mask: (B, N) torch tensor, true when index is a valid particle
    # :param eef_mask: (B, N) torch tensor, true when index is a valid eef particle
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
    mask_1 = mask[:, :, None].repeat(1, 1, N)
    mask_2 = mask[:, None, :].repeat(1, N, 1)
    mask = mask_1 * mask_2
    dis[~mask] = 1e10  # avoid invalid particles to particles relations
    eef_mask_1 = eef_mask[:, :, None].repeat(1, 1, N)
    eef_mask_2 = eef_mask[:, None, :].repeat(1, N, 1)
    eef_mask = eef_mask_1 * eef_mask_2
    dis[eef_mask] = 1e10  # avoid eef to eef relations
    adj_matrix = ((dis - threshold) < 0).float()

    # remove self edge
    if no_self_edge:
        self_edge_mask = torch.eye(N, device=states.device, dtype=states.dtype)[None, :, :]
        adj_matrix = adj_matrix * (1 - self_edge_mask)

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


def preprocess_graph(args, data_dir, max_n=None, max_nobj=None, max_neef=None, max_nR=None, dense=True):  # save states, relations and attributes; use results of extract_pushes
    # data_dir = "../data/2023-09-04-18-42-27-707743/"
    if dense:
        print("using dense push")
        save_dir = os.path.join(data_dir, "dense_graph")
        obj_kp_dir = os.path.join(data_dir, "dense_obj_kypts")
        eef_kp_dir = os.path.join(data_dir, "dense_eef_kypts")
    else:
        print("using sparse push")
        save_dir = os.path.join(data_dir, "graph")
        obj_kp_dir = os.path.join(data_dir, "obj_kypts")
        eef_kp_dir = os.path.join(data_dir, "eef_kypts")

    os.makedirs(save_dir, exist_ok=True)

    obj_kp_paths = sorted(list(glob.glob(os.path.join(obj_kp_dir, '*.pkl'))))

    num_frames = len(obj_kp_paths) - 1  # last frame do not have corresponding eef keypoints

    # all in world space, don't need to load camera params
    graph_list = []
    for i in range(num_frames):
        # get keypoints 
        obj_kp_path = obj_kp_paths[i]
        obj_kp = pkl.load(open(obj_kp_path, 'rb')) # list of (rand_ptcl_num, 3)
        eef_kp_path = os.path.join(eef_kp_dir, f'{i:06}.npy')
        eef_kp = np.load(eef_kp_path).astype(np.float32)   # (2, 8, 3)
        eef_kp_num = eef_kp.shape[1]
        ptcl_per_eef = eef_kp.shape[1]

        top_k = 20
        obj_kp = [kp[:top_k] for kp in obj_kp]  # identical subsampling for all instances
        instance_num = len(obj_kp)
        ptcl_per_instance = obj_kp[0].shape[0]
        obj_kp = np.concatenate(obj_kp, axis=0) # (N = instance_num * rand_ptcl_num, 3)
        obj_kp_num = obj_kp.shape[0]

        if max_nobj is not None:
            # pad obj_kp
            obj_kp_pad = np.zeros((max_nobj, 3), dtype=np.float32)
            obj_kp_pad[:obj_kp_num] = obj_kp
            obj_kp = obj_kp_pad
            obj_kp_num = max_nobj
        else:
            print("not using max_nobj which only works for fixed number of obj keypoints")
        
        if max_neef is not None:
            # pad eef_kp
            eef_kp_pad = np.zeros((2, max_neef, 3), dtype=np.float32)
            eef_kp_pad[:, :eef_kp_num] = eef_kp
            eef_kp = eef_kp_pad
            eef_kp_num = max_neef
        else:
            print("not using max_neef which only works for fixed number of eef keypoints")
        
        if max_n is not None:
            max_instance_num = max_n
            assert max_instance_num >= instance_num
        else:
            print("not using max_n which only works for fixed number of objects")
            max_instance_num = instance_num

        state_mask = np.zeros((obj_kp_num + eef_kp_num), dtype=bool)
        state_mask[obj_kp_num : obj_kp_num + ptcl_per_eef] = True
        state_mask[:instance_num * ptcl_per_instance] = True

        eef_mask = np.zeros((obj_kp_num + eef_kp_num), dtype=bool)
        eef_mask[obj_kp_num : obj_kp_num + ptcl_per_eef] = True

        obj_mask = np.zeros((obj_kp_num,), dtype=bool)
        obj_mask[:instance_num * ptcl_per_instance] = True

        # construct instance information
        p_rigid = np.ones(max_instance_num, dtype=np.float32)  # TODO extend to nonrigid
        p_instance = np.zeros((obj_kp_num, max_instance_num), dtype=np.float32)
        for j in range(instance_num):
            p_instance[j * ptcl_per_instance : (j + 1) * ptcl_per_instance, j] = 1  # TODO different number of particles per instance
        physics_param = np.zeros(obj_kp_num, dtype=np.float32)  # 1-dim

        # construct attributes
        # attr_dim = max_instance_num + 1
        # assert attr_dim == args.attr_dim
        # attrs = np.zeros((obj_kp_num + eef_kp_num, attr_dim), dtype=np.float32)
        # for j in range(instance_num + 1):  # instances and end-effector
        #     # assert instance_num + 1 <= attr_dim  # TODO make attr_dim instance_num independent
        #     one_hot = np.zeros(attr_dim)
        #     if j == instance_num:  # end-effector
        #         one_hot[-1] = 1.
        #         attrs[obj_kp_num : obj_kp_num + ptcl_per_eef] = one_hot
        #     else:
        #         one_hot[j] = 1.
        #         attrs[j * ptcl_per_instance: (j + 1) * ptcl_per_instance] = one_hot
        attr_dim = 2
        assert attr_dim == args.attr_dim
        attrs = np.zeros((obj_kp_num + eef_kp_num, attr_dim), dtype=np.float32)
        attrs[:instance_num * ptcl_per_instance, 0] = 1.
        attrs[obj_kp_num : obj_kp_num + ptcl_per_eef, 1] = 1.

        # construct relations (density as hyperparameter)
        adj_thresh = 1
        states = np.concatenate([obj_kp, eef_kp[0]], axis=0)  # (N, 3)  # the current eef_kp
        Rr, Rs = construct_edges_from_states(torch.tensor(states).unsqueeze(0), adj_thresh, 
                                             mask=torch.tensor(state_mask).unsqueeze(0), 
                                             eef_mask=torch.tensor(eef_mask).unsqueeze(0),
                                             no_self_edge=True)
        Rr, Rs = Rr.squeeze(0).numpy(), Rs.squeeze(0).numpy()

        # action encoded as state_delta (only stored in eef keypoints)
        states_delta = np.zeros((obj_kp_num + eef_kp_num, states.shape[-1]), dtype=np.float32)
        states_delta[obj_kp_num : obj_kp_num + ptcl_per_eef] = eef_kp[1] - eef_kp[0]

        # next state
        pred_gap = 1  # TODO make this code compatible with pred_gap > 1
        obj_kp_path_next = obj_kp_paths[i + pred_gap]
        obj_kp_next = pkl.load(open(obj_kp_path_next, 'rb')) # list of (rand_ptcl_num, 3)
        obj_kp_next = [kp[:top_k] for kp in obj_kp_next]
        obj_kp_next = np.concatenate(obj_kp_next, axis=0) # (N = instance_num * rand_ptcl_num, 3)
        # eef_kp_path_next = os.path.join(eef_kp_dir, f'{i+pred_gap:06}.npy')
        # eef_kp_next = np.load(eef_kp_path_next).astype(np.float32)   # (2, 8, 3)
        # assert eef_kp_next is not None, print(eef_kp_path_next)
        # assert np.max(eef_kp_next[0] - eef_kp[1]) < 4e-2, print(np.max(eef_kp_next[0] - eef_kp[1]))  # check consistency (need to handle jumps)
        # states_next = np.concatenate([obj_kp_next, eef_kp_next[0]], axis=0)
        # states_next = np.concatenate([obj_kp_next, np.zeros_like(eef_kp[0])], axis=0)
        states_next = obj_kp_next
        if max_nobj is not None:
            # pad states_next
            states_pad = np.zeros((max_nobj, 3), dtype=np.float32)
            states_pad[:states_next.shape[0]] = states_next
            states_next = states_pad

        # history state
        states = states[None]  # n_his = 1  # TODO make this code compatible with n_his > 1

        # save graph
        graph = {
            "attrs": attrs, 
            "state": states, 
            "action": states_delta,
            "Rr": Rr, 
            "Rs": Rs,
            "p_rigid": p_rigid,
            "p_instance": p_instance,
            "physics_param": physics_param,
            "particle_den": np.array([adj_thresh]),
            "state_next": states_next,
            "obj_mask": obj_mask,
        }
        # print([f"{key}: {val.shape}" for key, val in graph.items()])
        graph_list.append(graph)

    if max_nR is not None:
        max_n_Rr = max_nR
        max_n_Rs = max_nR
    else:
        # print([graph['Rr'].shape[0] for graph in graph_list])
        max_n_Rr = max([graph['Rr'].shape[0] for graph in graph_list])
        max_n_Rs = max([graph['Rs'].shape[0] for graph in graph_list])

    print("max number of relations Rr: ", max_n_Rr)
    print("max number of relations Rs: ", max_n_Rs)

    for i, graph in enumerate(graph_list):
        graph["Rr"] = np.pad(graph["Rr"], ((0, max_n_Rr - graph["Rr"].shape[0]), (0, 0)), mode='constant')
        graph["Rs"] = np.pad(graph["Rs"], ((0, max_n_Rs - graph["Rs"].shape[0]), (0, 0)), mode='constant')
        save_path = os.path.join(save_dir, f'{i:06}.pkl')
        pkl.dump(graph, open(save_path, 'wb'))


def preprocess_graph_old(args, data_dir):  # save states, relations and attributes; use valid_list, deprecated
    # data_dir = "../data/2023-09-04-18-42-27-707743/"

    save_dir = os.path.join(data_dir, "graph")
    os.makedirs(save_dir, exist_ok=True)

    len_img_paths = len(list(glob.glob(os.path.join(data_dir, "camera_0/color/color_*.png"))))
    img_paths = [os.path.join(data_dir, "camera_0/color/color_{}.png".format(i)) for i in range(len_img_paths)]

    valid_list = np.loadtxt(os.path.join(data_dir, 'valid_frames.txt'), dtype=np.int32)
    print(len(valid_list), len(img_paths))

    pkl_paths = sorted(list(glob.glob(os.path.join(data_dir, 'obj_kypts_orig', '*.pkl'))))

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
    _ = gen_model(args, material_dict=None, debug=True)

    # load_keypoints(args, "../data/2023-09-04-18-26-45-932029")
    # preprocess_graph(args, "../data/2023-09-04-18-26-45-932029")
    # raise Exception

    # base_data_dir = "../data"
    # data_dir_list = glob.glob(os.path.join(base_data_dir, "*"))
    data_dir_list = [
        "../data/2023-08-30-03-04-49-509758",
        "../data/2023-08-23-12-08-12-201998",
        "../data/2023-08-30-02-02-53-617979",
        "../data/2023-08-30-01-45-16-257548",
        "../data/2023-08-23-12-23-07-775716",
        "../data/2023-08-30-02-20-39-572700",
        "../data/2023-08-30-03-44-14-098121",
        "../data/2023-08-30-00-54-02-790828",
        "../data/2023-08-30-02-48-27-532912",
        "../data/2023-08-29-17-49-04-904390",
        "../data/2023-08-30-03-27-55-702301",
        "../data/2023-08-29-18-07-15-165315",
        "../data/2023-08-23-12-17-53-370195",
        "../data/2023-09-04-18-42-27-707743",
        "../data/2023-08-30-04-16-04-497588",
        # "../data/2023-08-29-21-23-47-258600",
        "../data/2023-08-30-04-00-39-286249",
        "../data/2023-08-29-20-10-13-123194",
        "../data/2023-09-04-18-26-45-932029",
        "../data/2023-08-30-00-47-16-839238",
    ]
    for data_dir in data_dir_list:
        if os.path.isdir(data_dir):
            print(data_dir)
            # load_fingers(args, data_dir, idx=12)
            # load_pushes(args, data_dir, idx=12)
            # load_keypoints(args, data_dir)
            # extract_pushes(args, data_dir)
            extract_pushes_2(args, data_dir)
            # kypts_verification(args, data_dir)
            # preprocess_graph(args, data_dir, max_n=2, max_nobj=40, max_neef=8, max_nR=200, dense=True)
            # preprocess_graph(args, data_dir, max_n=2, max_nobj=40, max_neef=8, max_nR=200, dense=False)
