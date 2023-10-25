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


# shoe pushing
data_dir = "../data/rope"
prep_save_dir = "log/rope_debug_1/preprocess/rope"
save_dir = f"vis/graph-vis-{data_dir.split('/')[-1]}"
os.makedirs(save_dir, exist_ok=True)

obj_kypts_paths = []
eef_kypts_paths = []
eef_kypts_dir = os.path.join(prep_save_dir, 'kypts', 'eef_kypts')
eef_kypts_path_list = sorted(glob.glob(os.path.join(eef_kypts_dir, '*.npy')))
obj_kypts_dir = os.path.join(prep_save_dir, 'kypts', 'obj_kypts')
obj_kypts_path_list = sorted(glob.glob(os.path.join(obj_kypts_dir, '*.npy')))
obj_kypts_paths.extend(obj_kypts_path_list)
eef_kypts_paths.extend(eef_kypts_path_list)
print(f'Found {len(obj_kypts_path_list)} obj keypoint files in {obj_kypts_dir}')
print(f'Found {len(eef_kypts_path_list)} eef keypoint files in {eef_kypts_dir}')
assert len(obj_kypts_path_list) == len(eef_kypts_path_list)

# t_list = np.loadtxt(os.path.join(data_dir, 'push_t_list_dense.txt' if dense else 'push_t_list.txt'), dtype=np.int32)

colormap = label_colormap()

intr_list = [None] * 4
extr_list = [None] * 4
for cam in range(4):
    save_dir_cam = os.path.join(save_dir, f'camera_{cam+1}')
    os.makedirs(save_dir_cam, exist_ok=True)
    intr_list[cam] = np.load(os.path.join(data_dir, "camera_intrinsic_params.npy"))[cam]
    extr_list[cam] = np.load(os.path.join(data_dir, "camera_extrinsic_matrix.npy"))[cam]

# img_list = []  # (4, N)
for i in range(len(obj_kypts_paths)):
    obj_kp_path = obj_kypts_paths[i]
    eef_kp_path = eef_kypts_paths[i]
    # print(obj_kp_path)
    # print(eef_kp_path)

    # with open(obj_kp_path, 'rb') as f:
    #     obj_kp, obj_kp_next = pkl.load(f)
    obj_kp = np.load(obj_kp_path).astype(np.float32)
    obj_kp_next = obj_kp[1]
    obj_kp = obj_kp[0]
    eef_kp = np.load(eef_kp_path).astype(np.float32)

    # top_k = 20
    # # fps_idx_list = []
    # # for j in range(len(obj_kp)):
    # #     # farthest point sampling
    # #     particle_tensor = torch.from_numpy(obj_kp[j]).float()[None, ...]
    # #     fps_idx_tensor = farthest_point_sampler(particle_tensor, top_k, start_idx=np.random.randint(0, obj_kp[j].shape[0]))[0]
    # #     fps_idx = fps_idx_tensor.numpy().astype(np.int32)
    # #     fps_idx_list.append(fps_idx)
    # # obj_kp = [obj_kp[j][fps_idx] for j, fps_idx in enumerate(fps_idx_list)]
    # obj_kp = [kp[:top_k] for kp in obj_kp]
    # instance_num = len(obj_kp)
    # ptcl_per_instance = obj_kp[0].shape[0]
    # obj_kp = np.concatenate(obj_kp, axis=0) # (N = instance_num * rand_ptcl_num, 3)
    obj_kp_num = obj_kp.shape[0]
    eef_kp_num = eef_kp.shape[1]

    # # obj_kp_next = [obj_kp_next[j][fps_idx] for j, fps_idx in enumerate(fps_idx_list)]
    # obj_kp_next = [kp[:top_k] for kp in obj_kp_next]
    # obj_kp_next = np.concatenate(obj_kp_next, axis=0) # (N = instance_num * rand_ptcl_num, 3)
    # assert obj_kp_next.shape[0] == obj_kp_num

    episode_id = int(obj_kp_path.split('/')[-1].split('.')[0].split('_')[0])
    img_id = int(obj_kp_path.split('/')[-1].split('.')[0].split('_')[1])
    img_id_next = img_id + 1

    for cam in range(4):
        save_dir_cam = os.path.join(save_dir, f'camera_{cam+1}')
        intr = intr_list[cam]
        extr = extr_list[cam]

        for idx, (j, kp) in enumerate([(img_id, obj_kp), (img_id_next, obj_kp_next)]):
            img_path = os.path.join(data_dir, f'episode_{episode_id}', f'camera_{cam+1}', f'{j}_color.png')
            img = cv2.imread(img_path)

            # transform keypoints
            obj_kp_homo = np.concatenate([kp, np.ones((kp.shape[0], 1))], axis=1) # (N, 4)
            obj_kp_homo = obj_kp_homo @ extr.T  # (N, 4)

            obj_kp_homo[:, 1] *= -1
            obj_kp_homo[:, 2] *= -1

            # project keypoints
            fx, fy, cx, cy = intr
            obj_kp_proj = np.zeros((obj_kp_num, 2))
            obj_kp_proj[:, 0] = obj_kp_homo[:, 0] * fx / obj_kp_homo[:, 2] + cx
            obj_kp_proj[:, 1] = obj_kp_homo[:, 1] * fy / obj_kp_homo[:, 2] + cy

            for k in range(obj_kp_proj.shape[0]):
                cv2.circle(img, (int(obj_kp_proj[k, 0]), int(obj_kp_proj[k, 1])), 3, 
                    (int(colormap[k, 2]), int(colormap[k, 1]), int(colormap[k, 0])), -1)

            if idx == 0:
                # transform eef keypoints
                kpe = eef_kp.reshape(-1, 3)
                kpe[:, 2] *= -1
                eef_kp_homo = np.concatenate([kpe, np.ones((kpe.shape[0], 1))], axis=1) # (N, 4)
                eef_kp_homo = eef_kp_homo @ extr.T  # (N, 4)

                eef_kp_homo[:, 1] *= -1
                eef_kp_homo[:, 2] *= -1

                # project keypoints
                fx, fy, cx, cy = intr
                eef_kp_proj = np.zeros((2 * eef_kp_num, 2))
                eef_kp_proj[:, 0] = eef_kp_homo[:, 0] * fx / eef_kp_homo[:, 2] + cx
                eef_kp_proj[:, 1] = eef_kp_homo[:, 1] * fy / eef_kp_homo[:, 2] + cy

                eef_kp_proj = eef_kp_proj.reshape(2, eef_kp_num, 2)
                for k in range(eef_kp_num):
                    cv2.circle(img, (int(eef_kp_proj[0, k, 0]), int(eef_kp_proj[0, k, 1])), 3, 
                        (0, 0, 255), -1)
                    cv2.circle(img, (int(eef_kp_proj[1, k, 0]), int(eef_kp_proj[1, k, 1])), 3, 
                        (0, 255, 0), -1)
                    cv2.line(img, (int(eef_kp_proj[0, k, 0]), int(eef_kp_proj[0, k, 1])), (int(eef_kp_proj[1, k, 0]), int(eef_kp_proj[1, k, 1])), 
                        (0, 0, 255), 1)

            print(os.path.join(save_dir_cam, f'{episode_id:03}_{j:03}.jpg'))
            cv2.imwrite(os.path.join(save_dir_cam, f'{episode_id:03}_{j:03}.jpg'), img)
