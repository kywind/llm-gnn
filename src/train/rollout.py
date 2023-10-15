import glob
import numpy as np
import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import gen_args
from gnn.model_wrapper import gen_model
from gnn.utils import set_seed

import open3d as o3d

import cv2
import glob
from PIL import Image
import pickle as pkl
from dgl.geometry import farthest_point_sampler
from train.random_rigid_dataset import construct_edges_from_states
from train.train_rigid import truncate_graph
from data.utils import label_colormap


def rollout_rigid(args, data_dir, checkpoint, start_idx, rollout_steps, dense):
    set_seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    save_dir = f"vis/rollout-vis-{data_dir.split('/')[-1]}-dense" if dense else f"vis/rollout-vis-{data_dir.split('/')[-1]}"
    os.makedirs(save_dir, exist_ok=True)

    obj_kypts_paths = []
    eef_kypts_paths = []
    eef_kypts_dir = os.path.join(data_dir, 'dense_eef_kypts_2' if dense else 'eef_kypts_2')
    eef_kypts_path_list = sorted(glob.glob(os.path.join(eef_kypts_dir, '*.npy')))
    obj_kypts_dir = os.path.join(data_dir, 'dense_obj_kypts_2' if dense else 'obj_kypts_2')
    obj_kypts_path_list = sorted(glob.glob(os.path.join(obj_kypts_dir, '*.pkl')))
    obj_kypts_paths.extend(obj_kypts_path_list)
    eef_kypts_paths.extend(eef_kypts_path_list)
    print(f'Found {len(obj_kypts_path_list)} obj keypoint files in {obj_kypts_dir}')
    print(f'Found {len(eef_kypts_path_list)} eef keypoint files in {eef_kypts_dir}')
    assert len(obj_kypts_path_list) == len(eef_kypts_path_list)

    t_list = np.loadtxt(os.path.join(data_dir, 'push_t_list_2_dense.txt' if dense else 'push_t_list_2.txt'), dtype=np.int32)

    intr_list = [None] * 4
    extr_list = [None] * 4
    for cam in range(4):
        os.makedirs(os.path.join(save_dir, f'camera_{cam}'), exist_ok=True)
        intr_list[cam] = np.load(os.path.join(data_dir, f"camera_{cam}/camera_params.npy"))
        extr_list[cam] = np.load(os.path.join(data_dir, f"camera_{cam}/camera_extrinsics.npy"))
    
    # load model
    model, loss_funcs = gen_model(args, checkpoint=checkpoint, 
                                  material_dict=None, verbose=True, debug=False)
    model = model.to(device)
    model.eval()

    # load kypts
    fixed_idx = False
    top_k = 20
    fps_idx_list = []

    obj_kp_path = obj_kypts_paths[start_idx]
    eef_kp_path = eef_kypts_paths[start_idx]
    with open(obj_kp_path, 'rb') as f:
        obj_kp, obj_kp_next = pkl.load(f) # lists of (rand_ptcl_num, 3)
    eef_kp = np.load(eef_kp_path).astype(np.float32)   # (2, 8, 3)
    eef_kp_num = eef_kp.shape[1]

    # TODO replace this sampling to a random version
    if not fixed_idx or (fixed_idx and fps_idx_list == []):
        fps_idx_list = []
        for j in range(len(obj_kp)):
            # farthest point sampling
            particle_tensor = torch.from_numpy(obj_kp[j]).float()[None, ...]
            fps_idx_tensor = farthest_point_sampler(particle_tensor, top_k, start_idx=np.random.randint(0, obj_kp[j].shape[0]))[0]
            fps_idx = fps_idx_tensor.numpy().astype(np.int32)
            fps_idx_list.append(fps_idx)
    obj_kp = [obj_kp[j][fps_idx] for j, fps_idx in enumerate(fps_idx_list)]
    instance_num = len(obj_kp)
    ptcl_per_instance = obj_kp[0].shape[0]
    obj_kp = np.concatenate(obj_kp, axis=0) # (N = instance_num * rand_ptcl_num, 3)
    obj_kp_num = obj_kp.shape[0]

    # obj_kp_next = [kp[:top_k] for kp in obj_kp_next]
    obj_kp_next = [obj_kp_next[j][fps_idx] for j, fps_idx in enumerate(fps_idx_list)]
    obj_kp_next = np.concatenate(obj_kp_next, axis=0) # (N = instance_num * rand_ptcl_num, 3)
    assert obj_kp_next.shape[0] == obj_kp_num

    max_nobj = 40
    if max_nobj is not None:
        # pad obj_kp
        obj_kp_pad = np.zeros((max_nobj, 3), dtype=np.float32)
        obj_kp_pad[:obj_kp_num] = obj_kp
        obj_kp = obj_kp_pad
        # pad obj_kp_next
        obj_kp_next_pad = np.zeros((max_nobj, 3), dtype=np.float32)
        obj_kp_next_pad[:obj_kp_num] = obj_kp_next
        obj_kp_next = obj_kp_next_pad
    else:
        print("not using max_nobj which only works for fixed number of obj keypoints")
        max_nobj = obj_kp_num
    
    max_neef = 8
    if max_neef is not None:
        # pad eef_kp
        eef_kp_pad = np.zeros((2, max_neef, 3), dtype=np.float32)
        eef_kp_pad[:, :eef_kp_num] = eef_kp
        eef_kp = eef_kp_pad
    else:
        print("not using max_neef which only works for fixed number of eef keypoints")
        max_neef = eef_kp_num
    
    max_n = 2
    if max_n is not None:
        assert max_n >= instance_num
    else:
        print("not using max_n which only works for fixed number of objects")
        max_n = instance_num

    state_mask = np.zeros((max_nobj + max_neef), dtype=bool)
    state_mask[max_nobj : max_nobj + eef_kp_num] = True
    state_mask[:obj_kp_num] = True

    eef_mask = np.zeros((max_nobj + max_neef), dtype=bool)
    eef_mask[max_nobj : max_nobj + eef_kp_num] = True

    obj_mask = np.zeros((max_nobj,), dtype=bool)
    obj_mask[:obj_kp_num] = True

    # construct instance information
    p_rigid = np.ones(max_n, dtype=np.float32)  # clothes are nonrigid
    p_instance = np.zeros((max_nobj, max_n), dtype=np.float32)
    j_perm = np.random.permutation(instance_num)
    for j in range(instance_num):
        p_instance[j * ptcl_per_instance : (j + 1) * ptcl_per_instance, j_perm[j]] = 1  # TODO different number of particles per instance
    physics_param = np.zeros(max_nobj, dtype=np.float32)  # 1-dim

    # construct attributes
    attr_dim = 2
    assert attr_dim == args.attr_dim
    attrs = np.zeros((max_nobj + max_neef, attr_dim), dtype=np.float32)
    attrs[:obj_kp_num, 0] = 1.
    attrs[max_nobj : max_nobj + eef_kp_num, 1] = 1.

    # TODO construct relations here or after collate_fn?
    # construct relations (density as hyperparameter)
    # adj_thresh = np.random.uniform(*adj_thresh_range)
    adj_thresh = 0.15
    states = np.concatenate([obj_kp, eef_kp[0]], axis=0)  # (N, 3)  # the current eef_kp
    Rr, Rs = construct_edges_from_states(torch.tensor(states).unsqueeze(0), adj_thresh, 
                                        mask=torch.tensor(state_mask).unsqueeze(0), 
                                        eef_mask=torch.tensor(eef_mask).unsqueeze(0),
                                        no_self_edge=True)
    Rr, Rs = Rr.squeeze(0).numpy(), Rs.squeeze(0).numpy()

    # action encoded as state_delta (only stored in eef keypoints)
    states_delta = np.zeros((max_nobj + max_neef, states.shape[-1]), dtype=np.float32)
    states_delta[max_nobj : max_nobj + eef_kp_num] = eef_kp[1] - eef_kp[0]

    # history state
    states = states[None]  # n_his = 1  # TODO make this code compatible with n_his > 1

    max_nR = 500
    Rr = np.pad(Rr, ((0, max_nR - Rr.shape[0]), (0, 0)), mode='constant')
    Rs = np.pad(Rs, ((0, max_nR - Rs.shape[0]), (0, 0)), mode='constant')

    # generate graph (from dataloader)
    graph = {
        "attrs": attrs,  # (N+M, attr_dim)
        "state": states,  # (n_his, N+M, state_dim)
        "action": states_delta,  # (N+M, state_dim)
        "Rr": Rr,  # (n_rel, N+M)
        "Rs": Rs,  # (n_rel, N+M)
        "p_rigid": p_rigid,  # (n_instance,)
        "p_instance": p_instance,  # (N, n_instance)
        "physics_param": physics_param,  # (N,)
        "state_next": obj_kp_next,  # (N, state_dim)
        "obj_mask": obj_mask,  # (N,)
    }
    
    push_start = t_list[start_idx, 0]
    push_end = t_list[start_idx, 1]

    colormap = label_colormap()
    pred_kp_proj_last = []
    gt_kp_proj_last = []
    for cam in range(4):
        img_path = os.path.join(data_dir, f'camera_{cam}/color', f'color_{push_start}.png')
        img = cv2.imread(img_path)
        intr = intr_list[cam]
        extr = extr_list[cam]
        save_dir_cam = os.path.join(save_dir, f'camera_{cam}')

        # transform keypoints
        kp = states[0, :obj_kp_num]
        obj_kp_homo = np.concatenate([kp, np.ones((kp.shape[0], 1))], axis=1) # (N, 4)
        obj_kp_homo = obj_kp_homo @ extr.T  # (N, 4)

        # project keypoints
        fx, fy, cx, cy = intr
        obj_kp_proj = np.zeros((obj_kp_homo.shape[0], 2))
        obj_kp_proj[:, 0] = obj_kp_homo[:, 0] * fx / obj_kp_homo[:, 2] + cx
        obj_kp_proj[:, 1] = obj_kp_homo[:, 1] * fy / obj_kp_homo[:, 2] + cy

        # visualize
        for k in range(obj_kp_proj.shape[0]):
            cv2.circle(img, (int(obj_kp_proj[k, 0]), int(obj_kp_proj[k, 1])), 3, 
                (int(colormap[k, 2]), int(colormap[k, 1]), int(colormap[k, 0])), -1)
        
        pred_kp_proj_last.append(obj_kp_proj)
        gt_kp_proj_last.append(obj_kp_proj)
        
        cv2.imwrite(os.path.join(save_dir_cam, f'{push_start:06}_pred.jpg'), img)
        cv2.imwrite(os.path.join(save_dir_cam, f'{push_start:06}_gt.jpg'), img)
        img = np.concatenate([img, img], axis=1)
        cv2.imwrite(os.path.join(save_dir_cam, f'{push_start:06}_both.jpg'), img)

    graph = {key: torch.from_numpy(graph[key]).unsqueeze(0).to(device) for key in graph.keys()}

    # iterative rollout
    gt_state_list = []
    pred_state_list = []
    gt_lineset = [[], [], [], []]
    pred_lineset = [[], [], [], []]
    with torch.no_grad():
        for i in range(start_idx + 1, start_idx + 1 + rollout_steps):
            # only show one step of lineset
            gt_lineset = [[], [], [], []]
            pred_lineset = [[], [], [], []]

            graph = truncate_graph(graph)
            gt_state = graph['state_next'].detach().cpu().numpy()
            gt_state_list.append(graph)
            pred_state, pred_motion = model(**graph)
            pred_state = pred_state.detach().cpu().numpy()
            pred_state_list.append(pred_state)

            # next step input
            obj_kp = pred_state[0][obj_mask]
            gt_kp = gt_state[0][obj_mask]

            push_i = t_list[i, 0]

            pred_kp_proj_list = []
            gt_kp_proj_list = []
            for cam in range(4):
                img_path = os.path.join(data_dir, f'camera_{cam}/color', f'color_{push_i}.png')
                img_orig = cv2.imread(img_path)
                img = img_orig.copy()
                intr = intr_list[cam]
                extr = extr_list[cam]
                save_dir_cam = os.path.join(save_dir, f'camera_{cam}')

                # transform keypoints
                obj_kp_homo = np.concatenate([obj_kp, np.ones((obj_kp.shape[0], 1))], axis=1) # (N, 4)
                obj_kp_homo = obj_kp_homo @ extr.T  # (N, 4)

                # project keypoints
                fx, fy, cx, cy = intr
                obj_kp_proj = np.zeros((obj_kp_homo.shape[0], 2))
                obj_kp_proj[:, 0] = obj_kp_homo[:, 0] * fx / obj_kp_homo[:, 2] + cx
                obj_kp_proj[:, 1] = obj_kp_homo[:, 1] * fy / obj_kp_homo[:, 2] + cy

                pred_kp_proj_list.append(obj_kp_proj)

                # visualize
                for k in range(obj_kp_proj.shape[0]):
                    cv2.circle(img, (int(obj_kp_proj[k, 0]), int(obj_kp_proj[k, 1])), 3, 
                        (int(colormap[k, 2]), int(colormap[k, 1]), int(colormap[k, 0])), -1)

                pred_kp_last = pred_kp_proj_last[cam]
                for k in range(obj_kp_proj.shape[0]):
                    pred_lineset[cam].append([int(obj_kp_proj[k, 0]), int(obj_kp_proj[k, 1]), int(pred_kp_last[k, 0]), int(pred_kp_last[k, 1]), 
                                         int(colormap[k, 2]), int(colormap[k, 1]), int(colormap[k, 0]), i])

                for k in range(len(pred_lineset[cam])):
                    ln = pred_lineset[cam][k]
                    cv2.line(img, (ln[0], ln[1]), (ln[2], ln[3]), (ln[4], ln[5], ln[6]), 1)
                
                cv2.imwrite(os.path.join(save_dir_cam, f'{push_i:06}_pred.jpg'), img)
                img_pred = img.copy()

                # visualize gt similarly
                img = img_orig.copy()
                gt_kp_homo = np.concatenate([gt_kp, np.ones((gt_kp.shape[0], 1))], axis=1) # (N, 4)
                gt_kp_homo = gt_kp_homo @ extr.T  # (N, 4)
                gt_kp_proj = np.zeros((gt_kp_homo.shape[0], 2))
                gt_kp_proj[:, 0] = gt_kp_homo[:, 0] * fx / gt_kp_homo[:, 2] + cx
                gt_kp_proj[:, 1] = gt_kp_homo[:, 1] * fy / gt_kp_homo[:, 2] + cy

                gt_kp_proj_list.append(gt_kp_proj)
                
                for k in range(gt_kp_proj.shape[0]):
                    cv2.circle(img, (int(gt_kp_proj[k, 0]), int(gt_kp_proj[k, 1])), 3, 
                        (int(colormap[k, 2]), int(colormap[k, 1]), int(colormap[k, 0])), -1)

                gt_kp_last = gt_kp_proj_last[cam]
                for k in range(gt_kp_proj.shape[0]):
                    gt_lineset[cam].append([int(gt_kp_proj[k, 0]), int(gt_kp_proj[k, 1]), int(gt_kp_last[k, 0]), int(gt_kp_last[k, 1]), 
                                       int(colormap[k, 2]), int(colormap[k, 1]), int(colormap[k, 0]), i])

                for k in range(len(gt_lineset[cam])):
                    ln = gt_lineset[cam][k]
                    cv2.line(img, (ln[0], ln[1]), (ln[2], ln[3]), (ln[4], ln[5], ln[6]), 1)

                cv2.imwrite(os.path.join(save_dir_cam, f'{push_i:06}_gt.jpg'), img)
                img_gt = img.copy()

                img = np.concatenate([img_pred, img_gt], axis=1)
                cv2.imwrite(os.path.join(save_dir_cam, f'{push_i:06}_both.jpg'), img)


            pred_kp_proj_last = pred_kp_proj_list
            gt_kp_proj_last = gt_kp_proj_list

            # generate next graph
            obj_kp_path = obj_kypts_paths[i]
            eef_kp_path = eef_kypts_paths[i]
            with open(obj_kp_path, 'rb') as f:
                _, obj_kp_next = pkl.load(f) # lists of (rand_ptcl_num, 3)
            eef_kp = np.load(eef_kp_path).astype(np.float32)   # (2, 8, 3)
            eef_kp_num = eef_kp.shape[1]

            # obj_kp_next = [kp[:top_k] for kp in obj_kp_next]
            obj_kp_next = [obj_kp_next[j][fps_idx] for j, fps_idx in enumerate(fps_idx_list)]
            obj_kp_next = np.concatenate(obj_kp_next, axis=0) # (N = instance_num * rand_ptcl_num, 3)
            assert obj_kp_next.shape[0] == obj_kp_num

            if max_nobj is not None:
                # pad obj_kp
                obj_kp_pad = np.zeros((max_nobj, 3), dtype=np.float32)
                obj_kp_pad[:obj_kp_num] = obj_kp
                obj_kp = obj_kp_pad
                # pad obj_kp_next
                obj_kp_next_pad = np.zeros((max_nobj, 3), dtype=np.float32)
                obj_kp_next_pad[:obj_kp_num] = obj_kp_next
                obj_kp_next = obj_kp_next_pad
            else:
                print("not using max_nobj which only works for fixed number of obj keypoints")
                max_nobj = obj_kp_num

            if max_neef is not None:
                # pad eef_kp
                eef_kp_pad = np.zeros((2, max_neef, 3), dtype=np.float32)
                eef_kp_pad[:, :eef_kp_num] = eef_kp
                eef_kp = eef_kp_pad
            else:
                print("not using max_neef which only works for fixed number of eef keypoints")
                max_neef = eef_kp_num

            states = np.concatenate([obj_kp, eef_kp[0]], axis=0)  # (N, 3)  # the current eef_kp
            Rr, Rs = construct_edges_from_states(torch.tensor(states).unsqueeze(0), adj_thresh, 
                                                mask=torch.tensor(state_mask).unsqueeze(0), 
                                                eef_mask=torch.tensor(eef_mask).unsqueeze(0),
                                                no_self_edge=True)
            Rr, Rs = Rr.squeeze(0).numpy(), Rs.squeeze(0).numpy()

            # action encoded as state_delta (only stored in eef keypoints)
            states_delta = np.zeros((max_nobj + max_neef, states.shape[-1]), dtype=np.float32)
            states_delta[max_nobj : max_nobj + eef_kp_num] = eef_kp[1] - eef_kp[0]

            # history state
            states = states[None]  # n_his = 1  # TODO make this code compatible with n_his > 1

            max_nR = 500
            Rr = np.pad(Rr, ((0, max_nR - Rr.shape[0]), (0, 0)), mode='constant')
            Rs = np.pad(Rs, ((0, max_nR - Rs.shape[0]), (0, 0)), mode='constant')

            
            next_graph = {
                "attrs": graph["attrs"],  # (N+M, attr_dim)
                "state": torch.from_numpy(states).unsqueeze(0).to(device),  # (n_his, N+M, state_dim)
                "action": torch.from_numpy(states_delta).unsqueeze(0).to(device),  # (N+M, state_dim)
                "Rr": torch.from_numpy(Rr).unsqueeze(0).to(device),  # (n_rel, N+M)
                "Rs": torch.from_numpy(Rs).unsqueeze(0).to(device),  # (n_rel, N+M)
                "p_rigid": graph["p_rigid"],  # (n_instance,)
                "p_instance": graph["p_instance"],  # (N, n_instance)
                "physics_param": graph["physics_param"],  # (N,)
                "state_next": torch.from_numpy(obj_kp_next).unsqueeze(0).to(device),  # (N, state_dim)
                "obj_mask": graph["obj_mask"],  # (N,)
            }
            graph = next_graph
    
    print(len(gt_state_list))
    print(len(pred_state_list))


if __name__ == "__main__":
    args = gen_args()
    start_idx = 0
    rollout_steps = 100
    data_dir = "../data/2023-08-23-12-08-12-201998"
    checkpoint = "../log/rigid_dense_debug_1/checkpoints/model_1000.pth"
    dense = True
    rollout_rigid(args, data_dir, checkpoint, start_idx, rollout_steps, dense)

    for cam in range(4):
        img_path = f"vis/rollout-vis-{data_dir.split('/')[-1]}-dense/camera_{cam}" if dense else f"vis/rollout-vis-{data_dir.split('/')[-1]}/camera_{cam}"
        frame_rate = 5
        height = 360
        width = 640
        pred_out_path = os.path.join(img_path, "pred.mp4")
        os.system(f"ffmpeg -loglevel panic -r {frame_rate} -f image2 -s {width}x{height} -pattern_type glob -i '{img_path}/*_pred.jpg' -vcodec libx264 -crf 25 -pix_fmt yuv420p {pred_out_path} -y")
        gt_out_path = os.path.join(img_path, "gt.mp4")
        os.system(f"ffmpeg -loglevel panic -r {frame_rate} -f image2 -s {width}x{height} -pattern_type glob -i '{img_path}/*_gt.jpg' -vcodec libx264 -crf 25 -pix_fmt yuv420p {gt_out_path} -y")
        both_out_path = os.path.join(img_path, "both.mp4")
        os.system(f"ffmpeg -loglevel panic -r {frame_rate} -f image2 -s {width}x{height} -pattern_type glob -i '{img_path}/*_both.jpg' -vcodec libx264 -crf 25 -pix_fmt yuv420p {both_out_path} -y")
