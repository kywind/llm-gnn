import glob
import numpy as np
import os
import torch
import torchvision
import torchvision.transforms as transforms

from config import gen_args
from gnn.model_wrapper import gen_model
from gnn.utils import set_seed

from data.multiview_dataparser import MultiviewDataparser
from data.multiview_tabletop_dataparser import MultiviewTabletopDataparser
from data.multiview_dataset import MultiviewParticleDataset
from llm.llm import LLM
import pyflex
import pickle
import open3d as o3d
import copy

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib

import cv2
import glob


def build_graph(args):
    set_seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    ## configs
    visualize = False
    verbose = False
    skip_segment = False
    tabletop_3in1_query = True

    camera_indices = [0, 1, 2, 3]
    data_dir = "../data/2023-09-13-15-19-50-765863/"
    vis_dir = "vis/multiview-tabletop-0/"
    dataset_name = "d3fields"

    if skip_segment:
        assert os.path.exists(vis_dir), "vis_dir does not exist"
        assert os.path.exists(os.path.join(vis_dir, 'mask_0_0.png')), "mask_0_0.png does not exist"
    os.makedirs(vis_dir, exist_ok=True)

    # initial dataparser
    init_parser = MultiviewTabletopDataparser(args, data_dir, dataset_name)
    llm = LLM(args)

    if not tabletop_3in1_query:
        init_parser.prepare_query_model()
        obj_list = []
        for camera_index in camera_indices:
            obj_list_view = []
            query_results = init_parser.query(
                texts='What objects are on the table? Answer:',
                camera_index=camera_index
            )
            objs = query_results.rstrip('.').split(',')
            for obj in objs:
                obj = obj.strip(' ')
                obj = obj.lstrip('and')
                obj = obj.strip(' ')
                if obj not in obj_list:
                    obj_list.append(obj)
                if obj not in obj_list_view:
                    obj_list_view.append(obj)
            obj_list.append(obj_list_view)
        init_parser.del_query_model()
        print('After query, obj_list:', obj_list)
    else:
        obj_list = []
        obj_list, mask_list, text_label_list = init_parser.tabletop_3in1_query()

    material_dict = {}
    for obj_name in obj_list:
        if obj_name not in material_dict:
            material_prompt = " ".join([
                "Classify the objects in the image as rigid objects, granular objects, deformable objects, or rope.",
                "Respond unknown if you are not sure."
                "\n\nQuery: coffee bean. Answer: granular.",
                "\n\nQuery: rope. Answer: rope.",
                "\n\nQuery: a wooden block. Answer: rigid.",
                "\n\nQuery: a banana. Answer: rigid.",
                "\n\nQuery: an orange. Answer: rigid.",
                "\n\nQuery: play-doh. Answer: deformable.",
                "\n\nQuery: sand. Answer: granular.",
                "\n\nQuery: a bottle. Answer: rigid.",
                "\n\nQuery: a t-shirt. Answer: deformable.",
                "\n\nQuery: rice. Answer: granular.",
                "\n\nQuery: laptop. Answer: rigid.",
                "\n\nQuery: " + obj_name + ". Answer:",
            ])
            material = llm.query(material_prompt)
            material = material.rstrip('.')
            material_dict[obj_name] = material

    # set gnn model
    model, model_loss = gen_model(args, material_dict)
    model.eval()
    model = model.to(device)

    mask_list = []
    text_labels_list = []

    if not skip_segment and not tabletop_3in1_query:
        for camera_index in camera_indices:
            segmentation_results, detection_results = init_parser.segment_gdino(
                obj_list=obj_list,
                camera_index=camera_index,
            )
            masks, _, text_labels = segmentation_results
            # _, _, labels = detection_results

            masks = masks.detach().cpu().numpy()  # (n_detect, H, W) boolean            
            for i in range(masks.shape[0]):
                mask = (masks[i] * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(vis_dir, 'mask_{}_{}.png'.format(camera_index, i)), mask)
                with open(os.path.join(vis_dir, 'text_labels_{}_{}.txt'.format(camera_index, i)), 'w') as f:
                    f.write(text_labels[i])
            
            mask_list.append(masks)
            text_labels_list.append(text_labels)

    elif skip_segment and not tabletop_3in1_query:
        for camera_index in camera_indices:
            masks = []
            text_labels = []
            mask_dirs = glob.glob(os.path.join(vis_dir, 'mask_{}*.png'.format(camera_index)))
            for i in range(len(mask_dirs)):
                mask = cv2.imread(os.path.join(vis_dir, 'mask_{}_{}.png'.format(camera_index, i)))
                mask = (mask > 0)[..., 0]  # (H, W) boolean
                masks.append(mask)
                with open(os.path.join(vis_dir, 'text_labels_{}_{}.txt'.format(camera_index, i)), 'r') as f:
                    text_labels.append(f.read())
            masks = np.stack(masks, axis=0)

            mask_list.append(masks)
            text_labels_list.append(text_labels)

    # set dataset
    cam_list = (init_parser.cam_params, init_parser.cam_extrinsics)

    data = MultiviewParticleDataset(
        args=args,
        depths=init_parser.depth_imgs,
        masks=mask_list,
        rgbs=init_parser.rgb_imgs,
        cams=cam_list,
        text_labels_list=text_labels_list,
        material_dict=material_dict,
        vis_dir=vis_dir,
        visualize=visualize,
        verbose=verbose,
    )

    import ipdb; ipdb.set_trace()

    for _ in range(1):
        
        # memory: B x mem_nlayer x particle_num x nf_memory
        # for now, only used as a placeholder
        # memory_init = model.init_memory(1, data.particle_num * data.n_instance)

        # model rollout
        rollout_len = 10
        with torch.set_grad_enabled(False):
            for step_id in range(rollout_len):

                # action_encoded = data.parse_action(actions[step_id])
                action_encoded = np.zeros((1, args.action_dim), dtype=np.float32)
                Rr_cur, Rs_cur = data.generate_relation()
                state = data.state
                attrs = data.attrs
                particle_den = data.particle_den
                particle_num = state.shape[0]

                state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # B x n_his x N x state_dim
                action_encoded = torch.tensor(action_encoded, device=device, dtype=torch.float32).unsqueeze(0) # B x action_dim
                Rr_cur = torch.tensor(Rr_cur, device=device, dtype=torch.float32).unsqueeze(0) # B x n_rel x n_p
                Rs_cur = torch.tensor(Rs_cur, device=device, dtype=torch.float32).unsqueeze(0) # B x n_rel x n_p
                attrs = torch.tensor(attrs, device=device, dtype=torch.float32).unsqueeze(0) # B x N x attr_dim
                particle_den = torch.tensor([particle_den], device=device, dtype=torch.float32) # B

                # initialize particle grouping
                # p_rigid, p_instance, physics_param = get_env_group(args, particle_num, scene_params)
                # p_instance: B x n_p x n_istance
                # p_rigid: B x n_instance
                n_p, n_instance, p_instance, p_rigid = data.get_grouping()
                p_instance = p_instance.to(device)
                p_rigid = p_rigid.to(device)

                # unsqueeze the batch dimension
                # attr: B x n_p x attr_dim
                # Rr_cur, Rs_cur: B x n_rel x n_p
                # state_cur (unnormalized): B x n_his x n_p x state_dim
                # attr, Rr_cur, Rs_cur  = prepare_input(
                #         state_cur[-1].cpu().numpy(), particle_num, n_shape, args, stdreg=args.stdreg)

                inputs = {
                    'state': state,
                    'attrs': attrs,
                    'Rr_cur': Rr_cur,
                    'Rs_cur': Rs_cur,
                    'p_instance': p_instance,
                    'p_rigid': p_rigid,
                    'action': action_encoded,
                    'particle_den': particle_den,
                }

                # pred_pos: B x n_p x state_dim
                # pred_motion: B x n_p x state_dim
                pred_pos, pred_motion = model.predict_dynamics(**inputs)

                # losses = [step_id]
                # for loss_func in model_loss:
                #     loss = loss_func(pred_pos_p, sample_pos_p)
                #     losses += [loss.item()]
                # loss_list.append(losses)

                # state_cur (unnormalized): B x n_his x n_p x state_dim
                # state_cur = torch.cat([state_cur[:, 1:], pred_pos.unsqueeze(1)], 1)
                # state_cur = state_cur.detach()[0]

                # get next state (same format as input state)
                next_state = pred_pos[0].detach().cpu().numpy()

                # visualize the particles on the image
                # draw_particles_multi(data, next_state, os.path.join(vis_dir, 'test_{}.png'.format(step_id)))
                draw_particles(data, next_state, os.path.join(vis_dir, 'test_{}.png'.format(step_id)))
 
                # record the prediction
                data.update(next_state)

        # visualize the movements of particles in a video
        # particles_set = [np.stack(data.history_states + [next_state], axis=0)]
        # plt_render(particles_set, particle_num, 'vis/vid_test_plt_2.gif')


def draw_particles_multi(data, next_state, save_path):
    state = data.state
    particle_num_multi = state.shape[0]
    particle_num = data.particle_num
    attrs = data.attrs
    Rr = data.Rr
    Rs = data.Rs
    action = data.action
    cam_params = data.cam_params
    cam_extrinsics = data.cam_extrinsics
    import ipdb; ipdb.set_trace()

def draw_particles(data, next_state, save_path):
    state = data.state
    particle_num = state.shape[0]
    attrs = data.attrs
    Rr = data.Rr
    Rs = data.Rs
    action = data.action
    cam_params = data.cam_params
    cam_extrinsics = data.cam_extrinsics

    # PIL to cv2
    img = np.array(data.image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_orig = img.copy()

    def pts_to_xy(pts):
        # pts: [N, 3]
        fx, fy, cx, cy = cam_params
        xy = np.zeros((pts.shape[0], 3))
        xy[:, 0] = pts[:, 0] * fx / pts[:, 2] + cx
        xy[:, 1] = pts[:, 1] * fy / pts[:, 2] + cy
        return xy
    
    state_xy = pts_to_xy(state)
    for i in range(particle_num):
        # green points
        cv2.circle(img, (int(state_xy[i, 0]), int(state_xy[i, 1])), 3, (0, 255, 0), -1)
    
    for i in range(Rs.shape[0]):
        # red lines
        assert Rs[i].sum() == 1
        idx1 = Rs[i].argmax()
        idx2 = Rr[i].argmax()
        cv2.line(img, (int(state_xy[idx1, 0]), int(state_xy[idx1, 1])), (int(state_xy[idx2, 0]), int(state_xy[idx2, 1])), (0, 0, 255), 1)
    # save image
    img_save = np.concatenate([img_orig, img], axis=1)
    cv2.imwrite(save_path, img_save)


def plt_render(particles_set, n_particle, render_path):
    n_frames = particles_set[0].shape[0]
    rows = 1
    cols = 3

    def visualize_points(ax, all_points, n_particles):
        points = ax.scatter(all_points[:n_particles, 0], all_points[:n_particles, 2], all_points[:n_particles, 1], c='b', s=10)
        shapes = None
        # shapes = ax.scatter(all_points[n_particles+9:, 0], all_points[n_particles+9:, 2], all_points[n_particles+9:, 1], c='r', s=20)

        extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
        sz = extents[:, 1] - extents[:, 0]
        centers = np.mean(extents, axis=1)
        maxsize = max(abs(sz))
        r = 0.25  # maxsize / 2
        for ctr, dim in zip(centers, 'xyz'):
            getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

        return points, shapes

    fig, big_axes = plt.subplots(rows, 1, figsize=(9, 9))
    row_titles = ['GT', 'Sample', 'Prediction']
    views = [(90, 90), (0, 90), (45, 135)]
    plot_info_all = {}
    for i in range(rows):
        axes = big_axes[i] if rows > 1 else big_axes
        axes.set_title(row_titles[i], fontweight='semibold')
        axes.axis('off')

        plot_info = []
        for j in range(cols):
            ax = fig.add_subplot(rows, cols, i * cols + j + 1, projection='3d')
            ax.view_init(*views[j])
            points, shapes = visualize_points(ax, particles_set[i][0], n_particle)
            plot_info.append((points, shapes))

        plot_info_all[row_titles[i]] = plot_info

    plt.tight_layout()
    # plt.show()

    def update(step):
        outputs = []
        for i in range(rows):
            states = particles_set[i]
            for j in range(cols):
                points, shapes = plot_info_all[row_titles[i]][j]
                points._offsets3d = (
                states[step, :n_particle, 0], states[step, :n_particle, 2], states[step, :n_particle, 1])
                # shapes._offsets3d = (
                # states[step, n_particle:, 0], states[step, n_particle:, 2], states[step, n_particle:, 1])
                outputs.append(points)
                # outputs.append(shapes)
        return outputs

    anim = animation.FuncAnimation(fig, update, frames=np.arange(0, n_frames), blit=False)

    # plt.show()
    anim.save(render_path, writer=animation.PillowWriter(fps=20))



if __name__ == '__main__':
    args = gen_args()
    build_graph(args)

