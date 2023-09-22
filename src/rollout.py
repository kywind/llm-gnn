import glob
import numpy as np
import os
import torch

from config import gen_args
from gnn.model import Model, EarthMoverLoss, ChamferLoss, HausdorffLoss
from gnn.utils import set_seed, Tee, count_parameters
from gnn.utils import load_data, get_env_group, get_scene_info, prepare_input

from data.dataparser import Dataparser
from data.particle_dataparser import ParticleDataparser
from data.multi_particle_dataparser import MultiParticleDataparser
from llm.llm import LLM
import pyflex
import pickle

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib

import cv2


def rollout(args):
    set_seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # single_material = True
    # img_dir = "../../dyn-res-pile-manip/data/gnn_dyn_data/0/0_color.png"  # blipv2 cannot identify this image
    # init_img_dir = "vis/g1.png"
    # depth_dir = "../../dyn-res-pile-manip/data/gnn_dyn_data/0/0_depth.png"
    # action_dir = "../../dyn-res-pile-manip/data/gnn_dyn_data/0/actions.p"
    # vis_dir = "vis/granular-dynres-0/"

    # single_material = False
    # img_dir = "vis/inputs/apples2_640.png"
    # init_img_dir = img_dir
    # depth_dir = "vis/inputs_depth/apples2_640-dpt_beit_large_512.png"
    # action_dir = "../../dyn-res-pile-manip/data/gnn_dyn_data/0/actions.p"
    # vis_dir = "vis/apples2_640-0/"

    single_material = False
    img_dir = "vis/inputs/dough_640.png"
    init_img_dir = img_dir
    depth_dir = "vis/inputs_depth/dough_640-dpt_beit_large_512.png" # MiDaS
    action_dir = "../../dyn-res-pile-manip/data/gnn_dyn_data/0/actions.p"
    vis_dir = "vis/dough_640-0/"

    os.makedirs(vis_dir, exist_ok=True)

    # initial dataparser
    init_parser = Dataparser(args, init_img_dir, detection=True, query=True)

    llm = LLM(args)

    query_results = init_parser.query(
        texts='What are the objects in the image? Answer:'
    )
    print('query_results:', query_results)

    obj_prompt = " ".join([
        "What are the individual objects mentioned in the query?",
        "Respond unknown if you are not sure."
        "\n\nQuery: bottle. Answer: a bottle.",
        "\n\nQuery: a knife, and a board. Answer: a knife, a board.",
        "\n\nQuery: a rope, a scissor, and a bag of bananas. Answer: a rope, scissor, a banana.",
        "\n\nQuery: a pile of sand. Answer: sand.",
        "\n\nQuery: oranges. Answer: oranges.",
        "\n\nQuery: rices. Answer: rices.",
        "\n\nQuery: a pile of sand, a pile of play-doh, and a pile of coffee beans. Answer:sand, play-doh, coffee bean.",
        "\n\nQuery: " + query_results + ". Answer:",
    ])
    objs_str = llm.query(obj_prompt)
    print('objs_str:', objs_str)
    objs = objs_str.rstrip('.').split(',')

    obj_name_list = []
    material_list = []
    for obj_name in objs:
        obj_name = obj_name.strip(' ')
    
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
        material_list.append(material)
        obj_name_list.append(obj_name)
        # print(obj_name, material)

    print('material_list:', material_list)
    # args.material = material_list

    if not single_material:
        # segmentation_results: predictions=list(crop_img, mask), boxes, scores, labels
        segmentation_results = init_parser.segment(
            texts='|'.join([','.join(obj_name_list), ','.join(material_list)])
        )

    args.material = 'granular'  # TODO ONLY FOR DEBUGGING
    ### LLM START ### set_initial_args
    ### LLM END ###

    # set gnn model
    model = Model(args)
    model.eval()
    model = model.to(device)

    if args.material == 'deformable':
        emd_loss = EarthMoverLoss()
        chamfer_loss = ChamferLoss()
        h_loss = HausdorffLoss()
        model_loss = [emd_loss, chamfer_loss, h_loss]

    elif args.material in ['rigid', 'multi_rigid', 'granular', 'rope', 'cloth']:
        mse_loss = torch.nn.MSELoss()
        model_loss = [mse_loss]

    else:
        model_loss = []
        ### LLM START ### set_loss
        ### LLM END ###

    if args.material == 'granular':
        args.attr_dim = 1
        args.n_his = 1
        args.state_dim = 0
        args.action_dim = 3
        args.load_action = True  # long actions
        args.nf_particle = 150
        args.nf_relation = 150
        args.nf_effect = 150
        args.pstep = 3
        args.time_step = 1
        args.dt = None
        args.sequence_length = 4
        args.scene_params_dim = 1

    elif args.material == 'deformable':
        args.attr_dim = 3
        args.n_his = 4
        args.state_dim = 3
        args.action_dim = 0
        args.load_action = False  # timestep actions
        args.nf_particle = 150
        args.nf_relation = 150
        args.nf_effect = 150
        args.pstep = 2
        args.time_step = 119
        args.dt = 1. / 60.
        args.sequence_length = 6
        args.scene_params_dim = 1

    else:
        args.attr_dim = None
        args.n_his = None
        args.state_dim = None
        ### LLM START ### set_arg_dimensions
        ### LLM END ###

    # set dataset
    if args.material == 'granular':
        screenWidth = 720
        screenHeight = 720
        headless = True
        pyflex.set_screenWidth(screenWidth)
        pyflex.set_screenHeight(screenHeight)
        pyflex.set_light_dir(np.array([0.1, 2.0, 0.1]))
        pyflex.set_light_fov(70.)
        pyflex.init(headless)

        cam_idx = 0
        rad = np.deg2rad(cam_idx * 20.)
        global_scale = 24
        cam_dis = 0.0 * global_scale / 8.0
        cam_height = 6.0 * global_scale / 8.0
        camPos = np.array([np.sin(rad) * cam_dis, cam_height, np.cos(rad) * cam_dis])
        camAngle = np.array([rad, -np.deg2rad(90.), 0.])
        pyflex.set_camPos(camPos)
        pyflex.set_camAngle(camAngle)

        projMat = pyflex.get_projMatrix().reshape(4, 4).T
        cx = screenWidth / 2.0
        cy = screenHeight / 2.0
        fx = projMat[0, 0] * cx
        fy = projMat[1, 1] * cy
        cam_params =  [fx, fy, cx, cy]

        cam_extrinsics = np.array(pyflex.get_viewMatrix()).reshape(4, 4).T
        cam = [cam_params, cam_extrinsics]

        if single_material:
            data = ParticleDataparser(
                args=args,
                img_dir=img_dir,
                depth_dir=depth_dir,
                cam=cam
            )
        else:
            data = MultiParticleDataparser(
                args=args,
                img_dir=img_dir,
                depth_dir=depth_dir,
                cam=cam,
                segmentation=segmentation_results,
                material=material_list
            )

        with open(action_dir, 'rb') as fp:
            actions = pickle.load(fp)
    else:
        raise NotImplementedError
    
    # import ipdb; ipdb.set_trace()

    for _ in range(1):
        
        # memory: B x mem_nlayer x particle_num x nf_memory
        # for now, only used as a placeholder
        memory_init = model.init_memory(1, data.particle_num * data.n_instance)

        # model rollout
        rollout_len = 10
        with torch.set_grad_enabled(False):
            for step_id in range(rollout_len):

                action_encoded = data.parse_action(actions[step_id])
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
        cv2.circle(img, (int(state_xy[i, 0]), int(state_xy[i, 1])), 10, (0, 255, 0), -1)
    
    for i in range(Rs.shape[0]):
        # red lines
        assert Rs[i].sum() == 1
        idx1 = Rs[i].argmax()
        idx2 = Rr[i].argmax()
        cv2.line(img, (int(state_xy[idx1, 0]), int(state_xy[idx1, 1])), (int(state_xy[idx2, 0]), int(state_xy[idx2, 1])), (0, 0, 255), 2)
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
    rollout(args)

