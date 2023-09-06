import glob
import numpy as np
import os
import torch

from config import gen_args
from model import Model, EarthMoverLoss, ChamferLoss, HausdorffLoss
from utils import set_seed, Tee, count_parameters
from utils import load_data, get_env_group, get_scene_info, prepare_input
from utils import subsample_ptcl, downsample_pcd, fps, recenter, depth2fgpcd
# from visualize import train_plot_curves, eval_plot_curves, eval_plot_curves, plt_render


def perception_function(args):  # TODO for LLM and VLM
    n_particle = 0
    n_shape = 0
    p_sample = []
    scene_params = []
    ### LLM START ### set_perception_function
    ### LLM END ###
    return n_particle, n_shape, p_sample, scene_params

def load_action_seq(args):  # TODO
    action_cur = np.load(os.path.join(args.dataf, 'action_seq.npy'))
    action_cur = torch.from_numpy(action_cur).float()
    return action_cur

def gen_action_tensor(args, state_cur, action_cur):  # TODO
    action_tensor = torch.zeros((args.n_rollout, args.n_his, args.particle_num, 3))
    for i in range(args.n_rollout):
        for j in range(args.n_his):
            action_tensor[i, j] = action_cur[i * args.n_his + j]
    return action_tensor

def subsample_ptcl(state, n_p, n_sample, p_rigid, p_instance):  # TODO
    bsz = state.shape[0]
    batch_sampled_ptcl = np.zeros((bsz, n_sample, 3))
    batch_particle_r = np.zeros((bsz, ))
    for i in range(bsz):
        fgpcd = state[i, :n_p, :]
        fgpcd = downsample_pcd(fgpcd, 0.01)
        sampled_ptcl, particle_r = fps(fgpcd, n_sample)
        batch_sampled_ptcl[i] = recenter(fgpcd, sampled_ptcl, r = min(0.02, 0.5 * particle_r))
        batch_particle_r[i] = particle_r
    
    p_rigid_sp = []  # TODO
    p_instance_sp = []
    return batch_sampled_ptcl, batch_particle_r, p_rigid_sp, p_instance_sp


def load_data_gt(args):  # TODO
    p_gt = np.load(os.path.join(args.dataf, 'gt.npy'))
    return p_gt


def load_data_samples(args):  # TODO
    p_sample = np.load(os.path.join(args.dataf, 'sample.npy'))
    return p_sample


def evaluate(args, inputs):

    ### LLM START ### set_initial_args
    ### LLM END ###

    set_seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

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

    elif args.material == 'open-vocab':
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

    elif args.material == 'open-vocab':
        args.attr_dim = None
        args.n_his = None
        args.state_dim = None
        ### LLM START ### set_arg_dimensions
        ### LLM END ###
    
    else:
        raise NotImplementedError

    loss_list_over_episodes = []
    for _ in range(args.n_rollout):
        loss_list = []

        n_instance, n_particle, n_shape, scene_params = perception_function(args)  # TODO left for LLM and VLM
        args.n_instance = n_instance

        # initialize particle grouping
        p_rigid, p_instance, physics_param = get_env_group(args, n_particle, scene_params)

        # memory: B x mem_nlayer x (n_particle + n_shape) x nf_memory
        # for now, only used as a placeholder
        memory_init = model.init_memory(1, n_particle + n_shape)

        # model rollout
        st_idx = args.n_his
        ed_idx = args.time_step

        # action sequence
        if args.load_action:
            action_seq = load_action_seq(args)
            action_seq = action_seq.to(device)
        
        if args.load_gt:
            p_gt = load_data_gt(args)  # TODO
            p_gt = torch.from_numpy(p_gt).float().to(device)
        p_sample = load_data_samples(args)  # TODO
        p_sample = torch.from_numpy(p_sample).float().to(device)

        with torch.set_grad_enabled(False):
            for step_id in range(st_idx, ed_idx):
                if step_id == st_idx:
                    # state_cur (unnormalized): n_his x (n_p + n_s) x state_dim
                    state_cur = p_sample[step_id - args.n_his:step_id]
                    state_cur = state_cur.to(device)
                
                if args.load_action:
                    action_cur = action_seq[step_id - args.n_his:step_id]
                    action_cur = action_cur.to(device)

                if args.material == 'granular':
                    n_sample = 30
                    state_cur_sp, particle_r, p_rigid_sp, p_instance_sp = subsample_ptcl(
                            state_cur, n_particle, p_rigid, p_instance, n_sample=n_sample)
                    particle_den = np.array([1 / (particle_r * particle_r)])[0]
                    n_particle_sp = state_cur_sp.shape
                elif args.material == 'open-vocab':
                    state_cur_sp = None
                    n_particle_sp = None
                    p_rigid_sp = None
                    p_instance_sp = None
                    ### LLM START ### set_subsampling
                    ### LLM END ###
                else:
                    state_cur_sp = state_cur
                    n_particle_sp = n_particle
                    p_rigid_sp = p_rigid
                    p_instance_sp = p_instance

                # unsqueeze the batch dimension
                # attr: B x (n_p + n_s) x attr_dim
                # Rr_cur, Rs_cur: B x n_rel x (n_p + n_s)
                # state_cur (unnormalized): B x n_his x (n_p + n_s) x state_dim
                attr, Rr_cur, Rs_cur  = prepare_input(
                        state_cur_sp[-1].cpu().numpy(), n_particle_sp, n_shape, args, stdreg=args.stdreg)

                attr = attr.to(device).unsqueeze(0)
                Rr_cur = Rr_cur.to(device).unsqueeze(0)
                Rs_cur = Rs_cur.to(device).unsqueeze(0)
                Rn_cur = Rn_cur.to(device).unsqueeze(0)
                state_cur = state_cur.unsqueeze(0)

                if args.load_action:
                    action_cur_tensor = gen_action_tensor(args, state_cur_sp, action_cur) # (n_sample * n_batch, particle_num, 3)

                inputs = [
                    state_cur_sp, 
                    attr, 
                    Rr_cur,
                    Rs_cur,
                    action_cur_tensor,
                    memory_init, 
                    p_rigid_sp,
                    p_instance_sp,
                    physics_param,
                    particle_den,
                ]

                # pred_pos (unnormalized): B x n_p x state_dim
                # pred_motion_norm (normalized): B x n_p x state_dim
                if args.sequence_length > args.n_his + 1:
                    pred_pos_p, pred_motion_norm = model.predict_dynamics(*inputs)
                else:
                    pred_pos_p, pred_motion_norm = model.predict_dynamics(*inputs)

                # concatenate the state of the shapes
                # pred_pos (unnormalized): B x (n_p + n_s) x state_dim
                sample_pos = p_sample[step_id].to(device).unsqueeze(0)
                sample_pos_p = sample_pos[:, :n_particle]
                pred_pos = torch.cat([pred_pos_p, sample_pos[:, n_particle:]], 1)

                # sample_motion_norm (normalized): B x (n_p + n_s) x state_dim
                # pred_motion_norm (normalized): B x (n_p + n_s) x state_dim
                sample_motion = (p_sample[step_id] - p_sample[step_id - 1]).unsqueeze(0)
                sample_motion = sample_motion.to(device)

                mean_d, std_d = model.stat[2:]
                sample_motion_norm = (sample_motion - mean_d) / std_d
                pred_motion_norm = torch.cat([pred_motion_norm, sample_motion_norm[:, n_particle:]], 1)

                losses = [step_id]
                for loss_func in model_loss:
                    loss = loss_func(pred_pos_p, sample_pos_p)
                    losses += [loss.item()]
                loss_list.append(losses)

                # state_cur (unnormalized): B x n_his x (n_p + n_s) x state_dim
                state_cur = torch.cat([state_cur[:, 1:], pred_pos.unsqueeze(1)], 1)
                state_cur = state_cur.detach()[0]

                # record the prediction
                # p_pred[step_id] = state_cur[-1].detach().cpu()

        loss_list_over_episodes.append(loss_list)

        # visualization
        # group_info = [d.data.cpu().numpy()[0, ...] for d in group_info]
        # if args.gt_particles:
        #     p_pred = np.concatenate((p_gt.numpy()[:st_idx], p_pred.numpy()[st_idx:ed_idx]))
        # else:
        #     p_pred = np.concatenate((p_sample.numpy()[:st_idx], p_pred.numpy()[st_idx:ed_idx]))
        # p_sample = p_sample.numpy()[:ed_idx]
        # if load_gt: 
        #     p_gt = p_gt.numpy()[:ed_idx]
        # vid_path = os.path.join(args.dataf, 'vid', str(idx_episode).zfill(3))
        # render_path = os.path.join(eval_out_path, 'render', f'vid_{idx_episode}_plt.gif')
        # if args.vis == 'plt':
        #     plt_render([p_gt, p_sample, p_pred], n_particle, render_path)
        # else:
        #     raise NotImplementedError

    # plot the loss curves for training and evaluating
    # with open(os.path.join(args.outf, 'train.npy'), 'rb') as f:
    #     train_log = np.load(f, allow_pickle=True)
    #     train_log = train_log[None][0]
    #     train_plot_curves(train_log['iters'], train_log['loss'], path=os.path.join(eval_out_path, 'plot', 'train_loss_curves.png'))

    # loss_list_over_episodes = np.array(loss_list_over_episodes)
    # loss_mean = np.mean(loss_list_over_episodes, axis=0)
    # loss_std = np.std(loss_list_over_episodes, axis=0)
    # eval_plot_curves(loss_mean[:, :-1], loss_std[:, :-1], path=os.path.join(eval_out_path, 'plot', 'eval_loss_curves.png'))

    # print(f"\nAverage emd loss at last frame: {np.mean(loss_list_over_episodes[:, -1, 1])} (+- {np.std(loss_list_over_episodes[:, -1, 1])})")
    # print(f"Average chamfer loss at last frame: {np.mean(loss_list_over_episodes[:, -1, 2])} (+- {np.std(loss_list_over_episodes[:, -1, 2])})")
    # print(f"Average hausdorff loss at last frame: {np.mean(loss_list_over_episodes[:, -1, 3])} (+- {np.std(loss_list_over_episodes[:, -1, 3])})")
    # print(f"\nAverage emd loss over episodes: {np.mean(loss_list_over_episodes[:, :, 1])} (+- {np.std(loss_list_over_episodes[:, :, 1])})")
    # print(f"Average chamfer loss over episodes: {np.mean(loss_list_over_episodes[:, :, 2])} (+- {np.std(loss_list_over_episodes[:, :, 2])})")
    # print(f"Average hausdorff loss over episodes: {np.mean(loss_list_over_episodes[:, :, 3])} (+- {np.std(loss_list_over_episodes[:, :, 3])})")
