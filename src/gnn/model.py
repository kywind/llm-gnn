import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy import optimize
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        s = x.size()
        x = self.model(x.view(-1, s[-1]))
        return x.view(list(s[:-1]) + [-1])


class Propagator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Propagator, self).__init__()

        self.linear = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, res=None):
        s_x = x.size()

        x = self.linear(x.view(-1, s_x[-1]))

        if res is not None:
            s_res = res.size()
            x += res.view(-1, s_res[-1])

        x = self.relu(x).view(list(s_x[:-1]) + [-1])
        return x


class ParticlePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ParticlePredictor, self).__init__()

        self.linear_0 = nn.Linear(input_size, hidden_size)
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        s_x = x.size()

        x = x.view(-1, s_x[-1])
        x = self.relu(self.linear_0(x))
        x = self.relu(self.linear_1(x))

        return self.linear_2(x).view(list(s_x[:-1]) + [-1])


class DynamicsPredictor(nn.Module):
    def __init__(self, args):

        super(DynamicsPredictor, self).__init__()

        self.args = args

        self.nf_particle = args.nf_particle
        self.nf_relation = args.nf_relation
        self.nf_effect = args.nf_effect

        self.quat_offset = torch.FloatTensor([1., 0., 0., 0.]).to(args.device)

        # ParticleEncoder
        if args.material == 'deformable':
            self.n_his = args.n_his  # default: n_his = 4
            self.attr_dim = args.attr_dim  # default: attr_dim = 3
            self.phys_dim = 1
            self.density_dim = 0
            self.state_dim = args.state_dim  # default: state_dim = 3
            self.offset_dim = self.state_dim  # default: offset_dim = 3 (x, y, z)
            self.action_dim = args.action_dim  # default: 0
            self.mem_dim = args.nf_effect * args.mem_nlayer  # default: mem_nlayer = 2
            self.state_normalize = True

        elif args.material == 'granular':
            self.n_his = args.n_his  # default: 1
            self.attr_dim = 1  # default: 1
            self.phys_dim = 0
            self.density_dim = 1  # particle density, default = 1
            self.state_dim = args.state_dim  # defualt: 0
            self.offset_dim = 0
            self.action_dim = args.action_dim  # pusher movement xyz
            self.mem_dim = 0
            self.state_normalize = False

        elif args.material in ['rigid', 'multi-rigid']:
            self.attr_dim = args.attr_dim  # default: 2 (movable, fixed)
            self.n_his = args.n_his  # default: 4 (use_history) or 1 (no history, use velocity)
            self.state_dim = args.state_dim  # default: 2/3 or 4/6 (with velocity)
            self.offset_dim = args.state_dim
            self.mem_dim = args.nf_effect * args.mem_nlayer  # default: mem_nlayer = 2 or 0
            self.action_dim = args.action_dim  # default: 2/3 or 0
            self.phys_dim = 1
            self.density_dim = 0

        elif args.material == 'rope':
            self.attr_dim = args.attr_dim
            self.state_dim = args.state_dim
            self.action_dim = args.action_dim

        elif args.material == 'cloth':
            self.attr_dim = args.attr_dim
            self.state_dim = args.state_dim
            self.action_dim = args.action_dim
        
        else:
            self.attr_dim = None
            self.state_dim = None
            ### LLM START ### set particle dimensions
            ### LLM END ###

        input_dim = self.attr_dim + self.phys_dim + self.density_dim + self.n_his * self.state_dim \
                + self.n_his * self.offset_dim + self.action_dim + self.mem_dim
        self.particle_encoder = Encoder(input_dim, self.nf_particle, self.nf_effect)

        # RelationEncoder
        if args.material in ['deformable', 'rigid', 'multi-rigid', 'rope', 'cloth']:
            self.rel_particle_dim = input_dim
            self.rel_attr_dim = 0
            self.rel_group_dim = 1  # sum of difference of group one-hot vector
            self.rel_distance_dim = 0
            self.rel_density_dim = 0
            
        elif args.material == 'granular':
            self.rel_particle_dim = 0
            self.rel_attr_dim = 1  # default: 1
            self.rel_group_dim = 0
            self.rel_distance_dim = 3  # difference of sender and receiver position
            self.rel_density_dim = 1  # same as particle density
        
        else:
            self.rel_particle_dim = None
            self.rel_attr_dim = None
            ### LLM START ### set relation dimensions
            ### LLM END ###

        rel_input_dim = self.rel_particle_dim * 2 + self.rel_attr_dim * 2 \
                + self.rel_group_dim + self.rel_distance_dim + self.rel_density_dim
        self.relation_encoder = Encoder(rel_input_dim, self.nf_relation, self.nf_effect)


        # ParticlePropagator
        self.particle_propagator = Propagator(self.nf_effect * 2, self.nf_effect)

        # RelationPropagator
        self.relation_propagator = Propagator(self.nf_effect * 3, self.nf_effect)
        # NOTE: dyn-res-manip adds a particle density to the relation propagator, 
        # which I think is unnecessary since the density is already included in the particle encoder.

        # ParticlePredictor
        if args.material in ['rigid', 'multi-rigid', 'deformable']:
            self.non_rigid_predictor = None
            self.rigid_predictor = ParticlePredictor(self.nf_effect, self.nf_effect, 7)
        elif args.material in ['granular', 'rope', 'cloth']:
            self.non_rigid_predictor = ParticlePredictor(self.nf_effect, self.nf_effect, self.state_dim)
            self.rigid_predictor = None
        elif args.material in ['deformable']:
            self.non_rigid_predictor = ParticlePredictor(self.nf_effect, self.nf_effect, self.state_dim)
            self.rigid_predictor = ParticlePredictor(self.nf_effect, self.nf_effect, 7)
        else:
            pass
            ### LLM START ### set rigid predictor
            ### LLM END ###

    # @profile
    def forward(self, state, attrs, Rr_cur, Rs_cur, p_instance, p_rigid, 
            action=None, memory=None, offset=None, physics_param=None, particle_den=None, verbose=False):

        args = self.args
        verbose = args.verbose_model

        B, N = attrs.size(0), attrs.size(1)  # batch size, total particle num
        n_instance = p_instance.size(2)  # number of instances
        n_p = p_instance.size(1)  # number of object particles (that need prediction)
        n_s = attrs.size(1) - n_p  # number of shape particles that do not need prediction
        n_rel = Rr_cur.size(1)  # number of relations

        # attrs: B x N x attr_dim
        # state: B x n_his x N x state_dim
        # Rr_cur, Rs_cur: B x n_rel x N
        # memory: B x mem_nlayer x N x nf_memory
        # p_rigid: B x n_instance
        # p_instance: B x n_particle x n_instance
        # physics_param: B x n_particle

        # Rr_cur_t, Rs_cur_t: B x N x n_rel
        Rr_cur_t = Rr_cur.transpose(1, 2).contiguous()
        Rs_cur_t = Rs_cur.transpose(1, 2).contiguous()

        # particle belongings and rigidness
        # p_rigid_per_particle: B x n_p x 1
        p_rigid_per_particle = torch.sum(p_instance * p_rigid[:, None, :], 2, keepdim=True)

        # state_res: B x (n_his - 1) x N x state_dim, state_cur: B x 1 x N x state_dim     
        state_res = state[:, 1:] - state[:, :-1]
        state_cur = state[:, -1:]

        if self.state_normalize:
            # *_p: absolute scale to model scale, *_d: residual scale to model scale
            mean_p, std_p, mean_d, std_d = self.mean_p, self.std_p, self.mean_d, self.std_d
            state_res_norm = (state_res - mean_d) / std_d
            state_res_norm[:, :, :n_p, :] = 0  # n_p = 301?
            state_cur_norm = (state_cur - mean_p) / std_p
        else:
            state_res_norm = state_res
            state_cur_norm = state_cur

        # state_norm: B x n_his x N x state_dim
        # [0, n_his - 1): state_residual
        # [n_his - 1, n_his): the current position
        state_norm = torch.cat([state_res_norm, state_cur_norm], 1)
        state_norm_t = state_norm.transpose(1, 2).contiguous().view(B, N, self.n_his * self.state_dim)

        # p_inputs: B x N x (attr_dim + n_his * state_dim)
        p_inputs = torch.cat([attrs, state_norm_t], 2)

        # other inputs
        if offset is not None:
            # add offset to center-of-mass for rigids to attr
            # offset: B x N x (n_his * state_dim)
            offset = torch.zeros(B, N, self.n_his * self.state_dim).to(args.device)

            # instance_center: B x n_instance x (n_his * state_dim)
            instance_center = p_instance.transpose(1, 2).bmm(state_norm_t[:, :n_p])
            instance_center /= torch.sum(p_instance, 1).unsqueeze(-1) + args.eps

            # c_per_particle: B x n_p x (n_his * state_dim)
            # particle offset: B x n_p x (n_his * state_dim)
            c_per_particle = p_instance.bmm(instance_center)
            c = (1 - p_rigid_per_particle) * state_norm_t[:, :n_p] + p_rigid_per_particle * c_per_particle
            offset[:, :n_p] = state_norm_t[:, :n_p] - c

            # p_inputs: B x N x (attr_dim + 2 * n_his * state_dim)
            p_inputs = torch.cat([p_inputs, offset], 2)

        if memory is not None:
            # memory_t: B x N x (mem_nlayer * nf_memory)
            memory_t = memory.transpose(1, 2).contiguous().view(B, N, -1)

            # p_inputs: B x N x (... + mem_nlayer * nf_memory)
            p_inputs = torch.cat([p_inputs, memory_t], 2)
        
        if physics_param is not None:
            # physics_param: B x N x 1
            physics_param_s = torch.zeros(B, n_s, 1).to(args.device)
            physics_param = torch.cat([physics_param[:, :, None], physics_param_s], 1)

            # p_inputs: B x N x (... + phys_dim)
            p_inputs = torch.cat([p_inputs, physics_param], 2)

        # TODO what if n_instance is not fixed?
        # group info
        # if p_instance is not None:
        #     # g: B x N x n_instance
        #     # for particle i in group j, g[:, i, j] = 1, otherwise 0
        #     g = p_instance
        #     g_s = torch.zeros(B, n_s, n_instance).to(args.device)
        #     g = torch.cat([g, g_s], 1)
        #     # p_inputs: B x N x (... + n_instance)
        #     p_inputs = torch.cat([p_inputs, g], 2)
        
        # action
        if action is not None:
            # action: B x N x action_dim
            action_s = torch.zeros(B, n_s, self.action_dim).to(args.device)
            action = torch.cat([action, action_s], 1)

            # p_inputs: B x N x (... + action_dim)
            p_inputs = torch.cat([p_inputs, action], 2)

        if particle_den is not None:
            particle_den = particle_den / 5000.

            # particle_den: B x N x 1
            particle_den = particle_den[:, None, None].repeat(1, n_p, 1)
            particle_den_s = torch.zeros(B, n_s, 1).to(args.device)
            particle_den = torch.cat([particle_den, particle_den_s], 1)

            # p_inputs: B x N x (... + density_dim)
            p_inputs = torch.cat([p_inputs, particle_den], 2)
        # Finished preparing p_inputs

        # Preparing rel_inputs
        rel_inputs = torch.empty((B, n_rel, 0), dtype=torch.float32).to(args.device)
        if self.rel_particle_dim > 0:
            assert self.rel_particle_dim == 2 * p_inputs.size(2)
            # p_inputs_r: B x n_rel x -1
            # p_inputs_s: B x n_rel x -1
            p_inputs_r = Rr_cur.bmm(p_inputs)
            p_inputs_s = Rs_cur.bmm(p_inputs)

            # rel_inputs: B x n_rel x (2 x rel_particle_dim)
            rel_inputs = torch.cat([rel_inputs, p_inputs_r, p_inputs_s], 2)

        if self.rel_attr_dim > 0:
            # attr_r: B x n_rel x attr_dim
            # attr_s: B x n_rel x attr_dim
            attrs_r = Rr_cur.bmm(attrs)
            attrs_s = Rs_cur.bmm(attrs)

            # rel_inputs: B x n_rel x (... + 2 x rel_attr_dim)
            rel_inputs = torch.cat([rel_inputs, attrs_r, attrs_s], 2)

        if self.rel_group_dim > 0:
            assert self.rel_group_dim == 1
            # receiver_group, sender_group
            # group_r: B x n_rel x -1
            # group_s: B x n_rel x -1
            g = torch.cat([p_instance, torch.zeros(B, n_s, n_instance).to(args.device)], 1)
            group_r = Rr_cur.bmm(g)
            group_s = Rs_cur.bmm(g)
            group_diff = torch.sum(torch.abs(group_r - group_s), 2, keepdim=True)

            # rel_inputs: B x n_rel x (... + 1)
            rel_inputs = torch.cat([rel_inputs, group_diff], 2)
        
        if self.rel_distance_dim > 0:
            assert self.rel_distance_dim == 3
            # receiver_pos, sender_pos
            # pos_r: B x n_rel x -1
            # pos_s: B x n_rel x -1
            pos_r = Rr_cur.bmm(state_norm_t)
            pos_s = Rs_cur.bmm(state_norm_t)
            pos_diff = pos_r - pos_s

            # rel_inputs: B x n_rel x (... + 3)
            rel_inputs = torch.cat([rel_inputs, pos_diff], 2)
        
        if self.rel_density_dim > 0:
            assert self.rel_density_dim == 1
            # receiver_density, sender_density
            # dens_r: B x n_rel x -1
            # dens_s: B x n_rel x -1
            dens_r = Rr_cur.bmm(particle_den)
            dens_s = Rs_cur.bmm(particle_den)
            dens_diff = dens_r - dens_s

            # rel_inputs: B x n_rel x (... + 1)
            rel_inputs = torch.cat([rel_inputs, dens_diff], 2)

        # particle encode
        particle_encode = self.particle_encoder(p_inputs)
        particle_effect = particle_encode
        if verbose:
            print("particle encode:", particle_encode.size())

        # calculate relation encoding
        relation_encode = self.relation_encoder(rel_inputs)
        if verbose:
            print("relation encode:", relation_encode.size())

        for i in range(args.pstep):
            if verbose:
                print("pstep", i)

            # effect_r, effect_s: B x n_rel x nf
            effect_r = Rr_cur.bmm(particle_effect)
            effect_s = Rs_cur.bmm(particle_effect)

            # calculate relation effect
            # effect_rel: B x n_rel x nf
            effect_rel = self.relation_propagator(
                torch.cat([relation_encode, effect_r, effect_s], 2))
            if verbose:
                print("relation effect:", effect_rel.size())

            # calculate particle effect by aggregating relation effect
            # effect_rel_agg: B x N x nf
            effect_rel_agg = Rr_cur_t.bmm(effect_rel)

            # calculate particle effect
            # particle_effect: B x N x nf
            particle_effect = self.particle_propagator(
                torch.cat([particle_encode, effect_rel_agg], 2),
                res=particle_effect)
            if verbose:
                 print("particle effect:", particle_effect.size())

        # non-rigid motion
        if self.non_rigid_predictor is not None:
            # non_rigid_motion: B x n_p x state_dim
            non_rigid_motion = self.non_rigid_predictor(particle_effect[:, :n_p].contiguous())

        # rigid motion
        if self.rigid_predictor is not None:
            # aggregate effects of particles in the same instance
            # instance effect: B x n_instance x nf_effect
            instance_effect = p_instance.transpose(1, 2).bmm(particle_effect[:, :n_p])

            # instance_rigid_params: (B * n_instance) x 7
            instance_rigid_params = self.rigid_predictor(instance_effect).view(B * n_instance, 7)

            # decode rotation
            # R: (B * n_instance) x 3 x 3
            R = self.rotation_matrix_from_quaternion(instance_rigid_params[:, :4] + self.quat_offset)
            if verbose:
                print("Rotation matrix", R.size(), "should be (B x n_instance, 3, 3)")

            # decode translation
            b = instance_rigid_params[:, 4:]
            if self.state_normalize:  # denormalize
                b = b * std_d + mean_d
            b = b.view(B * n_instance, 1, self.state_dim)
            if verbose:
                print("b", b.size(), "should be (B x n_instance, 1, state_dim)")

            # current particle state
            # p_0: B x 1 x n_p x state_dim -> (B * n_instance) x n_p x state_dim
            p_0 = state[:, -1:, :n_p]
            p_0 = p_0.repeat(1, n_instance, 1, 1).view(B * n_instance, n_p, self.state_dim)
            if verbose:
                print("p_0", p_0.size(), "should be (B x n_instance, n_p, state_dim)")

            # current per-instance center state
            c = instance_center[:, :, -3:]
            if self.state_normalize:  # denormalize
                c = c * std_p + mean_p
            c = c.view(B * n_instance, 1, self.state_dim)
            if verbose:
                print("c", c.size(), "should be (B x n_instance, 1, state_dim)")

            # updated state after rigid motion
            p_1 = torch.bmm(p_0 - c, R) + b + c
            if verbose:
                print("p_1", p_1.size(), "should be (B x n_instance, n_p, state_dim)")

            # compute difference for per-particle rigid motion
            # rigid_motion: B x n_instance x n_p x state_dim
            rigid_motion = (p_1 - p_0).view(B, n_instance, n_p, self.state_dim)
            if self.state_normalize:  # normalize
                rigid_motion = (rigid_motion - mean_d) / std_d
        
        # aggregate motions
        rigid_part = p_rigid_per_particle[..., 0].bool()

        pred_motion = torch.zeros(B, n_p, self.state_dim).to(args.device)
        if self.non_rigid_predictor is not None:
            pred_motion[~rigid_part] = non_rigid_motion[~rigid_part]
        if self.rigid_predictor is not None:
            pred_motion[rigid_part] = torch.sum(p_instance.transpose(1, 2)[..., None] * rigid_motion, 1)[rigid_part]
        if self.state_normalize:  # denormalize
            pred_motion = pred_motion * std_d + mean_d
        pred_pos = state[:, -1, :n_p] + torch.clamp(pred_motion, max=0.025, min=-0.025)
        if verbose:
            print('pred_pos', pred_pos.size())

        # pred_pos (denormalized): B x n_p x state_dim
        # pred_motion (denormalized): B x n_p x state_dim
        return pred_pos, pred_motion


class Model(nn.Module):
    def __init__(self, args):

        super(Model, self).__init__()

        self.args = args

        self.dt = torch.FloatTensor([args.dt]).to(args.device)
        self.mean_p = torch.FloatTensor(args.mean_p).to(args.device)
        self.std_p = torch.FloatTensor(args.std_p).to(args.device)
        self.mean_d = torch.FloatTensor(args.mean_d).to(args.device)
        self.std_d = torch.FloatTensor(args.std_d).to(args.device)

        # PropNet to predict forward dynamics
        self.dynamics_predictor = DynamicsPredictor(args)

    def init_memory(self, B, N):
        """
        memory  (B, mem_layer, N, nf_memory)
        """
        mem = torch.zeros(B, self.args.mem_nlayer, N, self.args.nf_effect).to(self.args.device)
        return mem

    def predict_dynamics(self, **inputs):
        """
        return:
        ret - predicted position of all particles, shape (n_particles, 3)
        """
        ret = self.dynamics_predictor(**inputs, verbose=self.args.verbose_model)
        return ret


class ChamferLoss(torch.nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def chamfer_distance(self, x, y):
        # x: [B, N, D]
        # y: [B, M, D]
        x = x[:, :, None, :].repeat(1, 1, y.size(1), 1) # x: [B, N, M, D]
        y = y[:, None, :, :].repeat(1, x.size(1), 1, 1) # y: [B, N, M, D]
        dis = torch.norm(torch.add(x, -y), 2, dim=3)    # dis: [B, N, M]
        dis_xy = torch.mean(torch.min(dis, dim=2)[0])   # dis_xy: mean over N
        dis_yx = torch.mean(torch.min(dis, dim=1)[0])   # dis_yx: mean over M

        return dis_xy + dis_yx

    def __call__(self, pred, label):
        # pred: [B, N, D]
        # label: [B, M, D]
        return self.chamfer_distance(pred, label)


class EarthMoverLoss(torch.nn.Module):
    def __init__(self):
        super(EarthMoverLoss, self).__init__()

    def em_distance(self, x, y):
        # x: [B, N, D]
        # y: [B, M, D]
        x_ = x[:, :, None, :].repeat(1, 1, y.size(1), 1)  # x: [B, N, M, D]
        y_ = y[:, None, :, :].repeat(1, x.size(1), 1, 1)  # y: [B, N, M, D]
        dis = torch.norm(torch.add(x_, -y_), 2, dim=3)  # dis: [B, N, M]
        x_list = []
        y_list = []
        # x.requires_grad = True
        # y.requires_grad = True
        for i in range(dis.shape[0]):
            cost_matrix = dis[i].detach().cpu().numpy()
            try:
                ind1, ind2 = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=False)
            except:
                # pdb.set_trace()
                print("Error in linear sum assignment!")
            x_list.append(x[i, ind1])
            y_list.append(y[i, ind2])
            # x[i] = x[i, ind1]
            # y[i] = y[i, ind2]
        new_x = torch.stack(x_list)
        new_y = torch.stack(y_list)
        # print(f"EMD new_x shape: {new_x.shape}")
        # print(f"MAX: {torch.max(torch.norm(torch.add(new_x, -new_y), 2, dim=2))}")
        emd = torch.mean(torch.norm(torch.add(new_x, -new_y), 2, dim=2))
        return emd

    def __call__(self, pred, label):
        # pred: [B, N, D]
        # label: [B, M, D]
        return self.em_distance(pred, label)


class HausdorffLoss(torch.nn.Module):
    def __init__(self):
        super(HausdorffLoss, self).__init__()

    def hausdorff_distance(self, x, y):
        # x: [B, N, D]
        # y: [B, M, D]
        x = x[:, :, None, :].repeat(1, 1, y.size(1), 1) # x: [B, N, M, D]
        y = y[:, None, :, :].repeat(1, x.size(1), 1, 1) # y: [B, N, M, D]
        dis = torch.norm(torch.add(x, -y), 2, dim=3)    # dis: [B, N, M]
        # print(dis.shape)
        dis_xy = torch.max(torch.min(dis, dim=2)[0])   # dis_xy: mean over N
        dis_yx = torch.max(torch.min(dis, dim=1)[0])   # dis_yx: mean over M

        return dis_xy + dis_yx

    def __call__(self, pred, label):
        # pred: [B, N, D]
        # label: [B, M, D]
        return self.hausdorff_distance(pred, label)
