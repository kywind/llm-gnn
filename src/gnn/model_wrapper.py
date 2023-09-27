import numpy as np
import torch
from gnn.model import Model, EarthMoverLoss, ChamferLoss, HausdorffLoss


def gen_model(args, material_dict):
    args.material = 'rigid'  # TODO debug

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
        args.dt = 1. / 60.
        args.sequence_length = 4
        args.scene_params_dim = 1
    
    elif args.material == 'rigid':  # TODO
        args.attr_dim = 1
        args.n_his = 1
        args.state_dim = 3
        args.action_dim = 3
        args.load_action = True  # long actions
        args.nf_particle = 150
        args.nf_relation = 150
        args.nf_effect = 150
        args.pstep = 3
        args.time_step = 1
        args.dt = 1. / 60.
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
    
    model = Model(args)
    return model, model_loss
