import numpy as np
import torch
from gnn.model import Model, DynamicsPredictor, EarthMoverLoss, ChamferLoss, HausdorffLoss


def gen_model(args, material_dict, material='rigid', checkpoint=None, verbose=False, debug=False):
    # args.material = 'rigid'  # TODO debug
    args.material = material

    # particle encoder and relation encoder

    if args.material == 'granular':
        # particle encoder
        args.attr_dim = 1  # single attribute
        args.n_his = 1  # do not consider history
        args.state_dim = 0  # no state, use action dim to represent
        args.offset_dim = 0  # same as state_dim
        args.action_dim = 3  # delta x, y, z caused by action (long actions)
        args.pstep = 3
        args.time_step = 1
        args.dt = 1. / 60.
        args.sequence_length = 4
        args.phys_dim = 1  # density
        args.density_dim = 1  # particle density

        # relation encoder
        args.rel_particle_dim = 0  # no particle
        args.rel_attr_dim = 1  # default: 1
        args.rel_group_dim = 0  # no group
        args.rel_distance_dim = 3  # difference of sender and receiver position
        args.rel_density_dim = 1  # same as particle density
        
    elif args.material == 'rigid':
        # particle encoder
        args.attr_dim = 2  # object and end effector
        args.n_his = 1  # TODO consider history
        args.state_dim = 0  # x, y, z (no absolute position)
        args.offset_dim = 0  # same as state_dim (no absolute position)
        args.action_dim = 3
        args.pstep = 3
        args.time_step = 1
        args.dt = 1. / 60.
        args.sequence_length = 4
        args.phys_dim = 0  # TODO friction, density
        args.density_dim = 0  # particle density

        # relation encoder
        # args.rel_particle_dim = -1  # input dim
        args.rel_particle_dim = 0
        args.rel_attr_dim = 2  # no attribute
        args.rel_group_dim = 1  # sum of difference of group one-hot vector
        args.rel_distance_dim = 3  # no distance
        args.rel_density_dim = 0  # no density
    
    elif args.material == 'cloth':
        # particle encoder
        args.attr_dim = 2  # object and end effector
        args.n_his = 1  # TODO consider history
        args.state_dim = 3  # x, y, z
        args.offset_dim = 0
        args.action_dim = 3
        args.pstep = 6
        args.time_step = 1
        args.dt = 1. / 60.
        args.sequence_length = 4
        args.phys_dim = 0  # TODO friction, density
        args.density_dim = 0  # particle density

        # relation encoder
        args.rel_particle_dim = -1  # input dim
        args.rel_attr_dim = 0  # no attribute
        args.rel_group_dim = 0  # sum of difference of group one-hot vector
        args.rel_distance_dim = 0  # no distance
        args.rel_density_dim = 0  # no density
    
    elif args.material in ['deformable', 'rope']:
        raise NotImplementedError

    else:
        pass
        ### LLM START ### set_arg_dimensions
        ### LLM END ###
    
    kwargs = {}

    if args.material in ['rigid', 'multi-rigid']:
        kwargs.update({
            "predict_rigid": True,
            "predict_non_rigid": False,
            "rigid_out_dim": 7,
            "non_rigid_out_dim": 0,
        })
    elif args.material in ['granular', 'rope', 'cloth']:
        kwargs.update({
            "predict_rigid": False,
            "predict_non_rigid": True,
            "rigid_out_dim": 0,
            "non_rigid_out_dim": args.state_dim,
        })
    elif args.material in ['deformable']:
        kwargs.update({
            "predict_rigid": True,
            "predict_non_rigid": True,
            "rigid_out_dim": 7,
            "non_rigid_out_dim": args.state_dim,
        })
    else:
        pass
        ### LLM START ### set rigid predictor
        ### LLM END ###
    
    if debug:
        return None, None

    if args.material == 'deformable':
        emd_loss = EarthMoverLoss()
        chamfer_loss = ChamferLoss()
        h_loss = HausdorffLoss()
        model_loss = [emd_loss, chamfer_loss, h_loss]

    elif args.material in ['rigid', 'multi_rigid', 'granular', 'rope', 'cloth']:
        mse_loss = torch.nn.MSELoss()
        model_loss = [mse_loss]

    else:
        pass
        ### LLM START ### set_loss
        ### LLM END ###

    model = DynamicsPredictor(args, verbose=verbose, **kwargs)
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))
    return model, model_loss
