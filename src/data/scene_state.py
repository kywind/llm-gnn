import numpy as np
import torch
from typing import List, Tuple, Dict, Any

from data.point_cloud import PointCloud
from gnn.utils import find_relations_neighbor


class SceneState:

    pcs: Dict[str, PointCloud]
    items: List[str]
    attrs: Dict[str, Any]

    def __init__(self, args):
        self.args = args
        pass

    def step_perception(self):
        n_instance, n_particle, n_shape, scene_params = 0, 0, 0, None

    def init_perception(self):
        pass
    
    def subsample(self, n):
        self.pcs['obj'] = self.pcs['obj'].subsample(n)
        particle_den = np.array([1 / (particle_r * particle_r)])[0]
        self.attrs['density'] = particle_den
    
    def get_R(self):
        args = self.args
        n_particle = sum([len(self.pcs[k]) for k in self.pcs.keys()])
        pos = self.pcs['obj'].points  # TODO multi obj

        if args.shape_aug:
            attr[n_particle: n_particle + 9, 1] = 1
            attr[n_particle + 9:, 2] = 1
            # pos = positions.data.cpu().numpy() if var else positions
            # floor to points
            for ind in range(9):
                dis = pos[:n_particle, 1] - pos[n_particle+ind, 1]
                #np.linalg.norm(pos[:n_particle] - pos[n_particle + ind], 2, axis=1)
                nodes = np.nonzero(dis < args.neighbor_radius)[0]
                # print(nodes)
                # if ind == 8:
                #     import pdb; pdb.set_trace()
                #     visualize_neighbors(pos, pos, 0, nodes)
                floor = np.ones(nodes.shape[0], dtype=np.int) * (n_particle + ind)
                rels += [np.stack([nodes, floor], axis=1)]
                rels2 += [np.stack([nodes, floor], axis=1)]
            for ind in range(22):
                # to primitive
                disp1 = np.sqrt(np.sum((pos[:n_particle] - pos[n_particle + 9 + ind]) ** 2, 1))
                nodes1 = np.nonzero(disp1 < (args.neighbor_radius + args.gripper_extra_neighbor_radius))[0]
                # detect how many grippers touching
                nodes2 = np.nonzero(disp1 < args.neighbor_radius)[0]
                # print('visualize prim1 neighbors')
                # print(nodes1)
                # if ind == 15:
                    # import pdb; pdb.set_trace()
                # visualize_neighbors(pos, pos, 0, nodes1)
                prim1 = np.ones(nodes1.shape[0], dtype=np.int) * (n_particle + 9 + ind)
                rels += [np.stack([nodes1, prim1], axis=1)]
                prim2 = np.ones(nodes2.shape[0], dtype=np.int) * (n_particle + 9 + ind)
                rels2 += [np.stack([nodes2, prim2], axis=1)]
            
            queries = np.arange(n_particle)
            anchors = np.arange(n_particle)
        
        rels += find_relations_neighbor(pos, queries, anchors, args.neighbor_radius, 2, var)
        rels2 += find_relations_neighbor(pos, queries, anchors, args.neighbor_radius, 2, var)
        # rels += find_k_relations_neighbor(args.neighbor_k, pos, queries, anchors, args.neighbor_radius, 2, var)

        if len(rels) > 0:
            rels = np.concatenate(rels, 0)

        if len(rels2) > 0:
            rels2 = np.concatenate(rels2, 0)

        if verbose:
            print("Relations neighbor", rels.shape)

        n_rel = rels.shape[0]
        Rr = torch.zeros(n_rel, n_particle + n_shape)
        Rs = torch.zeros(n_rel, n_particle + n_shape)
        Rr[np.arange(n_rel), rels[:, 0]] = 1
        Rs[np.arange(n_rel), rels[:, 1]] = 1

        Rn = torch.zeros(rels2.shape[0], n_particle + n_shape)
        Rn[np.arange(rels2.shape[0]), rels2[:, 1]] = 1

        if verbose:
            print("Object attr:", np.sum(attr, axis=0))
            print("Particle attr:", np.sum(attr[:n_particle], axis=0))
            print("Shape attr:", np.sum(attr[n_particle:n_particle + n_shape], axis=0))

        if verbose:
            print("Particle positions stats")
            print("  Shape", positions.shape)
            print("  Min", np.min(positions[:n_particle], 0))
            print("  Max", np.max(positions[:n_particle], 0))
            print("  Mean", np.mean(positions[:n_particle], 0))
            print("  Std", np.std(positions[:n_particle], 0))

        if var:
            particle = positions
        else:
            particle = torch.FloatTensor(positions)

        if verbose:
            for i in range(count_nodes - 1):
                if np.sum(np.abs(attr[i] - attr[i + 1])) > 1e-6:
                    print(i, attr[i], attr[i + 1])

        attr = torch.FloatTensor(attr)
        if stdreg:
            cluster_onehot = torch.FloatTensor(cluster_onehot)
        else:
            cluster_onehot = None
        assert attr.size(0) == count_nodes
        assert attr.size(1) == args.attr_dim

        # attr: (n_p + n_s) x attr_dim
        # particle (unnormalized): (n_p + n_s) x state_dim
        # Rr, Rs: n_rel x (n_p + n_s)
        return attr, particle, Rr, Rs, Rn, cluster_onehot
