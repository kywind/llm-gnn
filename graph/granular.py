import numpy as np
from graph.utils import depth2fgpcd, downsample_pcd, fps, recenter
from graph.base import Particle


def gen_granular_particle(obs, downsample=True, use_density=True, use_action=True):
    """Generate a particle."""
    def obs2ptcl_fixed_num_batch(self, obs, particle_num, batch_size):
        assert type(obs) == np.ndarray
        assert obs.shape[-1] == 5
        assert obs[..., :3].max() <= 255.0
        assert obs[..., :3].min() >= 0.0
        assert obs[..., :3].max() >= 1.0
        assert obs[..., -1].max() >= 0.7 * self.global_scale
        assert obs[..., -1].max() <= 0.8 * self.global_scale
        depth = obs[..., -1] / self.global_scale

        batch_sampled_ptcl = np.zeros((batch_size, particle_num, 3))
        batch_particle_r = np.zeros((batch_size, ))
        for i in range(batch_size):
            fgpcd = depth2fgpcd(depth, depth<0.599/0.8, self.get_cam_params())
            fgpcd = downsample_pcd(fgpcd, 0.01)
            sampled_ptcl, particle_r = fps(fgpcd, particle_num)
            batch_sampled_ptcl[i] = recenter(fgpcd, sampled_ptcl, r = min(0.02, 0.5 * particle_r))
            batch_particle_r[i] = particle_r
        return batch_sampled_ptcl, batch_particle_r
    
    batch_size = 1
    particle_num = 30
    batch_sampled_ptcl, batch_particle_r = obs2ptcl_fixed_num_batch(obs, particle_num, batch_size)
    particle_den = np.array([1 / (batch_particle_r * batch_particle_r)])[0]
    particle = Particle(
        x = batch_sampled_ptcl[0, :, 0],
        y = batch_sampled_ptcl[0, :, 1],  # upward
        z = batch_sampled_ptcl[0, :, 2],
        density = particle_den
    )
    return particle

def gen_granular_relations():
    pass
