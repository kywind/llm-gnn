import lmp

from graph.base import Graph, Particle, Relation
from graph.granular import gen_granular_particle
from perception import get_observation

# Query: Construct graph relations using nearest neighbors search bounded by a threshold, and use particle attributes and their relative positions as relation attributes.
# Code:
observation = perception.get_observation()
gen_granular_particle(observation, downsample=True, use_density=True, use_action=True)

# Query: Construct graph particles without using downsampling, and do not use action as particle attributes.
# Code:
observation = perception.get_observation()
gen_granular_particle(observation, downsample=False, use_action=False)

# Query: [[ instruction ]]
# Code:
