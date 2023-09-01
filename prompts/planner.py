import lmp


# Query: The object is a pile of coffee beans.
# Reasoning: 
# Let's think step by step.
# 1. The object is a pile of coffee beans, so it belongs to the class of granular objects.
# For granular objects, since there are too many object particles.
# Therefore, when constructing graph particles, we need downsampling to reduce complexity.
# We need to use the graph density and action info as the particle attributes.
# 2. Then, when constructing graph relations, we need to use nearest neighbors search bounded by a threshold.
# We need to use particle attributes and their relative positions as the relation attributes.
# 3. Finally, we can construct a graph.
# Code:
lmp.get_particle("Construct graph particles using downsampling, and use graph density and action info as particle attributes.")
lmp.get_relation("Construct graph relations using nearest neighbors search bounded by a threshold, and use particle attributes and their relative positions as relation attributes.")
lmp.get_graph("Construct a graph using graph particles and graph relations.")

# Query: The object is an apple.
# Reasoning:
# Let's think step by step.
# 1. The object is an apple, so it belongs to the class of rigid objects.
# For rigid objects, since there are not too many object particles.
# Therefore, when constructing graph particles, we don't need downsampling.
# We need to use the object pose as the particle attributes.
# 2. Then, when constructing graph relations, we need to use nearest neighbors search bounded by a threshold.
# We need to use particle attributes and their relative positions as the relation attributes.
# 3. Finally, we can construct a graph.
# Code:
lmp.get_particle("Construct graph particles without downsampling, and use object pose as particle attributes.")
lmp.get_relation("Construct graph relations using nearest neighbors search bounded by a threshold, and use particle attributes and their relative positions as relation attributes.")
lmp.get_graph("Construct a graph using graph particles and graph relations.")

# Query: [[ detection_results ]]
# Reasoning:
# Let's think step by step.
