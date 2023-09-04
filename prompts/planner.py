from lmp.gen_gnn import gen_gnn


# Query: The object is a pile of coffee beans.
# Reasoning: (write in comments)
# Let's think step by step.
# 1. The object is a pile of coffee beans, so it belongs to the class of granular objects.
# For granular objects, since there are too many object particles.
# Therefore, when constructing graph particles, we need downsampling to reduce complexity.
# We need to use the graph density and action info as the particle attributes.
# 2. Then, when constructing graph relations, we need to use nearest neighbors search bounded by a threshold.
# We need to use particle attributes and their relative positions as the relation attributes.
# 3. Finally, we can construct a graph of middle size since the object contains many particles.
# Code:
info = []
info.append("The object is a pile of coffee beans.")
info.append("Construct graph particles using downsampling, and use graph density and action info as particle attributes.")
info.append("Construct graph relations using nearest neighbors search bounded by a threshold, and use particle attributes and their relative positions as relation attributes.")
info.append("Construct a propagation network of middle size, with one final predictor network outputting particle xyz.")
gen_gnn(info)

# Query: The object is an apple.
# Reasoning: (write in comments)
# Let's think step by step.
# 1. The object is an apple, so it belongs to the class of rigid objects.
# For rigid objects, since there are not too many object particles.
# Therefore, when constructing graph particles, we don't need downsampling.
# We need to use the object pose as the particle attributes.
# 2. Then, when constructing graph relations, we need to use nearest neighbors search bounded by a threshold.
# We need to use particle attributes and their relative positions as the relation attributes.
# 3. Finally, we can construct a graph of small size since the object motion is simple.
# Code:
info = []
info.append("The object is an apple.")
info.append("Construct graph particles without downsampling, and use object pose as particle attributes.")
info.append("Construct graph relations using nearest neighbors search bounded by a threshold, and use particle attributes and their relative positions as relation attributes.")
info.append("Construct a propagation network of small size, with one final predictor network outputting particle xyz.")
gen_gnn(info)

# Query: [[ detection_results ]]
# Reasoning: (write in comments)
# Let's think step by step.
