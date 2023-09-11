from lmp.gen_gnn import gen_gnn


# Query: The object is a pile of coffee beans.
# Task: Use a pusher to push the object pile to the target position.
# Reasoning: (write in comments)
# Let's think step by step.
# 1. The object is a pile of coffee beans, so it belongs to the class of granular objects.
# 2. The task is to push the object pile, so we need to model detailed dynamics within the object pile.
# 3. Therefore, we need to construct a high-resolution network.
# Code:
info = []
info.append("The object is a pile of coffee beans.")
info.append("It belongs to the class of granular objects.")
info.append("The task is to push the object pile.")
info.append("We need to construct a network with high resolution.")
gen_gnn(info)

# Query: The object is an apple.
# Task: Use a gripper to grasp the object and put it in the target position.
# Reasoning: (write in comments)
# Let's think step by step.
# 1. The object is an apple, so it belongs to the class of rigid objects.
# 2. The task is to grasp the rigid object, so we do not need to model complex movements of the object.
# 3. Therefore, we can view the object as a single particle, but with detailed geometry and pose.
# Code:
info = []
info.append("The object is an apple.")
info.append("It belongs to the class of rigid objects.")
info.append("The task is to grasp the object.")
info.append("We need to model the object as a single particle with detailed geometry and pose.")
gen_gnn(info)

# Query: [[ detection_results ]]
# Reasoning: (write in comments)
# Let's think step by step.
