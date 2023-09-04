import lmp
from config import gen_args
args = gen_args()

# Query: The object is a pile of coffee beans.
# Reasoning: (write in comments)
# Let's think step by step.
# 1. The object is a pile of coffee beans, so it belongs to the class of granular objects.
args.material = 'granular'

# Query: The object is an apple.
# Reasoning: (write in comments)
# Let's think step by step.
# 1. The object is an apple, so it belongs to the class of rigid objects.
args.material = 'rigid'
