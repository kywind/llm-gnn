from lmp.gen_gnn import gen_gnn
from config import gen_args

args = gen_args()

# Info: The object is a pile of coffee beans. It belongs to the class of granular objects.
# Code:
args.material = "granular"

# Info: The object is an apple. It belongs to the class of rigid objects.
# Code:
args.material = "rigid"

# Info: [[ info 1 ]] [[ info 2 ]]
# Code:
