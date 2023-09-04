import os
import openai

from lmp.planner import planner
from config import gen_args

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.system('rm lmp_output/*.py')


def initial_perception_function():
    return "The object is a piece of dough."

args = gen_args()
with open(args.api_key, 'r') as f:
    api_key = f.read().strip()
openai.api_key = api_key

'''
step 0: generate initial perception that describes the overall scene
'''
detection_results = initial_perception_function()

'''
step 1: generate planner output
'''
planner(args, detection_results, out_file='lmp_output/planner.py')

'''
step 2: generate gen_gnn output
'''
out_file = f'lmp_output/gen_gnn.py'
os.system(f"python lmp_output/planner.py lmp_output/gen_gnn.py {args.llm} {api_key}")

'''
final step: run executable
'''
def run_gnn(*args):  # TODO
    pass

run_gnn(
    # 'lmp_output/gen_gnn.py',
    'lmp_output/eval.py',
    'lmp_output/model.py',
    'gnn/utils.py',
)
