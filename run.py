import os
import openai

import lmp
from argp import get_args

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.system('rm lmp_output/*')

args = get_args()
with open(args.api_key, 'r') as f:
    openai.api_key = f.read().strip()

detection_results = "The object is a pile of beans."
idx = 0
lmp.planner(args, detection_results, out_file=f'lmp_output/{idx}.py')

for _ in range(100):
    with open(f'lmp_output/{idx}.py', 'r') as f:
        next_lmp = f.read()
    if not next_lmp.find('\nlmp.'):
        break
    os.system(f"python lmp_output/{idx}.py lmp_output/{idx+1}.py")
    idx += 1
