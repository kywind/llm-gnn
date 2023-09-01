import os
import sys
import openai


def planner(args, detection_results, out_file):
    with open('prompts/planner.py', 'r') as f:
        planner_prompts = f.read()

    planner_prompts_all = planner_prompts.replace('[[ detection_results ]]', detection_results)
    # print(planner_prompts_all)
    planner_message = [{"role": "user", "content": planner_prompts_all}]
    planner_response = openai.ChatCompletion.create(
        model=args.model,
        messages=planner_message,
    )
    planner_response_text = planner_response['choices'][0]['message']['content']
    # print('\n\n')
    # print(planner_response_text)
    with open(out_file, 'a') as f:
        f.write('import lmp\n\n')
        f.write(planner_response_text)
        f.write('\n\n')


def get_graph(instruction):
    out_file = sys.argv[1]
    response_text = 'get_graph end'

    if not os.path.exists(out_file):
        output_text = 'import lmp\n\n' + response_text + '\n\n'
    else:
        output_text = response_text + '\n\n'

    with open(out_file, 'a') as f:
        f.write('import lmp\n\n')
        f.write(response_text)
        f.write('\n\n')


def get_particle(instruction):
    out_file = sys.argv[1]
    response_text = 'get_graph end'
    
    with open(out_file, 'a') as f:
        f.write('import lmp\n\n')
        f.write(response_text)
        f.write('\n\n')


def get_relation(instruction):
    out_file = sys.argv[1]
    response_text = 'get_graph end'
    
    if not os.path.exists(out_file):
        output_text = 'import lmp\n\n' + response_text + '\n\n'
    else:
        output_text = response_text + '\n\n'

    with open(out_file, 'a') as f:
        f.write('import lmp\n\n')
        f.write(response_text)
        f.write('\n\n')
