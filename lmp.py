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
        f.write('import os, sys\nsys.path.append(os.getcwd())\nimport lmp\n\n')
        f.write(planner_response_text)
        f.write('\n\n')


def get_graph(instruction):
    out_file = sys.argv[1]
    model = sys.argv[2]
    openai.api_key = sys.argv[3]
    response_text = '# get_graph called with instruction: ' + instruction

    # with open('prompts/get_graph.py', 'r') as f:
    #     prompts = f.read()
    # prompts_all = prompts.replace('[[ instruction ]]', instruction)
    # # print(prompts_all)
    # message = [{"role": "user", "content": prompts_all}]
    # response = openai.ChatCompletion.create(
    #     model=model,
    #     messages=message,
    # )

    new_file = not os.path.exists(out_file)
    with open(out_file, 'a') as f:
        if new_file:
            f.write('import os, sys\nsys.path.append(os.getcwd())\nimport lmp\n\n')
        f.write(response_text)
        f.write('\n\n')


def get_particle(instruction):
    out_file = sys.argv[1]
    model = sys.argv[2]
    openai.api_key = sys.argv[3]
    # response_text = '# get_particle called with instruction: ' + instruction

    with open('prompts/get_particle.py', 'r') as f:
        prompts = f.read()
    prompts_all = prompts.replace('[[ instruction ]]', instruction)
    # print(prompts_all)
    message = [{"role": "user", "content": prompts_all}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=message,
    )
    response_text = response['choices'][0]['message']['content']

    new_file = not os.path.exists(out_file)
    with open(out_file, 'a') as f:
        if new_file:
            f.write('import os, sys\nsys.path.append(os.getcwd())\nimport lmp\n\n')
        f.write(response_text)
        f.write('\n\n')


def get_relation(instruction):
    out_file = sys.argv[1]
    model = sys.argv[2]
    openai.api_key = sys.argv[3]
    response_text = '# get_relation called with instruction: ' + instruction

    # with open('prompts/get_relation.py', 'r') as f:
    #     prompts = f.read()
    # prompts_all = prompts.replace('[[ instruction ]]', instruction)
    # # print(prompts_all)
    # message = [{"role": "user", "content": prompts_all}]
    # response = openai.ChatCompletion.create(
    #     model=model,
    #     messages=message,
    # )
    # response_text = response['choices'][0]['message']['content']
    
    new_file = not os.path.exists(out_file)
    with open(out_file, 'a') as f:
        if new_file:
            f.write('import os, sys\nsys.path.append(os.getcwd())\nimport lmp\n\n')
        f.write(response_text)
        f.write('\n\n')
