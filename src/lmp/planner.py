import os
import sys
import openai


def planner(args, detection_results, out_file):
    with open('prompts/planner.py', 'r') as f:
        prompts = f.read()

    prompts_all = prompts.replace('[[ detection_results ]]', detection_results)
    # print(prompts_all)
    message = [{"role": "user", "content": prompts_all}]
    response = openai.ChatCompletion.create(
        model=args.llm,
        messages=message,
    )
    response_text = response['choices'][0]['message']['content']
    # print('\n\n')
    # print(response_text)
    with open(out_file, 'a') as f:
        f.write('import os, sys\nsys.path.append(os.getcwd())\nimport lmp\nfrom lmp.gen_gnn import gen_gnn\n\n')
        f.write(response_text)
        f.write('\n\n')
