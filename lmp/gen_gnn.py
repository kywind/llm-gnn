import os
import sys
import openai
import re


def gen_gnn(info):
    out_file = sys.argv[1]
    llm = sys.argv[2]
    openai.api_key = sys.argv[3]

    with open('prompts/gen_gnn.py', 'r') as f:
        prompts = f.read()
    
    '''
    gen gnn/eval.py
    '''
    template = 'gnn/eval.py'
    with open(template, 'r') as f:
        tpl = f.read()
    blank_st = [m.end() for m in re.finditer('### LLM START ###', tpl)]
    blank_ed = [m.start() for m in re.finditer('### LLM END ###', tpl)]
    assert len(blank_st) == len(blank_ed)
    blank_info = [tpl[blank_st[i]:blank_ed[i]].strip() for i in range(len(blank_st))]

    # TODO: replace with LLM output
    ptr = 0
    tpl_list = []
    indent_list = []
    blank_ed.append(-1)
    for i in range(len(blank_ed) - 1):
        substr = tpl[ptr:blank_ed[i]]
        tpl_list.append(substr)
        n_indent = len(substr) - len(substr.rstrip(' '))
        indent_list.append(n_indent)
        ptr = blank_ed[i]

    res = []
    for i in range(len(blank_st)):
        print(blank_info[i])
        res.append(tpl_list[i])
        res.append('hello world\n' + ' ' * indent_list[i])
        res.append('second line\n' + ' ' * indent_list[i])
    res.append(tpl_list[-1])
    res = ''.join(res)

    with open('lmp_output/eval.py', 'w') as f:
        f.write(res)
    
    '''
    gen gnn/model.py
    '''
    template = 'gnn/model.py'
    with open(template, 'r') as f:
        tpl = f.read()
    blank_st = [m.end() for m in re.finditer('### LLM START ###', tpl)]
    blank_ed = [m.start() for m in re.finditer('### LLM END ###', tpl)]
    assert len(blank_st) == len(blank_ed)
    blank_info = [tpl[blank_st[i]:blank_ed[i]].strip() for i in range(len(blank_st))]
    
    # TODO: replace with LLM output
    ptr = 0
    tpl_list = []
    indent_list = []
    blank_ed.append(-1)
    for i in range(len(blank_ed) - 1):
        substr = tpl[ptr:blank_ed[i]]
        tpl_list.append(substr)
        n_indent = len(substr) - len(substr.rstrip(' '))
        indent_list.append(n_indent)
        ptr = blank_ed[i]

    res = []
    for i in range(len(blank_st)):
        print(blank_info[i])
        res.append(tpl_list[i])
        res.append('hello world\n' + ' ' * indent_list[i])
        res.append('second line\n' + ' ' * indent_list[i])
    res.append(tpl_list[-1])
    res = ''.join(res)

    with open('lmp_output/model.py', 'w') as f:
        f.write(res)

    '''
    detection_results = info[0]
    prompts_all = prompts.replace('[[ detection_results ]]', detection_results)
    # print(prompts_all)
    message = [{"role": "user", "content": prompts_all}]
    response = openai.ChatCompletion.create(
        model=llm,
        messages=message,
    )
    response_text = response['choices'][0]['message']['content']
    # print('\n\n')
    # print(response_text)
    with open(out_file, 'a') as f:
        f.write('import os, sys\nsys.path.append(os.getcwd())\nimport lmp\n\n')
        f.write(response_text)
        f.write('\n\n')
    '''

