import os
import sys
import openai
import re


def gen_gnn(info):
    out_file = sys.argv[1]
    llm = sys.argv[2]
    openai.api_key = sys.argv[3]

    tgt_files = ['eval.py', 'model.py']
    for tgt_file in tgt_files:
        template = 'gnn/' + tgt_file
        with open(template, 'r') as f:
            tpl = f.read()
        blank_st = [m.end() for m in re.finditer('### LLM START ###', tpl)]
        blank_ed = [m.start() for m in re.finditer('### LLM END ###', tpl)]
        assert len(blank_st) == len(blank_ed)
        blank_info_list = [tpl[blank_st[i]:blank_ed[i]].strip() for i in range(len(blank_st))]

        ptr = 0
        tpl_list = []
        indent_list = []
        for i in range(len(blank_ed)):
            substr = tpl[ptr:blank_ed[i]]
            tpl_list.append(substr)
            n_indent = len(substr) - len(substr.rstrip(' '))
            indent_list.append(n_indent)
            ptr = blank_ed[i]
        tpl_list.append(tpl[ptr:])

        code_dict = {}
        for blank_info in blank_info_list:
            if not os.path.exists(f'prompts/gen_gnn/{blank_info}.py'):
                continue
            with open(f'prompts/gen_gnn/{blank_info}.py', 'r') as f:
                prompts = f.read()
            for i in range(len(info)):
                prompts = prompts.replace(f'[[ info {i+1} ]]', info[i])
            prompts_all = prompts
            # prompts_all = prompts.replace('[[ detection_results ]]', info[0]) # TODO find another way to add info
            # print(prompts_all)
            message = [{"role": "user", "content": prompts_all}]
            response = openai.ChatCompletion.create(
                model=llm,
                messages=message,
            )
            response_text = response['choices'][0]['message']['content']
            # print('\n\n')
            # print(response_text)
            code_dict[blank_info] = response_text

        res = []
        for i in range(len(blank_st)):
            blank_info = blank_info_list[i]
            print(blank_info)
            res.append(tpl_list[i])
            # res.append(f'hello world\n' + ' ' * indent_list[i])
            if blank_info in code_dict:
                code = code_dict[blank_info].strip('\n').split('\n')
                for j in range(len(code)):
                    line = code[j]
                    res.append(f'{line}\n' + ' ' * indent_list[i])
        res.append(tpl_list[-1])
        res = ''.join(res)

        with open('lmp_output/' + tgt_file, 'w') as f:
            f.write(res)

