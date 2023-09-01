import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0613')
    parser.add_argument('--api-key', type=str, default='files/api_key.txt')
    # parser.add_argument('--base-prompt', type=str, default='files/base_prompt.txt')
    # parser.add_argument('--func-list', type=str, default='files/func_list.json')
    return parser.parse_args()
