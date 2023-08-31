import os
import json

def read_file_str(file_path, strip=False, split=None):
    if not os.path.isfile(file_path):
        return ''
    with open(file_path, 'r') as f:
        data = f.read()
    if strip:
        data = data.strip()
    if split is not None:
        data = data.split(split)
    return data

def write_file_str(file_path, string):
    if not os.path.isdir(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, 'w') as f:
        f.write(string)

def read_file_json(file_path, strip=False):
    if not os.path.isfile(file_path):
        return {}
    with open(file_path, 'r') as f:
        data = json.load(f)
    if strip:
        data = data.strip()
    return data
