import argparse
import json
import os

import sys
sys.path.append("..")
from config import path_to_result_file_for

def label_data(weak_model_result_path, strong_model_result_path):
    with open(weak_model_result_path) as f:
        weak_result = json.load(f)

    with open(strong_model_result_path) as f:
        strong_result = json.load(f)

    assert strong_result.keys() == weak_result.keys()
    problem_indices = strong_result.keys()
    vs_result = list(map(
                         lambda pair_of_results: pair_of_results[0] > pair_of_results[1], # Strong win when its pass rate is greater / binary labels also can be compared directly as True > False
                         zip(strong_result.values(), weak_result.values())
                        ))
 
    return vs_result, problem_indices

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weak_model', required=True)
    parser.add_argument('-s', '--strong_model', required=True)
    parser.add_argument('-b', '--benchmark')
    args = parser.parse_args()

    base_dir = f'{args.benchmark.lower()}/route_data'
    
    strong_model = args.strong_model
    strong_result_file_path = path_to_result_file_for(args.benchmark, strong_model)

    weak_model = args.weak_model
    weak_result_file_path = path_to_result_file_for(args.benchmark, weak_model)
    
    vs_result, keys_list = label_data(weak_result_file_path, strong_result_file_path)

    os.makedirs(f'{base_dir}/pairwise', exist_ok=True)
    vs_data_path = f'{base_dir}/pairwise/{strong_model}_vs_{weak_model}.json'

    vs_data = list()

    for i, key in enumerate(keys_list):
        vs_data.append({
            "model_a": strong_model,
            "model_b": weak_model,
            "idx": i,
            "key": key,
            "winner": "model_a" if vs_result[i] else "model_b"
        })

    with open(vs_data_path, 'w') as f:
        json.dump(vs_data, f, indent=4)
