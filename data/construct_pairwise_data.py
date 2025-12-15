import argparse
import json
import os

import sys
sys.path.append("..")
from config import path_to_result_file_for

def load_code_generation_data(weak_model_result_path, strong_model_result_path):
    with open(weak_model_result_path) as f:
        weak_result = json.load(f)

    with open(strong_model_result_path) as f:
        strong_result = json.load(f)

    assert strong_result.keys() == weak_result.keys()
    problem_indices = strong_result.keys()
    vs_result = list(map(
                         lambda pair_of_pass_rates: pair_of_pass_rates[0] > pair_of_pass_rates[1], # Strong win only when its pass rate is greater
                         zip(strong_result.values(), weak_result.values())
                        ))
    
    return vs_result, problem_indices

def load_humaneval_data(weak_model_result_path, strong_model_result_path):
    return load_code_generation_data(weak_model_result_path, strong_model_result_path)
    
def load_apps_data(weak_model_result_path, strong_model_result_path):
    return load_code_generation_data(weak_model_result_path, strong_model_result_path)

def load_testeval_overall_coverage_data(weak_model_result_path, strong_model_result_path):
    with open(weak_model_result_path) as f:
        weak_result = json.load(f)
        weak_line_coverage = map(lambda cov_dict: cov_dict["1"]["line_cov"] if "1" in cov_dict else 0.0, weak_result.values())

    with open(strong_model_result_path) as f:
        strong_result = json.load(f)
        strong_line_coverage = map(lambda cov_dict: cov_dict["1"]["line_cov"] if "1" in cov_dict else 0.0, strong_result.values())

    assert strong_result.keys() == weak_result.keys()
    problem_indices = strong_result.keys()
    vs_result = list(map(
                         lambda pair_of_coverage: pair_of_coverage[0] > pair_of_coverage[1],
                         zip(strong_line_coverage, weak_line_coverage)
                        ))
    
    return vs_result, problem_indices

def load_testeval_targeted_branch_coverage_data(weak_model_result_path, strong_model_result_path):
    with open(weak_model_result_path) as f:
        weak_result = json.load(f)
        weak_branch_covered = map(lambda cov_dict: cov_dict["covered"], weak_result.values())

    with open(strong_model_result_path) as f:
        strong_result = json.load(f)
        strong_branch_covered = map(lambda cov_dict: cov_dict["covered"], strong_result.values())

    assert strong_result.keys() == weak_result.keys()
    problem_indices = strong_result.keys()
    
    vs_result = list(map(
                         lambda pair_of_covered: not pair_of_covered[1] and pair_of_covered[0], # Strong win only when 1) the strong model covers & 2) the weak model does not cover
                         zip(strong_branch_covered, weak_branch_covered)
                        ))
    
    return vs_result, problem_indices

def load_binary_data(weak_model_result_path, strong_model_result_path):
    with open(weak_model_result_path) as f:
        weak_result = json.load(f)

    with open(strong_model_result_path) as f:
        strong_result = json.load(f)

    assert strong_result.keys() == weak_result.keys()
    problem_indices = strong_result.keys()
    vs_result = list(map(
                         lambda pair_of_covered: float(pair_of_covered[0]) > float(pair_of_covered[1]), # True -> 1.0, False -> 0.0 / for path coverage, path similarity float is compared
                         zip(strong_result.values(), weak_result.values())
                        ))
    
    return vs_result, problem_indices

def load_testeval_targeted_line_coverage_data(weak_model_result_path, strong_model_result_path):
    return load_binary_data(weak_model_result_path, strong_model_result_path)

def load_testeval_targeted_path_coverage_data(weak_model_result_path, strong_model_result_path):
    return load_binary_data(weak_model_result_path, strong_model_result_path)

def load_libro_d4j_data(weak_model_result_path, strong_model_result_path):
    return load_binary_data(weak_model_result_path, strong_model_result_path)

def retrieve_pairwise_data_loader(benchmark):
    return {
        'HumanEval': load_humaneval_data,
        'APPS': load_apps_data,
        'TestEval_total': load_testeval_overall_coverage_data,
        'TestEval_line': load_testeval_targeted_line_coverage_data,
        'TestEval_branch': load_testeval_targeted_branch_coverage_data,
        'TestEval_path': load_testeval_targeted_path_coverage_data,
        'LIBRO_d4j': load_libro_d4j_data,
    }[benchmark]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weak_model', required=True)
    parser.add_argument('-s', '--strong_model', required=True)
    parser.add_argument('-b', '--benchmark')
    args = parser.parse_args()

    base_dir = f'{args.benchmark.lower()}/route_data'
    
    loader = retrieve_pairwise_data_loader(args.benchmark)

    strong_model = args.strong_model
    strong_result_file_path = path_to_result_file_for(args.benchmark, strong_model)

    weak_model = args.weak_model
    weak_result_file_path = path_to_result_file_for(args.benchmark, weak_model)
    
    vs_result, keys_list = loader(weak_result_file_path, strong_result_file_path)

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
