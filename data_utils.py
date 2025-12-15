import json
from config import REPO_PATH

def load_keys(benchmark, strong_model_name, weak_model_name):
    labels_path = f'{REPO_PATH}/data/{benchmark.lower()}/route_data/pairwise/{strong_model_name}_vs_{weak_model_name}.json'
    with open(labels_path) as f:
        data = json.load(f)
        keys = [item['key'] for item in data]

    return keys

def load_code_generation_result(path, keys):
    with open(path) as f:
        strong_result = json.load(f)

    return [1 if strong_result[key] > 0 else 0 for key in keys]

def load_testeval_overall_coverage_result(path, keys):
    with open(path) as f:
        result = json.load(f)

    return [result[key]['1']['line_cov'] if '1' in result[key] else 0.0 for key in keys]

def load_testeval_targeted_branch_coverage_results(path, keys):
    with open(path) as f:
        result = json.load(f)

    return [1 if result[key]["covered"] else 0 for key in keys]

def load_binary_results(path, keys):
    with open(path) as f:
        result = json.load(f)

    return [1 if result[key] else 0 for key in keys]

def load_testeval_targeted_line_coverage_results(path, keys):
    return load_binary_results(path, keys)

def load_testeval_targeted_path_coverage_results(path, keys):
    with open(path) as f:
        result = json.load(f)
    return [1 if result[key] == 1.0 else 0 for key in keys] # exact match

def load_libro_d4j_results(path, keys):
    return load_binary_results(path, keys)

def retrieve_corresponding_result_loader(benchmark):
    return {
        'HumanEval': load_code_generation_result,
        'APPS': load_code_generation_result,
        'TestEval_total': load_testeval_overall_coverage_result,
        'TestEval_line': load_testeval_targeted_line_coverage_results,
        'TestEval_branch': load_testeval_targeted_branch_coverage_results,
        'TestEval_path': load_testeval_targeted_path_coverage_results,
        'LIBRO_d4j': load_libro_d4j_results,
    }[benchmark]