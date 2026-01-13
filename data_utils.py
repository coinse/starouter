import json
from config import REPO_PATH

def load_keys(benchmark, strong_model_name, weak_model_name):
    labels_path = f'{REPO_PATH}/data/{benchmark.lower()}/route_data/pairwise/{strong_model_name}_vs_{weak_model_name}.json'
    with open(labels_path) as f:
        data = json.load(f)
        keys = [item['key'] for item in data]

    return keys

def load_success_at_any_result(path, keys):
    with open(path) as f:
        strong_result = json.load(f)

    return [1 if strong_result[key] > 0 else 0 for key in keys]

def load_testeval_overall_coverage_result(path, keys):
    with open(path) as f:
        result = json.load(f)

    return [result[key] for key in keys]

def retrieve_corresponding_result_loader(benchmark):
    return {
        'HumanEval': load_success_at_any_result,
        'APPS': load_success_at_any_result,
        'TestEval_total': load_testeval_overall_coverage_result,
        'TestEval_line': load_success_at_any_result,
        'TestEval_branch': load_success_at_any_result,
        'TestEval_path': load_success_at_any_result,
        'LIBRO_d4j': load_success_at_any_result,
    }[benchmark]
