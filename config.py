REPO_PATH = '/root/se-router'

TESTING_BENCHMARKS = ['TestEval_line', 'TestEval_branch', 'TestEval_path', 'TestEval_total', 'LIBRO_d4j']
CODE_GENERATION_BENCHMARKS = ['HumanEval', 'APPS']

STRONG_MODELS = ['gpt-4o', 'gemini-2.5-flash-lite']

WEAK_MODELS = ['llama3-8b', 'llama3.1-8b', 'mistral-nemo-12b', 'qwen2.5-coder-7b', 'phi4-14b']

WEAK_MODEL_HUGGINGFACE_MAP = {
    'llama3-8b': "meta-llama/Meta-Llama-3-8B-Instruct", 
    'llama3.1-8b': "meta-llama/Meta-Llama-3.1-8B-Instruct", 
    'mistral-nemo-12b': "mistralai/Mistral-Nemo-Instruct-2407", 
    'phi4-14b': "microsoft/phi-4",
    'qwen2.5-coder-7b': "Qwen/Qwen2.5-Coder-7B-Instruct", 
}

WEAK_MODEL_TOTAL_LAYERS_MAP = {
    'llama3-8b': 32,
    'llama3.1-8b': 32,
    'mistral-nemo-12b': 40,
    'phi4-14b': 40,
    'qwen2.5-coder-7b': 32,
}

RESULT_FILES_MAP = {
    'HumanEval': {
        'gpt-4o': 'data/humaneval/combined_results/HumanEval_gpt-4o_base@10.json',
        'gemini-2.5-flash-lite': 'data/humaneval/combined_results/HumanEval_gemini-2.5-flash-lite_base@10.json',
        'llama3-8b': 'data/humaneval/combined_results/HumanEval_Meta-Llama-3-8B-Instruct_base@10.json',
        'llama3.1-8b': 'data/humaneval/combined_results/HumanEval_Meta-Llama-3.1-8B-Instruct_base@10.json',
        'mistral-nemo-12b': 'data/humaneval/combined_results/HumanEval_Mistral-Nemo-Instruct-2407_base@10.json',
        'phi4-14b': 'data/humaneval/combined_results/HumanEval_phi-4_base@10.json',   
        'qwen2.5-coder-7b': 'data/humaneval/combined_results/HumanEval_Qwen2.5-Coder-7B-Instruct_base@10.json',
    },
    'APPS': {
        'gpt-4o': 'data/apps/combined_results/APPS_gpt-4o_base@10.json',
        'gemini-2.5-flash-lite': 'data/apps/combined_results/APPS_gemini-2.5-flash-lite_base@10.json',
        'llama3-8b': 'data/apps/combined_results/APPS_Meta-Llama-3-8B-Instruct_base@10.json',
        'llama3.1-8b': 'data/apps/combined_results/APPS_Meta-Llama-3.1-8B-Instruct_base@10.json',
        'mistral-nemo-12b': 'data/apps/combined_results/APPS_Mistral-Nemo-Instruct-2407_base@10.json',
        'phi4-14b': 'data/apps/combined_results/APPS_phi-4_base@10.json',   
        'qwen2.5-coder-7b': 'data/apps/combined_results/APPS_Qwen2.5-Coder-7B-Instruct_base@10.json',
    },
    'TestEval_total': {   
        'gpt-4o': 'data/testeval_total/combined_results/totalcov_gpt-4o_result.json',
        'gemini-2.5-flash-lite': 'data/testeval_total/combined_results/totalcov_gemini-2.5-flash-lite_result.json',
        'llama3-8b': 'data/testeval_total/combined_results/totalcov_Meta-Llama-3-8B-Instruct_result.json',
        'llama3.1-8b': 'data/testeval_total/combined_results/totalcov_Meta-Llama-3.1-8B-Instruct_result.json',
        'mistral-nemo-12b': 'data/testeval_total/combined_results/totalcov_Mistral-Nemo-Instruct-2407_result.json',
        'phi4-14b': 'data/testeval_total/combined_results/totalcov_phi-4_result.json',
        'qwen2.5-coder-7b': 'data/testeval_total/combined_results/totalcov_Qwen2.5-Coder-7B-Instruct_result.json',
    },
    'TestEval_line': {
        'gpt-4o': 'data/testeval_line/combined_results/linecov_gpt-4o_result.json',
        'gemini-2.5-flash-lite': 'data/testeval_line/combined_results/linecov_gemini-2.5-flash-lite_result.json',
        'llama3-8b': 'data/testeval_line/combined_results/linecov_Meta-Llama-3-8B-Instruct_result.json',
        'llama3.1-8b': 'data/testeval_line/combined_results/linecov_Meta-Llama-3.1-8B-Instruct_result.json',
        'mistral-nemo-12b': 'data/testeval_line/combined_results/linecov_Mistral-Nemo-Instruct-2407_result.json',
        'phi4-14b': 'data/testeval_line/combined_results/linecov_phi-4_result.json',   
        'qwen2.5-coder-7b': 'data/testeval_line/combined_results/linecov_Qwen2.5-Coder-7B-Instruct_result.json',
    },
    'TestEval_branch': {
        'gpt-4o': 'data/testeval_branch/combined_results/branchcov_gpt-4o_result.json',
        'gemini-2.5-flash-lite': 'data/testeval_branch/combined_results/branchcov_gemini-2.5-flash-lite_result.json',
        'llama3-8b': 'data/testeval_branch/combined_results/branchcov_Meta-Llama-3-8B-Instruct_result.json',
        'llama3.1-8b': 'data/testeval_branch/combined_results/branchcov_Meta-Llama-3.1-8B-Instruct_result.json',
        'mistral-nemo-12b': 'data/testeval_branch/combined_results/branchcov_Mistral-Nemo-Instruct-2407_result.json',
        'phi4-14b': 'data/testeval_branch/combined_results/branchcov_phi-4_result.json',
        'qwen2.5-coder-7b': 'data/testeval_branch/combined_results/branchcov_Qwen2.5-Coder-7B-Instruct_result.json',
    },
    'TestEval_path': {
        'gpt-4o': 'data/testeval_path/combined_results/pathcov_gpt-4o_result.json',
        'gemini-2.5-flash-lite': 'data/testeval_path/combined_results/pathcov_gemini-2.5-flash-lite_result.json',
        'llama3-8b': 'data/testeval_path/combined_results/pathcov_Meta-Llama-3-8B-Instruct_result.json',
        'llama3.1-8b': 'data/testeval_path/combined_results/pathcov_Meta-Llama-3.1-8B-Instruct_result.json',
        'mistral-nemo-12b': 'data/testeval_path/combined_results/pathcov_Mistral-Nemo-Instruct-2407_result.json',
        'phi4-14b': 'data/testeval_path/combined_results/pathcov_phi-4_result.json',
        'qwen2.5-coder-7b': 'data/testeval_path/combined_results/pathcov_Qwen2.5-Coder-7B-Instruct_result.json',
    },
    'LIBRO_d4j': {
        'gpt-4o': 'data/libro_d4j/combined_results/simple_result_d4j_gpt-4o.json',
        'gemini-2.5-flash-lite': 'data/libro_d4j/combined_results/simple_result_d4j_gemini-2.5-flash-lite.json',
        'llama3-8b': 'data/libro_d4j/combined_results/simple_result_d4j_llama3-8b.json',
        'llama3.1-8b': 'data/libro_d4j/combined_results/simple_result_d4j_llama3.1-8b.json',
        'mistral-nemo-12b': 'data/libro_d4j/combined_results/simple_result_d4j_mistral-nemo-12b.json',
        'phi4-14b': 'data/libro_d4j/combined_results/simple_result_d4j_phi4-14b.json',
        'qwen2.5-coder-7b': 'data/libro_d4j/combined_results/simple_result_d4j_qwen2.5-coder-7b.json',
    }
}

EMBEDDING_MODELS = ['nomic-embed-text', 'text-embedding-3-small', 'text-embedding-3-large']
ALGORITHMS = ["Logistic Regression", "SVM", "MLP", "Ensemble", "RouteLLM"]
PROMPT_TYPES = ['entire', 'input_only']

DEFAULT_PROMPT_TYPE = 'entire'
DEFAULT_INTERNAL_STATE_TYPE = '3/4'
DEFAULT_EMBEDDING_TYPE = 'text-embedding-3-large'

def total_number_of_layers_of(weak_model):
    if weak_model not in WEAK_MODEL_TOTAL_LAYERS_MAP:
        raise Exception(f"No such model: {weak_model}")
    return WEAK_MODEL_TOTAL_LAYERS_MAP[weak_model]

def path_to_result_file_for(benchmark, model):
    if benchmark not in RESULT_FILES_MAP:
        raise Exception(f"No such benchmark: {benchmark}")
    if model not in RESULT_FILES_MAP[benchmark]:
        raise Exception(f"No corresponding {benchmark} result for model: {model}")
    return f"{REPO_PATH}/{RESULT_FILES_MAP[benchmark][model]}"