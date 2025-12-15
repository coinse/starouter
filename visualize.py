import argparse
import os
from config import *
from data_utils import load_keys, retrieve_corresponding_result_loader
from metric import compute_routed_performance, compute_random_performance_for_given_ratio, compute_best_and_worst_routing_performance

import pandas as pd
import matplotlib.pyplot as plt
import tqdm

def visualize_performance(probs_df, strong_model_name, strong_result, weak_result, title, metric, fig_path):
    assert len(weak_result) == len(strong_result)
    _, ax = plt.subplots(figsize=(6, 4), dpi=300)
    for algorithm in ALGORITHMS:
        weak_win_probs = probs_df[probs_df['Model'] == algorithm]["CV Probabilities"].iloc[0][:, 1]
        ratios, routed_perf = compute_routed_performance(weak_win_probs, strong_result, weak_result)
        ax.plot(ratios, routed_perf, label=algorithm)
    random_perf = [compute_random_performance_for_given_ratio(strong_result, weak_result, r) for r in ratios]
    ax.plot(ratios, random_perf, label='Random')

    x_indices, best_performances, worst_performances = compute_best_and_worst_routing_performance(weak_result, strong_result)
    ax.plot(x_indices, best_performances, linestyle='--', label='Best')
    ax.plot(x_indices, worst_performances, linestyle='--', label='Worst')
    
    ax.set_xlabel(f"% calls to {strong_model_name}")
    ax.set_ylabel(f"{metric}(Total {len(strong_result)})")
    ax.grid(True)
    ax.set_title(title)
    ax.legend(loc='lower right')
    plt.savefig(fig_path)

def retrieve_metric_name(benchmark):
    return {
        'HumanEval': "pass@any",
        'APPS': "pass@any",
        'TestEval_total': "line-cov@1",
        'TestEval_line': "lines covered",
        'TestEval_branch': "branches covered",
        'TestEval_path': "paths covered",
        'LIBRO_d4j': "reproduced@any",
    }[benchmark]     

def iterate_over_embeddings(benchmark, strong_model_name, strong_model_result_path, weak_model_name, weak_model_result_path, prompt_type):
    for embedding_model in tqdm.tqdm(EMBEDDING_MODELS):
        pkl_path = f'./results/{benchmark.lower()}/preset/{embedding_model}_{strong_model_name}_{weak_model_name}_classifiers_{prompt_type}.pkl'
        perf_path = f'./results/{benchmark.lower()}/preset/{embedding_model}_{strong_model_name}_{weak_model_name}_routing_performance_{prompt_type}.png'
        if not os.path.isfile(pkl_path):
            print(f"{pkl_path} does not exist!")
            continue
        if os.path.isfile(perf_path):
            print(f"{perf_path} already exists!")
            continue
        keys = load_keys(benchmark, strong_model_name, weak_model_name)
        embedding_df = pd.read_pickle(pkl_path)
        loader = retrieve_corresponding_result_loader(benchmark) 
        strong_result = loader(strong_model_result_path, keys)
        weak_result = loader(weak_model_result_path, keys)
        title = f'{benchmark} / {strong_model_name} vs. {weak_model_name}\nembedding ({embedding_model} / {prompt_type})'
        visualize_performance(
            probs_df=embedding_df, 
            strong_model_name=strong_model_name,
            strong_result=strong_result, 
            weak_result=weak_result, 
            title=title, 
            metric=retrieve_metric_name(benchmark),
            fig_path=perf_path,
        )

def iterate_over_internal_states(benchmark, strong_model_name, strong_model_result_path, weak_model_name, weak_model_result_path, total_layers, prompt_type):
    layers = [round(total_layers / 2), round(total_layers * 2 / 3), round(total_layers * 3 / 4)]
    
    for layer in tqdm.tqdm(layers):
        pkl_path = f'./results/{benchmark.lower()}/preset/internal_state_layer{layer}_{strong_model_name}_{weak_model_name}_classifiers_{prompt_type}.pkl'
        perf_path = f'./results/{benchmark.lower()}/preset/internal_state_layer{layer}_{strong_model_name}_{weak_model_name}_routing_performance_{prompt_type}.png'
        if not os.path.isfile(pkl_path):
            print(f"{pkl_path} does not exist!")
            continue
        if os.path.isfile(perf_path):
            print(f"{perf_path} already exists!")
            continue
        keys = load_keys(benchmark, strong_model_name, weak_model_name)
        internal_state_df = pd.read_pickle(pkl_path)
        loader = retrieve_corresponding_result_loader(benchmark) 
        strong_result = loader(strong_model_result_path, keys)
        weak_result = loader(weak_model_result_path, keys)
        title = f'{benchmark} / {strong_model_name} vs. {weak_model_name}\ninternal state (Layer {layer} / {prompt_type})'
        visualize_performance(
            probs_df=internal_state_df, 
            strong_model_name=strong_model_name,
            strong_result=strong_result, 
            weak_result=weak_result, 
            title=title, 
            metric=retrieve_metric_name(benchmark),
            fig_path=perf_path,
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prompt_type', default='entire', choices=PROMPT_TYPES)
    parser.add_argument('--code_generation', action='store_true')
    args = parser.parse_args()

    if args.code_generation:
        benchmarks = CODE_GENERATION_BENCHMARKS
    else:
        benchmarks = TESTING_BENCHMARKS
    
    for benchmark in benchmarks:
        print(f'Now processing {benchmark}')
        for strong_model_name in STRONG_MODELS:
            for weak_model_name in WEAK_MODELS:
                strong_model_result_path = path_to_result_file_for(benchmark, strong_model_name)
                weak_model_result_path = path_to_result_file_for(benchmark, weak_model_name)
                
                iterate_over_embeddings(
                    benchmark=benchmark,
                    strong_model_name=strong_model_name,
                    strong_model_result_path=strong_model_result_path,
                    weak_model_name=weak_model_name,
                    weak_model_result_path=weak_model_result_path,
                    prompt_type=args.prompt_type,
                )
                iterate_over_internal_states(
                    benchmark=benchmark,
                    strong_model_name=strong_model_name,
                    strong_model_result_path=strong_model_result_path,
                    weak_model_name=weak_model_name,
                    weak_model_result_path=weak_model_result_path,
                    total_layers=total_number_of_layers_of(weak_model_name),
                    prompt_type=args.prompt_type,
                )
