#!/bin/bash

benchmarks=("HumanEval" "APPS" "LIBRO_d4j" "TestEval_total" "TestEval_line" "TestEval_branch" "TestEval_path")
weak_models=("llama3-8b" "llama3.1-8b" "mistral-nemo-12b" "phi4-14b" "qwen2.5-coder-7b")
prompt_types=("entire" "input_only")

for benchmark in "${benchmarks[@]}"; do
    for weak_model in "${weak_models[@]}"; do
        for prompt_type in "${prompt_types[@]}"; do
            python extract_internal_state.py -b "$benchmark" -m "$weak_model" --prompt_type="$prompt_type"
        done
    done
done