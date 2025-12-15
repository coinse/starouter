#!/bin/bash

benchmarks=("HumanEval" "APPS" "LIBRO_d4j" "TestEval_total" "TestEval_line" "TestEval_branch" "TestEval_path")
strong_models=("gpt-4o" "gemini-2.5-flash-lite")
weak_models=("llama3-8b" "llama3.1-8b" "mistral-nemo-12b" "phi4-14b" "qwen2.5-coder-7b")

for benchmark in "${benchmarks[@]}"; do
    for strong_model in "${strong_models[@]}"; do
        for weak_model in "${weak_models[@]}"; do
            python construct_pairwise_data.py -b "$benchmark" -w "$weak_model" -s "$strong_model"
        done
    done
done