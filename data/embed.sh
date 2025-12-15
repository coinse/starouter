#!/bin/bash
benchmarks=("HumanEval" "APPS" "LIBRO_d4j" "TestEval_total" "TestEval_line" "TestEval_branch" "TestEval_path")
embedding_models=("nomic-embed-text" "text-embedding-3-small" "text-embedding-3-large")
prompt_types=("entire" "input_only")

for benchmark in "${benchmarks[@]}"; do
    for embedding_model in "${embedding_models[@]}"; do
        for prompt_type in "${prompt_types[@]}"; do
            python embed.py -b "$benchmark" -t "$embedding_model" --prompt_type="$prompt_type"
        done
    done
done