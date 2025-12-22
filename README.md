# STARouter: Internal State based LLM Router for Software Testing Tasks

Replication package for the paper "STARouter: Internal State based LLM Router for Software Testing Tasks"

### Experimental Results
We include notebooks that quickly walk you through our experimental results. For each Research Question, refer to
- **RQ1. Effectiveness**: To what extent does our router approximate the optimal routing scenario? [notebook](./notebooks/RQ1_Effectiveness.ipynb)
- **RQ2. Generalizability**: Does our approach generalize across different contexts? [notebook](./notebooks/RQ2_Generalizability.ipynb)
- **RQ3. Configurations**: How do variations in the input configuration affect router performance? [notebook](./notebooks/RQ3_Input_Variations.ipynb)  

We also provide a synthetic routing example about the definition of [Router Optimality](./notebooks/Router_Optimality.ipynb)

Cost-performance curves for individual runs (model pair/benchmark/input configurations) are stored under `results/{BENCHMARK}/preset` directories.

### Repository Structure

Under `data` directory, all scripts required to
1. Label win model: [construct_pairwise_data.py](./data/construct_pairwise_data.py)
2. Extract internal states from SLMs: [extract_internal_state.py](./data/extract_internal_state.py)
3. Embed prompts: [embed.py](./data/embed.py)
are included, along with resulting data files under each benchmark directory.

Note that we exclude benchmark implementations for brevity, please refer to the original implementations:
- LIBRO: https://github.com/coinse/libro
- TestEval: https://github.com/LLM4SoftwareTesting/TestEval
- APPS: https://github.com/hendrycks/apps
- HumanEval: https://github.com/openai/human-eval

Main scripts are presented in the repository, each containing
- [experiment.py](./experiment.py): Train and test routers based on internal states and prompt embeddings. Since hyperparameter tuning process takes long, we ran it on subset of tasks as a preliminary exploration and set the preset values. We recommend you to add `--skip_hyperparameter_tuning` option to speed up.
- [visualize.py](./visualize.py): Draw cost-performance(proportion of strong model calls) curve for individual runs. As hyperparameter tuning runs do not store resulting probabilities, by default, we plot results for experiments on preset.
- [generalize.py](./generalize.py): Test cross-benchmark generalization by training a router on one and test on all others.

Configuration and helper functions are included in
- [config.py](./config.py): main configuration file that contains list of models and benchmarks. **You must set the value of** `REPO_PATH` based on your machine, which is now set as `/root/starouter`.
- [data_utils.py](./data_utils.py) & [metric.py](./metric.py): Load model performance metrics on each benchmark and compute RO/CPT based on Trapezoidal rule.
