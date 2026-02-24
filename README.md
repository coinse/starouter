# STARouter: Internal State based LLM Router for Software Testing Tasks

Replication package for the paper "STARouter: Internal State based LLM Router for Software Testing Tasks"

### Experimental Results
We include notebooks that quickly walk you through our experimental results under `notebooks` directory. For each Research Question, refer to
- **RQ1. Effectiveness**: To what extent does our router approximate the optimal routing scenario? [notebook](./notebooks/RQ1_Effectiveness.ipynb)
- **RQ2. Generalizability**: Does our approach generalize across different contexts? [notebook](./notebooks/RQ2_Generalizability.ipynb)
- **Additional**: [Sensitivity analysis on Input Variations](./notebooks/Input_Variations.ipynb) and a [Router Optimality demo](./notebooks/Router_Optimality.ipynb).

> **Note**: We **bundle all extracted internal states, prompt embeddings, and final predictions** within this artifact. You can reproduce all tables and figures in the paper using the notebooks above **without** running any data processing or experiment scripts, but only by updating the `REPO_PATH` in `config.py`. 

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

### Detailed Instruction
To reproduce the experimental results, we recommend you to use `python>=3.9` - we specifically used `python=3.9.19`.  
All required modules are listed in `requirements.txt` along with their versions. You may download them via `pip install -r requirements.txt`.  

#### Data Processing
1. Label win model
   * Vanilla results of all models are included under `data/{BENCHMARK}/combined_results`.
   * Labelled data is stored under `data/{BENCHMARK}/route_data/pairwise` directory.
2. Extract internal state
   * Downloading open-source language models from HuggingFace requires your HF token along with proper access to each language model. 
   * Note that downloading all five SLMs may take up ~200GB of disk storage.
3. Embed prompts
   * Requires [Ollama](https://ollama.com/), with which `nomic-embed-text` model is pulled using `ollama pull nomic-embed-text` command. We assume that the Ollama endpoint is set to its default value (`endpoint = 'http://localhost:11434/api/embeddings'`), you may modify this value within `data/embed.py`
   * Requires OpenAI api key, which we load from the default environmental variable `OPENAI_API_KEY`.

For all three steps, we provide bash scripts(`data/*.sh`) that iterate over models/benchmarks along with their python scripts.
Currently, scripts for internal state extracting and embedding check the existence of corresponding `*.pt` files before executing - which they are - to avoid redundant processing cost.

#### Experiments
We provide a single script `run.sh` that executes all routing experiments along with the visualization of individual runs.  

```bash
python experiment.py --skip_hyperparameter_tuning -p entire
python experiment.py --skip_hyperparameter_tuning --code_generation -p entire

python visualize.py -p entire
python visualize.py --code_generation -p entire

python generalize.py -p entire
```
All above scripts also check the resulting file existence to avoid redundant executions.
1. Experiment routing (`experiment.py`)
   - As noted above, we recommend you to add `--skip_hyperparameter_tuning` option when running the experiments by yourself.
   - `--code_generation` flag tests routing on code generation benchmarks, if set to true. Otherwise, the default behavior is to experiment on testing benchmarks.
2. Visualize individual runs (`visualize.py`)
3. Conduct cross-benchmark generalization experiments (`generalize.py`)
   - By default, we iterate over all benchmarks including both testing and code generation tasks
   - For RQ2-1, where we analyze the transferability of router predictions, we do not need extra experiments, but only the resulting predictions. In turn, all such processing is done within the python notebook.