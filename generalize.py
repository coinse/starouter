from experiment import * 

def load_embeddings_with_labels(benchmark, strong_model_name, weak_model_name, embedding_model, prompt_type):
    def sort_embeddings(embeddings, pairwise_data):
        return [embeddings[elem['key']] for elem in pairwise_data]
    
    if prompt_type == 'prompt_with_response':
        short_name_for_weak_model = '-'.join(weak_model_name.split('-')[:-1])
        embeddings_path = f'./data/{benchmark.lower()}/route_data/{embedding_model}_{short_name_for_weak_model}_embeddings.pt'
    else:
        embeddings_path = f'./data/{benchmark.lower()}/route_data/{embedding_model}_{prompt_type}_embeddings.pt'
    embeddings = torch.load(embeddings_path)
    labels_path = f'./data/{benchmark.lower()}/route_data/pairwise/{strong_model_name}_vs_{weak_model_name}.json'
    with open(labels_path) as f:
        data = json.load(f)
        labels = [1 if item['winner'] == 'model_b' else 0 for item in data]
        keys = [item['key'] for item in data]

    return sort_embeddings(embeddings, data), labels, keys

def find_best_model_with_hyperparameter_preset_without_cross_validation(train_X, train_y, test_sets):
    if not isinstance(train_X, np.ndarray): train_X = np.array(train_X) 
    if not isinstance(train_y, np.ndarray): train_y = np.array(train_y)

    n_components = min(24, train_X.shape[1] - 1)
        
    pipelines = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components, random_state=42)),
            ('model', LogisticRegression(random_state=42, max_iter=2000, solver='liblinear', penalty='l1', C=0.1))
        ]),
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components, random_state=42)),
            ('model', SVC(probability=True, random_state=42, C=0.1, gamma='scale'))
        ]),
        'MLP': Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPClassifier(early_stopping=True, n_iter_no_change=10, random_state=42, hidden_layer_sizes=(32,), learning_rate_init=0.005, batch_size=16, max_iter=200))
        ]), 
        'Ensemble': VotingClassifier(
            estimators=[
                ('lr', Pipeline([
                    ('scaler', StandardScaler()),
                    ('pca', PCA(n_components=n_components, random_state=42)),
                    ('model', LogisticRegression(random_state=42, max_iter=2000, solver='liblinear', penalty='l1', C=0.1))
                ])),
                ('svm', Pipeline([
                    ('scaler', StandardScaler()),
                    ('pca', PCA(n_components=n_components, random_state=42)),
                    ('model', SVC(probability=True, random_state=42, C=0.1, gamma='scale'))
                ])),
                ('mlp', Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', MLPClassifier(early_stopping=True, n_iter_no_change=10, random_state=42, hidden_layer_sizes=(32,), learning_rate_init=0.005, batch_size=16, max_iter=200))
                ]))
            ],
            voting='soft',
        ),
        'RouteLLM': RouteLLMClassifier(random_state=42, dim=2, epochs=20, learning_rate=0.01, batch_size=8),
    }

    results = [[] for _ in range(len(test_sets))]
    print("\n--- Model Evaluation (Fixed Parameters) ---")
    for name, pipeline in pipelines.items():
        print(f"\n[+] Evaluating {name}...")
        
        pipeline.fit(train_X, train_y)
        for i, (test_X, test_y) in enumerate(test_sets):
            if not isinstance(test_X, np.ndarray): test_X = np.array(test_X) 
            if not isinstance(test_y, np.ndarray): test_y = np.array(test_y)
            test_probabilities = pipeline.predict_proba(test_X)
        
            results[i].append({
                'Model': name,
                'Best Params': dict(pipeline.named_steps['model'].get_params()) if hasattr(pipeline, 'named_steps') else dict(pipeline.get_params()),
                'CV Probabilities': test_probabilities
            })
        del pipeline
        gc.collect()
    
    pipelines.clear()
    del pipelines
    gc.collect()

    dataframes = list()
    for result in results:
        dataframes.append(
            pd.DataFrame(result)
        )
    del results
    gc.collect()
    
    return dataframes

def generalize_cross_benchamrks(benchmarks, prompt_type, strong_model_name):
    for benchmark in benchmarks:
        os.makedirs(f'./results/{benchmark.lower()}/benchmark_generalization/', exist_ok=True)
    
    for weak_model_name in WEAK_MODELS:
        embedding_model = DEFAULT_EMBEDDING_TYPE
        dataset = list()
        for benchmark in benchmarks:
            embeddings, labels, _ = load_embeddings_with_labels(benchmark, strong_model_name, weak_model_name, embedding_model, prompt_type)
            dataset.append((embeddings, labels))
            
        for i in range(len(benchmarks)):
            train_benchmark = benchmarks[i]

            if all([os.path.isfile(f'./results/{test_benchmark.lower()}/benchmark_generalization/{embedding_model}_from_{train_benchmark}_{strong_model_name}_{weak_model_name}_classifiers_{prompt_type}.pkl') for test_benchmark in benchmarks[:i] + benchmarks[i + 1:]]):
                print(f'Cross bench results for {embedding_model}/{strong_model_name}/{weak_model_name}/{train_benchmark} already exist!')
                continue

            train_X, train_y = dataset[i]
            dataframes = find_best_model_with_hyperparameter_preset_without_cross_validation(train_X, train_y, dataset[:i] + dataset[i + 1:])
            for test_benchmark, df in zip(benchmarks[:i] + benchmarks[i + 1:], dataframes):
                pkl_path = f'./results/{test_benchmark.lower()}/benchmark_generalization/{embedding_model}_from_{train_benchmark}_{strong_model_name}_{weak_model_name}_classifiers_{prompt_type}.pkl'
                pd.to_pickle(df, pkl_path)

        total_layers = total_number_of_layers_of(weak_model_name)
        layer = round(total_layers * 3 / 4) # DEFAULT_INTERNAL_STATE_TYPE
        dataset = list()
        for benchmark in benchmarks:
            internal_states, labels, _ = load_internal_states_with_layer(benchmark, strong_model_name, weak_model_name, layer, prompt_type)
            dataset.append((internal_states, labels))
            
        for i in range(len(benchmarks)):
            train_benchmark = benchmarks[i]

            if all([os.path.isfile(f'./results/{test_benchmark.lower()}/benchmark_generalization/internal_state_layer{layer}_from_{train_benchmark}_{strong_model_name}_{weak_model_name}_classifiers_{prompt_type}.pkl') for test_benchmark in benchmarks[:i] + benchmarks[i + 1:]]):
                print(f'Cross bench results for layer{layer}/{strong_model_name}/{weak_model_name}/{train_benchmark} already exist!')
                continue

            train_X, train_y = dataset[i]
            dataframes = find_best_model_with_hyperparameter_preset_without_cross_validation(train_X, train_y, dataset[:i] + dataset[i + 1:])
            for test_benchmark, df in zip(benchmarks[:i] + benchmarks[i + 1:], dataframes):
                pkl_path = f'./results/{test_benchmark.lower()}/benchmark_generalization/internal_state_layer{layer}_from_{train_benchmark}_{strong_model_name}_{weak_model_name}_classifiers_{prompt_type}.pkl'
                pd.to_pickle(df, pkl_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prompt_type', default='entire', choices=PROMPT_TYPES)
    args = parser.parse_args()

    benchmarks = TESTING_BENCHMARKS + CODE_GENERATION_BENCHMARKS
    prompt_type = args.prompt_type
    strong_model_name = 'gpt-4o'
    
    generalize_cross_benchamrks(benchmarks, prompt_type, strong_model_name)
